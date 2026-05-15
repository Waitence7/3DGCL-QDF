#!/usr/bin/env python3
"""Per-slot QDF inference for every MMFF conformer in a DGCL dataset.

For each molecule in :class:`~dig.threedgraph.dataset.MoleculeNet` (e.g. ESOL)
we feed the four MMFF conformer coordinate slabs
(``data.max{1..4}pos_mmff``) plus the atomic numbers (``data.z``) into a
**pretrained QDF checkpoint** to obtain per-(smiles, slot) predictions.

Two prediction targets are supported via ``--qdf-property``:

``homolumo`` (legacy)
    Predicts (HOMO, LUMO) ⇒ CSV columns ``smiles,slot,homo,lumo``. Best paired
    with ``QM9under7atoms_homolumo_eV`` checkpoints (operation=mean, dim=200).
    Across MMFF rotamers of the same molecule QDF HOMO/LUMO usually differs
    by only ~1 meV which makes the resulting weights nearly uniform unless
    rescued by ``compute_mmff_weights.py --normalize zscore`` (default).

``atomization`` (recommended for ESOL)
    Predicts atomization energy [eV] ⇒ CSV columns ``smiles,slot,energy``.
    Best paired with ``QM9under14atoms_atomizationenergy_eV`` checkpoints
    (operation=sum, dim=250). Atomization energy is far more
    conformer-sensitive than HOMO/LUMO (the relaxed geometry sits in a
    deeper potential well), so per-slot weights become genuinely
    informative even before normalization.

Notes
-----
* QDF source files are not modified – we re-use ``preprocess.py`` helpers
  (``create_sphere`` / ``create_field`` / ``create_distancematrix`` /
  ``create_potential``) and ``train.QuantumDeepField`` from the existing
  ``QuantumDeepField_molecule`` tree.
* Molecules whose elements or orbital types are not present in the
  pretrained ``orbital_dict`` (e.g. P/S/Cl/Br/I outside QM9under7atoms) are
  silently skipped – downstream ``WeightedMMFFView(weight_mode='auto')``
  falls back to Boltzmann from MMFF energies for those rows.
* Conformer slots whose ``maxKpos_mmff`` is missing, empty, or contains
  NaN/Inf are also skipped.

Examples
--------

::

    cd c:\\DGCL\\3DGCL
    # default: atomization energy with the under14atoms checkpoint
    .venv\\Scripts\\python.exe examples\\sslgraph\\bench\\qdf_mmff_predict.py ^
        --dataset esol --batch-size 8
    # -> dataset/esol_qdf_mmff_preds_atomization.csv

    # legacy HOMO/LUMO
    .venv\\Scripts\\python.exe examples\\sslgraph\\bench\\qdf_mmff_predict.py ^
        --dataset esol --batch-size 8 --qdf-property homolumo
    # -> dataset/esol_qdf_mmff_preds_homolumo.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parents[2]
QDF_TRAIN = REPO / "QuantumDeepField_molecule" / "train"
for _p in (REPO, BENCH, QDF_TRAIN):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train as qdf_train  # noqa: E402
import preprocess as qdf_preprocess  # noqa: E402

from dig.threedgraph.dataset import MoleculeNet  # noqa: E402


SLOT_POS_ATTRS = {
    1: "max1pos_mmff",
    2: "max2pos_mmff",
    3: "max3pos_mmff",
    4: "max4pos_mmff",
}


def _parse_basis(basis_set: str) -> tuple[int, int]:
    """Replicate QDF preprocess parsing: ``'6-31G'`` -> (inner=6, outer=3+1=4)."""
    digits = basis_set[:-1].replace("-", "")  # strip trailing 'G' / 'g'
    nums = [int(b) for b in digits]
    if not nums:
        raise ValueError(f"unparseable basis set: {basis_set!r}")
    return nums[0], sum(nums[1:])


def _build_conformer_record(
    idx_str: str,
    z: torch.Tensor,
    pos: torch.Tensor,
    sphere: np.ndarray,
    orbital_dict: dict,
    inner: int,
    outer: int,
) -> list | None:
    """Build the 6-element QDF input record for one (atoms, coords) pair.

    Returns ``None`` when an element or orbital key is absent from
    ``orbital_dict`` (skip + fall back to Boltzmann downstream).
    """
    z_list = z.detach().cpu().tolist()
    p = pos.detach().cpu().to(torch.float64).numpy()
    if not np.isfinite(p).all():
        return None
    if p.shape[0] != len(z_list) or p.shape[1] != 3:
        return None

    atomic_numbers = []
    n_electrons = 0
    atomic_coords = []
    atomic_orbitals: list[int] = []
    orbital_coords = []
    quantum_numbers: list[int] = []

    all_atoms = qdf_preprocess.all_atoms
    for an, xyz in zip(z_list, p.tolist()):
        an = int(an)
        if not 1 <= an <= len(all_atoms):
            return None
        atom = all_atoms[an - 1]
        atomic_numbers.append([an])
        n_electrons += an
        atomic_coords.append(xyz)

        if an <= 2:
            aqs = [(atom + "1s" + str(i), 1) for i in range(outer)]
        else:
            aqs = (
                [(atom + "1s" + str(i), 1) for i in range(inner)]
                + [(atom + "2s" + str(i), 2) for i in range(outer)]
                + [(atom + "2p" + str(i), 2) for i in range(outer)]
            )
        for o_key, q in aqs:
            o_idx = orbital_dict.get(o_key)
            if o_idx is None:
                return None
            atomic_orbitals.append(int(o_idx))
            orbital_coords.append(xyz)
            quantum_numbers.append(q)

    atomic_coords_np = np.asarray(atomic_coords, dtype=np.float64)
    orbital_coords_np = np.asarray(orbital_coords, dtype=np.float64)
    atomic_numbers_np = np.asarray(atomic_numbers, dtype=np.int64)
    atomic_orbitals_np = np.asarray(atomic_orbitals, dtype=np.int64)
    quantum_numbers_np = np.asarray([quantum_numbers], dtype=np.float32)
    n_electrons_np = np.asarray([[n_electrons]], dtype=np.float32)

    field_coords = qdf_preprocess.create_field(sphere, atomic_coords_np)
    dm_orb = qdf_preprocess.create_distancematrix(field_coords, orbital_coords_np)
    dm_atom = qdf_preprocess.create_distancematrix(field_coords, atomic_coords_np)
    potential = qdf_preprocess.create_potential(dm_atom, atomic_numbers_np)
    n_field = int(field_coords.shape[0])

    return [
        idx_str,
        atomic_orbitals_np,
        dm_orb.astype(np.float32),
        quantum_numbers_np.astype(np.float32),
        n_electrons_np.astype(np.float32),
        n_field,
        # property/potential slots filled but never read in predict=True path
        np.zeros((1, 2), dtype=np.float32),
        potential.astype(np.float32).reshape(n_field, 1),
    ]


def _predict(model, records: list[list]) -> np.ndarray:
    """Run ``model.forward(data, predict=True)`` on a batch of records.

    Mirrors ``train.collate_fn``: data = list(zip(*records)). Returns a
    ``[B, N_output]`` numpy array.
    """
    data = list(zip(*records))
    with torch.no_grad():
        _ids, E_ = model.forward(tuple(data), predict=True)
    return E_.detach().cpu().numpy()


def _pick_device(name: str) -> torch.device:
    if name == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


# Default QDF setups per --qdf-property. Each maps to a known good checkpoint
# in ``QuantumDeepField_molecule/output/`` (clean, no Windows " - 복사본"
# artefacts) and the matching architecture hyperparameters.
_PROPERTY_PRESETS = {
    "homolumo": {
        "qdf_trained": "QM9under7atoms_homolumo_eV",
        "dim": 200,
        "hidden_hk": 200,
        "layer_hk": 3,
        "operation": "mean",
        "n_output": 2,
        "csv_header": ["smiles", "slot", "homo", "lumo"],
        "checkpoint": (
            "QuantumDeepField_molecule/output/"
            "model--QM9under7atoms_homolumo_eV--6-31G--radius0.75--"
            "grid_interval0.3--dim200--layer_functional3--hidden_HK200--"
            "layer_HK3--mean--batch_size8--lr1e-4--lr_decay0.5--step_size200--"
            "iteration2000"
        ),
    },
    "atomization": {
        "qdf_trained": "QM9under14atoms_atomizationenergy_eV",
        "dim": 250,
        "hidden_hk": 250,
        "layer_hk": 3,
        "operation": "sum",
        "n_output": 1,
        "csv_header": ["smiles", "slot", "energy"],
        "checkpoint": (
            "QuantumDeepField_molecule/output/"
            "model--QM9under14atoms_atomizationenergy_eV--6-31G--radius0.75--"
            "grid_interval0.3--dim250--layer_functional3--hidden_HK250--"
            "layer_HK3--sum--batch_size4--lr1e-4--lr_decay0.5--step_size200--"
            "iteration2000"
        ),
    },
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esol",
                    help="MoleculeNet dataset name (e.g. esol, freesolv, lipo)")
    ap.add_argument("--root", default="dataset/")
    ap.add_argument("--qdf-property", choices=tuple(_PROPERTY_PRESETS.keys()),
                    default="atomization",
                    help="Which QDF target to predict per MMFF slot. "
                         "'atomization' (default) uses the larger "
                         "QM9under14atoms checkpoint and is far more "
                         "conformer-sensitive than 'homolumo'.")
    ap.add_argument("--qdf-trained", default=None,
                    help="QDF dataset name used at pretraining; orbital_dict lives at "
                         "QuantumDeepField_molecule/dataset/<this>/orbitaldict_<basis>.pickle. "
                         "Defaults to the preset for --qdf-property.")
    ap.add_argument("--basis-set", default="6-31G")
    ap.add_argument("--radius", type=float, default=0.75)
    ap.add_argument("--grid-interval", type=float, default=0.3)
    ap.add_argument("--dim", type=int, default=None,
                    help="QDF feature dim (preset per --qdf-property).")
    ap.add_argument("--layer-functional", type=int, default=3)
    ap.add_argument("--hidden-hk", type=int, default=None,
                    help="QDF HK hidden dim (preset per --qdf-property).")
    ap.add_argument("--layer-hk", type=int, default=None,
                    help="QDF HK layer count (preset per --qdf-property).")
    ap.add_argument("--operation", default=None,
                    help="QDF readout op (preset per --qdf-property: "
                         "mean for homolumo, sum for atomization).")
    ap.add_argument("--checkpoint", default=None,
                    help="Path to QDF state_dict. Defaults to the preset "
                         "checkpoint for --qdf-property.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most this many molecules (0 = all)")
    ap.add_argument("--device", default="auto",
                    choices=("auto", "xpu", "cuda", "cpu"))
    ap.add_argument("--out", type=Path, default=None,
                    help="Output CSV path. Default: "
                         "dataset/<name>_qdf_mmff_preds_<property>.csv")
    ap.add_argument("--progress-every", type=int, default=200)
    args = ap.parse_args()

    preset = _PROPERTY_PRESETS[args.qdf_property]
    if args.qdf_trained is None:
        args.qdf_trained = preset["qdf_trained"]
    if args.dim is None:
        args.dim = preset["dim"]
    if args.hidden_hk is None:
        args.hidden_hk = preset["hidden_hk"]
    if args.layer_hk is None:
        args.layer_hk = preset["layer_hk"]
    if args.operation is None:
        args.operation = preset["operation"]
    if args.checkpoint is None:
        args.checkpoint = preset["checkpoint"]
    n_output = preset["n_output"]
    csv_header = preset["csv_header"]

    device = _pick_device(args.device)
    print(f"[qdf-mmff] property={args.qdf_property}  device={device}")
    print(f"[qdf-mmff] arch: dim={args.dim} hidden_hk={args.hidden_hk} "
          f"layer_hk={args.layer_hk} op={args.operation} N_out={n_output}")

    orbital_dict_path = (REPO / "QuantumDeepField_molecule" / "dataset" /
                         args.qdf_trained /
                         f"orbitaldict_{args.basis_set}.pickle")
    if not orbital_dict_path.exists():
        raise SystemExit(f"orbital_dict not found: {orbital_dict_path}")
    with orbital_dict_path.open("rb") as fh:
        orbital_dict = pickle.load(fh)
    N_orbitals = len(orbital_dict)
    print(f"[qdf-mmff] orbital_dict: {orbital_dict_path.name}  "
          f"(N_orbitals={N_orbitals})")

    inner, outer = _parse_basis(args.basis_set)
    sphere = qdf_preprocess.create_sphere(args.radius, args.grid_interval)
    print(f"[qdf-mmff] sphere points: {sphere.shape}  "
          f"(radius={args.radius}, grid={args.grid_interval})")

    model = qdf_train.QuantumDeepField(
        device, N_orbitals,
        args.dim, args.layer_functional, args.operation, n_output,
        args.hidden_hk, args.layer_hk,
    ).to(device)
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = (REPO / ckpt).resolve()
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    state = torch.load(str(ckpt), map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"[qdf-mmff] loaded checkpoint: {ckpt.name}")

    ds = MoleculeNet(root=args.root, name=args.dataset)
    N = len(ds) if not args.limit else min(args.limit, len(ds))
    print(f"[qdf-mmff] {args.dataset}: scanning {N}/{len(ds)} molecules x "
          f"4 slots = up to {N*4} forward passes")

    out_csv = args.out or (
        Path(args.root) /
        f"{args.dataset}_qdf_mmff_preds_{args.qdf_property}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    batch_meta: list[tuple[str, int]] = []
    batch_records: list[list] = []
    n_rows = 0
    n_skip = 0
    t0 = time.perf_counter()

    def _flush(writer) -> None:
        nonlocal n_rows
        if not batch_records:
            return
        E = _predict(model, batch_records)
        for (smi, slot), e in zip(batch_meta, E):
            if n_output == 1:
                writer.writerow([smi, slot, float(e[0])])
            else:
                writer.writerow([smi, slot, float(e[0]), float(e[1])])
            n_rows += 1
        batch_meta.clear()
        batch_records.clear()

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(csv_header)

        for i in range(N):
            data = ds[i]
            smi = getattr(data, "smiles", None)
            z = getattr(data, "z", None)
            if smi is None or z is None:
                n_skip += 4
                continue

            for slot in (1, 2, 3, 4):
                pos = getattr(data, SLOT_POS_ATTRS[slot], None)
                if pos is None or pos.numel() == 0:
                    n_skip += 1
                    continue
                rec = _build_conformer_record(
                    f"{i}_{slot}", z, pos, sphere,
                    orbital_dict, inner, outer,
                )
                if rec is None:
                    n_skip += 1
                    continue
                batch_meta.append((str(smi), slot))
                batch_records.append(rec)

                if len(batch_records) >= args.batch_size:
                    _flush(writer)

            if args.progress_every and (i + 1) % args.progress_every == 0:
                dt = time.perf_counter() - t0
                eta = dt / max(i + 1, 1) * max(N - i - 1, 0)
                print(f"  [{i + 1}/{N}]  rows={n_rows}  skipped={n_skip}  "
                      f"elapsed={dt:.1f}s  eta={eta:.1f}s")

        _flush(writer)

    dt = time.perf_counter() - t0
    print(f"[qdf-mmff] done. rows={n_rows}  skipped(slots)={n_skip}  "
          f"elapsed={dt:.1f}s")
    print(f"[saved] {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
