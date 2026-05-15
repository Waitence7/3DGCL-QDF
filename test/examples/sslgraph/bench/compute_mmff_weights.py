#!/usr/bin/env python3
"""Pre-compute per-molecule, per-MMFF-slot weights for ``MMFFweighted`` view.

Writes a ``.pt`` file containing ``{smiles: tensor[K]}`` that downstream code
(e.g. ``apply_mmff_weights``) attaches to PyG ``Data`` objects as
``data.mmff_weights``. ``WeightedMMFFView`` then samples each graph's MMFF
conformer slot using this categorical distribution.

Sources
-------

``--source boltzmann`` (default, **no extra model needed**)
    For each molecule reads ``max{1..K}_energy`` (already in MoleculeNet) and
    writes Boltzmann weights ``w_k ∝ exp(-(E_k - min E) / kT)``. Higher
    stability (lower MMFF energy) ⇒ higher selection probability.

``--source qdf --pred-csv <path>``
    Reads a CSV produced by ``qdf_mmff_predict.py``. For molecules whose
    SMILES isn't in the CSV (typically because the QDF ``orbital_dict``
    doesn't cover their elements — QM9 checkpoints support only H/C/N/O/F,
    so ESOL's P/S/Cl/Br/I species are skipped) you can choose how to fill
    the gap with ``--fallback`` (default ``boltzmann``). The CSV's columns
    automatically select between two property modes:

    * ``smiles,slot,homo,lumo``   (legacy HOMO/LUMO checkpoint)
        score uses any python expression over ``(homo, lumo, gap)``.
        Default ``--score-expr "-(lumo - homo)"`` (larger HOMO-LUMO gap
        → higher weight). On RDKit MMFF rotamers QDF predictions only
        differ by ~1 meV, so always pair with ``--normalize zscore``.

    * ``smiles,slot,energy``      (atomization-energy checkpoint, **recommended**)
        score uses any python expression over ``(energy,)``.
        Default ``--score-expr "-energy"`` because QDF outputs a signed
        total-energy-difference where *more negative* values correspond
        to *more stable* conformers. Atomization energy is far more
        conformer-sensitive than HOMO/LUMO (spread ~0.3 eV vs ~1 meV),
        but ``--normalize zscore`` is still the safer default.

    Example score expressions::

        # homolumo CSV
        --score-expr "-(lumo - homo)"          # gap (default)
        --score-expr "-abs(homo + 0.27)"       # closeness to target HOMO
        # atomization CSV
        --score-expr "-energy"                 # more negative = more stable (default)
        --score-expr "-abs(energy + 100)"      # closeness to target energy

Output
------

* ``dataset/<name>_mmff_weights_<source>.pt`` (or path from ``--out``)
* Optional ``.npz`` mirror with ``smiles, weights[N,K]`` arrays (``--also-npz``)

The file is consumed by ``apply_mmff_weights(dataset, path)`` (see
``dig.sslgraph.method.contrastive.views_fn.mmff_weights_io``) before the
DataLoader is built. The view itself stays unaware of how weights were
produced – it just reads ``batch.mmff_weights``.

Example
-------

::

    cd c:\\DGCL\\3DGCL
    .venv\\Scripts\\python.exe examples\\sslgraph\\bench\\compute_mmff_weights.py \\
        --dataset esol --source boltzmann --kT 0.5
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parents[2]
for _p in (REPO, BENCH):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dig.threedgraph.dataset import MoleculeNet

# Recognised CSV schemas produced by ``qdf_mmff_predict.py``. Selected
# automatically from the CSV header (no extra CLI flag).
QDF_CSV_SCHEMAS: dict[str, dict] = {
    "homolumo": {
        "required": {"smiles", "slot", "homo", "lumo"},
        "value_cols": ("homo", "lumo"),
        "default_score_expr": "-(lumo - homo)",
        "score_namespace": ("homo", "lumo", "gap"),
    },
    "atomization": {
        "required": {"smiles", "slot", "energy"},
        "value_cols": ("energy",),
        # QDF outputs a signed total-energy-difference where *more negative*
        # values correspond to *more stable* conformers (lower potential well).
        # Therefore the default score is ``-energy`` so the most negative slot
        # gets the highest weight after softmax.
        "default_score_expr": "-energy",
        "score_namespace": ("energy",),
    },
}

SLOT_KEYS = ["MMFF1", "MMFF2", "MMFF3", "MMFF4"]


def _energies(data) -> list[float]:
    out = []
    for k in (1, 2, 3, 4):
        v = getattr(data, f"max{k}_energy", None)
        if v is None:
            return []
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().float().item()
        out.append(float(v))
    return out


def boltzmann_weights(dataset, kT: float) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    skipped = 0
    for data in dataset:
        e = _energies(data)
        if not e:
            skipped += 1
            continue
        e_t = torch.tensor(e, dtype=torch.float32)
        e_t = e_t - e_t.min()
        w = torch.softmax(-e_t / max(kT, 1e-6), dim=0)
        smi = getattr(data, "smiles", None)
        if smi is None:
            skipped += 1
            continue
        out[str(smi).strip()] = w
    print(f"[boltzmann] populated {len(out)} smiles, skipped {skipped}")
    return out


_ALLOWED_SCORE_NAMES = {"abs": abs, "min": min, "max": max, "pow": pow,
                        "exp": math.exp, "log": math.log, "sqrt": math.sqrt}


def _detect_qdf_schema(fieldnames: list[str]) -> tuple[str, dict]:
    """Pick the QDF CSV schema from the header columns. Order matters:
    homolumo requires both ``homo`` and ``lumo`` so it cannot be confused
    with the simpler ``energy``-only atomization schema."""
    cols = set(fieldnames or [])
    for prop, spec in QDF_CSV_SCHEMAS.items():
        if spec["required"].issubset(cols):
            return prop, spec
    raise SystemExit(
        f"qdf CSV must contain one of the recognised schemas; got {fieldnames!r}. "
        f"Recognised: {[sorted(s['required']) for s in QDF_CSV_SCHEMAS.values()]}"
    )


def _safe_eval(expr: str, namespace: dict[str, float],
               allowed: tuple[str, ...]) -> float:
    code = compile(expr, "<score-expr>", "eval")
    for name in code.co_names:
        if name not in allowed and name not in _ALLOWED_SCORE_NAMES:
            raise ValueError(f"score-expr uses disallowed name {name!r}")
    return float(eval(code, {"__builtins__": {}, **_ALLOWED_SCORE_NAMES},
                      namespace))


def qdf_weights(
    csv_path: Path,
    kT: float,
    score_expr: str | None,
    normalize: str = "zscore",
) -> tuple[dict[str, torch.Tensor], str, str]:
    """Build per-molecule slot weights from a QDF prediction CSV.

    The CSV's columns (``homo,lumo`` vs ``energy``) auto-select between
    HOMO/LUMO and atomization-energy schemas. ``score_expr=None`` falls
    back to the schema's default (``-(lumo - homo)`` for homolumo,
    ``energy`` for atomization).

    Returns ``(weights, property, effective_score_expr)``.

    QDF outputs on RDKit MMFF conformers of the **same** molecule are usually
    only millivolts apart (the model is trained on QM9 ground-state geometries
    so it's nearly conformer-invariant). A naive ``softmax(score / kT)`` with
    kT≈1 then produces ~uniform weights.

    The ``normalize`` knob rescues information from those tiny but
    *reproducible* per-slot differences:

    * ``"none"``   : original behaviour ``softmax(score / kT)``.
    * ``"zscore"`` : standardize each molecule's 4 scores to zero mean / unit
                     std *before* the softmax. Recommended for QDF.
    * ``"rank"``   : replace scores by their per-molecule rank ``0..K-1``,
                     so the resulting distribution depends only on slot
                     ordering (deterministic spread, independent of QDF noise
                     magnitude).
    """
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        prop, spec = _detect_qdf_schema(list(reader.fieldnames or []))
        value_cols: tuple[str, ...] = spec["value_cols"]
        if score_expr is None:
            score_expr = spec["default_score_expr"]
        allowed_names = spec["score_namespace"]
        rows: dict[str, dict[int, tuple[float, ...]]] = {}
        for row in reader:
            smi = row["smiles"].strip()
            slot = int(row["slot"])
            if slot not in (1, 2, 3, 4):
                continue
            rows.setdefault(smi, {})[slot] = tuple(
                float(row[c]) for c in value_cols
            )

    if normalize not in ("none", "zscore", "rank"):
        raise SystemExit(f"--normalize must be one of none|zscore|rank; got {normalize!r}")

    out: dict[str, torch.Tensor] = {}
    skipped = 0
    n_constant = 0
    for smi, slots in rows.items():
        if set(slots.keys()) != {1, 2, 3, 4}:
            skipped += 1
            continue
        scores = []
        for k in (1, 2, 3, 4):
            vals = slots[k]
            ns: dict[str, float] = dict(zip(value_cols, vals))
            if prop == "homolumo":
                ns["gap"] = ns["lumo"] - ns["homo"]
            scores.append(_safe_eval(score_expr, ns, allowed_names))
        s = torch.tensor(scores, dtype=torch.float32)

        if normalize == "zscore":
            std = float(s.std(unbiased=False))
            if std < 1e-9:
                # All 4 QDF predictions identical — keep uniform so downstream
                # code falls back to either Boltzmann (weight_mode='auto') or
                # treats all slots equally.
                w = torch.full_like(s, 1.0 / s.numel())
                n_constant += 1
            else:
                z = (s - s.mean()) / std
                w = torch.softmax(z / max(kT, 1e-6), dim=0)
        elif normalize == "rank":
            ranks = torch.argsort(torch.argsort(s)).float()  # 0..K-1
            w = torch.softmax(ranks / max(kT, 1e-6), dim=0)
        else:
            w = torch.softmax(s / max(kT, 1e-6), dim=0)
        out[smi] = w
    print(
        f"[qdf:{prop}] populated {len(out)} smiles, skipped {skipped} "
        f"(score={score_expr!r}, kT={kT}, normalize={normalize}, "
        f"constant_score_molecules={n_constant})"
    )
    return out, prop, score_expr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esol")
    ap.add_argument("--root", default="dataset/")
    ap.add_argument("--source", choices=["boltzmann", "qdf"], default="boltzmann")
    ap.add_argument("--kT", type=float, default=1.0,
                    help="Boltzmann temperature (also used as softmax temperature for QDF)")
    ap.add_argument("--pred-csv", type=Path, default=None,
                    help="QDF HOMO/LUMO prediction CSV (required for --source qdf); "
                         "must be from a homolumo checkpoint, not atomization.")
    ap.add_argument(
        "--score-expr", default=None,
        help="Python expression producing the per-slot score; passed to "
             "softmax. Variables depend on the QDF CSV schema: (homo, lumo, "
             "gap) for homolumo CSVs, (energy,) for atomization CSVs. "
             "Default per schema: '-(lumo - homo)' (larger gap = higher) "
             "for homolumo, 'energy' (higher = more stable) for atomization."
    )
    ap.add_argument(
        "--normalize", choices=("none", "zscore", "rank"), default="zscore",
        help="Per-molecule rescaling of the QDF score before softmax. "
             "QDF outputs differ by only ~1 meV across MMFF rotamers of the "
             "same molecule, so the default 'zscore' standardises each "
             "molecule's 4 scores before softmax — preserves relative slot "
             "ranking irrespective of magnitude. 'none' = legacy behaviour. "
             "'rank' = score replaced by 0..K-1 rank (deterministic spread)."
    )
    ap.add_argument(
        "--fallback", choices=("boltzmann", "uniform", "skip"), default="boltzmann",
        help="What to do for dataset molecules with no QDF prediction. "
             "QM9-trained QDF checkpoints only cover H/C/N/O/F, so ESOL's "
             "P/S/Cl/Br/I species are silently dropped. 'boltzmann' (default) "
             "fills them with MMFF-energy Boltzmann weights — covers 100%% "
             "of molecules. 'uniform' assigns 0.25 each (identical to side A "
             "behaviour). 'skip' is the legacy behaviour: those molecules are "
             "left out of the .pt and WeightedMMFFView falls back to uniform."
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--also-npz", action="store_true")
    args = ap.parse_args()

    out_path = args.out or (
        Path(args.root) / f"{args.dataset}_mmff_weights_{args.source}.pt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    qdf_property: str | None = None
    score_expr_used: str | None = None
    fallback_meta: dict | None = None
    if args.source == "boltzmann":
        dataset = MoleculeNet(root=args.root, name=args.dataset)
        weights = boltzmann_weights(dataset, kT=args.kT)
    else:
        if args.pred_csv is None:
            raise SystemExit("--pred-csv is required for --source qdf")
        weights, qdf_property, score_expr_used = qdf_weights(
            args.pred_csv, kT=args.kT, score_expr=args.score_expr,
            normalize=args.normalize,
        )
        # Coverage fix: QDF checkpoints trained on QM9 (H/C/N/O/F) silently
        # skip molecules with other elements (P/S/Cl/Br/I in ESOL, ~46% of
        # the set). Without this, WeightedMMFFView falls back to uniform for
        # those rows, which is identical to side A and dilutes the QDF
        # signal. ``--fallback boltzmann`` covers them with MMFF-energy
        # weights so all 1128 molecules get a non-uniform distribution.
        if args.fallback != "skip":
            dataset = MoleculeNet(root=args.root, name=args.dataset)
            n_added = 0
            n_total = 0
            n_uniform = 0
            for data in dataset:
                smi = getattr(data, "smiles", None)
                if smi is None:
                    continue
                key = str(smi).strip()
                n_total += 1
                if key in weights:
                    continue
                if args.fallback == "boltzmann":
                    e = _energies(data)
                    if e:
                        e_t = torch.tensor(e, dtype=torch.float32)
                        e_t = e_t - e_t.min()
                        weights[key] = torch.softmax(
                            -e_t / max(args.kT, 1e-6), dim=0
                        )
                        n_added += 1
                        continue
                if args.fallback == "uniform" or (
                    args.fallback == "boltzmann" and not e
                ):
                    weights[key] = torch.full((4,), 0.25, dtype=torch.float32)
                    n_uniform += 1
            fallback_meta = {
                "policy": args.fallback,
                "n_added": int(n_added),
                "n_uniform": int(n_uniform),
                "n_total": int(n_total),
            }
            print(
                f"[fallback:{args.fallback}] +{n_added} boltzmann, "
                f"+{n_uniform} uniform; final weights={len(weights)} "
                f"(dataset has {n_total} molecules)"
            )

    payload = {
        "source": args.source,
        "kT": args.kT,
        "score_expr": score_expr_used,
        "normalize": args.normalize if args.source == "qdf" else None,
        "qdf_property": qdf_property,
        "pred_csv": str(args.pred_csv) if args.pred_csv is not None else None,
        "fallback": fallback_meta,
        "slots": SLOT_KEYS,
        "weights": weights,
    }
    torch.save(payload, out_path)
    print(f"[saved] {out_path}  (n={len(weights)}, property={qdf_property})")

    if args.also_npz and weights:
        import numpy as np
        smiles = list(weights.keys())
        W = torch.stack([weights[s] for s in smiles], dim=0).numpy()
        npz_path = out_path.with_suffix(".npz")
        np.savez_compressed(npz_path, smiles=np.array(smiles, dtype=object), weights=W)
        print(f"[saved] {npz_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
