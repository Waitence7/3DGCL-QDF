#!/usr/bin/env python3
"""Compare the original NumPy/SciPy preprocess geometry path against the Rust
``qdf_io`` implementation on the same molecules.

This is meant as a correctness smoke test before trusting ``--backend rust``.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
QDF_ROOT = REPO_ROOT / "QuantumDeepField_molecule"
sys.path.insert(0, str(QDF_ROOT / "train"))

import preprocess as pp  # noqa: E402


def check_parse_matches(blocks: list[str], basis_set: str, has_property: bool) -> int:
    """Rust ``parse_molecule_block_rust`` + ``create_orbitals`` vs pure Python parse."""
    mismatches = 0
    for i, block in enumerate(blocks):
        od_py = defaultdict(lambda: len(od_py))
        os.environ["QDF_FORCE_PYTHON_PARSE"] = "1"
        try:
            m_py = pp._parse_molecule_block(block, basis_set, od_py, property=has_property)
        finally:
            os.environ.pop("QDF_FORCE_PYTHON_PARSE", None)

        od_rs = defaultdict(lambda: len(od_rs))
        m_rs = pp._parse_molecule_block(block, basis_set, od_rs, property=has_property)

        if m_py["idx"] != m_rs["idx"]:
            print(f"[FAIL-parse] idx py={m_py['idx']!r} rs={m_rs['idx']!r}")
            mismatches += 1
            continue
        if not np.allclose(m_py["atomic_coords"], m_rs["atomic_coords"], rtol=0, atol=0):
            print(f"[FAIL-parse] {m_py['idx']}: atomic_coords mismatch")
            mismatches += 1
            continue
        if not np.allclose(m_py["orbital_coords"], m_rs["orbital_coords"], rtol=0, atol=0):
            print(f"[FAIL-parse] {m_py['idx']}: orbital_coords mismatch")
            mismatches += 1
            continue
        if not np.array_equal(m_py["atomic_numbers"], m_rs["atomic_numbers"]):
            print(f"[FAIL-parse] {m_py['idx']}: atomic_numbers mismatch")
            mismatches += 1
            continue
        if not np.array_equal(m_py["atomic_orbitals"], m_rs["atomic_orbitals"]):
            print(f"[FAIL-parse] {m_py['idx']}: atomic_orbitals mismatch")
            mismatches += 1
            continue
        if not np.allclose(m_py["quantum_numbers"], m_rs["quantum_numbers"], rtol=0, atol=0):
            print(f"[FAIL-parse] {m_py['idx']}: quantum_numbers mismatch")
            mismatches += 1
            continue
        if not np.allclose(m_py["N_electrons"], m_rs["N_electrons"], rtol=0, atol=0):
            print(f"[FAIL-parse] {m_py['idx']}: N_electrons mismatch")
            mismatches += 1
            continue
        if has_property:
            pv_py = m_py["property_values"]
            pv_rs = m_rs["property_values"]
            if pv_py is None or pv_rs is None:
                print(f"[FAIL-parse] {m_py['idx']}: property_values None mismatch")
                mismatches += 1
                continue
            if not np.allclose(pv_py, pv_rs, rtol=0, atol=0):
                print(f"[FAIL-parse] {m_py['idx']}: property_values mismatch")
                mismatches += 1
                continue
        elif m_py["property_values"] != m_rs["property_values"]:
            print(f"[FAIL-parse] {m_py['idx']}: property_values mismatch (no prop)")
            mismatches += 1
            continue

        if (i + 1) % 50 == 0 or i == 0:
            print(f"parse ok [{i+1}/{len(blocks)}] {m_py['idx']}")

    if mismatches:
        print(f"\n{mismatches} parse mismatches out of {len(blocks)} molecules.")
        return 1
    print(f"\nAll {len(blocks)} molecules: Rust parse matches Python.")
    return 0


def load_dataset_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    return text.split("\n\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="QM9under14atoms_atomizationenergy_eV")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--basis-set", default="6-31G")
    parser.add_argument("--radius", type=float, default=0.75)
    parser.add_argument("--grid-interval", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--check-parse",
        action="store_true",
        help="Compare Rust text parser vs pure-Python _parse_molecule_block (same orbital_dict per mol).",
    )
    args = parser.parse_args()

    try:
        import qdf_io  # noqa: WPS433
    except Exception as e:  # pragma: no cover
        print("ERROR: qdf_io is not installed/built:", e)
        return 2

    dataset_dir = QDF_ROOT / "dataset" / args.dataset
    src_path = dataset_dir / f"{args.split}.txt"
    if not src_path.exists():
        print(f"ERROR: missing {src_path}")
        return 1

    blocks = load_dataset_blocks(src_path)
    if args.limit:
        blocks = blocks[: args.limit]

    has_property = args.split in {"train", "val", "test"} and args.dataset.endswith("_eV")
    orbital_dict = defaultdict(lambda: len(orbital_dict))

    if args.check_parse:
        return check_parse_matches(blocks, args.basis_set, has_property)

    sphere = pp.create_sphere(args.radius, args.grid_interval)
    sphere64 = np.ascontiguousarray(sphere, dtype=np.float64)

    mismatches = 0
    for i, block in enumerate(blocks):
        mol = pp._parse_molecule_block(block, args.basis_set, orbital_dict, property=has_property)

        atomic_coords = mol["atomic_coords"]
        orbital_coords = mol["orbital_coords"]
        atomic_numbers = mol["atomic_numbers"]

        field = pp.create_field(sphere, atomic_coords)
        dm_a = pp.create_distancematrix(field, atomic_coords)
        pot_np = pp.create_potential(dm_a, atomic_numbers)
        dm_o_np = pp.create_distancematrix(field, orbital_coords)

        dm_o_rs, pot_rs, n_field_rs = qdf_io.preprocess_molecule_rust(
            atomic_coords, orbital_coords, atomic_numbers, sphere64
        )

        dm_o_rs = np.asarray(dm_o_rs, dtype=np.float32)
        pot_rs = np.asarray(pot_rs, dtype=np.float32)

        if int(n_field_rs) != int(len(field)):
            print(f"[FAIL] {mol['idx']}: n_field rust={n_field_rs} numpy={len(field)}")
            mismatches += 1
            continue

        if not np.allclose(dm_o_np.astype(np.float32), dm_o_rs, rtol=args.rtol, atol=args.atol):
            diff = np.max(np.abs(dm_o_np.astype(np.float32) - dm_o_rs))
            print(f"[FAIL] {mol['idx']}: distance_matrix max abs diff = {diff}")
            mismatches += 1
            continue

        if has_property:
            if not np.allclose(pot_np.astype(np.float32), pot_rs, rtol=args.rtol, atol=args.atol):
                diff = np.max(np.abs(pot_np.astype(np.float32) - pot_rs))
                print(f"[FAIL] {mol['idx']}: potential max abs diff = {diff}")
                mismatches += 1
                continue

        if (i + 1) % 50 == 0 or i == 0:
            print(f"ok [{i+1}/{len(blocks)}] {mol['idx']}")

    if mismatches:
        print(f"\n{mismatches} mismatches out of {len(blocks)} molecules.")
        return 1

    print(f"\nAll {len(blocks)} molecules match within rtol={args.rtol}, atol={args.atol}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
