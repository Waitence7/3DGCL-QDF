#!/usr/bin/env python3
"""Compare the original NumPy/SciPy preprocess geometry path against the Rust
``qdf_io`` implementation on the same molecules.

This is meant as a correctness smoke test before trusting ``--backend rust``.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
QDF_ROOT = REPO_ROOT / "QuantumDeepField_molecule"
sys.path.insert(0, str(QDF_ROOT / "train"))

import preprocess as pp  # noqa: E402


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
