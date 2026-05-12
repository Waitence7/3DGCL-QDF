#!/usr/bin/env python3
"""Convert a preprocessed ``.npy`` dataset (one file per molecule, as produced
by ``train/preprocess.py``) into a single binary shard usable by
``train/dataset_shard.MyDatasetShard``.

The original ``.npy`` directory is left untouched.

Example (from repo root)::

    .\.venv\\Scripts\\python.exe QuantumDeepField_molecule\\bench\\convert_to_shard.py \
        --dataset QM9under14atoms_atomizationenergy_eV \
        --basis-set 6-31G --radius 0.75 --grid-interval 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QDF_ROOT = REPO_ROOT / "QuantumDeepField_molecule"
sys.path.insert(0, str(QDF_ROOT / "train"))

from dataset_shard import default_shard_path, write_shard  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Dataset folder name under QuantumDeepField_molecule/dataset/")
    parser.add_argument("--basis-set", default="6-31G")
    parser.add_argument("--radius", default="0.75")
    parser.add_argument("--grid-interval", default="0.3")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    field = f"{args.basis_set}_{args.radius}sphere_{args.grid_interval}grid"
    dataset_dir = QDF_ROOT / "dataset" / args.dataset

    summaries = []
    for split in args.splits:
        npy_dir = dataset_dir / f"{split}_{field}"
        if not npy_dir.exists():
            print(f"[skip] {npy_dir} not found (run preprocess.py first?)")
            continue
        shard_path = default_shard_path(npy_dir)
        if shard_path.exists() and not args.overwrite:
            print(f"[skip] {shard_path.name} already exists (use --overwrite to rebuild)")
            continue
        print(f"[write] {npy_dir.name} -> {shard_path.name}")
        info = write_shard(npy_dir, shard_path)
        size_mb = info["bytes_written"] / (1024 * 1024)
        print(f"        {info['n_molecules']} molecules, "
              f"{size_mb:.1f} MiB, "
              f"{info['elapsed_sec']:.1f} sec "
              f"(has_property={info['has_property']}, n_output={info['n_output']})")
        summaries.append((split, info))

    if not summaries:
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
