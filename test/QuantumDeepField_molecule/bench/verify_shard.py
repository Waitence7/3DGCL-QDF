#!/usr/bin/env python3
"""Sanity-check that ``MyDatasetShard`` returns numerically identical data to
``MyDataset`` for the same molecules. Used to catch shard format mistakes
before running a benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
QDF_ROOT = REPO_ROOT / "QuantumDeepField_molecule"
sys.path.insert(0, str(QDF_ROOT / "train"))

import train as qdf_train  # noqa: E402
from dataset_shard import MyDatasetShard, default_shard_path  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--basis-set", default="6-31G")
    parser.add_argument("--radius", default="0.75")
    parser.add_argument("--grid-interval", default="0.3")
    parser.add_argument("--n-check", type=int, default=20,
                        help="Number of molecules to verify (sampled across the split).")
    args = parser.parse_args()

    field = f"{args.basis_set}_{args.radius}sphere_{args.grid_interval}grid"
    npy_dir = QDF_ROOT / "dataset" / args.dataset / f"{args.split}_{field}"
    shard_path = default_shard_path(npy_dir)
    assert npy_dir.exists(), f"missing {npy_dir}"
    assert shard_path.exists(), f"missing {shard_path}"

    ds_npy = qdf_train.MyDataset(str(npy_dir))
    ds_shard = MyDatasetShard(shard_path)
    assert len(ds_npy) == len(ds_shard), f"length {len(ds_npy)} != {len(ds_shard)}"

    # ``MyDataset`` sorts files by mtime; the shard writer sorts by name.
    # Build a lookup by idx to align the two views.
    shard_index = {ds_shard[i][0]: i for i in range(len(ds_shard))}
    assert len(shard_index) == len(ds_shard), "Duplicate idx in shard"

    n_total = len(ds_npy)
    idxs = sorted(set(np.linspace(0, n_total - 1, args.n_check).astype(int).tolist()))
    print(f"Verifying {len(idxs)} / {n_total} molecules ({args.split} split)")

    mismatched = 0
    for i in idxs:
        rec_npy = ds_npy[i]
        idx_str = str(rec_npy[0])
        j = shard_index[idx_str]
        rec_shard = ds_shard[j]
        if len(rec_npy) != len(rec_shard):
            print(f"  [FAIL] {idx_str}: field count {len(rec_npy)} vs {len(rec_shard)}")
            mismatched += 1
            continue
        for k, (a, b) in enumerate(zip(rec_npy, rec_shard)):
            if isinstance(a, (str, int)):
                if a != b:
                    print(f"  [FAIL] {idx_str}: field {k} scalar mismatch {a!r} vs {b!r}")
                    mismatched += 1
                    break
            else:
                a_arr = np.asarray(a)
                b_arr = np.asarray(b)
                if a_arr.shape != b_arr.shape:
                    print(f"  [FAIL] {idx_str}: field {k} shape {a_arr.shape} vs {b_arr.shape}")
                    mismatched += 1
                    break
                if a_arr.dtype != b_arr.dtype:
                    print(f"  [FAIL] {idx_str}: field {k} dtype {a_arr.dtype} vs {b_arr.dtype}")
                    mismatched += 1
                    break
                if not np.array_equal(a_arr, b_arr):
                    diff = np.abs(a_arr.astype(np.float64) - b_arr.astype(np.float64)).max()
                    print(f"  [FAIL] {idx_str}: field {k} max abs diff = {diff}")
                    mismatched += 1
                    break
        else:
            print(f"  ok  {idx_str}  fields={len(rec_npy)}")

    if mismatched == 0:
        print(f"\nAll {len(idxs)} molecules match.")
        return 0
    print(f"\n{mismatched} mismatches.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
