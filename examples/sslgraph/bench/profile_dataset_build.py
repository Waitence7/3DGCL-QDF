#!/usr/bin/env python3
"""Benchmark loading + simple iteration for ``MoleculeNet`` vs
``MoleculeNetShard``.

Assumes a processed ``data.pt`` already exists (run the upstream ``MoleculeNet``
once first). Will build ``data.shard`` on the fly if missing.

Example::

    .\\.venv\\Scripts\\python.exe examples\\sslgraph\\bench\\profile_dataset_build.py \\
        --name esol --root dataset/ --iters 2
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="esol")
    parser.add_argument("--root", default="dataset/")
    parser.add_argument(
        "--iters",
        type=int,
        default=1,
        help="Re-run the load+iterate measurement this many times.",
    )
    parser.add_argument(
        "--build-shard",
        action="store_true",
        help="If set, (re)build data.shard from the processed data.pt first.",
    )
    args = parser.parse_args()

    from dig.threedgraph.dataset import (
        MoleculeNet,
        MoleculeNetShard,
        convert_inmemory_to_shard,
        default_shard_path,
    )

    shard_path = default_shard_path(args.root, args.name)
    if args.build_shard or not shard_path.is_file():
        print(f"[build] writing shard {shard_path}")
        src = MoleculeNet(root=args.root, name=args.name)
        info = convert_inmemory_to_shard(src, shard_path, progress=False)
        print(f"[build] wrote n={info['n_written']} in {info['bytes'] / 1024:.1f} KiB")
        del src
        gc.collect()

    print(f"[bench] dataset={args.name} iters={args.iters}")
    for it in range(args.iters):
        gc.collect()
        t0 = time.perf_counter()
        d_pt = MoleculeNet(root=args.root, name=args.name)
        t_load_pt = time.perf_counter() - t0
        n_pt = len(d_pt)
        t0 = time.perf_counter()
        for i in range(n_pt):
            _ = d_pt[i]
        t_iter_pt = time.perf_counter() - t0

        del d_pt
        gc.collect()

        t0 = time.perf_counter()
        d_sh = MoleculeNetShard(root=args.root, name=args.name)
        t_load_sh = time.perf_counter() - t0
        n_sh = len(d_sh)
        t0 = time.perf_counter()
        for i in range(n_sh):
            _ = d_sh[i]
        t_iter_sh = time.perf_counter() - t0
        del d_sh
        gc.collect()

        print(
            f"  iter {it}: "
            f"pt(load={1000*t_load_pt:6.1f}ms iter={1000*t_iter_pt:6.1f}ms n={n_pt})  "
            f"shard(load={1000*t_load_sh:6.1f}ms iter={1000*t_iter_sh:6.1f}ms n={n_sh})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
