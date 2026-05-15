#!/usr/bin/env python3
"""Build a ``data.shard`` next to an already-processed ``data.pt`` for a
DGCL molecular dataset.

The original ``data.pt`` is left in place; this script only adds a sibling
``processed/data.shard`` for :class:`dig.threedgraph.dataset.MoleculeNetShard`
to mmap. Re-run after each ``MoleculeNet.process`` to refresh.

Example::

    .\\.venv\\Scripts\\python.exe examples\\sslgraph\\convert_dataset_to_shard.py \
        --name esol --root dataset/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", required=True,
        help="dataset key recognised by MoleculeNet (esol, freesolv, lipo, hiv, ...).",
    )
    parser.add_argument("--root", default="dataset/")
    parser.add_argument(
        "--dataset-kind",
        choices=["moleculenet", "qm"],
        default="moleculenet",
        help="Which DGCL dataset class to wrap. Currently only MoleculeNet/QM share the "
             "PyG Data fields the shard knows about.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override shard output path (default: {root}/{name}/processed/data.shard).",
    )
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    import dig_io  # noqa: F401  (will raise with a clear message if not built)

    from dig.threedgraph.dataset import (
        MoleculeNet,
        QM,
        convert_inmemory_to_shard,
        default_shard_path,
    )

    if args.dataset_kind == "moleculenet":
        ds = MoleculeNet(root=args.root, name=args.name)
    else:
        ds = QM(root=args.root, name=args.name)

    out_path = Path(args.out) if args.out else default_shard_path(args.root, args.name)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"src dataset: {ds!r} (n={len(ds)})")
    print(f"writing shard: {out_path}")
    t0 = time.perf_counter()
    info = convert_inmemory_to_shard(ds, out_path, progress=not args.no_progress)
    elapsed = time.perf_counter() - t0
    print(
        f"done in {elapsed:.2f}s — wrote={info['n_written']}, skipped={info['n_skipped']}, "
        f"size={info['bytes'] / 1024:.1f} KiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
