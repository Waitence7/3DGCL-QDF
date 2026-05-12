#!/usr/bin/env python3
"""Decompose one pretraining epoch into dataloader / views_fn / forward / backward / step
timings, so we can see *where* the wall clock actually goes before reaching for Rust.

Run from repo root, e.g.::

    .\\.venv\\Scripts\\python.exe examples\\sslgraph\\bench\\profile_pretrain.py \\
        --dataset esol --epochs 2 --warmup 1

The script mirrors the configuration used in ``examples/sslgraph/pretrain.ipynb``
(``GraphCL`` + SchNet + ``MMFFrandom`` augmentation) but stops the optimiser after
a few iterations so we get an actionable timing breakdown in <1 minute.

Env switches honoured (so the same script can validate later patches):
    DATALOADER_NUM_WORKERS, PIN_MEMORY, PRETRAIN_AMP=bf16|fp16, MMFFRANDOM_FAST=1

For A/B wall + CPU + XPU + plots, use ``compare_pretrain_ab.py`` (does not alter
your shell environment between runs).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BENCH = Path(__file__).resolve().parent
for _p in (REPO, BENCH):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pretrain_bench_core import run_pretrain_benchmark  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="esol")
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=1, help="measured epochs (after warmup)")
    parser.add_argument("--warmup", type=int, default=1, help="warmup epochs not timed")
    parser.add_argument("--root", default="dataset/")
    parser.add_argument(
        "--max-iters", type=int, default=0,
        help="stop measurement after N iterations per epoch (0 = full epoch)",
    )
    args = parser.parse_args()

    import torch  # noqa: E402

    device = __import__(
        "dig.sslgraph.utils.device", fromlist=["pick_torch_device"]
    ).pick_torch_device()
    print(f"[device] {device}  torch={torch.__version__}")
    amp_mode = os.environ.get("PRETRAIN_AMP", "").strip().lower()
    pin_mem = os.environ.get("PIN_MEMORY", "").strip().lower() in ("1", "true", "yes", "on")
    nw = int(os.environ.get("DATALOADER_NUM_WORKERS", "0") or "0")
    mmff = os.environ.get("MMFFRANDOM_FAST", "").strip().lower() in ("1", "true", "yes", "on")
    print(f"[knobs ] MMFF_FAST={mmff}  AMP={amp_mode or 'off'}  PIN_MEMORY={pin_mem}  NUM_WORKERS={nw}")

    out = run_pretrain_benchmark(
        dataset=args.dataset,
        root=args.root,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup,
        measure_epochs=args.epochs,
        max_iters_per_epoch=args.max_iters,
        env_overrides={},
    )

    totals = out["totals"]
    n_it = int(totals["n_iter"])
    print()
    for i, row in enumerate(out.get("per_epoch", [])):
        print(
            f"  [EPOCH {i + 1}] wall={row['wall']*1000:8.1f} ms  n_iter={int(row['n_iter'])}  "
            f"data={row['data']*1000:6.1f}  to_dev={row['to_dev']*1000:6.1f}  "
            f"views={row['views']*1000:6.1f}  fwd={row['fwd']*1000:6.1f}  "
            f"bwd={row['bwd']*1000:6.1f}  step={row['step']*1000:6.1f}"
        )

    print()
    print(f"=== Summary over {args.epochs} measured epoch(s) ===")
    if n_it == 0:
        print("no iterations recorded.")
        return 1
    rows = [
        ("dataloader.next()", totals["data"]),
        ("batch.to(device) ", totals["to_dev"]),
        ("views_fn         ", totals["views"]),
        ("forward + loss   ", totals["fwd"]),
        ("backward         ", totals["bwd"]),
        ("optim + sched    ", totals["step"]),
    ]
    sum_t = sum(v for _, v in rows)
    print(f"{'stage':<20s} {'total (ms)':>12s} {'per-iter (ms)':>14s}  share")
    print("-" * 64)
    for name, t in rows:
        share = (t / sum_t * 100) if sum_t > 0 else 0
        print(f"{name:<20s} {t*1000:12.1f} {t*1000/n_it:14.2f}  {share:5.1f}%")
    print("-" * 64)
    print(f"{'TOTAL stages':<20s} {sum_t*1000:12.1f}")
    print(f"{'wall (epochs)':<20s} {totals['wall']*1000:12.1f}")
    print(f"  view[0]={out.get('view_impl_0')}")
    print(f"  iters/epoch ~ {n_it / args.epochs:.1f}, graphs/iter ~ {totals['n_graphs']/n_it:.1f}, "
          f"nodes/iter ~ {totals['n_nodes']/n_it:.1f}")
    return 0


if __name__ == "__main__":
    os.chdir(REPO)
    raise SystemExit(main())
