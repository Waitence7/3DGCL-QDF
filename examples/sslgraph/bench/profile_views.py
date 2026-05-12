#!/usr/bin/env python3
"""Micro-benchmark the DGCL contrastive view fns: Python vs Rust (``dig_io``).

Builds a synthetic batch of small graphs (configurable size/edge-count) and
times :class:`UniformSample`, :class:`RWSample`, and :class:`EdgePerturbation`
in both backends. We do not depend on RDKit or any dataset on disk, so this
script is safe to run in any environment with the Rust extension installed.

Run from repo root::

    .\\.venv\\Scripts\\python.exe examples\\sslgraph\\bench\\profile_views.py \\
        --n-graphs 256 --n-nodes 40 --n-edges 200 --repeat 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dig.sslgraph.method.contrastive.views_fn.sample import UniformSample, RWSample  # noqa: E402
from dig.sslgraph.method.contrastive.views_fn.structure import EdgePerturbation  # noqa: E402


def make_random_batch(n_graphs: int, n_nodes: int, n_edges: int, dim: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    datas = []
    for _ in range(n_graphs):
        x = torch.from_numpy(rng.standard_normal(size=(n_nodes, dim)).astype(np.float32))
        src = rng.integers(0, n_nodes, size=n_edges, dtype=np.int64)
        dst = rng.integers(0, n_nodes, size=n_edges, dtype=np.int64)
        ei = torch.from_numpy(np.stack([src, dst], axis=0)).long()
        datas.append(Data(x=x, edge_index=ei))
    return Batch.from_data_list(datas)


def time_calls(fn, batch, repeat: int) -> list[float]:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = fn(batch)
        times.append(time.perf_counter() - t0)
    return times


def fmt(seq):
    if not seq:
        return "n/a"
    return f"{1000 * statistics.mean(seq):8.2f} ms  (min {1000 * min(seq):.2f}, max {1000 * max(seq):.2f})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-graphs", type=int, default=256)
    parser.add_argument("--n-nodes", type=int, default=40)
    parser.add_argument("--n-edges", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--kernels",
        nargs="+",
        choices=["uniform", "rw", "edge"],
        default=["uniform", "rw", "edge"],
    )
    args = parser.parse_args()

    print(
        f"batch: n_graphs={args.n_graphs}, n_nodes={args.n_nodes}, "
        f"n_edges={args.n_edges}, repeat={args.repeat} (+{args.warmup} warmup)"
    )
    batch = make_random_batch(args.n_graphs, args.n_nodes, args.n_edges)

    impls = ("python", "rust")
    configs = []
    if "uniform" in args.kernels:
        configs.append((
            "UniformSample(gin)",
            lambda impl: UniformSample(encoder="gin", ratio=0.1, device="cpu", impl=impl),
        ))
    if "rw" in args.kernels:
        configs.append((
            "RWSample(ratio=0.5)",
            lambda impl: RWSample(ratio=0.5, impl=impl),
        ))
    if "edge" in args.kernels:
        configs.append((
            "EdgePerturbation(add+drop)",
            lambda impl: EdgePerturbation(add=True, drop=True, ratio=0.1, impl=impl),
        ))

    for label, build in configs:
        print(f"\n[{label}]")
        for impl in impls:
            view = build(impl)
            # warmup
            for _ in range(args.warmup):
                try:
                    _ = view(batch)
                except Exception as e:  # original python EdgePerturbation NameError, etc.
                    print(f"  {impl:<6s}: skipped ({type(e).__name__}: {e})")
                    break
            else:
                times = time_calls(view, batch, args.repeat)
                print(f"  {impl:<6s}: {fmt(times)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
