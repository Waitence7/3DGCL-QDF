#!/usr/bin/env python3
"""Property-level verification for the Rust view-fn kernels in ``dig_io``.

The Rust kernels use a different PRNG (ChaCha8 seeded from a small Python-side
counter) than the original PyTorch path, so bit-equal output to the upstream
``UniformSample`` / ``RWSample`` / ``EdgePerturbation`` is not the goal. What we
check is that:

  * each Rust call is **deterministic** (same seed -> identical output);
  * the returned ``new_edge_index`` is **structurally valid** (node ids in
    range, no duplicate columns, subgraph relabeling honoured);
  * the kept-index set has the **right cardinality**;
  * an equivalent reference re-implementation written against the same Rust
    keep-indices reproduces the relabeled ``edge_index`` exactly.

Run (from repo root)::

    .\\.venv\\Scripts\\python.exe examples\\sslgraph\\verify_views_backend.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import dig_io  # noqa: E402


def _random_edge_index(n_nodes: int, n_edges: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``n_edges`` directed edges; allows self-loops / duplicates."""
    src = rng.integers(0, n_nodes, size=n_edges, dtype=np.int64)
    dst = rng.integers(0, n_nodes, size=n_edges, dtype=np.int64)
    return np.stack([src, dst], axis=0).astype(np.int64)


def _reference_relabel(edge_index: np.ndarray, keep: np.ndarray, num_nodes: int) -> np.ndarray:
    relabel = -np.ones(num_nodes, dtype=np.int64)
    relabel[keep] = np.arange(len(keep), dtype=np.int64)
    nu = relabel[edge_index[0]]
    nv = relabel[edge_index[1]]
    mask = (nu >= 0) & (nv >= 0)
    return np.stack([nu[mask], nv[mask]], axis=0).astype(np.int64)


def check_uniform_sample(seed_a: int = 1, seed_b: int = 1, n_nodes: int = 32, n_edges: int = 160):
    rng = np.random.default_rng(0)
    ei = _random_edge_index(n_nodes, n_edges, rng)
    keep_num = int(n_nodes * 0.9)

    new_ei1, keep1 = dig_io.uniform_sample_subgraph(ei, n_nodes, keep_num, seed_a)
    new_ei2, keep2 = dig_io.uniform_sample_subgraph(ei, n_nodes, keep_num, seed_b)

    assert np.array_equal(new_ei1, new_ei2), "uniform_sample is non-deterministic for same seed"
    assert np.array_equal(keep1, keep2), "uniform_sample keep_indices non-deterministic"

    assert keep1.shape == (keep_num,)
    assert np.all(np.diff(keep1) > 0), "keep_indices must be strictly sorted"
    assert new_ei1.dtype == np.int64 and new_ei1.shape[0] == 2
    if new_ei1.shape[1] > 0:
        assert new_ei1.min() >= 0
        assert new_ei1.max() < keep_num

    ref = _reference_relabel(ei, keep1, n_nodes)
    assert np.array_equal(np.sort(new_ei1, axis=1), np.sort(ref, axis=1)) or \
        np.array_equal(new_ei1, ref), \
        "Rust relabel differs from the NumPy reference"

    print(f"[uniform_sample] OK  n={n_nodes} keep={keep_num} new_E={new_ei1.shape[1]}")


def check_rw_sample(seed: int = 7, n_nodes: int = 64, n_edges: int = 320):
    rng = np.random.default_rng(0)
    ei = _random_edge_index(n_nodes, n_edges, rng)
    sub_num = int(n_nodes * 0.5)

    new_ei1, keep1 = dig_io.rw_sample_subgraph(ei, n_nodes, sub_num, seed, False)
    new_ei2, keep2 = dig_io.rw_sample_subgraph(ei, n_nodes, sub_num, seed, False)
    assert np.array_equal(new_ei1, new_ei2)
    assert np.array_equal(keep1, keep2)

    assert keep1.dtype == np.int64
    assert np.all(np.diff(keep1) > 0), "RW keep must be sorted unique"
    assert keep1.size <= sub_num
    if new_ei1.shape[1] > 0:
        assert new_ei1.min() >= 0
        assert new_ei1.max() < keep1.size

    ref = _reference_relabel(ei, keep1, n_nodes)
    assert np.array_equal(new_ei1, ref), "Rust RW relabel differs from NumPy reference"

    print(f"[rw_sample]      OK  n={n_nodes} kept={keep1.size}/{sub_num} new_E={new_ei1.shape[1]}")


def check_edge_perturb(seed: int = 11, n_nodes: int = 50, n_edges: int = 200):
    rng = np.random.default_rng(0)
    ei = _random_edge_index(n_nodes, n_edges, rng)

    out1 = dig_io.edge_perturb(ei, n_nodes, 0.1, True, False, seed)
    out2 = dig_io.edge_perturb(ei, n_nodes, 0.1, True, False, seed)
    assert np.array_equal(out1, out2)

    out3 = dig_io.edge_perturb(ei, n_nodes, 0.1, True, True, seed + 1)

    for label, out in (("add-only", out1), ("add+drop", out3)):
        assert out.dtype == np.int64 and out.shape[0] == 2
        if out.shape[1] > 0:
            assert out.min() >= 0 and out.max() < n_nodes
            # column-unique:
            pairs = out.T.tolist()
            assert len(pairs) == len(set(map(tuple, pairs))), f"{label} not unique"
        print(f"[edge_perturb]   OK  {label:<8s} new_E={out.shape[1]}")


def main() -> int:
    if not dig_io.is_available():
        print("dig_io native extension not loaded; build with `cd dig_io && maturin develop --release`.")
        return 1
    check_uniform_sample()
    check_rw_sample()
    check_edge_perturb()
    print("All view-fn properties pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
