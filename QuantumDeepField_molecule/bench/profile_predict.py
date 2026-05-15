#!/usr/bin/env python3
"""Profile the QDF *inference* loop (mirrors ``predict/predict.py``).

Same toggles as ``profile_train.py`` so you can compare NumPy and Rust
back-ends on the predict path:

  --loader {npy,shard}              data loader backend
  --pad-impl {python,rust,rust-pad-only}   LCAO host-side helpers

This script does NOT require a trained checkpoint -- it builds a fresh
model and measures the forward-pass throughput only. The original
``predict/predict.py`` code is left untouched; toggles can be wired into
the production script separately.

Phases::

    A. dataloader-only iteration       (pure IO / batch assembly cost)
    B. dataloader + ``model.forward``   (full inference path, no_grad)

For each batch, B is broken down into IO time and forward time, both
synchronized against the active device.

Example (from repo root)::

    .\.venv\Scripts\python.exe QuantumDeepField_molecule\bench\profile_predict.py \
        --dataset QM9full_homolumo_eV --split test \
        --basis-set 6-31G --radius 0.75 --grid-interval 0.3 \
        --operation mean --batch-size 8 --max-batches 2000 \
        --loader shard --pad-impl python
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
QDF_ROOT = REPO_ROOT / "QuantumDeepField_molecule"
sys.path.insert(0, str(QDF_ROOT / "train"))

import train as qdf_train  # noqa: E402


def pick_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_sync(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="QM9under7atoms_homolumo_eV")
    parser.add_argument("--split", default="test",
                        help="Which preprocessed split to iterate (predict typically uses 'test').")
    parser.add_argument("--basis-set", default="6-31G")
    parser.add_argument("--radius", default="0.75")
    parser.add_argument("--grid-interval", default="0.3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--layer-functional", type=int, default=3)
    parser.add_argument("--hidden-HK", type=int, default=200)
    parser.add_argument("--layer-HK", type=int, default=3)
    parser.add_argument("--operation", default="sum")
    parser.add_argument("--warmup-batches", type=int, default=2,
                        help="Batches at the start excluded from averages (XPU JIT/warmup). "
                             "Capped automatically when a split has fewer batches than this.")
    parser.add_argument("--min-counted-batches", type=int, default=16,
                        help="Phase B keeps cycling the dataloader until this many counted "
                             "(post-warmup) batches are collected, unless --max-batches stops it first.")
    parser.add_argument("--max-loader-passes", type=int, default=50_000,
                        help="Safety cap: max full passes over the split in phase B.")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Stop after this many batches in each phase.")
    parser.add_argument("--loader", choices=["npy", "shard"], default="npy")
    parser.add_argument("--pad-impl",
                        choices=["python", "rust", "rust-pad-only"],
                        default="python")
    parser.add_argument("--predict-mode", choices=["predict", "test"], default="test",
                        help="'predict' = model.forward(data, predict=True) -> (idx, E_); "
                             "'test'    = model.forward(data) -> (idx, E, E_), matches train.Tester.")
    parser.add_argument("--lcao-breakdown", action="store_true",
                        help="Wrap LCAO sub-methods and print per-method totals.")
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"XPU device: {torch.xpu.get_device_name(0)}")

    field = f"{args.basis_set}_{args.radius}sphere_{args.grid_interval}grid"
    dataset_dir = QDF_ROOT / "dataset" / args.dataset
    split_dir = dataset_dir / f"{args.split}_{field}"
    if not split_dir.exists():
        sys.exit(f"Missing preprocessed dir: {split_dir}\n"
                 f"Run train/preprocess.sh or predict/preprocess.sh first.")

    import pickle
    with open(dataset_dir / f"orbitaldict_{args.basis_set}.pickle", "rb") as f:
        orbital_dict = pickle.load(f)
    n_orbitals = len(orbital_dict)
    print(f"N_orbitals: {n_orbitals}")
    print(f"Loader backend: {args.loader}   split={args.split}")

    if args.loader == "npy":
        ds = qdf_train.MyDataset(str(split_dir))
    else:
        from dataset_shard import MyDatasetShard, default_shard_path
        shard = default_shard_path(split_dir)
        if not shard.exists():
            sys.exit(
                f"Missing shard file: {shard}\n"
                "Build it first with bench/convert_to_shard.py."
            )
        ds = MyDatasetShard(shard)
    n_mols = len(ds)
    print(f"{args.split} dataset: {n_mols} molecules")
    batches_per_epoch = max(1, (n_mols + args.batch_size - 1) // args.batch_size)
    warm_eff = min(args.warmup_batches, max(0, batches_per_epoch - 1))
    if warm_eff != args.warmup_batches:
        print(
            f"Note: --warmup-batches {args.warmup_batches} -> {warm_eff} "
            f"(only {batches_per_epoch} batch(es)/epoch for len={n_mols}, "
            f"batch_size={args.batch_size})."
        )

    # Use the same dataloader helper as training for parity with predict.py.
    dl = qdf_train.mydataloader(ds, args.batch_size, args.num_workers)

    # Infer N_output from one sample (skip pickling overhead by indexing directly).
    sample = ds[0]
    if len(sample) == 8:
        n_output = int(np.asarray(sample[6]).shape[1])
    else:
        n_output = 1
    print(f"N_output: {n_output}")

    model = qdf_train.QuantumDeepField(
        device, n_orbitals,
        args.dim, args.layer_functional, args.operation, n_output,
        args.hidden_HK, args.layer_HK,
    ).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    print(f"LCAO helpers (--pad-impl): {args.pad_impl}")
    if args.pad_impl != "python":
        from model_patches import apply_rust_lcao
        if args.pad_impl == "rust":
            apply_rust_lcao(model, what=("pad", "list_to_batch"))
        elif args.pad_impl == "rust-pad-only":
            apply_rust_lcao(model, what=("pad",))

    lcao_bucket: dict[str, float] = defaultdict(float)
    lcao_count: dict[str, int] = defaultdict(int)
    if args.lcao_breakdown:
        def make_timer(orig, key):
            def wrapped(*a, **kw):
                t0 = time.perf_counter()
                out = orig(*a, **kw)
                device_sync(device)
                lcao_bucket[key] += time.perf_counter() - t0
                lcao_count[key] += 1
                return out
            return wrapped
        model.list_to_batch = make_timer(model.list_to_batch, "list_to_batch")
        model.pad = make_timer(model.pad, "pad")
        model.basis_matrix = make_timer(model.basis_matrix, "basis_matrix")
        model.functional = make_timer(model.functional, "functional")
        model.HKmap = make_timer(model.HKmap, "HKmap")

    # --- Phase A: dataloader-only ------------------------------------------
    print(f"\n[phase A] dataloader-only iteration over '{args.split}' split (no model)")
    t0 = time.perf_counter()
    batches_seen = 0
    samples_seen = 0
    for batch in dl:
        batches_seen += 1
        samples_seen += len(batch[0])
        if args.max_batches is not None and batches_seen >= args.max_batches:
            break
    wall_A = time.perf_counter() - t0
    print(f"  batches={batches_seen}, samples={samples_seen}, "
          f"wall={wall_A:.3f}s ({1000*wall_A/max(batches_seen,1):.2f} ms/batch, "
          f"{1000*wall_A/max(samples_seen,1):.3f} ms/sample)")

    # --- Phase B: inference -------------------------------------------------
    print(
        f"\n[phase B] inference forward pass ({args.predict_mode} mode), "
        f"warmup_batches={warm_eff} (effective), "
        f"target_counted>={args.min_counted_batches}, "
        f"batches/epoch≈{batches_per_epoch}"
    )
    bucket = defaultdict(float)
    bucket_count = defaultdict(int)

    t_iter_start = time.perf_counter()
    total_batches = 0
    global_batch_idx = 0
    loader_passes = 0
    stop_phase_b = False

    with torch.inference_mode():
        while loader_passes < args.max_loader_passes and not stop_phase_b:
            it = iter(dl)
            got_batch = False
            for data in it:
                got_batch = True
                t_iter_end = time.perf_counter()
                io_t = t_iter_end - t_iter_start
                counted = global_batch_idx >= warm_eff

                t = time.perf_counter()
                if args.predict_mode == "predict":
                    idx, E_ = model.forward(data, predict=True)
                    # Match predict.py: collect into Python via .tolist()
                    _ = E_.detach().cpu().tolist()
                else:
                    idx, E, E_ = model.forward(data)
                    _ = (E.detach().cpu().tolist(), E_.detach().cpu().tolist())
                device_sync(device)
                fwd_t = time.perf_counter() - t

                total_batches += 1
                global_batch_idx += 1
                if counted:
                    bucket["io_next_batch"] += io_t
                    bucket["forward"] += fwd_t
                    bucket_count["io_next_batch"] += 1
                    bucket_count["forward"] += 1

                t_iter_start = time.perf_counter()
                if args.max_batches is not None and total_batches >= args.max_batches:
                    stop_phase_b = True
                    break

                if bucket_count.get("forward", 0) >= args.min_counted_batches:
                    stop_phase_b = True
                    break

            loader_passes += 1
            if not got_batch:
                break
            if stop_phase_b:
                break

    counted_batches = bucket_count.get("forward", 0)
    print(
        f"  phase B: {counted_batches} counted batch(es) across {loader_passes} loader pass(es); "
        f"total_batches={total_batches}"
    )

    if counted_batches == 0:
        print(
            "No counted batches (empty split, or raise --max-batches / "
            "check batch_size vs dataset length)."
        )
        return 1
    if counted_batches < args.min_counted_batches:
        print(
            f"Note: only {counted_batches} counted batches "
            f"(target {args.min_counted_batches}); increase --max-batches or lower "
            f"--min-counted-batches for short splits."
        )

    print("\n--- Averages over counted batches (warmup excluded) ---")
    rows = []
    for name in ["io_next_batch", "forward"]:
        total = bucket[name]
        rows.append((name, total, 1000.0 * total / counted_batches))
    total_counted = sum(t for _, t, _ in rows)
    print(f"{'phase':<18s} {'total(s)':>10s} {'%':>6s} {'ms/batch':>10s}")
    for name, total, per in rows:
        share = 100.0 * total / total_counted if total_counted > 0 else 0.0
        print(f"{name:<18s} {total:10.3f} {share:6.1f} {per:10.3f}")
    print("-" * 50)
    print(f"{'total counted':<18s} {total_counted:10.3f} {100.0:6.1f}")

    if args.lcao_breakdown and lcao_bucket:
        print("\n--- LCAO / model sub-method totals ---")
        print(f"{'method':<16s} {'calls':>8s} {'total(s)':>10s} {'ms/call':>10s}")
        for name, total in sorted(lcao_bucket.items(), key=lambda kv: -kv[1]):
            n = max(lcao_count[name], 1)
            print(f"{name:<16s} {lcao_count[name]:>8d} {total:10.3f} {1000.0*total/n:10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
