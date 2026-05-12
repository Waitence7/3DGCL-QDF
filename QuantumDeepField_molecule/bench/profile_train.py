#!/usr/bin/env python3
"""Profile one (or a few) training epochs of the QDF model.

The QDF training loop in ``train/train.py`` interleaves:

  * disk reads (one ``.npy`` per molecule via ``MyDataset``)
  * Python-side block-diagonal padding inside ``model.pad`` (CPU->XPU copies)
  * model forward / backward on XPU
  * optimizer step

This script breaks those phases apart so we know where the time goes.

Run example (from repo root):

    .\.venv\Scripts\python.exe QuantumDeepField_molecule\bench\profile_train.py \
        --dataset QM9under7atoms_homolumo_eV \
        --basis-set 6-31G --radius 0.75 --grid-interval 0.3 \
        --batch-size 8 --epochs 3
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

import train as qdf_train  # noqa: E402  (train/train.py)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="QM9under7atoms_homolumo_eV")
    parser.add_argument("--basis-set", default="6-31G")
    parser.add_argument("--radius", default="0.75")
    parser.add_argument("--grid-interval", default="0.3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--layer-functional", type=int, default=3)
    parser.add_argument("--hidden-HK", type=int, default=200)
    parser.add_argument("--layer-HK", type=int, default=3)
    parser.add_argument("--operation", default="mean")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--step-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=1,
                        help="Batches at the start of each epoch excluded from averages (XPU JIT/warmup).")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Stop each epoch (and phase A) after this many batches. Use to keep large-dataset profiling short.")
    parser.add_argument("--lcao-breakdown", action="store_true",
                        help="Patch model.LCAO to time list_to_batch / pad / basis_matrix / matmul / normalize separately.")
    parser.add_argument("--loader", choices=["npy", "shard"], default="npy",
                        help="Data loader backend. 'npy' = original MyDataset (one .npy per molecule); "
                             "'shard' = new MyDatasetShard backed by the Rust mmap reader. "
                             "Both implementations are kept; this flag only selects which is used.")
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")
    if device.type == "xpu":
        print(f"XPU device: {torch.xpu.get_device_name(0)}")

    field = f"{args.basis_set}_{args.radius}sphere_{args.grid_interval}grid"
    dataset_dir = QDF_ROOT / "dataset" / args.dataset

    train_dir = dataset_dir / f"train_{field}"
    val_dir = dataset_dir / f"val_{field}"
    test_dir = dataset_dir / f"test_{field}"
    for d in (train_dir, val_dir, test_dir):
        if not d.exists():
            sys.exit(f"Missing preprocessed dir: {d}\n"
                     f"Run train/preprocess.sh first or pass a preprocessed dataset.")

    import pickle
    with open(dataset_dir / f"orbitaldict_{args.basis_set}.pickle", "rb") as f:
        orbital_dict = pickle.load(f)
    n_orbitals = len(orbital_dict)
    print(f"N_orbitals: {n_orbitals}")

    print(f"Loader backend: {args.loader}")
    if args.loader == "npy":
        ds_train = qdf_train.MyDataset(str(train_dir))
        ds_val = qdf_train.MyDataset(str(val_dir))
        ds_test = qdf_train.MyDataset(str(test_dir))
    else:
        from dataset_shard import MyDatasetShard, default_shard_path
        train_shard = default_shard_path(train_dir)
        val_shard = default_shard_path(val_dir)
        test_shard = default_shard_path(test_dir)
        for sp in (train_shard, val_shard, test_shard):
            if not sp.exists():
                sys.exit(
                    f"Missing shard file: {sp}\n"
                    "Build it first with bench/convert_to_shard.py."
                )
        ds_train = MyDatasetShard(train_shard)
        ds_val = MyDatasetShard(val_shard)
        ds_test = MyDatasetShard(test_shard)
    print(f"train/val/test = {len(ds_train)} / {len(ds_val)} / {len(ds_test)}")

    dl_train = qdf_train.mydataloader(ds_train, args.batch_size, args.num_workers, shuffle=True)

    # Infer N_output by reading one sample (skip the dataloader to keep profile clean).
    sample = ds_train[0]
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.lr_decay)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Optional fine-grained LCAO breakdown by wrapping the model's sub-methods.
    # The original methods are kept (we just record total time in a dict).
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
        # Wrap the methods that contain CPU work / host->device copies / GPU compute.
        model.list_to_batch = make_timer(model.list_to_batch, "list_to_batch")
        model.pad = make_timer(model.pad, "pad")
        model.basis_matrix = make_timer(model.basis_matrix, "basis_matrix")
        # Wrap functional / HKmap to know how much fwd is the deep stack vs LCAO.
        model.functional = make_timer(model.functional, "functional")
        model.HKmap = make_timer(model.HKmap, "HKmap")

    # --- Phase A: pure data-loading throughput (just iterate the dataloader) -----
    print("\n[phase A] dataloader-only iteration (no model)")
    t0 = time.perf_counter()
    batches_seen = 0
    samples_seen = 0
    for batch in dl_train:
        batches_seen += 1
        samples_seen += len(batch[0])
        if args.max_batches is not None and batches_seen >= args.max_batches:
            break
    wall_A = time.perf_counter() - t0
    print(f"  batches={batches_seen}, samples={samples_seen}, "
          f"wall={wall_A:.3f}s ({1000*wall_A/max(batches_seen,1):.2f} ms/batch, "
          f"{1000*wall_A/max(samples_seen,1):.3f} ms/sample)")

    # --- Phase B: training, split into IO / LCAO / functional / backward / step --
    print(f"\n[phase B] training profile across {args.epochs} epoch(s) "
          f"(warmup batches per epoch: {args.warmup_batches})")
    bucket = defaultdict(float)
    bucket_count = defaultdict(int)

    for epoch in range(args.epochs):
        ep_t0 = time.perf_counter()
        ep_loss_E = 0.0
        ep_loss_V = 0.0
        ep_batches = 0

        it = iter(dl_train)
        t_iter_start = time.perf_counter()
        for batch_idx, data in enumerate(it):
            t_iter_end = time.perf_counter()
            io_t = t_iter_end - t_iter_start
            counted = batch_idx >= args.warmup_batches

            # --- target=E ---
            t = time.perf_counter()
            loss_E = model.forward(data, train=True, target="E")
            device_sync(device)
            fwd_E = time.perf_counter() - t

            t = time.perf_counter()
            optimizer.zero_grad()
            loss_E.backward()
            device_sync(device)
            bwd_E = time.perf_counter() - t

            t = time.perf_counter()
            optimizer.step()
            device_sync(device)
            step_E = time.perf_counter() - t

            ep_loss_E += float(loss_E.detach().cpu().item())

            # --- target=V ---
            t = time.perf_counter()
            loss_V = model.forward(data, train=True, target="V")
            device_sync(device)
            fwd_V = time.perf_counter() - t

            t = time.perf_counter()
            optimizer.zero_grad()
            loss_V.backward()
            device_sync(device)
            bwd_V = time.perf_counter() - t

            t = time.perf_counter()
            optimizer.step()
            device_sync(device)
            step_V = time.perf_counter() - t

            ep_loss_V += float(loss_V.detach().cpu().item())
            ep_batches += 1

            if counted:
                bucket["io_next_batch"] += io_t
                bucket_count["io_next_batch"] += 1
                bucket["fwd_E"] += fwd_E
                bucket["bwd_E"] += bwd_E
                bucket["step_E"] += step_E
                bucket["fwd_V"] += fwd_V
                bucket["bwd_V"] += bwd_V
                bucket["step_V"] += step_V

            t_iter_start = time.perf_counter()
            if args.max_batches is not None and (batch_idx + 1) >= args.max_batches:
                break

        scheduler.step()
        ep_wall = time.perf_counter() - ep_t0
        print(f"  epoch {epoch}: batches={ep_batches}, wall={ep_wall:.3f}s, "
              f"loss_E={ep_loss_E:.4f}, loss_V={ep_loss_V:.4f}")

    print("\n--- Averages over counted batches (warmup excluded) ---")
    counted_batches = bucket_count.get("io_next_batch", 0)
    if counted_batches == 0:
        print("No counted batches (try lowering --warmup-batches).")
        return
    rows = []
    for name in ["io_next_batch", "fwd_E", "bwd_E", "step_E", "fwd_V", "bwd_V", "step_V"]:
        total = bucket[name]
        rows.append((name, total, 1000.0 * total / counted_batches))
    total_counted = sum(t for _, t, _ in rows)
    print(f"{'phase':<18s} {'total(s)':>10s} {'%':>6s} {'ms/batch':>10s}")
    for name, total, per in rows:
        share = 100.0 * total / total_counted
        print(f"{name:<18s} {total:10.3f} {share:6.1f} {per:10.3f}")
    print("-" * 50)
    print(f"{'total counted':<18s} {total_counted:10.3f} {100.0:6.1f}")

    print(
        "\nTip: if 'io_next_batch' dominates, the .npy-per-molecule pipeline is the bottleneck."
        "  If 'fwd_*' dominates, the XPU compute is the bottleneck (do not rewrite that in Rust)."
        "  If 'bwd_*' is high vs 'fwd_*', autograd graph (LCAO pad/scatter) is the suspect."
    )

    if args.lcao_breakdown and lcao_bucket:
        print("\n--- LCAO / model sub-method totals (counted batches only is NOT applied here) ---")
        print(f"{'method':<16s} {'calls':>8s} {'total(s)':>10s} {'ms/call':>10s}")
        for name, total in sorted(lcao_bucket.items(), key=lambda kv: -kv[1]):
            n = max(lcao_count[name], 1)
            print(f"{name:<16s} {lcao_count[name]:>8d} {total:10.3f} {1000.0*total/n:10.3f}")


if __name__ == "__main__":
    main()
