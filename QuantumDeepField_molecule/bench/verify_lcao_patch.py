#!/usr/bin/env python3
"""Numerically compare ``QuantumDeepField`` outputs with and without the Rust
LCAO patches, on the same batches. Used as a smoke test before benchmarking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
QDF_ROOT = REPO_ROOT / "QuantumDeepField_molecule"
sys.path.insert(0, str(QDF_ROOT / "train"))

import train as qdf_train  # noqa: E402
from dataset_shard import MyDatasetShard, default_shard_path  # noqa: E402
from model_patches import apply_rust_lcao, unapply_rust_lcao  # noqa: E402


def pick_device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="QM9under14atoms_atomizationenergy_eV")
    parser.add_argument("--basis-set", default="6-31G")
    parser.add_argument("--radius", default="0.75")
    parser.add_argument("--grid-interval", default="0.3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-batches", type=int, default=5)
    parser.add_argument("--operation", default="sum")
    parser.add_argument("--dim", type=int, default=200)
    parser.add_argument("--layer-functional", type=int, default=3)
    parser.add_argument("--hidden-HK", type=int, default=200)
    parser.add_argument("--layer-HK", type=int, default=3)
    parser.add_argument("--tol", type=float, default=5e-4,
                        help="Max relative error tolerance for losses.")
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    field = f"{args.basis_set}_{args.radius}sphere_{args.grid_interval}grid"
    dataset_dir = QDF_ROOT / "dataset" / args.dataset
    train_dir = dataset_dir / f"train_{field}"

    # Use the shard reader so both runs see byte-identical data.
    ds = MyDatasetShard(default_shard_path(train_dir))
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=qdf_train.collate_fn,
    )

    import pickle
    with open(dataset_dir / f"orbitaldict_{args.basis_set}.pickle", "rb") as f:
        n_orbitals = len(pickle.load(f))

    n_output = ds.n_output
    print(f"N_orbitals={n_orbitals}, N_output={n_output}")

    # Cache the first N batches once so both runs see identical inputs.
    batches: list = []
    for i, data in enumerate(dl):
        batches.append(data)
        if len(batches) >= args.n_batches:
            break

    def run_pass(patch: str) -> dict:
        torch.manual_seed(1729)
        if device.type == "xpu":
            torch.xpu.manual_seed_all(1729)
        elif device.type == "cuda":
            torch.cuda.manual_seed_all(1729)

        model = qdf_train.QuantumDeepField(
            device, n_orbitals,
            args.dim, args.layer_functional, args.operation, n_output,
            args.hidden_HK, args.layer_HK,
        ).to(device)

        if patch == "rust":
            apply_rust_lcao(model, what=("pad", "list_to_batch"))
        elif patch == "rust-pad-only":
            apply_rust_lcao(model, what=("pad",))

        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        losses_E = []
        losses_V = []
        for data in batches:
            loss_E = model.forward(data, train=True, target="E")
            opt.zero_grad()
            loss_E.backward()
            opt.step()
            losses_E.append(float(loss_E.detach().cpu().item()))

            loss_V = model.forward(data, train=True, target="V")
            opt.zero_grad()
            loss_V.backward()
            opt.step()
            losses_V.append(float(loss_V.detach().cpu().item()))

        return {"losses_E": losses_E, "losses_V": losses_V}

    print("\n[run 1] pad-impl=python")
    ref = run_pass("python")
    print("  losses_E:", [f"{x:.4f}" for x in ref["losses_E"]])
    print("  losses_V:", [f"{x:.4f}" for x in ref["losses_V"]])

    failures = 0
    for patch in ("rust-pad-only", "rust"):
        print(f"\n[run] pad-impl={patch}")
        got = run_pass(patch)
        print("  losses_E:", [f"{x:.4f}" for x in got["losses_E"]])
        print("  losses_V:", [f"{x:.4f}" for x in got["losses_V"]])
        for name in ("losses_E", "losses_V"):
            a = np.asarray(ref[name])
            b = np.asarray(got[name])
            absdiff = np.abs(a - b)
            reldiff = absdiff / (np.abs(a) + 1e-9)
            max_rel = float(reldiff.max())
            print(f"  {name}: max abs diff = {absdiff.max():.3e}, max rel diff = {max_rel:.3e}")
            if max_rel > args.tol:
                print(f"    [FAIL] {patch}/{name} exceeds tolerance {args.tol}")
                failures += 1

    if failures:
        print(f"\n{failures} mismatches.")
        return 1
    print("\nAll runs match python within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
