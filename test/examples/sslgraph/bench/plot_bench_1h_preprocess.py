#!/usr/bin/env python3
"""Plot ``bench_1h_preprocess.jsonl`` from ``run_1h_preprocess_repeat.py``.

Writes ``figs/bench_1h_preprocess_overview.png`` next to this script.

RAM note (``profile_preprocess.py`` + ``--no-save``):
  - NumPy path: one molecule at a time; peak RAM is roughly a few molecules'
    worth of dense distance / field arrays (size grows with grid), not 11k×.
  - Rust path: buffers up to ``rust_batch_size`` molecules before calling
    ``preprocess_batch_rust``; default 64 in the 1h driver.
  - ``orbital_dict`` grows over the full split (string keys); usually modest
    compared to geometry arrays.
  For actual RSS over time, re-run the driver with psutil sampling (not in jsonl).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=Path(__file__).resolve().parent / "bench_1h_preprocess.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "figs" / "bench_1h_preprocess_overview.png",
    )
    args = ap.parse_args()

    if not args.jsonl.is_file():
        print("Missing:", args.jsonl)
        return 2

    rows = []
    with args.jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows in", args.jsonl)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    ax1 = fig.add_subplot(gs[0, :])
    for backend, color in (("numpy", "#4C72B0"), ("rust", "#55A868")):
        sub = df[df["backend"] == backend]
        ax1.plot(sub["round"], sub["wall_sec"], "o-", label=backend, color=color, alpha=0.85)
    ax1.set_xlabel("round")
    ax1.set_ylabel("wall time (s)")
    ax1.set_title("Per-run wall time (full train split, --no-save)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    df.boxplot(column="wall_sec", by="backend", ax=ax2)
    ax2.set_title("Wall time distribution by backend")
    ax2.set_xlabel("")
    plt.suptitle("")

    ax3 = fig.add_subplot(gs[1, 1])
    n_r = len(df[df["backend"] == "rust"])
    n_n = len(df[df["backend"] == "numpy"])
    ax3.bar(
        ["numpy\n(sum)", "rust\n(sum)"],
        [df.loc[df["backend"] == "numpy", "wall_sec"].sum(), df.loc[df["backend"] == "rust", "wall_sec"].sum()],
        color=["#4C72B0", "#55A868"],
    )
    ax3.set_ylabel("total wall (s)")
    ax3.set_title(f"Cumulative over log ({n_n} numpy, {n_r} rust runs)")

    ram_note = (
        "RAM (qualitative, this jsonl has no RSS):\n"
        "• --no-save: no 11k× disk arrays in RAM at once.\n"
        "• NumPy: one molecule in flight; arrays scale with grid size.\n"
        "• Rust: buffer ≤ rust_batch_size (64) molecules per native call.\n"
        "• orbital_dict grows across the split; usually modest."
    )
    fig.text(0.5, 0.02, ram_note, ha="center", va="bottom", fontsize=9, family="monospace")

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("Wrote", args.out)
    print(df.groupby("backend")["wall_sec"].describe().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
