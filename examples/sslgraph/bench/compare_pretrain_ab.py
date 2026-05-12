#!/usr/bin/env python3
"""A/B compare pretrain micro-benchmark: baseline vs optimized env (non-destructive).

Runs two isolated configurations (each restores ``os.environ`` after), samples
CPU / XPU (Windows PDH via QDF ``bench/sampler.py`` when available) and wall
time, then writes JSON + matplotlib figures similar to QDF ``run_pipeline.ipynb``.

Example::

    cd c:\\DGCL\\3DGCL
    .venv\\Scripts\\python.exe examples\\sslgraph\\bench\\compare_pretrain_ab.py \\
        --dataset esol --warmup 1 --epochs 1 --max-iters 3

Default A clears ``MMFFRANDOM_FAST`` and ``PRETRAIN_AMP`` (legacy PyG path).
Default B sets ``MMFFRANDOM_FAST=1`` and ``PRETRAIN_AMP=bf16``.
Override with ``--a-env KEY=VAL`` (repeat) or ``--a-clear KEY`` (repeat).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parents[2]
for _p in (REPO, BENCH_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib.pyplot as plt
import numpy as np

from pretrain_bench_core import run_pretrain_benchmark, snapshot_relevant_env
from pretrain_process_metrics import measure_callable


def _parse_kv(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError(f"expected KEY=VAL, got {s!r}")
    k, v = s.split("=", 1)
    return k.strip(), v.strip()


def _smooth(arr: np.ndarray, dt: float, window_s: float = 0.5) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.size == 0 or dt <= 0:
        return a
    w = max(1, int(round(window_s / dt)))
    if w <= 1 or a.size < w:
        return a
    return np.convolve(a, np.ones(w) / w, mode="same")


def _summary_dict(s: Any) -> dict[str, float]:
    return {
        "wall_s": float(getattr(s, "wall_s", 0.0)),
        "peak_rss_mb": float(getattr(s, "peak_rss_mb", 0.0)),
        "mean_rss_mb": float(getattr(s, "mean_rss_mb", 0.0)),
        "mean_cpu_pct": float(getattr(s, "mean_cpu_pct", 0.0)),
        "peak_cpu_pct": float(getattr(s, "peak_cpu_pct", 0.0)),
        "mean_xpu_pct": float(getattr(s, "mean_xpu_pct", 0.0)),
        "peak_xpu_pct": float(getattr(s, "peak_xpu_pct", 0.0)),
        "n_samples": int(getattr(s, "n_samples", 0)),
    }


def main() -> int:
    os.chdir(REPO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esol")
    ap.add_argument("--root", default="dataset/")
    ap.add_argument("--batch-size", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--interval", type=float, default=0.1, help="resource sampler interval (s)")
    ap.add_argument("--out-dir", default="", help="default: examples/sslgraph/bench/figs/pretrain_ab_<ts>/")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--a-label", default="baseline (PyG views)")
    ap.add_argument("--b-label", default="optimized (MMFF_FAST + bf16)")
    ap.add_argument("--a-clear", action="append", default=[], metavar="KEY",
                    help="unset env var for run A (repeatable). Default adds MMFFRANDOM_FAST,PRETRAIN_AMP")
    ap.add_argument("--b-set", action="append", default=[], metavar="KEY=VAL",
                    help="set env for run B (repeatable). Default MMFFRANDOM_FAST=1 PRETRAIN_AMP=bf16")
    args = ap.parse_args()

    ts = time.strftime("%m%d_%H%M")
    out = Path(args.out_dir) if args.out_dir else (
        REPO / "examples" / "sslgraph" / "bench" / "figs" / f"pretrain_ab_{ts}"
    )
    out.mkdir(parents=True, exist_ok=True)

    a_clear = list(args.a_clear)
    if not a_clear:
        a_clear = ["MMFFRANDOM_FAST", "PRETRAIN_AMP"]
    a_overrides: dict[str, str | None] = {k: None for k in a_clear}

    b_set = list(args.b_set)
    if not b_set:
        b_overrides = {"MMFFRANDOM_FAST": "1", "PRETRAIN_AMP": "bf16"}
    else:
        b_overrides = dict(_parse_kv(s) for s in b_set)

    print("=== Host env snapshot (unchanged after this script) ===")
    print(json.dumps(snapshot_relevant_env(), indent=2))

    def run_side(label: str, overrides: dict[str, str | None]):
        print(f"\n>>> [{label}] env overrides: {overrides}")

        def work():
            return run_pretrain_benchmark(
                dataset=args.dataset,
                root=args.root,
                batch_size=args.batch_size,
                warmup_epochs=args.warmup,
                measure_epochs=args.epochs,
                max_iters_per_epoch=args.max_iters,
                env_overrides=overrides,
            )

        t0 = time.perf_counter()
        bench, samples, summary = measure_callable(work, interval=args.interval)
        outer_wall = time.perf_counter() - t0
        bench["outer_wall_s"] = outer_wall
        bench["resource_summary"] = _summary_dict(summary)
        print(
            f"    wall={summary.wall_s:.3f}s  mean_cpu={summary.mean_cpu_pct:.1f}%  "
            f"mean_xpu={summary.mean_xpu_pct:.1f}%  peak_rss={summary.peak_rss_mb:.0f}MB  "
            f"view[0]={bench.get('view_impl_0')}"
        )
        return bench, samples, summary

    run_a, samp_a, sum_a = run_side(args.a_label, a_overrides)
    run_b, samp_b, sum_b = run_side(args.b_label, b_overrides)

    payload = {
        "ts": ts,
        "a_label": args.a_label,
        "b_label": args.b_label,
        "a_overrides": a_overrides,
        "b_overrides": b_overrides,
        "args": vars(args),
        "run_a": run_a,
        "run_b": run_b,
        "summary_a": _summary_dict(sum_a),
        "summary_b": _summary_dict(sum_b),
    }
    json_path = out / "pretrain_ab_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[saved] {json_path}")

    if args.no_plot:
        return 0

    # ---- Bar chart (QDF-style scalar comparison) ----
    labels_m = ["wall (s)", "mean CPU%", "mean XPU%", "peak RSS (MB)"]
    fmts_m = ["{:.2f}", "{:.1f}", "{:.1f}", "{:.0f}"]
    va = [sum_a.wall_s, sum_a.mean_cpu_pct, sum_a.mean_xpu_pct, sum_a.peak_rss_mb]
    vb = [sum_b.wall_s, sum_b.mean_cpu_pct, sum_b.mean_xpu_pct, sum_b.peak_rss_mb]
    x = np.arange(len(labels_m))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.7))
    bars_a = ax.bar(x - w / 2, va, width=w, label=args.a_label)
    bars_b = ax.bar(x + w / 2, vb, width=w, label=args.b_label)
    for container, vals in ((bars_a, va), (bars_b, vb)):
        ax.bar_label(
            container,
            labels=[f.format(v) for f, v in zip(fmts_m, vals)],
            padding=2, fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_m)
    ax.set_ylabel("value (see x labels for units)")
    ax.set_title(f"Pretrain A/B — {args.dataset} batch={args.batch_size}")
    ax.legend()
    ax.margins(y=0.18)
    fig.tight_layout()
    bar_path = out / f"bars_pretrain_ab_{ts}.png"
    fig.savefig(bar_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {bar_path}")

    # ---- Per-stage wall split from bench totals ----
    def stage_ms(tot: dict[str, float]) -> list[float]:
        return [
            tot.get("data", 0) * 1000,
            tot.get("to_dev", 0) * 1000,
            tot.get("views", 0) * 1000,
            tot.get("fwd", 0) * 1000,
            tot.get("bwd", 0) * 1000,
            tot.get("step", 0) * 1000,
        ]

    stages = ["data", "to_dev", "views", "fwd", "bwd", "step"]
    sa = stage_ms(run_a["totals"])
    sb = stage_ms(run_b["totals"])
    x2 = np.arange(len(stages))
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    ax2.bar(x2 - w / 2, sa, width=w, label=args.a_label)
    ax2.bar(x2 + w / 2, sb, width=w, label=args.b_label)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(stages)
    ax2.set_ylabel("ms (summed over measured epochs)")
    ax2.set_title("Stage timing (micro-bench loop)")
    ax2.legend()
    fig2.tight_layout()
    st_path = out / f"bars_pretrain_stages_{ts}.png"
    fig2.savefig(st_path, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    print(f"[saved] {st_path}")

    # ---- Timeseries CPU / XPU (two rows, QDF-like smoothing) ----
    def _dt(t_list: list[float]) -> float:
        if len(t_list) < 2:
            return args.interval
        return max(1e-6, float(np.median(np.diff(t_list))))

    fig3, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    for col, (lab, samp, sm) in enumerate(
        [(args.a_label, samp_a, sum_a), (args.b_label, samp_b, sum_b)]
    ):
        t = np.asarray(samp.t, dtype=float)
        cpu = np.asarray(samp.cpu_pct, dtype=float)
        xpu = np.asarray(samp.xpu_pct, dtype=float)
        dt = _dt(list(samp.t))
        axes[0, col].plot(t, _smooth(cpu, dt), color="#4C78A8", lw=1.2)
        axes[0, col].set_ylabel("CPU % (system)")
        axes[0, col].set_title(f"{lab}\nwall={sm.wall_s:.2f}s  mean_cpu={sm.mean_cpu_pct:.1f}%")
        axes[0, col].grid(True, alpha=0.3)
        axes[1, col].plot(t, _smooth(xpu, dt), color="#F58518", lw=1.2)
        axes[1, col].set_ylabel("XPU % (GPU engines Σ)")
        axes[1, col].set_xlabel("time (s)")
        axes[1, col].set_title(f"mean_xpu={sm.mean_xpu_pct:.1f}%  peak={sm.peak_xpu_pct:.1f}%")
        axes[1, col].grid(True, alpha=0.3)
    fig3.suptitle(f"Resource trace — {args.dataset}  sampler={args.interval}s")
    fig3.tight_layout()
    ts_path = out / f"timeseries_cpu_xpu_{ts}.png"
    fig3.savefig(ts_path, dpi=160, bbox_inches="tight")
    plt.close(fig3)
    print(f"[saved] {ts_path}")

    speed = sum_a.wall_s / max(sum_b.wall_s, 1e-9)
    print(f"\n=== Ratio (A wall / B wall) = {speed:.2f}x  (higher = B faster) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
