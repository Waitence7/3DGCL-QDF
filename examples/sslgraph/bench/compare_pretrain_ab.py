#!/usr/bin/env python3
"""A/B/C compare pretrain micro-benchmark: 2- or 3-way, non-destructive.

Runs 2–3 isolated configurations (each restores ``os.environ`` after), samples
CPU / XPU (Windows PDH via QDF ``bench/sampler.py`` when available) and wall
time, then writes JSON + matplotlib figures similar to QDF
``run_pipeline.ipynb``.

Examples
--------

A vs B (default 2-way, unchanged from before)::

    cd c:\\DGCL\\3DGCL
    .venv\\Scripts\\python.exe examples\\sslgraph\\bench\\compare_pretrain_ab.py \\
        --dataset esol --warmup 1 --epochs 1 --max-iters 3

3-way (baseline / FAST / WEIGHTED), e.g. to study MMFF weighted view::

    .venv\\Scripts\\python.exe examples\\sslgraph\\bench\\compare_pretrain_ab.py \\
        --dataset esol --warmup 1 --epochs 1 --max-iters 3 \\
        --c-label "weighted (MMFF_WEIGHTED + bf16)" \\
        --c-set MMFFRANDOM_WEIGHTED=1 \\
        --c-set PRETRAIN_AMP=bf16

Defaults
--------

* A: clears ``MMFFRANDOM_FAST`` and ``PRETRAIN_AMP``  (legacy PyG path)
* B: sets ``MMFFRANDOM_FAST=1``, ``PRETRAIN_AMP=bf16``
* C: only runs when ``--c-label`` or ``--c-set`` is supplied
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


def _fmt_ms(v: float) -> str:
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def main() -> int:
    os.chdir(REPO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esol")
    ap.add_argument("--root", default="dataset/")
    ap.add_argument("--batch-size", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--interval", type=float, default=0.1,
                    help="resource sampler interval (s)")
    ap.add_argument("--out-dir", default="",
                    help="default: examples/sslgraph/bench/figs/pretrain_ab_<ts>/")
    ap.add_argument("--no-plot", action="store_true")

    ap.add_argument("--a-label", default="baseline (PyG views)")
    ap.add_argument("--a-clear", action="append", default=[], metavar="KEY",
                    help="unset env for run A (repeatable). "
                         "Default clears MMFFRANDOM_FAST,MMFFRANDOM_WEIGHTED,PRETRAIN_AMP")
    ap.add_argument("--a-set", action="append", default=[], metavar="KEY=VAL",
                    help="set env for run A (repeatable, default empty)")

    ap.add_argument("--b-label", default="optimized (MMFF_FAST + bf16)")
    ap.add_argument("--b-clear", action="append", default=[], metavar="KEY",
                    help="unset env for run B (repeatable, default empty)")
    ap.add_argument("--b-set", action="append", default=[], metavar="KEY=VAL",
                    help="set env for run B (repeatable). "
                         "Default MMFFRANDOM_FAST=1 PRETRAIN_AMP=bf16")

    ap.add_argument("--c-label", default="",
                    help="If non-empty (or --c-set given), runs a 3rd config C.")
    ap.add_argument("--c-clear", action="append", default=[], metavar="KEY")
    ap.add_argument("--c-set", action="append", default=[], metavar="KEY=VAL")

    args = ap.parse_args()

    ts = time.strftime("%m%d_%H%M")
    out = Path(args.out_dir) if args.out_dir else (
        REPO / "examples" / "sslgraph" / "bench" / "figs" / f"pretrain_ab_{ts}"
    )
    out.mkdir(parents=True, exist_ok=True)

    # ---- build env overrides per side ----
    a_clear = list(args.a_clear) or [
        "MMFFRANDOM_FAST", "MMFFRANDOM_WEIGHTED", "PRETRAIN_AMP",
    ]
    a_overrides: dict[str, str | None] = {k: None for k in a_clear}
    a_overrides.update(dict(_parse_kv(s) for s in args.a_set))

    b_clear_extra = list(args.b_clear)
    b_set_kv = list(args.b_set) or ["MMFFRANDOM_FAST=1", "PRETRAIN_AMP=bf16"]
    b_overrides: dict[str, str | None] = {k: None for k in b_clear_extra}
    b_overrides.update(dict(_parse_kv(s) for s in b_set_kv))

    run_c_enabled = bool(args.c_label or args.c_set or args.c_clear)
    if run_c_enabled:
        c_label = args.c_label or "third (custom)"
        c_overrides: dict[str, str | None] = {k: None for k in args.c_clear}
        c_overrides.update(dict(_parse_kv(s) for s in args.c_set))
    else:
        c_label = ""
        c_overrides = {}

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
    sides: list[tuple[str, dict, Any, Any]] = [
        (args.a_label, run_a, samp_a, sum_a),
        (args.b_label, run_b, samp_b, sum_b),
    ]
    if run_c_enabled:
        run_c, samp_c, sum_c = run_side(c_label, c_overrides)
        sides.append((c_label, run_c, samp_c, sum_c))
    else:
        run_c = samp_c = sum_c = None

    payload: dict[str, Any] = {
        "ts": ts,
        "labels": [s[0] for s in sides],
        "a_label": args.a_label,
        "b_label": args.b_label,
        "c_label": c_label if run_c_enabled else None,
        "a_overrides": a_overrides,
        "b_overrides": b_overrides,
        "c_overrides": c_overrides if run_c_enabled else None,
        "args": vars(args),
        "run_a": run_a,
        "run_b": run_b,
        "run_c": run_c,
        "summary_a": _summary_dict(sum_a),
        "summary_b": _summary_dict(sum_b),
        "summary_c": _summary_dict(sum_c) if run_c_enabled else None,
    }
    json_path = out / "pretrain_ab_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[saved] {json_path}")

    if args.no_plot:
        return 0

    n_sides = len(sides)
    palette = ["#4C78A8", "#F58518", "#54A24B"][:n_sides]
    width = 0.36 if n_sides == 2 else 0.27
    offsets = (
        [-width / 2, +width / 2] if n_sides == 2 else [-width, 0.0, +width]
    )

    # ---- Bar chart (QDF-style scalar comparison) ----
    labels_m = ["wall (s)", "mean CPU%", "mean XPU%", "peak RSS (MB)"]
    fmts_m = ["{:.2f}", "{:.1f}", "{:.1f}", "{:.0f}"]
    x = np.arange(len(labels_m))
    fig, ax = plt.subplots(figsize=(9.5 if n_sides == 2 else 10.5, 4.7))
    for k, (label, _bench, _samp, sm) in enumerate(sides):
        vals = [sm.wall_s, sm.mean_cpu_pct, sm.mean_xpu_pct, sm.peak_rss_mb]
        bars = ax.bar(x + offsets[k], vals, width=width,
                      label=label, color=palette[k])
        ax.bar_label(
            bars,
            labels=[f.format(v) for f, v in zip(fmts_m, vals)],
            padding=2, fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels_m)
    ax.set_ylabel("value (see x labels for units)")
    ax.set_title(f"Pretrain {'A/B/C' if n_sides == 3 else 'A/B'} — "
                 f"{args.dataset} batch={args.batch_size}")
    ax.legend(loc="upper left", fontsize=9)
    ax.margins(y=0.2)
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
    x2 = np.arange(len(stages))
    fig2, ax2 = plt.subplots(figsize=(10.0 if n_sides == 2 else 11.0, 4.8))
    for k, (label, bench, _samp, _sm) in enumerate(sides):
        vals = stage_ms(bench["totals"])
        bars = ax2.bar(x2 + offsets[k], vals, width=width,
                       label=label, color=palette[k])
        ax2.bar_label(
            bars,
            labels=[f"{_fmt_ms(v)} ms" for v in vals],
            padding=2, fontsize=8,
        )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(stages)
    ax2.set_ylabel("ms (summed over measured epochs)")
    ax2.set_title("Stage timing (micro-bench loop)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.margins(y=0.22)
    fig2.tight_layout()
    st_path = out / f"bars_pretrain_stages_{ts}.png"
    fig2.savefig(st_path, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    print(f"[saved] {st_path}")

    # ---- Timeseries CPU / XPU (one column per side) ----
    def _dt(t_list: list[float]) -> float:
        if len(t_list) < 2:
            return args.interval
        return max(1e-6, float(np.median(np.diff(t_list))))

    fig3, axes = plt.subplots(
        2, n_sides,
        figsize=(5.5 * n_sides, 7),
        sharex=False, squeeze=False,
    )
    for col, (label, _bench, samp, sm) in enumerate(sides):
        t = np.asarray(samp.t, dtype=float)
        cpu = np.asarray(samp.cpu_pct, dtype=float)
        xpu = np.asarray(samp.xpu_pct, dtype=float)
        dt = _dt(list(samp.t))
        axes[0, col].plot(t, _smooth(cpu, dt), color="#4C78A8", lw=1.2)
        axes[0, col].set_ylabel("CPU % (system)")
        axes[0, col].set_title(
            f"{label}\nwall={sm.wall_s:.2f}s  mean_cpu={sm.mean_cpu_pct:.1f}%"
        )
        axes[0, col].grid(True, alpha=0.3)
        axes[1, col].plot(t, _smooth(xpu, dt), color="#F58518", lw=1.2)
        axes[1, col].set_ylabel("XPU % (GPU engines Σ)")
        axes[1, col].set_xlabel("time (s)")
        axes[1, col].set_title(
            f"mean_xpu={sm.mean_xpu_pct:.1f}%  peak={sm.peak_xpu_pct:.1f}%"
        )
        axes[1, col].grid(True, alpha=0.3)
    fig3.suptitle(f"Resource trace — {args.dataset}  sampler={args.interval}s")
    fig3.tight_layout()
    ts_path = out / f"timeseries_cpu_xpu_{ts}.png"
    fig3.savefig(ts_path, dpi=160, bbox_inches="tight")
    plt.close(fig3)
    print(f"[saved] {ts_path}")

    # ---- Ratios ----
    ratio_ab = sum_a.wall_s / max(sum_b.wall_s, 1e-9)
    print(f"\n=== Ratio (A wall / B wall) = {ratio_ab:.2f}x  (higher = B faster) ===")
    if run_c_enabled:
        ratio_ac = sum_a.wall_s / max(sum_c.wall_s, 1e-9)
        ratio_bc = sum_b.wall_s / max(sum_c.wall_s, 1e-9)
        print(f"=== Ratio (A wall / C wall) = {ratio_ac:.2f}x  (higher = C faster) ===")
        print(f"=== Ratio (B wall / C wall) = {ratio_bc:.2f}x  (>1: C faster than B) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
