#!/usr/bin/env python3
"""One-shot QDF→MMFF weights→GraphCL micro-bench on **CPU** and **XPU** (two passes).

Creates a **single** output directory::

    examples/sslgraph/bench/figs/qdf_ensemble_cpu_xpu_<YYYYMMDD_HHMMSS>/
        manifest.json
        cpu/steps.json
        cpu/qdf_preds.csv
        cpu/mmff_weights.pt
        cpu/pretrain_ab/   (compare_pretrain_ab outputs + PNGs)
        xpu/...
        compare_wall_by_step.png
        compare_resources.png
        SUMMARY.md

Requires: ESOL (or chosen) ``MoleculeNet`` under ``dataset/``, QDF checkpoint
for ``--qdf-property`` (default ``atomization``), and for XPU a ``torch+xpu``
build where ``torch.xpu.is_available()`` is true.

Reuses existing tools: ``qdf_mmff_predict.py``, ``compute_mmff_weights.py``,
``compare_pretrain_ab.py`` (see ``compare_pretrain_quality.ipynb`` for the
full quality pipeline).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parents[2]
FIGS = BENCH / "figs"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _run(
    *,
    label: str,
    argv: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*64}", flush=True)
    print(f"[{label}] starting …", flush=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            [sys.executable, *[str(a) for a in argv]],
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            logf.write(line)
        proc.wait()
    wall = time.perf_counter() - t0
    rc = proc.returncode
    print(f"[{label}] {'OK' if rc == 0 else 'FAILED (rc=' + str(rc) + ')'}  wall={wall:.1f}s", flush=True)
    return {
        "step": label,
        "wall_sec": wall,
        "returncode": rc,
        "log": str(log_path.relative_to(REPO)).replace("\\", "/"),
    }


def _torch_xpu_available() -> bool:
    try:
        import torch

        xm = getattr(torch, "xpu", None)
        if xm is None:
            return False
        return bool(getattr(xm, "is_available", lambda: False)())
    except Exception:
        return False


def _plot_bars(cpu_steps: list[dict], xpu_steps: list[dict], out_png: Path) -> None:
    names = [s["step"] for s in cpu_steps]
    y_cpu = [s["wall_sec"] for s in cpu_steps]
    y_xpu = [s["wall_sec"] for s in xpu_steps]
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, y_cpu, width=w, label="CPU (TORCH_DEVICE=cpu)", color="#4C72B0")
    ax.bar(x + w / 2, y_xpu, width=w, label="XPU (TORCH_DEVICE=xpu)", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("wall time (s)")
    ax.set_title("QDF ensemble pipeline — step wall times (CPU vs XPU)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_resources(cpu_json: Path, xpu_json: Path, out_png: Path) -> None:
    def load(path: Path) -> dict:
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    ca, cx = load(cpu_json), load(xpu_json)
    labels = ["wall (s)", "mean CPU%", "mean XPU%", "peak RSS (MB)"]
    keys = [("summary_a", "wall_s"), ("summary_a", "mean_cpu_pct"), ("summary_a", "mean_xpu_pct"), ("summary_a", "peak_rss_mb")]
    kb = [("summary_b", "wall_s"), ("summary_b", "mean_cpu_pct"), ("summary_b", "mean_xpu_pct"), ("summary_b", "peak_rss_mb")]

    def pick(d: dict, sk: tuple[str, str]) -> float:
        sec, k = sk
        v = (d.get(sec) or {}).get(k)
        return float(v) if v is not None else 0.0

    va = [pick(ca, keys[i]) for i in range(4)]
    vb = [pick(ca, kb[i]) for i in range(4)]
    vxa = [pick(cx, keys[i]) for i in range(4)]
    vxb = [pick(cx, kb[i]) for i in range(4)]

    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - 1.5 * w, va, width=w, label="CPU — A (baseline)", color="#9ecae9")
    ax.bar(x - 0.5 * w, vb, width=w, label="CPU — B (fast+bf16)", color="#3182bd")
    ax.bar(x + 0.5 * w, vxa, width=w, label="XPU — A", color="#fdd0a2")
    ax.bar(x + 1.5 * w, vxb, width=w, label="XPU — B", color="#e6550d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("value (see label)")
    ax.set_title("compare_pretrain_ab resource summary (from pretrain_ab_results.json)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_summary_md(
    path: Path,
    *,
    run_root: Path,
    cpu_steps: list[dict],
    xpu_steps: list[dict],
    xpu_skipped: bool,
    xpu_reason: str,
) -> None:
    lines = [
        "# QDF ensemble CPU vs XPU — run summary",
        "",
        f"- **Run root:** `{run_root.as_posix()}`",
        f"- **XPU branch:** {'skipped — ' + xpu_reason if xpu_skipped else 'completed'}",
        "",
        "## Step wall times (seconds)",
        "",
        "| step | CPU | XPU |",
        "|------|-----|-----|",
    ]
    for i, a in enumerate(cpu_steps):
        if xpu_skipped or i >= len(xpu_steps):
            xv = "—"
        else:
            xv = f"{xpu_steps[i]['wall_sec']:.3f}"
        lines.append(f"| {a['step']} | {a['wall_sec']:.3f} | {xv} |")
    lines += ["", "Logs under `cpu/logs/` and `xpu/logs/`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_mode(
    *,
    mode: str,
    run_root: Path,
    base_env: dict[str, str],
    args: argparse.Namespace,
) -> list[dict]:
    mdir = run_root / mode
    logs = mdir / "logs"
    mdir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'#'*64}", flush=True)
    print(f"# MODE: {mode.upper()}  ({3} steps)", flush=True)
    print(f"{'#'*64}", flush=True)

    pred_csv = mdir / "qdf_mmff_preds.csv"
    weights_pt = mdir / "mmff_weights.pt"
    pre_ab = mdir / "pretrain_ab"

    steps: list[dict] = []

    env = dict(base_env)
    if mode == "cpu":
        env["TORCH_DEVICE"] = "cpu"
        env["TORCH_DISABLE_XPU_DEFAULT"] = "1"
        dev_arg = "cpu"
    else:
        env["TORCH_DEVICE"] = "xpu"
        env.pop("TORCH_DISABLE_XPU_DEFAULT", None)
        dev_arg = "xpu"

    env["MMFF_WEIGHTS_PATH"] = str(weights_pt.resolve())
    env.setdefault("DATALOADER_NUM_WORKERS", "0")
    env.setdefault("PIN_MEMORY", "0")

    argv_pred = [
        str(BENCH / "qdf_mmff_predict.py"),
        "--dataset",
        args.dataset,
        "--root",
        args.root,
        "--qdf-property",
        args.qdf_property,
        "--device",
        dev_arg,
        "--out",
        str(pred_csv),
        "--batch-size",
        str(args.qdf_batch_size),
    ]
    if args.limit > 0:
        argv_pred += ["--limit", str(args.limit)]
    if args.checkpoint:
        argv_pred += ["--checkpoint", str(Path(args.checkpoint).resolve())]

    steps.append(
        _run(
            label="01_qdf_mmff_predict",
            argv=argv_pred,
            cwd=REPO,
            env=env,
            log_path=logs / "01_qdf_mmff_predict.log",
        )
    )
    if steps[-1]["returncode"] != 0:
        (mdir / "steps.json").write_text(json.dumps(steps, indent=2), encoding="utf-8")
        return steps

    argv_w = [
        str(BENCH / "compute_mmff_weights.py"),
        "--dataset",
        args.dataset,
        "--root",
        args.root,
        "--source",
        "qdf",
        "--pred-csv",
        str(pred_csv),
        "--kT",
        str(args.weight_kT),
        "--normalize",
        args.weight_normalize,
        "--fallback",
        args.weight_fallback,
        "--out",
        str(weights_pt),
    ]
    steps.append(
        _run(
            label="02_compute_mmff_weights",
            argv=argv_w,
            cwd=REPO,
            env=env,
            log_path=logs / "02_compute_mmff_weights.log",
        )
    )
    if steps[-1]["returncode"] != 0:
        (mdir / "steps.json").write_text(json.dumps(steps, indent=2), encoding="utf-8")
        return steps

    argv_ab = [
        str(BENCH / "compare_pretrain_ab.py"),
        "--dataset",
        args.dataset,
        "--root",
        args.root,
        "--batch-size",
        str(args.pretrain_batch_size),
        "--warmup",
        str(args.pretrain_warmup),
        "--epochs",
        str(args.pretrain_epochs),
        "--max-iters",
        str(args.pretrain_max_iters),
        "--out-dir",
        str(pre_ab),
        "--interval",
        str(args.sampler_interval),
    ]
    steps.append(
        _run(
            label="03_compare_pretrain_ab",
            argv=argv_ab,
            cwd=REPO,
            env=env,
            log_path=logs / "03_compare_pretrain_ab.log",
        )
    )

    (mdir / "steps.json").write_text(json.dumps(steps, indent=2), encoding="utf-8")
    return steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="esol")
    ap.add_argument("--root", default="dataset/")
    ap.add_argument("--qdf-property", default="atomization", choices=("atomization", "homolumo"))
    ap.add_argument("--checkpoint", default="", help="Optional absolute/relative QDF checkpoint path")
    ap.add_argument("--limit", type=int, default=0, help="QDF predict limit (0=all). Use 256 for smoke.")
    ap.add_argument("--qdf-batch-size", type=int, default=8)
    ap.add_argument("--weight-kT", type=float, default=0.5)
    ap.add_argument("--weight-normalize", default="zscore")
    ap.add_argument("--weight-fallback", default="boltzmann")
    ap.add_argument("--pretrain-batch-size", type=int, default=128)
    ap.add_argument("--pretrain-warmup", type=int, default=1)
    ap.add_argument("--pretrain-epochs", type=int, default=1)
    ap.add_argument("--pretrain-max-iters", type=int, default=8)
    ap.add_argument("--sampler-interval", type=float, default=0.15)
    ap.add_argument("--run-root", type=Path, default=None, help="Override output directory")
    ap.add_argument(
        "--no-skip-xpu",
        action="store_true",
        help="Attempt XPU even when torch.xpu.is_available() is false (+xpu wheels / driver quirks).",
    )
    ap.add_argument("--force-xpu", action="store_true", help="Alias of --no-skip-xpu")
    args = ap.parse_args()

    run_root = args.run_root
    if run_root is None:
        FIGS.mkdir(parents=True, exist_ok=True)
        run_root = FIGS / f"qdf_ensemble_cpu_xpu_{_ts()}"
    run_root.mkdir(parents=True, exist_ok=False)

    base_env = os.environ.copy()
    base_env["PYTHONIOENCODING"] = "utf-8"
    base_env["PYTHONUTF8"] = "1"

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "args": vars(args),
        "run_root": str(run_root),
        "torch_version": None,
    }
    try:
        import torch

        manifest["torch_version"] = torch.__version__
    except Exception:
        pass

    print(f"[orchestrator] run_root={run_root}", flush=True)
    print(f"[orchestrator] dataset={args.dataset}  limit={args.limit}  max_iters={args.pretrain_max_iters}", flush=True)

    cpu_steps = run_mode(mode="cpu", run_root=run_root, base_env=base_env, args=args)
    manifest["cpu_steps"] = cpu_steps

    xpu_skipped = False
    xpu_reason = ""
    xpu_steps: list[dict] = []
    try_xpu = args.no_skip_xpu or args.force_xpu or _torch_xpu_available()
    if not try_xpu:
        xpu_skipped = True
        xpu_reason = "torch.xpu.is_available() is false — pass --no-skip-xpu to try anyway"
        print(f"\n[orchestrator] XPU skipped: {xpu_reason}", flush=True)
    else:
        xpu_steps = run_mode(mode="xpu", run_root=run_root, base_env=base_env, args=args)
        if any(s.get("returncode", 1) != 0 for s in xpu_steps):
            xpu_reason = "one or more XPU steps failed — see xpu/logs/"

    manifest["xpu_steps"] = xpu_steps
    manifest["xpu_skipped"] = xpu_skipped
    manifest["xpu_skip_reason"] = xpu_reason
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if len(xpu_steps) == len(cpu_steps) and len(cpu_steps) > 0:
        _plot_bars(cpu_steps, xpu_steps, run_root / "compare_wall_by_step.png")
        cj = run_root / "cpu" / "pretrain_ab" / "pretrain_ab_results.json"
        xj = run_root / "xpu" / "pretrain_ab" / "pretrain_ab_results.json"
        if cj.is_file() and xj.is_file():
            _plot_resources(cj, xj, run_root / "compare_resources.png")
    else:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.bar([s["step"] for s in cpu_steps], [s["wall_sec"] for s in cpu_steps], color="#4C72B0")
        ax.set_ylabel("wall (s)")
        ax.set_title("CPU-only steps (XPU skipped)")
        plt.xticks(rotation=15, ha="right")
        fig.tight_layout()
        fig.savefig(run_root / "compare_wall_by_step.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    _write_summary_md(
        run_root / "SUMMARY.md",
        run_root=run_root,
        cpu_steps=cpu_steps,
        xpu_steps=xpu_steps if not xpu_skipped else [],
        xpu_skipped=xpu_skipped,
        xpu_reason=xpu_reason,
    )

    try:
        import pandas as pd

        rows = [{"mode": "cpu", **s} for s in cpu_steps] + [{"mode": "xpu", **s} for s in xpu_steps]
        pd.DataFrame(rows).to_csv(run_root / "step_walls.csv", index=False)
    except Exception:
        pass

    print(f"\n{'='*64}", flush=True)
    print(f"[orchestrator] ALL DONE  run_root={run_root}", flush=True)
    if any(s.get("returncode", 1) != 0 for s in cpu_steps):
        return 1
    if not xpu_skipped and xpu_steps and any(s.get("returncode", 1) != 0 for s in xpu_steps):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
