#!/usr/bin/env python3
"""
Benchmark wall-clock time for one Finetune.evaluate() run (ESOL regression, from-scratch).

- Runs each device in a **fresh subprocess** (clean torch / XPU state).
- Uses DATALOADER_NUM_WORKERS via env (same as the notebooks).

**Intel XPU 주의:** 이 레포는 `pyproject.toml`에 `default-groups = ["cpu"]`가 있어서,
`uv run benchmark_cpu_vs_xpu.py`만 치면 **CPU용 torch로 다시 맞춰** XPU 측정이 실패할 수 있습니다.

XPU 비교를 하려면 **레포 루트**에서 먼저:

  uv sync --no-default-groups --group xpu

그다음 **둘 중 하나**로 실행 (권장: venv python 직접):

  .venv/Scripts/python.exe examples/sslgraph/benchmark_cpu_vs_xpu.py --devices cpu xpu:0

또는:

  uv run --no-default-groups --group xpu python examples/sslgraph/benchmark_cpu_vs_xpu.py --devices cpu xpu:0
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def _repo_root(script_path: Path) -> Path:
    # .../examples/sslgraph/benchmark_cpu_vs_xpu.py -> repo root is parents[3]
    return script_path.resolve().parent.parent.parent


def _sslgraph_examples_dir(script_path: Path) -> Path:
    return script_path.resolve().parent


def _build_benchmark_args_ns(
    *,
    device_str: str,
    batch_size: int,
    epochs: int,
    n_folds: int,
    n_times: int,
    z_dim: int,
    cutoff: float,
    num_layers: int,
    num_filters: int,
    num_gaussians: int,
    dropout_rate: float,
    f_lr: float,
    f_weight_decay: float,
    finetune: bool,
    model_path: str,
    edge_weight: bool,
) -> SimpleNamespace:
    import torch

    # Mirror downstream/finetune notebook fields required by Finetune + Encoder (schnet).
    args = SimpleNamespace()
    args.finetune = finetune
    args.seed = 2222
    args.model_path = model_path

    args.device = torch_device_from_str(device_str, torch)

    args.dataset = "esol"
    args.batch_size = batch_size

    args.encoder = "schnet"
    args.cutoff = cutoff
    args.num_layers = num_layers
    args.num_filters = num_filters
    args.num_gaussians = num_gaussians
    args.z_dim = z_dim
    args.edge_weight = edge_weight

    args.n_times = n_times
    args.n_folds = n_folds
    args.f_epoch = epochs
    args.f_lr = f_lr
    args.aug_1, args.aug_2 = "MMFFrandom", "MMFFrandom"
    args.aug_ratio = 0.2
    args.tau = 0.2
    args.proj = "schnet"

    args.dropout_rate = dropout_rate
    args.f_optim = "ExponentialLR"
    args.f_weight_decay = f_weight_decay
    args.f_lr_decay_step_size = 20
    args.f_lr_decay_factor = 0.5
    args.expo_gamma = 0.95

    args.T_0 = 100
    args.T_mult = 1
    args.eta_max = 0.05
    args.T_up = 10
    args.gamma = 0.5
    args.target = "y"
    return args


def torch_device_from_str(s: str, torch):
    s = (s or "cpu").strip().lower()
    if s == "cpu":
        return torch.device("cpu")
    if s.startswith("xpu"):
        idx = "0"
        if ":" in s:
            idx = s.split(":", 1)[1].strip() or "0"
        return torch.device(f"xpu:{idx}")
    if s.startswith("cuda"):
        idx = "0"
        if ":" in s:
            idx = s.split(":", 1)[1].strip() or "0"
        return torch.device(f"cuda:{idx}")
    return torch.device(s)


def _wants_accelerator(cli: argparse.Namespace) -> bool:
    if cli.mode == "measure":
        d = (cli.device or "cpu").strip().lower()
        return d.startswith("xpu") or d.startswith("cuda")
    for raw in cli.devices:
        d = raw.strip().lower()
        if d.startswith("xpu") or d.startswith("cuda"):
            return True
    return False


def _warn_if_cpu_only_torch_for_accel(cli: argparse.Namespace) -> None:
    if not _wants_accelerator(cli):
        return
    try:
        import torch
    except Exception:
        return
    v = (torch.__version__ or "").lower()
    if "+cpu" in v and "+xpu" not in v and "+cu" not in v:
        sys.stderr.write(
            f"\n[benchmark] 경고: 현재 python={sys.executable}\n"
            f"            torch={torch.__version__} (CPU 전용 휠로 보임)\n"
            "XPU/CUDA 측정이 필요하면 레포 루트에서 `uv sync --no-default-groups --group xpu` 후\n"
            "`.venv/Scripts/python.exe`로 이 스크립트를 실행하거나,\n"
            "`uv run --no-default-groups --group xpu python ...` 를 쓰세요.\n"
            "(`uv run`만 쓰면 default cpu 그룹 때문에 torch가 +cpu로 덮어씌워질 수 있음.)\n\n"
        )


def _validate_accelerator(torch, torch_device) -> None:
    """Fail fast instead of dumping long dispatcher/backend lists from PyTorch."""
    ver = (torch.__version__ or "").lower()
    dt = torch_device.type

    if dt == "xpu" and "+cpu" in ver and "+xpu" not in ver:
        raise RuntimeError(
            "XPU 디바이스를 요청했지만 현재 PyTorch는 CPU 전용 빌드입니다 "
            f"({torch.__version__}).\n"
            "프로젝트에서: `uv sync --no-default-groups --group xpu` 후 같은 venv에서 "
            "`python -c \"import torch; print(torch.__version__)\"`가 `+xpu`를 포함하는지 확인하세요.\n"
            "`uv run`이 다른 환경을 쓰면 CPU 휠로 덮어씌워질 수 있습니다."
        )

    if dt == "xpu":
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError(
                "XPU를 사용할 수 없습니다 (torch.xpu 없음 또는 is_available=False). "
                f"torch={torch.__version__}"
            )
        dc = int(getattr(torch.xpu, "device_count", lambda: 0)())
        idx = int(torch_device.index if torch_device.index is not None else 0)
        if dc <= idx:
            raise RuntimeError(f"XPU device_count={dc}, 요청 인덱스={idx}")

    if dt == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 디바이스를 요청했지만 torch.cuda.is_available() == False 입니다.\n"
            "NVIDIA CUDA용 torch를 설치했는지 확인하세요."
        )


def run_measure(cli: argparse.Namespace) -> dict:
    import time as time_mod

    import torch

    os.environ.setdefault("TORCH_SKIP_NPU", "1")
    os.environ["DATALOADER_NUM_WORKERS"] = str(cli.num_workers)
    os.environ.setdefault("TQDM_DISABLE", "1")

    ssl_dir = _sslgraph_examples_dir(Path(__file__))
    repo = _repo_root(Path(__file__))
    os.chdir(ssl_dir)
    sys.path.insert(0, str(repo))

    args = _build_benchmark_args_ns(
        device_str=cli.device,
        batch_size=cli.batch_size,
        epochs=cli.epochs,
        n_folds=cli.n_folds,
        n_times=cli.n_times,
        z_dim=cli.z_dim,
        cutoff=cli.cutoff,
        num_layers=cli.num_layers,
        num_filters=cli.num_filters,
        num_gaussians=cli.num_gaussians,
        dropout_rate=cli.dropout,
        f_lr=cli.lr,
        f_weight_decay=cli.weight_decay,
        finetune=cli.finetune,
        model_path=cli.model_path or ".",
        edge_weight=cli.edge_weight,
    )

    torch_device = args.device
    print(f"[measure] cwd={ssl_dir}", flush=True)
    print(f"[measure] torch={torch.__version__} device={torch_device}", flush=True)

    _validate_accelerator(torch, torch_device)
    if torch_device.type == "xpu":
        print(
            f"[measure] xpu is_available={torch.xpu.is_available()} "
            f"device_count={getattr(torch.xpu, 'device_count', lambda: 'n/a')()}",
            flush=True,
        )

    from dig.sslgraph.evaluation.finetune import Finetune

    evaluator = Finetune(args=args)
    evaluator.setup_train_config(
        batch_size=args.batch_size,
        cutoff=args.cutoff,
        num_layers=args.num_layers,
        num_filters=args.num_filters,
        num_gaussians=args.num_gaussians,
        z_dim=args.z_dim,
        dropout_rate=args.dropout_rate,
        target=args.target,
        f_lr=args.f_lr,
        f_weight_decay=args.f_weight_decay,
    )

    t0 = time_mod.perf_counter()
    loss_m, loss_sd, *_ = evaluator.evaluate()
    t1 = time_mod.perf_counter()
    elapsed = t1 - t0

    out = {
        "device_requested": cli.device,
        "device_used": str(torch_device),
        "num_workers": int(cli.num_workers),
        "epochs": int(cli.epochs),
        "n_folds": int(cli.n_folds),
        "n_times": int(cli.n_times),
        "batch_size": int(cli.batch_size),
        "seconds_wall": round(elapsed, 3),
        "test_rmse_mean": float(loss_m) if loss_m == loss_m else None,
        "test_rmse_sd": float(loss_sd) if loss_sd == loss_sd else None,
        "torch_version": torch.__version__,
    }
    print("__BENCH_RESULT__:" + json.dumps(out), flush=True)
    return out


def orchestrate(cli: argparse.Namespace) -> None:
    script = Path(__file__).resolve()
    results = []
    for dev in cli.devices:
        dev_clean = dev.strip()
        dl = dev_clean.lower()
        if dl.startswith("xpu") and ":" not in dev_clean:
            dev_clean = "xpu:0"
        env = os.environ.copy()
        env.setdefault("TORCH_SKIP_NPU", "1")
        env["DATALOADER_NUM_WORKERS"] = str(cli.num_workers)
        env.setdefault("TQDM_DISABLE", "1")

        cmd = [
            sys.executable,
            str(script),
            "--mode",
            "measure",
            "--device",
            dev_clean,
            "--epochs",
            str(cli.epochs),
            "--batch-size",
            str(cli.batch_size),
            "--n-folds",
            str(cli.n_folds),
            "--n-times",
            str(cli.n_times),
            "--z-dim",
            str(cli.z_dim),
            "--cutoff",
            str(cli.cutoff),
            "--num-layers",
            str(cli.num_layers),
            "--num-filters",
            str(cli.num_filters),
            "--num-gaussians",
            str(cli.num_gaussians),
            "--num-workers",
            str(cli.num_workers),
            "--lr",
            str(cli.lr),
            "--weight-decay",
            str(cli.weight_decay),
            "--dropout",
            str(cli.dropout),
        ]
        if cli.edge_weight:
            cmd.append("--edge-weight")
        if cli.finetune:
            cmd.append("--finetune")
        if cli.model_path:
            cmd.extend(["--model-path", cli.model_path])

        print(f"\n=== subprocess: {' '.join(cmd)} ===\n", flush=True)
        p = subprocess.run(cmd, env=env, cwd=str(script.parent), text=True, capture_output=True)
        combined = (p.stdout or "") + (p.stderr or "")
        if p.returncode != 0:
            print(combined)
            raise SystemExit(p.returncode)
        bench_line = None
        for line in combined.splitlines():
            if line.startswith("__BENCH_RESULT__:"):
                bench_line = line[len("__BENCH_RESULT__:") :]
        if bench_line is None:
            print(combined)
            raise RuntimeError(f"No benchmark result marker for device {dev_clean}")

        payload = json.loads(bench_line)
        results.append(payload)
        print(json.dumps(payload, indent=2), flush=True)

    print("\n=== summary ===")
    baseline = results[0]["seconds_wall"] if results else None
    for r in results:
        speedup = (
            round(baseline / r["seconds_wall"], 3)
            if baseline and r["seconds_wall"]
            else None
        )
        print(
            f"{r['device_used']:>12} | {r['seconds_wall']:>8.3f}s | "
            f"RMSE(mean)={r['test_rmse_mean']} | workers={r['num_workers']}"
            + (f" | vs_first={speedup}x" if speedup else "")
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--mode",
        choices=["orchestrate", "measure"],
        default="orchestrate",
        help="orchestrate: run subprocess per device; measure: timed single run (internal)",
    )
    p.add_argument(
        "--devices",
        nargs="+",
        default=["cpu", "xpu:0"],
        help="devices for orchestrate mode (examples: cpu xpu:0)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="single device for measure mode (examples: cpu, xpu:0, cuda:0)",
    )
    p.add_argument("--num-workers", type=int, default=0, help="maps to DATALOADER_NUM_WORKERS")
    p.add_argument("--epochs", type=int, default=2, help="f_epoch passed to Finetune")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-folds", type=int, default=1)
    p.add_argument("--n-times", type=int, default=1)
    p.add_argument("--z-dim", type=int, default=64)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-filters", type=int, default=128)
    p.add_argument("--num-gaussians", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--edge-weight", action="store_true", default=False)
    p.add_argument("--finetune", action="store_true", default=False)
    p.add_argument("--model-path", default="", help="checkpoint path when --finetune")
    return p.parse_args(argv)


def main(argv: list[str]) -> None:
    cli = parse_args(argv)
    _warn_if_cpu_only_torch_for_accel(cli)
    if cli.mode == "measure":
        run_measure(cli)
        return
    orchestrate(cli)


if __name__ == "__main__":
    main(sys.argv[1:])
