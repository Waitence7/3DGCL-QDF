#!/usr/bin/env python3
"""Repeat QDF profile_preprocess (full QM9under14 train split) toward a target wall time.

Default targets ~1 hour: numpy + rust each full pass ~64s → ~56 pairs ≈ 60 min.

Usage (repo root)::

    .venv\\Scripts\\python.exe examples\\sslgraph\\bench\\run_1h_preprocess_repeat.py --target-seconds 3600
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PRE = REPO / "QuantumDeepField_molecule" / "bench" / "profile_preprocess.py"
DATASET = "QM9under14atoms_atomizationenergy_eV"


def one_run(backend: str, rust_batch: int) -> tuple[float, int, str]:
    args = [
        sys.executable,
        str(PRE),
        "--dataset",
        DATASET,
        "--split",
        "train",
        "--limit",
        "0",
        "--no-save",
        "--backend",
        backend,
    ]
    if backend == "rust":
        args += ["--rust-batch-size", str(rust_batch)]
    t0 = time.perf_counter()
    cp = subprocess.run(
        args,
        cwd=str(REPO),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    wall = time.perf_counter() - t0
    tail = (cp.stdout or "")[-800:]
    return wall, cp.returncode, tail


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target-seconds", type=float, default=3600.0)
    p.add_argument("--rust-batch-size", type=int, default=64)
    p.add_argument(
        "--log-jsonl",
        type=Path,
        default=REPO / "examples" / "sslgraph" / "bench" / "bench_1h_preprocess.jsonl",
    )
    args = p.parse_args()

    if not PRE.is_file():
        print("Missing", PRE, file=sys.stderr)
        return 2

    round_idx = 0
    t_start = time.perf_counter()
    args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.log_jsonl.open("w", encoding="utf-8") as log:
        while time.perf_counter() - t_start < args.target_seconds:
            round_idx += 1
            for backend in ("numpy", "rust"):
                wall, rc, tail = one_run(backend, args.rust_batch_size)
                rec = {
                    "round": round_idx,
                    "backend": backend,
                    "wall_sec": wall,
                    "returncode": rc,
                    "elapsed_total_sec": time.perf_counter() - t_start,
                }
                log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                log.flush()
                print(
                    f"[{rec['elapsed_total_sec']:8.1f}s] round={round_idx} {backend:5s} "
                    f"wall={wall:8.2f}s rc={rc}",
                    flush=True,
                )
                if rc != 0:
                    print("STDERR tail:\n", tail[-2000:], file=sys.stderr)
                    return 1

    total = time.perf_counter() - t_start
    print(f"Done. rounds={round_idx} total_wall={total:.1f}s log={args.log_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
