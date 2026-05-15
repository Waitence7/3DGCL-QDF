#!/usr/bin/env python3
"""Smoke pretrain+finetune then a mid-budget finetune (CLI helper for the notebook).

Matches ``compare_pretrain_quality.ipynb`` cell 1 when QUALITY_BUDGET='smoke',
then runs finetune again with longer f_epoch (between smoke and two_hour).
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parents[2]
os.chdir(REPO)
for p in (REPO, BENCH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch

import pretrain_quality_core as q
from dig.sslgraph.utils.device import pick_torch_device

DEVICE = pick_torch_device()
FIG_TS = datetime.now().strftime("%m%d_%H%M")
RUN_DIR = REPO / "examples/sslgraph/bench/figs" / f"pretrain_quality_{FIG_TS}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ROOT = REPO / "models" / f"quality_{FIG_TS}"
MODEL_ROOT.mkdir(parents=True, exist_ok=True)

DATASET = "esol"
DATASET_ROOT = REPO / "dataset"
BATCH_SIZE = 400
FINETUNE_BATCH = 128
WEIGHTS_PT = REPO / "dataset/esol_mmff_weights_qdf_atomization.pt"
SIDES = ("A", "C")
SEED = 2222
ESOL_FILTER_QM78 = False
ESOL_TEST_MAX_ATOMS = 14

# smoke (same as notebook QUALITY_BUDGET == 'smoke')
P_EPOCH, F_EPOCH = 5, 18
N_FOLDS, N_TIMES = 2, 1
PRETRAIN_ENCODER = "gin"
PRETRAIN_CUTOFF = 3.0
SIDE_C_AUG_2 = "top12"
WEIGHT_KT = 0.5
WEIGHT_MODE = "auto"
WEIGHT_NORM = "zscore"
PRETRAIN_AMP = DEVICE.type != "cpu"
PRETRAIN_PIN_MEMORY = DEVICE.type != "cpu"

os.environ["DATALOADER_NUM_WORKERS"] = "8"
if PRETRAIN_PIN_MEMORY:
    os.environ["PIN_MEMORY"] = "1"
else:
    os.environ.pop("PIN_MEMORY", None)

print("device:", DEVICE)
print("RUN_DIR:", RUN_DIR)
print("MODEL_ROOT:", MODEL_ROOT)
print(
    "smoke: P_EPOCH, F_EPOCH, N_FOLDS, N_TIMES =",
    P_EPOCH,
    F_EPOCH,
    N_FOLDS,
    N_TIMES,
)

pre_results: dict = {}
for side in SIDES:
    root = MODEL_ROOT / side
    root.mkdir(parents=True, exist_ok=True)
    kw = dict(
        dataset=DATASET,
        batch_size=BATCH_SIZE,
        p_epoch=P_EPOCH,
        model_root=root,
        device=DEVICE,
        weight_kT=WEIGHT_KT,
        weight_mode=WEIGHT_MODE,
        weight_norm=WEIGHT_NORM,
        p_pretrain_amp=PRETRAIN_AMP,
        encoder=PRETRAIN_ENCODER,
        cutoff=PRETRAIN_CUTOFF,
        side_c_aug_2=SIDE_C_AUG_2,
        esol_filter_qm78=ESOL_FILTER_QM78,
        dataset_root=DATASET_ROOT,
        seed=SEED,
    )
    if side in ("B", "C"):
        kw["weights_path"] = WEIGHTS_PT
    print(f"\n===== smoke pretrain side {side} =====")
    r = q.run_pretrain_side(side, **kw)
    pre_results[side] = r
    print("  best_ckpt:", r.get("best_ckpt"))
    print("  wall_s:", round(float(r.get("wall_s", 0)), 2))
    gc.collect()
    if hasattr(torch, "xpu"):
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass

with open(RUN_DIR / "pretrain_summary.json", "w", encoding="utf-8") as f:
    json.dump(pre_results, f, indent=2, default=str)
print("\n[saved]", RUN_DIR / "pretrain_summary.json")

# --- smoke finetune ---
fine_smoke: dict = {}
for side in SIDES:
    print(f"\n===== smoke finetune side {side} =====")
    t0 = time.perf_counter()
    r = q.run_finetune_side(
        pre_results[side],
        dataset=DATASET,
        batch_size=FINETUNE_BATCH,
        f_epoch=F_EPOCH,
        n_times=N_TIMES,
        n_folds=N_FOLDS,
        device=DEVICE,
        seed=SEED,
        esol_filter_qm78=ESOL_FILTER_QM78,
        dataset_root=DATASET_ROOT,
        test_max_atoms=ESOL_TEST_MAX_ATOMS,
    )
    fine_smoke[side] = r
    if r.get("rmse_mean") is not None:
        print(
            f"  RMSE = {r['rmse_mean']:.4f} ± {r['rmse_sd']:.4f}  "
            f"wall_s={r.get('wall_s', 0):.1f} elapsed={time.perf_counter() - t0:.1f}s"
        )
    else:
        print("  error:", r.get("error"))
    gc.collect()

with open(RUN_DIR / "finetune_summary_smoke.json", "w", encoding="utf-8") as f:
    json.dump(fine_smoke, f, indent=2, default=str)
print("[saved]", RUN_DIR / "finetune_summary_smoke.json")

# --- mid finetune: ~half of two_hour F_EPOCH (80), same N_FOLDS/N_TIMES as smoke ---
F_MID, NF_MID, NT_MID = 40, 2, 1
print(
    f"\n[mid] f_epoch={F_MID} n_folds={NF_MID} n_times={NT_MID} "
    "(reuses smoke checkpoints; between smoke F=18 and two_hour F=80)"
)
fine_mid: dict = {}
for side in SIDES:
    print(f"\n===== mid finetune side {side} =====")
    t0 = time.perf_counter()
    r = q.run_finetune_side(
        pre_results[side],
        dataset=DATASET,
        batch_size=FINETUNE_BATCH,
        f_epoch=F_MID,
        n_times=NT_MID,
        n_folds=NF_MID,
        device=DEVICE,
        seed=SEED,
        esol_filter_qm78=ESOL_FILTER_QM78,
        dataset_root=DATASET_ROOT,
        test_max_atoms=ESOL_TEST_MAX_ATOMS,
    )
    fine_mid[side] = r
    if r.get("rmse_mean") is not None:
        print(
            f"  RMSE = {r['rmse_mean']:.4f} ± {r['rmse_sd']:.4f}  "
            f"wall_s={r.get('wall_s', 0):.1f} elapsed={time.perf_counter() - t0:.1f}s"
        )
    else:
        print("  error:", r.get("error"))
    gc.collect()

with open(RUN_DIR / "finetune_summary_mid.json", "w", encoding="utf-8") as f:
    json.dump(fine_mid, f, indent=2, default=str)
print("[saved]", RUN_DIR / "finetune_summary_mid.json")
print("\nAll done.")
