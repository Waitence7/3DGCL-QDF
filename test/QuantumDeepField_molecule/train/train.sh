#!/usr/bin/env bash
#
# QM9full HOMO–LUMO (eV) pretrain — defaults tuned for Linux + GPU/XPU + shard data.
#
# Before first run (from QuantumDeepField_molecule/; first four args are positional):
#   python train/preprocess.py QM9full_homolumo_eV 6-31G 0.75 0.3 \
#     --backend rust --output-format both
#   (shard/both needs qdf_io: cd qdf_io && maturin develop --release)
#
# Optional overrides (export before bash train.sh):
#   MAX_TRAIN_SECONDS=   — empty = no wall-clock limit (train all iteration epochs)
#   TRAIN_LOADER=npy     — if shards are missing (slow on full)
#   TRAIN_PAD_IMPL=python — if rust LCAO is not built
#   num_workers=0       — Windows + RDKit issues in other stacks; QDF train is usually OK at 4

set -euo pipefail

# --- Dataset & field ---------------------------------------------------------
dataset=QM9full_homolumo_eV

basis_set=6-31G
radius=0.75
grid_interval=0.3

# --- Model -------------------------------------------------------------------
dim=200
layer_functional=3
hidden_HK=200
layer_HK=3
operation=mean

# --- Optimization ------------------------------------------------------------
batch_size=8
lr=1e-4
lr_decay=0.5
step_size=200
iteration=2000

num_workers=0

# --- Data loader & host LCAO -----------------------------------------------
TRAIN_LOADER="${TRAIN_LOADER:-shard}"
TRAIN_PAD_IMPL="${TRAIN_PAD_IMPL:-rust}"

# Wall-clock (seconds). Checked between epochs; one epoch may exceed the budget.
# Unlimited: run as  MAX_TRAIN_SECONDS= bash train.sh   (empty = no --max-train-seconds)
if [[ ! -v MAX_TRAIN_SECONDS ]]; then
  MAX_TRAIN_SECONDS=7200
fi

# Stop if mean val MAE does not improve for N epochs. Disable: EARLY_STOP_PATIENCE= bash train.sh
if [[ ! -v EARLY_STOP_PATIENCE ]]; then
  EARLY_STOP_PATIENCE=30
fi

# --- Build CLI ---------------------------------------------------------------
setting=${dataset}--${basis_set}--radius${radius}--grid_interval${grid_interval}--dim${dim}--layer_functional${layer_functional}--hidden_HK${hidden_HK}--layer_HK${layer_HK}--${operation}--batch_size${batch_size}--lr${lr}--lr_decay${lr_decay}--step_size${step_size}--iteration${iteration}

extra_args=(--loader "${TRAIN_LOADER}" --pad-impl "${TRAIN_PAD_IMPL}")

if [[ -n "${MAX_TRAIN_SECONDS}" ]]; then
  extra_args+=(--max-train-seconds "${MAX_TRAIN_SECONDS}")
fi
if [[ -n "${EARLY_STOP_PATIENCE}" ]]; then
  extra_args+=(--early-stop-patience "${EARLY_STOP_PATIENCE}")
fi

echo "dataset=${dataset}  loader=${TRAIN_LOADER}  pad=${TRAIN_PAD_IMPL}  workers=${num_workers}"
echo "max_train_seconds=${MAX_TRAIN_SECONDS:-<empty>}  early_stop_patience=${EARLY_STOP_PATIENCE:-<empty>}"
echo "iteration=${iteration} (may stop earlier)"
echo "setting=${setting}"
echo "-------------------------------------------------------------------"

cd "$(dirname "$0")"
python train.py "${dataset}" "${basis_set}" "${radius}" "${grid_interval}" \
  "${dim}" "${layer_functional}" "${hidden_HK}" "${layer_HK}" \
  "${operation}" "${batch_size}" "${lr}" "${lr_decay}" "${step_size}" "${iteration}" \
  "${setting}" "${num_workers}" \
  "${extra_args[@]}"
