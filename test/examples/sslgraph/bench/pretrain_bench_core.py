"""Shared pretrain micro-benchmark loop used by ``profile_pretrain.py`` and
``compare_pretrain_ab.py``. Does not mutate global ``os.environ`` — callers
pass explicit env overrides into :func:`run_pretrain_benchmark`.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch_geometric.loader import DataLoader

from dig.sslgraph.method import GraphCL
from dig.sslgraph.utils import Encoder
from dig.sslgraph.utils.dataloader_kw import accelerator_dataloader_kw
from dig.sslgraph.utils.device import pick_torch_device
from dig.threedgraph.dataset import MoleculeNet

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@contextmanager
def isolated_env(overrides: dict[str, str | None]) -> Iterator[None]:
    """Temporarily set ``os.environ`` keys; ``None`` means delete the key."""
    saved: dict[str, str | None] = {}
    for k in overrides:
        saved[k] = os.environ.get(k)
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _amp_ctx(device: torch.device) -> Any:
    mode = os.environ.get("PRETRAIN_AMP", "").strip().lower()
    if mode not in ("bf16", "fp16"):
        return nullcontext()
    dtype = torch.bfloat16 if mode == "bf16" else torch.float16
    device_type = device.type
    try:
        return torch.amp.autocast(device_type=device_type, dtype=dtype)
    except Exception:
        return nullcontext()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu" and hasattr(torch, "xpu"):
        try:
            torch.xpu.synchronize()
        except Exception:
            pass


def build_args(dataset_name: str, device: torch.device, batch_size: int) -> SimpleNamespace:
    args = SimpleNamespace()
    args.finetune = False
    args.seed = 2222
    args.device = device
    args.model_path = "./models"
    args.pretrain_dataset = dataset_name
    args.batch_size = batch_size
    args.encoder = "schnet"
    args.edge_weight = True
    args.feat_dim = 9
    args.cutoff = 5.0
    args.num_layers = 2
    args.num_filters = 128
    args.num_gaussians = 50
    args.z_dim = 32
    args.int_emb_size = 64
    args.basis_emb_size_dist = 8
    args.basis_emb_size_angle = 8
    args.basis_emb_size_torsion = 8
    args.out_emb_channels = 256
    args.num_spherical = 3
    args.num_radial = 6
    args.envelope_exponent = 5
    args.num_before_skip = 1
    args.num_after_skip = 2
    args.num_output_layers = 3
    args.use_node_features = True
    args.p_epoch = 1
    args.p_lr = 1e-3
    args.aug_1, args.aug_2 = "MMFFrandom", "MMFFrandom"
    args.aug_ratio = 0.25
    args.tau = 0.2
    args.proj = "spherenet"
    args.dropout_rate = 0.0
    args.p_optim = "ExponentialLR"
    args.p_weight_decay = 0
    args.expo_gamma = 0.95
    args.T_0 = 20
    args.T_mult = 2
    args.eta_max = 0.05
    args.T_up = 10
    args.gamma = 0.5
    args.pc = False
    return args


def run_pretrain_benchmark(
    *,
    dataset: str,
    root: str,
    batch_size: int,
    warmup_epochs: int,
    measure_epochs: int,
    max_iters_per_epoch: int,
    env_overrides: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Run the contrastive micro-loop for ``warmup_epochs + measure_epochs`` epochs.

    ``env_overrides`` is applied for the duration of this call only (via
    :func:`isolated_env`). Returns aggregate timing dict + per-epoch buckets.
    """
    env_overrides = env_overrides or {}

    with isolated_env(env_overrides):
        device = pick_torch_device()
        cfg = build_args(dataset, device, batch_size)
        ds = MoleculeNet(root=root, name=dataset)

        # Optional MMFF weighted view: attach precomputed per-slot weights to
        # ``ds`` so DataLoader collates ``batch.mmff_weights`` automatically.
        # Only applied when env ``MMFF_WEIGHTS_PATH`` points at an existing file.
        weights_path = os.environ.get("MMFF_WEIGHTS_PATH", "").strip()
        weights_meta: dict[str, Any] | None = None
        if weights_path:
            wp = Path(weights_path)
            if not wp.is_absolute():
                wp = (REPO_ROOT / wp).resolve()
            if wp.is_file():
                from dig.sslgraph.method.contrastive.views_fn.mmff_weights_io import (
                    load_weights, apply_mmff_weights,
                )
                w = load_weights(wp)
                n_attached, n_missing = apply_mmff_weights(ds, w, verbose=False)
                weights_meta = {
                    "path": str(wp), "n_attached": int(n_attached),
                    "n_missing": int(n_missing), "n_total": int(n_attached + n_missing),
                }
            else:
                weights_meta = {"path": str(wp), "error": "file_not_found"}

        dl_kw = dict(accelerator_dataloader_kw())
        if os.environ.get("PIN_MEMORY", "").strip().lower() in ("1", "true", "yes", "on"):
            dl_kw["pin_memory"] = True
        loader = DataLoader(ds, batch_size, shuffle=True, **dl_kw)

        encoder = Encoder(cfg).to(device)
        learner = GraphCL(cfg)
        views_fn = learner.views_fn
        proj_head = learner._get_proj(
            cfg.encoder, cfg.num_layers, cfg.proj, cfg.z_dim,
        ).to(device)
        learner.proj_head_g = proj_head
        loss_fn = learner.loss_fn

        encoder.train()
        proj_head.train()
        params = list(encoder.parameters()) + list(proj_head.parameters())
        optimiser = Adam(params, lr=cfg.p_lr, weight_decay=cfg.p_weight_decay)
        scheduler = ExponentialLR(optimiser, gamma=cfg.expo_gamma)

        view_impl = type(views_fn[0]).__name__ if views_fn else "?"

        def one_epoch(record: bool) -> dict[str, float]:
            bucket: dict[str, float] = {
                "data": 0.0, "to_dev": 0.0, "views": 0.0, "fwd": 0.0,
                "bwd": 0.0, "step": 0.0, "n_iter": 0.0, "n_graphs": 0.0, "n_nodes": 0.0,
            }
            _sync(device)
            epoch_start = time.perf_counter()
            last_t = epoch_start
            loader_iter = iter(loader)
            while True:
                try:
                    _sync(device)
                    t0 = time.perf_counter()
                    batch = next(loader_iter)
                    _sync(device)
                    t1 = time.perf_counter()
                except StopIteration:
                    break

                batch = batch.to(device)
                _sync(device)
                t2 = time.perf_counter()

                with _amp_ctx(device):
                    views = [vfn(batch) for vfn in views_fn]
                _sync(device)
                t3 = time.perf_counter()

                optimiser.zero_grad(set_to_none=True)
                with _amp_ctx(device):
                    zs = []
                    for v in views:
                        z = encoder(v.to(device))
                        zs.append(proj_head(z))
                    loss = loss_fn(zs, neg_by_crpt=False, tau=cfg.tau, pc=cfg.pc)
                _sync(device)
                t4 = time.perf_counter()

                loss.backward()
                _sync(device)
                t5 = time.perf_counter()

                optimiser.step()
                scheduler.step()
                _sync(device)
                t6 = time.perf_counter()

                if record:
                    bucket["data"] += t1 - t0
                    bucket["to_dev"] += t2 - t1
                    bucket["views"] += t3 - t2
                    bucket["fwd"] += t4 - t3
                    bucket["bwd"] += t5 - t4
                    bucket["step"] += t6 - t5
                    bucket["n_iter"] += 1.0
                    bucket["n_graphs"] += float(batch.num_graphs)
                    bucket["n_nodes"] += float(batch.num_nodes)

                last_t = t6
                if max_iters_per_epoch and int(bucket["n_iter"]) >= max_iters_per_epoch:
                    break

            _sync(device)
            bucket["wall"] = last_t - epoch_start
            return bucket

        totals: dict[str, float] = {
            "data": 0.0, "to_dev": 0.0, "views": 0.0, "fwd": 0.0,
            "bwd": 0.0, "step": 0.0, "wall": 0.0, "n_iter": 0.0,
            "n_graphs": 0.0, "n_nodes": 0.0,
        }
        epoch_rows: list[dict[str, float]] = []
        total_epochs = warmup_epochs + measure_epochs
        for e in range(total_epochs):
            record = e >= warmup_epochs
            b = one_epoch(record)
            if record:
                epoch_rows.append(dict(b))
                for k in totals:
                    totals[k] += b.get(k, 0.0)

        return {
            "device": str(device),
            "torch_version": torch.__version__,
            "view_impl_0": view_impl,
            "env_effective": {k: os.environ.get(k) for k in sorted(
                set(env_overrides) | {"MMFFRANDOM_FAST", "MMFFRANDOM_WEIGHTED",
                                      "PRETRAIN_AMP", "PIN_MEMORY",
                                      "DATALOADER_NUM_WORKERS",
                                      "MMFF_WEIGHTS_PATH"}
            )},
            "mmff_weights": weights_meta,
            "totals": totals,
            "epochs_measured": measure_epochs,
            "per_epoch": epoch_rows,
            "dataset": dataset,
            "n_dataset": len(ds),
            "batch_size": batch_size,
        }


def snapshot_relevant_env() -> dict[str, str | None]:
    keys = (
        "MMFFRANDOM_FAST", "MMFFRANDOM_WEIGHTED", "PRETRAIN_AMP",
        "PIN_MEMORY", "DATALOADER_NUM_WORKERS",
        "MMFF_WEIGHTS_PATH",
        "TORCH_DEVICE", "TORCH_SKIP_NPU",
    )
    return {k: os.environ.get(k) for k in keys}
