"""Shared runner for A/B/C pretrain + finetune **quality** comparison.

Used by ``compare_pretrain_quality.ipynb``. Each "side" is one full GraphCL
pretrain followed by one finetune grid (single config). Non-destructive:
each call uses a private ``model_root`` so checkpoints never collide between
sides.

Side definitions
----------------

* ``A`` baseline   : ``aug = 'MMFFrandom'``, no impl override
                     → original PyG ``RandomView`` + ``NodeTranslation``
* ``B`` top1+identity : ``aug_1 = 'MMFFweighted_top1'``, ``aug_2 = None``
                     → view1 = best-weight slot; view2 = raw ``data.pos`` (GraphCL
                     identity branch). One weighted slot pick per batch step; no
                     second-slot conformer swap.
* ``C`` weighted   : ``aug_1 = 'MMFFweighted'`` + ``mmff_weights`` attached
                     → ``WeightedMMFFView`` (per-graph categorical over 4 slots).
                     Optionally ``aug_2`` can differ (e.g. ``'noise'``, ``'top12'``)
                     so the two contrastive views are not two i.i.d. samples from
                     the same slot distribution.

Sides share the same seed and dataset; encoder ``cutoff`` / ``encoder`` and
``side_c_aug_2`` are controlled via ``run_pretrain_side`` kwargs.
"""
from __future__ import annotations

import gc
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import torch

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dig.sslgraph.evaluation import Pretrain, Finetune  # noqa: E402
from dig.sslgraph.method import GraphCL  # noqa: E402
from dig.sslgraph.utils import Encoder  # noqa: E402
from dig.sslgraph.utils.device import pick_torch_device  # noqa: E402
from dig.sslgraph.utils.seed import setup_seed  # noqa: E402
from dig.threedgraph.dataset import MoleculeNet  # noqa: E402
from dig.sslgraph.method.contrastive.views_fn.mmff_weights_io import (  # noqa: E402
    apply_mmff_weights, load_weights,
)


# Env vars graphcl.py / mmff_fast.py consult — we *always* clear them for the
# duration of a side call and let the explicit ``args.*`` knob below win.
_GUARDED_ENV = (
    "MMFFRANDOM_FAST", "MMFFRANDOM_WEIGHTED", "MMFF_WEIGHTS_PATH",
)


@contextmanager
def _clean_aug_env() -> Iterator[None]:
    saved: dict[str, str | None] = {k: os.environ.get(k) for k in _GUARDED_ENV}
    try:
        for k in _GUARDED_ENV:
            os.environ.pop(k, None)
        yield
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def build_pretrain_args(
    *,
    dataset: str,
    batch_size: int,
    p_epoch: int,
    model_root: Path,
    device: torch.device,
    seed: int = 2222,
    p_pretrain_amp: bool = False,
    encoder: str = "schnet",
    cutoff: float = 5.0,
) -> SimpleNamespace:
    """Same hyperparams as ``examples/sslgraph/pretrain.ipynb`` cell 3."""
    args = SimpleNamespace()
    args.finetune = False
    args.seed = seed
    args.model_path = str(model_root)
    args.device = device
    args.pretrain_dataset = dataset
    args.batch_size = batch_size
    args.encoder = str(encoder)
    args.edge_weight = True
    args.feat_dim = 9
    args.cutoff = float(cutoff)
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
    args.p_epoch = p_epoch
    args.p_lr = 1e-3
    args.aug_1, args.aug_2 = "MMFFrandom", "MMFFrandom"  # overridden per side
    args.aug_ratio = 0.25
    args.tau = 0.2
    args.proj = "spherenet"
    args.dropout_rate = 0.0
    args.p_optim = "ExponentialLR"
    args.p_weight_decay = 0
    args.p_lr_decay_step_size = 15
    args.p_lr_decay_factor = 0.5
    args.expo_gamma = 0.95
    args.T_0 = 20
    args.T_mult = 2
    args.eta_max = 0.05
    args.T_up = 10
    args.gamma = 0.5
    args.pc = False
    args.p_pretrain_amp = bool(p_pretrain_amp)
    return args


def configure_side(
    args: SimpleNamespace,
    side: str,
    *,
    weights_path: Path | None = None,
    weight_mode: str = "auto",
    weight_norm: str = "auto",
    weight_kT: float = 1.0,
    side_c_aug_2: str | None = None,
) -> SimpleNamespace:
    """Mutate ``args`` for one of {'A','B','C'} and return it."""
    if side == "A":
        args.aug_1 = args.aug_2 = "MMFFrandom"
        if hasattr(args, "mmffrandom_impl"):
            del args.mmffrandom_impl
    elif side == "B":
        args.aug_1 = "MMFFweighted_top1"
        args.aug_2 = None
        args.mmff_weight_mode = weight_mode
        _wn_b = (weight_norm or "auto").lower()
        if _wn_b in ("none", "zscore", "rank"):
            _wn_b = "auto"
        args.mmff_weight_norm = _wn_b
        args.mmff_weight_kT = weight_kT
        args.mmff_slots = ("MMFF1", "MMFF2", "MMFF3", "MMFF4")
        if hasattr(args, "mmffrandom_impl"):
            del args.mmffrandom_impl
    elif side == "C":
        _a2 = (side_c_aug_2 or "").strip()
        _a2_lo = _a2.lower()
        if _a2_lo in ("top12", "top1_top2", "1_2", "12"):
            # Deterministic (1st, 2nd) pair from the per-graph weight vector —
            # avoids the "two i.i.d. multinomial draws often hit the same
            # slot" degeneracy.
            args.aug_1 = "MMFFweighted_top1"
            args.aug_2 = "MMFFweighted_top2"
        elif _a2_lo in ("top1", "top1_only", "best", "argmax", "1"):
            # Both views = deterministic argmax (best slot only). The "second
            # place" slot is never read; only the top1 conformer enters the
            # contrastive pair. positive pair becomes the *same* coordinates
            # twice — a clean ablation of "what if augmentation is identity?"
            args.aug_1 = "MMFFweighted_top1"
            args.aug_2 = "MMFFweighted_top1"
        else:
            args.aug_1 = "MMFFweighted"
            if _a2 and _a2_lo not in ("none", "mmffweighted", "same"):
                args.aug_2 = _a2
            else:
                args.aug_2 = "MMFFweighted"
        args.mmff_weight_mode = weight_mode
        # ``weight_norm`` here is the *view-time* knob (auto/softmax/linear).
        # The notebook's ``WEIGHT_NORM`` (none/zscore/rank) is build-time only
        # (compute_mmff_weights.py --normalize); map those onto ``"auto"`` so
        # an accidental pass-through doesn't crash WeightedMMFFView.
        _wn = (weight_norm or "auto").lower()
        if _wn in ("none", "zscore", "rank"):
            _wn = "auto"
        args.mmff_weight_norm = _wn
        args.mmff_weight_kT = weight_kT
        args.mmff_slots = ("MMFF1", "MMFF2", "MMFF3", "MMFF4")
    else:
        raise ValueError(f"unknown side {side!r}; expected A/B/C")
    args._side = side
    args._weights_path = str(weights_path) if weights_path else None
    return args


def _find_best_ckpt(model_root: Path) -> Path | None:
    """Pick the freshest ``enc_best_epoch-*_loss-*.pkl`` under ``model_root``."""
    cands = sorted(
        model_root.rglob("enc_best_epoch-*_loss-*.pkl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0].resolve() if cands else None


def run_pretrain_side(
    side: str,
    *,
    dataset: str,
    batch_size: int,
    p_epoch: int,
    model_root: Path,
    device: torch.device,
    weights_path: Path | None = None,
    weight_kT: float = 1.0,
    weight_mode: str = "auto",
    weight_norm: str = "auto",
    seed: int = 2222,
    p_pretrain_amp: bool = False,
    encoder: str = "schnet",
    cutoff: float = 5.0,
    side_c_aug_2: str | None = None,
) -> dict[str, Any]:
    """Run one full pretrain configuration and return a summary dict."""
    model_root = Path(model_root)
    model_root.mkdir(parents=True, exist_ok=True)
    setup_seed(seed)
    random.seed(seed)
    args = build_pretrain_args(
        dataset=dataset, batch_size=batch_size, p_epoch=p_epoch,
        model_root=model_root, device=device, seed=seed,
        p_pretrain_amp=p_pretrain_amp,
        encoder=encoder,
        cutoff=cutoff,
    )
    configure_side(
        args, side, weights_path=weights_path,
        weight_mode=weight_mode, weight_kT=weight_kT,
        weight_norm=weight_norm, side_c_aug_2=side_c_aug_2,
    )

    backbone_name = str(encoder)

    # Construct learner under cleaned env so MMFFRANDOM_* shortcuts can't
    # override our explicit per-side ``args.mmffrandom_impl`` / aug names.
    with _clean_aug_env():
        enc_module = Encoder(args)
        learner = GraphCL(args)
        evaluator = Pretrain(args)

    # Attach weights to the *pretrain dataset* in place (no host env writes).
    weights_meta: dict[str, Any] | None = None
    if side in ("B", "C"):
        if weights_path is None or not Path(weights_path).is_file():
            raise FileNotFoundError(
                f"Sides B/C require an existing weights .pt; got {weights_path!r}. "
                "Build with examples/sslgraph/bench/compute_mmff_weights.py first."
            )
        w = load_weights(weights_path)
        n_attached, n_missing = apply_mmff_weights(
            evaluator.pretrain_dataset, w, verbose=False,
        )
        weights_meta = {
            "path": str(weights_path),
            "n_attached": int(n_attached),
            "n_missing": int(n_missing),
            "n_total": int(n_attached + n_missing),
            "kT": float(weight_kT),
            "mode": weight_mode,
        }

    view_classes = [type(v).__name__ for v in learner.views_fn]

    t0 = time.perf_counter()
    with _clean_aug_env():
        enc_module = evaluator.evaluate(learning_model=learner, encoder=enc_module)
    wall = time.perf_counter() - t0

    losses = list(getattr(learner, "last_epoch_losses", []))
    best_ckpt = _find_best_ckpt(model_root)

    gc.collect()
    if hasattr(torch, "xpu"):
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass

    return {
        "side": side,
        "label": _default_label(side, weights_meta),
        "view_classes": view_classes,
        "wall_s": wall,
        "losses_per_epoch": losses,
        "model_root": str(model_root),
        "best_ckpt": str(best_ckpt) if best_ckpt else None,
        "weights": weights_meta,
        "args_aug_1": args.aug_1,
        "args_aug_2": args.aug_2,
        "args_mmffrandom_impl": getattr(args, "mmffrandom_impl", None),
        "p_epoch": p_epoch,
        "batch_size": batch_size,
        "dataset": dataset,
        "p_pretrain_amp": bool(p_pretrain_amp),
        "pretrain_encoder": backbone_name,
        "pretrain_cutoff": float(cutoff),
        "aug_2_effective": str(args.aug_2),
    }


def _default_label(side: str, weights: dict | None) -> str:
    if side == "A":
        return "A baseline (PyG views)"
    if side == "B":
        return "B top1 + identity (aug2=None)"
    if side == "C":
        if weights and "path" in weights:
            return f"C weighted ({Path(weights['path']).stem})"
        return "C weighted"
    return side


def build_finetune_args(
    *,
    ckpt_path: Path,
    dataset: str,
    batch_size: int,
    f_epoch: int,
    n_times: int,
    n_folds: int,
    device: torch.device,
    seed: int = 2222,
    encoder: str = "schnet",
    cutoff: float = 5.0,
) -> SimpleNamespace:
    """Mirror ``examples/sslgraph/finetune.ipynb`` defaults; only knobs we vary
    between sides are ``model_path``, ``dataset``, ``f_epoch``."""
    args = SimpleNamespace()
    args.finetune = True
    args.seed = seed
    args.model_path = str(ckpt_path)
    args.device = device
    args.dataset = dataset
    args.batch_size = batch_size
    args.encoder = str(encoder)
    args.cutoff = float(cutoff)
    args.num_layers = 2
    args.num_filters = 128
    args.num_gaussians = 50
    args.z_dim = 32
    args.edge_weight = False
    args.feat_dim = 9
    args.n_times = n_times
    args.n_folds = n_folds
    args.f_epoch = f_epoch
    args.f_lr = 1e-3
    args.aug_1, args.aug_2 = "MMFFrandom", "MMFFrandom"
    args.aug_ratio = 0.2
    args.tau = 0.2
    args.proj = "schnet"
    args.dropout_rate = 0.0
    args.f_optim = "ExponentialLR"
    args.f_weight_decay = 5e-5
    args.f_lr_decay_step_size = 20
    args.f_lr_decay_factor = 0.5
    args.expo_gamma = 0.95
    args.T_0 = 100
    args.T_mult = 1
    args.eta_max = 0.05
    args.T_up = 10
    args.gamma = 0.5
    args.batch_lst = [batch_size]
    args.cutoff_lst = [float(cutoff)]
    args.num_layers_lst = [2]
    args.num_filters_lst = [128]
    args.num_gaussians_lst = [50]
    args.z_dim_lst = [32]
    args.dropout_rate_lst = [0.1]
    args.target_lst = ["y"]
    args.f_lr_lst = [2e-3]
    args.f_weight_decay_lst = [1e-3]
    # required by Finetune.__init__
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
    return args


def run_finetune_side(
    side_summary: dict[str, Any],
    *,
    dataset: str,
    batch_size: int,
    f_epoch: int,
    n_times: int,
    n_folds: int,
    device: torch.device,
    seed: int = 2222,
    encoder: str | None = None,
    cutoff: float | None = None,
) -> dict[str, Any]:
    """Run finetune grid_search with the side's checkpoint."""
    ckpt = side_summary.get("best_ckpt")
    if not ckpt:
        return {"side": side_summary["side"], "rmse_mean": None,
                "rmse_sd": None, "error": "no_checkpoint"}

    _enc = encoder if encoder is not None else side_summary.get("pretrain_encoder")
    if not _enc:
        _enc = "schnet"
    _cut = cutoff if cutoff is not None else side_summary.get("pretrain_cutoff")
    if _cut is None:
        _cut = 5.0

    args = build_finetune_args(
        ckpt_path=Path(ckpt), dataset=dataset, batch_size=batch_size,
        f_epoch=f_epoch, n_times=n_times, n_folds=n_folds, device=device, seed=seed,
        encoder=str(_enc), cutoff=float(_cut),
    )
    evaluator = Finetune(args=args, log_interval=10)
    # Skip the per-epoch eval over train_loader (~70% of per-epoch wall time on
    # ESOL). train_rmse is only used for the tqdm postfix; best-checkpoint
    # selection still uses val_rmse, and returned RMSE/preds are unchanged.
    evaluator.eval_train_per_epoch = False
    # ``Finetune.grid_search`` parses ``args.model_path`` with ``split('/')[2]``
    # for log filenames — that crashes on Windows absolute paths. Since we only
    # need a single hyperparam config per side, call ``evaluate()`` directly
    # after applying the single-point grid via ``setup_train_config``.
    evaluator.setup_train_config(
        batch_size=args.batch_lst[0],
        cutoff=args.cutoff_lst[0],
        num_layers=args.num_layers_lst[0],
        num_filters=args.num_filters_lst[0],
        num_gaussians=args.num_gaussians_lst[0],
        z_dim=args.z_dim_lst[0],
        dropout_rate=args.dropout_rate_lst[0],
        target=args.target_lst[0],
        f_lr=args.f_lr_lst[0],
        f_weight_decay=args.f_weight_decay_lst[0],
    )
    t0 = time.perf_counter()
    loss_m, loss_sd, test_trues, test_preds, test_smiles = evaluator.evaluate()
    wall = time.perf_counter() - t0

    rmse_mean = float(loss_m) if loss_m is not None else float("nan")
    rmse_sd = float(loss_sd) if loss_sd is not None else float("nan")

    gc.collect()
    if hasattr(torch, "xpu"):
        try:
            torch.xpu.empty_cache()
        except Exception:
            pass

    return {
        "side": side_summary["side"],
        "label": side_summary["label"],
        "ckpt": ckpt,
        "rmse_mean": rmse_mean,
        "rmse_sd": rmse_sd,
        "grid_size": 1,
        "wall_s": float(wall),
        "f_epoch": f_epoch,
        "n_times": n_times,
        "n_folds": n_folds,
        "dataset": dataset,
    }


# ---------------------------------------------------------------------------
# View distribution check — verify "random" actually became weighted
# ---------------------------------------------------------------------------

def _build_view_for_side(
    side: str,
    *,
    args_for_init: SimpleNamespace,
    device: torch.device,
) -> Any:
    """Construct the same view fn ``GraphCL`` would build for this side."""
    setup_seed(args_for_init.seed)
    random.seed(args_for_init.seed)
    learner = GraphCL(args_for_init)
    # Both aug_1 / aug_2 are the same view in our setup; take views_fn[0].
    return learner.views_fn[0]


def _detect_slot_picks(batch, out_batch, atol: float = 1e-5) -> list[int | None]:
    """For each graph in ``batch`` (assuming MMFF slabs exist), return which
    of MMFF1..MMFF4 the view picked. ``None`` if no slab matches (e.g. an
    augmentation that does not select a conformer)."""
    n_g = int(batch.num_graphs)
    ptr = batch.ptr.tolist()
    slabs = [getattr(batch, f"max{k}pos_mmff") for k in (1, 2, 3, 4)]
    out_pos = out_batch.pos
    picks: list[int | None] = []
    for g in range(n_g):
        a, b = ptr[g], ptr[g + 1]
        block = out_pos[a:b]
        match: int | None = None
        for k in range(4):
            if torch.allclose(block, slabs[k][a:b], atol=atol):
                match = k + 1
                break
        picks.append(match)
    return picks


def measure_slot_distribution(
    side: str,
    *,
    dataset: str,
    batch_size: int,
    n_calls: int,
    device: torch.device,
    weights_path: Path | None = None,
    weight_kT: float = 1.0,
    weight_mode: str = "auto",
    weight_norm: str = "auto",
    seed: int = 2222,
    encoder: str = "schnet",
    cutoff: float = 5.0,
    side_c_aug_2: str | None = None,
    p_pretrain_amp: bool = False,
) -> dict[str, Any]:
    """Run the side's view fn ``n_calls`` times on a single batch and report
    the empirical slot frequency over all (graph × call) decisions."""
    from torch_geometric.loader import DataLoader  # local import (slow)

    setup_seed(seed)
    random.seed(seed)
    ds = MoleculeNet(root="dataset/", name=dataset)
    if side in ("B", "C"):
        if weights_path is None or not Path(weights_path).is_file():
            raise FileNotFoundError(
                f"Sides B/C require an existing weights .pt; got {weights_path!r}."
            )
        w = load_weights(weights_path)
        apply_mmff_weights(ds, w, verbose=False)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader)).to(device)

    args = build_pretrain_args(
        dataset=dataset, batch_size=batch_size, p_epoch=1,
        model_root=Path("./_dist_unused"), device=device, seed=seed,
        p_pretrain_amp=p_pretrain_amp,
        encoder=encoder,
        cutoff=cutoff,
    )
    configure_side(
        args, side, weights_path=weights_path,
        weight_mode=weight_mode, weight_kT=weight_kT,
        weight_norm=weight_norm, side_c_aug_2=side_c_aug_2,
    )
    with _clean_aug_env():
        view_fn = _build_view_for_side(side, args_for_init=args, device=device)

    counter = np.zeros(5, dtype=np.int64)  # indices 1..4, 0 = no_match
    per_graph_hist = np.zeros((int(batch.num_graphs), 4), dtype=np.int64)
    for _ in range(n_calls):
        out = view_fn(batch)
        picks = _detect_slot_picks(batch, out)
        for g, k in enumerate(picks):
            if k is None:
                counter[0] += 1
            else:
                counter[k] += 1
                per_graph_hist[g, k - 1] += 1

    total_decisions = int(counter[1:].sum())
    freq = counter[1:].astype(float) / max(total_decisions, 1)
    if total_decisions > 0:
        eps = 1e-12
        # KL(observed || uniform=0.25) — high => more skewed
        kl = float(np.sum(freq * (np.log(freq + eps) - np.log(0.25))))
        # mean per-graph entropy in bits
        p = per_graph_hist.astype(float)
        p = p / np.clip(p.sum(axis=1, keepdims=True), 1, None)
        ent = -np.sum(p * np.log2(p + eps), axis=1)
        entropy_mean = float(np.mean(ent))
    else:
        kl = float("nan")
        entropy_mean = float("nan")

    view_name = type(view_fn).__name__
    return {
        "side": side,
        "view_class": view_name,
        "n_calls": int(n_calls),
        "n_graphs": int(batch.num_graphs),
        "total_decisions": total_decisions,
        "no_match": int(counter[0]),
        "slot_counts": counter[1:].astype(int).tolist(),
        "slot_freq": freq.tolist(),
        "kl_to_uniform_bits": float(kl / np.log(2.0)) if total_decisions else float("nan"),
        "per_graph_entropy_bits_mean": entropy_mean,
        "n_slots_used": int(np.count_nonzero(counter[1:])),
    }


__all__ = [
    "build_pretrain_args",
    "configure_side",
    "run_pretrain_side",
    "build_finetune_args",
    "run_finetune_side",
    "measure_slot_distribution",
]
