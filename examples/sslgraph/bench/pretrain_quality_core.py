"""Shared runner for A/B/C pretrain + finetune **quality** comparison.

Used by ``compare_pretrain_quality.ipynb``. Each "side" is one full GraphCL
pretrain followed by one finetune grid (single config). Non-destructive:
each call uses a private ``model_root`` so checkpoints never collide between
sides.

Side definitions
----------------

* ``A`` baseline   : ``aug = 'MMFFrandom'``, no impl override
                     → original PyG ``RandomView`` + ``NodeTranslation``.
                     Pretrain always uses ``p_epoch=5`` (frozen); finetune always
                     ``f_epoch=18``, ``n_folds=2``, ``n_times=1`` — independent of
                     ``QUALITY_BUDGET`` bumps for B/C.
* ``B`` top1+random   : ``aug_1 = 'MMFFweighted_top1'``, ``aug_2 = 'MMFFrandom'``
                     → view1 = deterministic best slot from ``.pt`` weights;
                     view2 = original ``RandomView`` over two pre-sampled MMFF slots.
* ``C`` weighted   : ``aug_1 = 'MMFFweighted'`` + ``mmff_weights`` attached
                     → ``WeightedMMFFView`` (per-graph categorical over 4 slots).
                     Optionally ``aug_2`` can differ (e.g. ``'noise'``, ``'top12'``)
                     so the two contrastive views are not two i.i.d. samples from
                     the same slot distribution.

When ``qdf_aux_lambda > 0`` and a multi-CSV ``.pt`` (``targets_members``) is supplied,
**B / C** use a **K-output** QDF auxiliary head (mean MSE across QDF ensemble members).
**A** never uses QDF aux (baseline GraphCL only).

Sides share the same seed and dataset; encoder ``cutoff`` / ``encoder`` and
``side_c_aug_2`` are controlled via ``run_pretrain_side`` kwargs.
"""
from __future__ import annotations

import gc
import hashlib
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping

import numpy as np
import torch

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Side ``A`` only — frozen legacy *short smoke* schedule (original GraphCL bench).
# B/C follow the caller's ``p_epoch`` / ``f_epoch`` / folds / times from the notebook.
BASELINE_A_P_EPOCH = 5
BASELINE_A_F_EPOCH = 18
BASELINE_A_N_FOLDS = 2
BASELINE_A_N_TIMES = 1

from dig.sslgraph.evaluation import Pretrain, Finetune  # noqa: E402
from dig.sslgraph.method import GraphCL  # noqa: E402
from dig.sslgraph.utils import Encoder  # noqa: E402
from dig.sslgraph.utils.device import pick_torch_device  # noqa: E402
from dig.sslgraph.utils.seed import setup_seed  # noqa: E402
from rdkit import Chem  # noqa: E402

from dig.threedgraph.dataset import MoleculeNet, QM  # noqa: E402
from dig.sslgraph.method.contrastive.views_fn.mmff_weights_io import (  # noqa: E402
    apply_mmff_weights, load_weights,
)

# Must stay aligned with ``compute_mmff_weights.QDF_CSV_SCHEMAS`` defaults.
_QDF_DEFAULT_SCORE_EXPR = {"homolumo": "gap", "atomization": "-energy"}


def desired_qdf_score_expr(qdf_property: str, score_expr: str | None) -> str:
    """Effective ``--score-expr`` for QDF weights when ``score_expr`` is None."""
    if score_expr is not None and str(score_expr).strip():
        return str(score_expr).strip()
    key = (qdf_property or "").strip().lower()
    if key not in _QDF_DEFAULT_SCORE_EXPR:
        raise ValueError(
            f"unknown QDF_PROPERTY {qdf_property!r}; expected one of "
            f"{sorted(_QDF_DEFAULT_SCORE_EXPR)}"
        )
    return _QDF_DEFAULT_SCORE_EXPR[key]


# Env vars graphcl.py / mmff_fast.py consult — we *always* clear them for the
# duration of a side call and let the explicit ``args.*`` knob below win.
_GUARDED_ENV = (
    "MMFFRANDOM_FAST", "MMFFRANDOM_WEIGHTED", "MMFF_WEIGHTS_PATH",
)


def _canon_smiles(raw: str | None) -> str | None:
    if raw is None:
        return None
    smi = str(raw).strip()
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def qm7_qm8_canonical_smiles_union(*, root: str | Path) -> frozenset[str]:
    """Canonical SMILES occurring in processed QM7 **or** QM8 (DIG ``QM`` loader)."""
    root = Path(root)
    out: set[str] = set()
    for name in ("qm7", "qm8"):
        ds = QM(root=str(root), name=name)
        for i in range(len(ds)):
            c = _canon_smiles(getattr(ds[i], "smiles", None))
            if c:
                out.add(c)
    return frozenset(out)


def esol_indices_in_canon_smiles_set(
    esol: MoleculeNet, canon: frozenset[str],
) -> list[int]:
    """Row indices into ``esol`` whose ``.smiles`` canonicalize to a member of ``canon``."""
    rows: list[int] = []
    for i in range(len(esol)):
        c = _canon_smiles(getattr(esol[i], "smiles", None))
        if c is not None and c in canon:
            rows.append(i)
    return rows


def molnet_esol_qm78_subset(
    *,
    root: str | Path,
    seed: int,
) -> tuple[MoleculeNet, dict[str, Any]]:
    """Return ESOL :class:`MoleculeNet` restricted to QM7∪QM8 (canonical SMILES), shuffled.

    First download/process QM7 and QM8 under ``root`` (same layout as ``QM``).
    """
    root = Path(root)
    qm = qm7_qm8_canonical_smiles_union(root=root)
    full = MoleculeNet(root=str(root), name="esol")
    idx = esol_indices_in_canon_smiles_set(full, qm)
    if not idx:
        raise RuntimeError(
            "ESOL ∩ (QM7 ∪ QM8) is empty after canonical SMILES matching. "
            "Build QM7/QM8 datasets under the given root, or disable esol_filter_qm78."
        )
    sub = full.copy(torch.tensor(idx, dtype=torch.long))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(sub))
    sub_shuf = sub.copy(torch.tensor(perm, dtype=torch.long))
    meta: dict[str, Any] = {
        "esol_filter_qm78": True,
        "n_esol_full": int(len(full)),
        "n_esol_kept": int(len(sub_shuf)),
        "n_qm7_qm8_union": int(len(qm)),
    }
    return sub_shuf, meta


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
    # GraphCL ``_get_proj``: GIN/GCN need the wider ``spherenet`` projection branch;
    # SchNet uses ``schnet``; SphereNet matches ``spherenet``.
    _enc = str(encoder).lower()
    if _enc in ("gin", "gcn"):
        args.proj = "spherenet"
    elif _enc == "spherenet":
        args.proj = "spherenet"
    else:
        args.proj = "schnet"
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
    args.qdf_aux_lambda = 0.0
    args.qdf_aux_pt = None
    args.qdf_aux_ensemble_k = 1
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
    side_b_aug_1: str | None = None,
) -> SimpleNamespace:
    """Mutate ``args`` for one of {'A','B','C'} and return it.

    ``side_b_aug_1`` overrides the default ``'MMFFweighted_top1'`` augmentation
    for side ``'B'``'s view-1. Use ``'MMFFweighted'`` to switch from deterministic
    argmax (top-1) to stochastic Boltzmann/multinomial sampling.
    """
    if side == "A":
        args.aug_1 = args.aug_2 = "MMFFrandom"
        if hasattr(args, "mmffrandom_impl"):
            del args.mmffrandom_impl
    elif side == "B":
        args.aug_1 = side_b_aug_1 if side_b_aug_1 else "MMFFweighted_top1"
        args.aug_2 = "MMFFrandom"
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
    side_b_aug_1: str | None = None,
    esol_filter_qm78: bool = False,
    dataset_root: str | Path | None = None,
    qdf_aux_lambda: float = 0.0,
    qdf_aux_pt: str | Path | None = None,
) -> dict[str, Any]:
    """Run one full pretrain configuration and return a summary dict.

    ``qdf_aux_lambda`` / ``qdf_aux_pt`` are ignored for side ``'A'`` (baseline
    GraphCL only). They apply only to ``'B'`` / ``'C'`` when given.

    Side ``'A'`` always uses ``p_epoch = BASELINE_A_P_EPOCH`` (5) regardless of
    the ``p_epoch`` argument so the baseline checkpoint stays on the original
    short-run curve while B/C may use longer budgets.
    """
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
        side_b_aug_1=side_b_aug_1,
    )
    args.qdf_aux_lambda = float(qdf_aux_lambda or 0.0)
    args.qdf_aux_pt = str(Path(qdf_aux_pt).resolve()) if qdf_aux_pt else None
    # Side A is the clean GraphCL baseline: no QDF auxiliary head / targets.
    if side == "A":
        args.qdf_aux_lambda = 0.0
        args.qdf_aux_pt = None
        args.qdf_aux_ensemble_k = 1
        args.p_epoch = int(BASELINE_A_P_EPOCH)

    # B/C: infer auxiliary head width **before** GraphCL builds Contrastive (multi-CSV
    # ``targets_members`` => K-output head + mean MSE over QDF teachers).
    if side in ("B", "C") and float(args.qdf_aux_lambda or 0.0) > 0 and args.qdf_aux_pt:
        pta_probe = Path(args.qdf_aux_pt)
        if pta_probe.is_file():
            try:
                b = torch.load(str(pta_probe), map_location="cpu", weights_only=False)
                if isinstance(b, dict):
                    meta = b.get("meta") or {}
                    km = meta.get("qdf_aux_ensemble_k")
                    tm = b.get("targets_members")
                    if isinstance(km, int) and km >= 1:
                        args.qdf_aux_ensemble_k = int(km)
                    elif isinstance(tm, list) and len(tm) >= 2:
                        args.qdf_aux_ensemble_k = len(tm)
                    else:
                        args.qdf_aux_ensemble_k = 1
            except Exception:
                args.qdf_aux_ensemble_k = 1
        else:
            args.qdf_aux_ensemble_k = 1
    else:
        args.qdf_aux_ensemble_k = 1

    backbone_name = str(encoder)

    # Construct learner under cleaned env so MMFFRANDOM_* shortcuts can't
    # override our explicit per-side ``args.mmffrandom_impl`` / aug names.
    with _clean_aug_env():
        enc_module = Encoder(args)
        learner = GraphCL(args)
        evaluator = Pretrain(args)

    qm78_meta: dict[str, Any] | None = None
    if esol_filter_qm78:
        if dataset != "esol":
            raise ValueError("esol_filter_qm78=True requires dataset=='esol'")
        droot = Path(dataset_root) if dataset_root is not None else (REPO_ROOT / "dataset")
        evaluator.pretrain_dataset, qm78_meta = molnet_esol_qm78_subset(
            root=droot, seed=seed,
        )

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

    qdf_aux_meta: dict[str, Any] | None = None
    if float(args.qdf_aux_lambda or 0.0) > 0 and args.qdf_aux_pt:
        pta = Path(args.qdf_aux_pt)
        if pta.is_file():
            from dig.sslgraph.method.contrastive.views_fn.qdf_aux_io import (
                apply_qdf_aux_from_pt,
            )
            na, nn, k_applied = apply_qdf_aux_from_pt(
                evaluator.pretrain_dataset, pta, verbose=False,
            )
            qdf_aux_meta = {
                "path": str(pta),
                "lambda": float(args.qdf_aux_lambda),
                "n_attached": int(na),
                "n_nan": int(nn),
                "ensemble_k": int(k_applied),
            }
        else:
            args.qdf_aux_lambda = 0.0
            qdf_aux_meta = {"error": f"missing qdf_aux file: {pta}"}

    view_classes = [type(v).__name__ for v in learner.views_fn]

    t0 = time.perf_counter()
    with _clean_aug_env():
        enc_module = evaluator.evaluate(learning_model=learner, encoder=enc_module)
    wall = time.perf_counter() - t0

    losses = list(getattr(learner, "last_epoch_losses", []))
    best_ckpt = _find_best_ckpt(model_root)
    _ck_sha: str | None = None
    if best_ckpt:
        _bp = Path(best_ckpt)
        if _bp.is_file():
            _ck_sha = hashlib.sha256(_bp.read_bytes()).hexdigest()

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
        "ckpt_sha256": _ck_sha,
        "weights": weights_meta,
        "args_aug_1": args.aug_1,
        "args_aug_2": args.aug_2,
        "args_mmffrandom_impl": getattr(args, "mmffrandom_impl", None),
        "p_epoch": int(args.p_epoch),
        "p_epoch_requested": int(p_epoch),
        "batch_size": batch_size,
        "dataset": dataset,
        "p_pretrain_amp": bool(p_pretrain_amp),
        "pretrain_encoder": backbone_name,
        "pretrain_cutoff": float(cutoff),
        "aug_2_effective": str(args.aug_2),
        "esol_qm78_filter": qm78_meta,
        "qdf_aux": qdf_aux_meta,
    }


def _default_label(side: str, weights: dict | None) -> str:
    if side == "A":
        return "A baseline (PyG views)"
    if side == "B":
        return "B top1 + MMFFrandom (mixed views)"
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
    test_max_atoms: int | None = None,
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
    # ``Finetune`` / ``PredictionModel``: same head family as backbone (see ``finetune.py``).
    _enc = str(encoder).lower()
    args.proj = "spherenet" if _enc == "spherenet" else "schnet"
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
    args.test_max_atoms = test_max_atoms
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
    esol_filter_qm78: bool = False,
    dataset_root: str | Path | None = None,
    test_max_atoms: int | None = None,
) -> dict[str, Any]:
    """Run finetune grid_search with the side's checkpoint.

    Side ``'A'`` always uses ``BASELINE_A_F_EPOCH`` / folds / times so downstream
    RMSE stays on the original short-run baseline; B/C use the ``f_epoch`` /
    ``n_folds`` / ``n_times`` arguments.
    """
    ckpt = side_summary.get("best_ckpt")
    if not ckpt:
        return {"side": side_summary["side"], "rmse_mean": None,
                "rmse_sd": None, "error": "no_checkpoint"}

    _side = str(side_summary.get("side") or "")
    if _side == "A":
        f_epoch = int(BASELINE_A_F_EPOCH)
        n_folds = int(BASELINE_A_N_FOLDS)
        n_times = int(BASELINE_A_N_TIMES)

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
        test_max_atoms=test_max_atoms,
    )
    evaluator = Finetune(args=args, log_interval=10)
    qm78_meta: dict[str, Any] | None = None
    if esol_filter_qm78:
        if dataset != "esol":
            raise ValueError("esol_filter_qm78=True requires dataset=='esol'")
        droot = Path(dataset_root) if dataset_root is not None else (REPO_ROOT / "dataset")
        evaluator.dataset, qm78_meta = molnet_esol_qm78_subset(root=droot, seed=seed)
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
        "f_epoch": int(f_epoch),
        "n_times": int(n_times),
        "n_folds": int(n_folds),
        "dataset": dataset,
        "esol_qm78_filter": qm78_meta,
        "test_max_atoms": test_max_atoms,
    }


# ---------------------------------------------------------------------------
# View distribution check — verify "random" actually became weighted
# ---------------------------------------------------------------------------

def _build_graphcl_views_for_side(
    side: str,
    *,
    args_for_init: SimpleNamespace,
    device: torch.device,
) -> tuple[Any, Any]:
    """Return GraphCL's two view callables (aug_1, aug_2), same as pretrain."""
    setup_seed(args_for_init.seed)
    random.seed(args_for_init.seed)
    learner = GraphCL(args_for_init)
    vfs = learner.views_fn
    if len(vfs) < 2:
        raise RuntimeError(
            f"Expected GraphCL.views_fn length >= 2, got {len(vfs)}"
        )
    return vfs[0], vfs[1]


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
    side_b_aug_1: str | None = None,
    p_pretrain_amp: bool = False,
    dataset_root: str | Path | None = None,
    esol_filter_qm78: bool = False,
) -> dict[str, Any]:
    """Run both GraphCL view fns ``n_calls`` times each round on one batch.

    Each round calls aug_1 then aug_2; slot picks from both are pooled into
    one histogram (same weight as two contrastive branches in training).

    Returns top-level ``slot_freq`` (pooled) plus nested ``view0`` / ``view1``
    dicts with the same keys (``slot_freq``, ``kl_to_uniform_bits``, etc.)
    for each branch alone.
    """
    from torch_geometric.loader import DataLoader  # local import (slow)

    setup_seed(seed)
    random.seed(seed)
    droot = Path(dataset_root) if dataset_root is not None else Path("dataset")
    if dataset == "esol" and esol_filter_qm78:
        ds, _qm_meta = molnet_esol_qm78_subset(root=droot, seed=seed)
    else:
        ds = MoleculeNet(root=str(droot), name=dataset)
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
        side_b_aug_1=side_b_aug_1,
    )
    with _clean_aug_env():
        view0, view1 = _build_graphcl_views_for_side(
            side, args_for_init=args, device=device,
        )

    n_g = int(batch.num_graphs)
    counter = np.zeros(5, dtype=np.int64)  # pooled; 0 = no_match
    counter_v0 = np.zeros(5, dtype=np.int64)
    counter_v1 = np.zeros(5, dtype=np.int64)
    per_graph_hist = np.zeros((n_g, 4), dtype=np.int64)
    per_graph_hist_v0 = np.zeros((n_g, 4), dtype=np.int64)
    per_graph_hist_v1 = np.zeros((n_g, 4), dtype=np.int64)

    def _accumulate(
        picks: list[int | None],
        c_all: np.ndarray,
        c_v: np.ndarray,
        h_v: np.ndarray,
        h_pool: np.ndarray,
    ) -> None:
        for g, k in enumerate(picks):
            if k is None:
                c_all[0] += 1
                c_v[0] += 1
            else:
                c_all[k] += 1
                c_v[k] += 1
                h_v[g, k - 1] += 1
                h_pool[g, k - 1] += 1

    for _ in range(n_calls):
        out0 = view0(batch)
        _accumulate(
            _detect_slot_picks(batch, out0),
            counter, counter_v0, per_graph_hist_v0, per_graph_hist,
        )
        out1 = view1(batch)
        _accumulate(
            _detect_slot_picks(batch, out1),
            counter, counter_v1, per_graph_hist_v1, per_graph_hist,
        )

    def _view_metrics(
        c_v: np.ndarray, h_v: np.ndarray,
    ) -> dict[str, Any]:
        total = int(c_v[1:].sum())
        freq = c_v[1:].astype(float) / max(total, 1)
        if total > 0:
            eps = 1e-12
            kl_nat = float(np.sum(freq * (np.log(freq + eps) - np.log(0.25))))
            kl_bits = float(kl_nat / np.log(2.0))
            p = h_v.astype(float)
            p = p / np.clip(p.sum(axis=1, keepdims=True), 1, None)
            ent = -np.sum(p * np.log2(p + eps), axis=1)
            h_mean = float(np.mean(ent))
        else:
            kl_bits = float("nan")
            h_mean = float("nan")
        return {
            "total_decisions": total,
            "no_match": int(c_v[0]),
            "slot_counts": c_v[1:].astype(int).tolist(),
            "slot_freq": freq.tolist(),
            "kl_to_uniform_bits": kl_bits,
            "per_graph_entropy_bits_mean": h_mean,
            "n_slots_used": int(np.count_nonzero(c_v[1:])),
        }

    total_decisions = int(counter[1:].sum())
    freq = counter[1:].astype(float) / max(total_decisions, 1)
    if total_decisions > 0:
        eps = 1e-12
        kl = float(np.sum(freq * (np.log(freq + eps) - np.log(0.25))))
        p = per_graph_hist.astype(float)
        p = p / np.clip(p.sum(axis=1, keepdims=True), 1, None)
        ent = -np.sum(p * np.log2(p + eps), axis=1)
        entropy_mean = float(np.mean(ent))
    else:
        kl = float("nan")
        entropy_mean = float("nan")

    v0n = type(view0).__name__
    v1n = type(view1).__name__
    view_class = f"{v0n}+{v1n}"
    m0 = _view_metrics(counter_v0, per_graph_hist_v0)
    m1 = _view_metrics(counter_v1, per_graph_hist_v1)
    return {
        "side": side,
        "view_class": view_class,
        "view_classes": [v0n, v1n],
        "views_per_round": 2,
        "n_calls": int(n_calls),
        "n_graphs": n_g,
        "total_decisions": total_decisions,
        "no_match": int(counter[0]),
        "slot_counts": counter[1:].astype(int).tolist(),
        "slot_freq": freq.tolist(),
        "kl_to_uniform_bits": float(kl / np.log(2.0)) if total_decisions else float("nan"),
        "per_graph_entropy_bits_mean": entropy_mean,
        "n_slots_used": int(np.count_nonzero(counter[1:])),
        # Per-view (aug_1 vs aug_2) — same schema as pooled for JSON / printing.
        "view0": m0,
        "view1": m1,
    }


def warn_bc_ckpt_byte_identical(pre_results: Mapping[str, Any]) -> None:
    """If B and C saved the same encoder bytes, downstream RMSE will match exactly.

    This happens when pretrain trajectories are bitwise identical (e.g. near-
    uniform MMFF weights so both sides see the same effective augmentations).
    It is **not** a finetune bug.
    """
    b = pre_results.get("B") or {}
    c = pre_results.get("C") or {}
    pb, pc = b.get("best_ckpt"), c.get("best_ckpt")
    if not pb or not pc:
        return
    pbp, pcp = Path(str(pb)), Path(str(pc))
    if not pbp.is_file() or not pcp.is_file():
        return
    if pbp.read_bytes() == pcp.read_bytes():
        print(
            "[warn] B와 C의 best 체크포인트 **파일 내용이 바이트 단위로 동일**합니다. "
            "그래서 finetune RMSE(소수점 전체)도 완전히 같게 나옵니다. "
            "pretrain `losses_per_epoch` 도 동일했는지 확인해 보세요. "
            "원인: homolumo 등으로 슬롯 가중치가 거의 균등하면 B/C 뷰가 사실상 같은 "
            "학습 경로로 수렴할 수 있습니다. 차이를 키우려면 atomization 가중치, "
            "epoch 증가, 또는 측면별 다른 `seed` 를 검토하세요."
        )


__all__ = [
    "BASELINE_A_P_EPOCH",
    "BASELINE_A_F_EPOCH",
    "BASELINE_A_N_FOLDS",
    "BASELINE_A_N_TIMES",
    "build_pretrain_args",
    "configure_side",
    "run_pretrain_side",
    "build_finetune_args",
    "run_finetune_side",
    "measure_slot_distribution",
    "warn_bc_ckpt_byte_identical",
    "molnet_esol_qm78_subset",
    "qm7_qm8_canonical_smiles_union",
    "esol_indices_in_canon_smiles_set",
]
