"""Attach per-graph QDF-derived scalar targets for DGCL auxiliary pretraining.

When :attr:`qdf_aux_lambda` > 0 in :class:`~dig.sslgraph.method.contrastive.model.Contrastive`,
the pretrain loop adds auxiliary loss using the **first** view's graph embedding ``z``
(before the contrastive projection head).

* **Single target** (legacy): ``lambda * MSE(Linear(z), qdf_aux)`` with ``qdf_aux`` shape
  ``[N, 1]``.
* **Ensemble members** (``targets_members`` in the ``.pt`` from
  ``build_qdf_aux_ensemble.py`` with multiple ``--pred-csv``): ``Linear(z, K)`` and
  ``lambda * mean_k MSE(pred[:, k], target[:, k])`` over finite entries.

Typical flow::

    # offline: ensemble QDF CSVs -> per-SMILES scalar (e.g. mean slot energy)
    python examples/sslgraph/bench/build_qdf_aux_ensemble.py \\
        --pred-csv a.csv --pred-csv b.csv --out dataset/esol_qdf_aux.pt

    from dig.sslgraph.method.contrastive.views_fn.qdf_aux_io import (
        apply_qdf_aux_from_pt,
    )
    apply_qdf_aux_from_pt(dataset, \"dataset/esol_qdf_aux.pt\")

    # args.qdf_aux_lambda = 0.05  # set on pretrain args before GraphCL.train
    # args.qdf_aux_ensemble_k = K  # set from .pt meta before GraphCL.train (B/C only)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def _inmem_storage(dataset):
    return getattr(dataset, "_data", None) or dataset.data


def apply_qdf_aux_scalars(
    dataset,
    smiles_to_scalar: Mapping[str, float],
    *,
    attr: str = "qdf_aux",
    verbose: bool = True,
) -> tuple[int, int]:
    """Attach ``attr`` of shape ``[N, 1]`` on InMemoryMoleculeNet-style datasets.

    Missing SMILES get ``nan`` so the contrastive loop can mask them out.
    Returns ``(n_attached, n_nan)``.
    """
    has_store = hasattr(dataset, "_data") or hasattr(dataset, "data")
    is_inmem = has_store and hasattr(dataset, "slices")
    if not is_inmem:
        raise RuntimeError("apply_qdf_aux_scalars expects an InMemoryDataset with slices")

    store = _inmem_storage(dataset)
    smiles_list = getattr(store, "smiles", None)
    if smiles_list is None:
        raise RuntimeError("dataset storage has no 'smiles' attribute")

    if isinstance(smiles_list, torch.Tensor):
        smiles_iter = [str(s) for s in smiles_list.tolist()]
    else:
        smiles_iter = list(smiles_list)

    N = len(smiles_iter)
    vals: list[float] = []
    n_attached = 0
    for smi in smiles_iter:
        key = str(smi).strip()
        if key in smiles_to_scalar:
            vals.append(float(smiles_to_scalar[key]))
            n_attached += 1
        else:
            vals.append(float("nan"))

    t = torch.tensor(vals, dtype=torch.float32).unsqueeze(1)  # [N, 1]
    setattr(store, attr, t)
    if not hasattr(dataset, "slices") or dataset.slices is None:
        dataset.slices = {}
    dataset.slices[attr] = torch.arange(N + 1, dtype=torch.long)
    if hasattr(dataset, "_data_list"):
        dataset._data_list = None

    n_nan = N - n_attached
    if verbose:
        print(f"[qdf_aux] attached={n_attached}  nan={n_nan}  total={N}  attr={attr!r}  K=1")
    return n_attached, n_nan


def apply_qdf_aux_members(
    dataset,
    members: Sequence[Mapping[str, float]],
    *,
    attr: str = "qdf_aux",
    verbose: bool = True,
) -> tuple[int, int, int]:
    """Attach ``attr`` of shape ``[N, K]`` — one scalar column per QDF ensemble member.

    Returns ``(n_attached, n_nan, K)`` where a row counts as attached if **any**
    member has a finite value for that SMILES.
    """
    if not members:
        raise ValueError("apply_qdf_aux_members: empty members")
    K = len(members)
    has_store = hasattr(dataset, "_data") or hasattr(dataset, "data")
    is_inmem = has_store and hasattr(dataset, "slices")
    if not is_inmem:
        raise RuntimeError("apply_qdf_aux_members expects an InMemoryDataset with slices")

    store = _inmem_storage(dataset)
    smiles_list = getattr(store, "smiles", None)
    if smiles_list is None:
        raise RuntimeError("dataset storage has no 'smiles' attribute")

    if isinstance(smiles_list, torch.Tensor):
        smiles_iter = [str(s) for s in smiles_list.tolist()]
    else:
        smiles_iter = list(smiles_list)

    N = len(smiles_iter)
    mat = torch.full((N, K), float("nan"), dtype=torch.float32)
    n_row_any = 0
    for i, smi in enumerate(smiles_iter):
        key = str(smi).strip()
        row_ok = False
        for j, dct in enumerate(members):
            if key in dct:
                mat[i, j] = float(dct[key])
                row_ok = True
        if row_ok:
            n_row_any += 1

    setattr(store, attr, mat)
    if not hasattr(dataset, "slices") or dataset.slices is None:
        dataset.slices = {}
    dataset.slices[attr] = torch.arange(N + 1, dtype=torch.long)
    if hasattr(dataset, "_data_list"):
        dataset._data_list = None

    n_nan = N - n_row_any
    if verbose:
        print(
            f"[qdf_aux] attached_rows={n_row_any}  nan_rows={n_nan}  total={N}  "
            f"attr={attr!r}  K={K}",
        )
    return n_row_any, n_nan, K


def apply_qdf_aux_from_pt(
    dataset,
    path: str | Path,
    *,
    attr: str = "qdf_aux",
    verbose: bool = True,
) -> tuple[int, int, int]:
    """Load ``.pt`` from ``build_qdf_aux_ensemble.py`` and attach ``qdf_aux``.

    Returns ``(n_attached, n_nan, ensemble_k)`` where ``ensemble_k`` is the auxiliary
    head width (1 for scalar-only files).
    """
    path = Path(path)
    blob: Any = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise TypeError(f"unsupported qdf aux file: {type(blob)}")

    tm = blob.get("targets_members")
    if isinstance(tm, list) and len(tm) >= 2 and all(isinstance(x, dict) for x in tm):
        # Normalize keys like scalar path
        norm: list[dict[str, float]] = []
        for d in tm:
            norm.append({str(k).strip(): float(v) for k, v in d.items()})
        return apply_qdf_aux_members(dataset, norm, attr=attr, verbose=verbose)

    if isinstance(tm, list) and len(tm) == 1 and isinstance(tm[0], dict):
        raw = {str(k).strip(): float(v) for k, v in tm[0].items()}
        na, nn = apply_qdf_aux_scalars(dataset, raw, attr=attr, verbose=verbose)
        return na, nn, 1

    if "targets" in blob:
        raw = blob["targets"]
    else:
        raw = {k: float(v) for k, v in blob.items() if not str(k).startswith("_")}
    smap = {str(k).strip(): float(v) for k, v in raw.items()}
    na, nn = apply_qdf_aux_scalars(dataset, smap, attr=attr, verbose=verbose)
    return na, nn, 1


__all__ = ["apply_qdf_aux_scalars", "apply_qdf_aux_members", "apply_qdf_aux_from_pt"]
