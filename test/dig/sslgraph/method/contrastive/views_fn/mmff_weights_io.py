"""Attach pre-computed MMFF slot weights to PyG ``Data`` objects.

Pairs with :class:`~dig.sslgraph.method.contrastive.views_fn.WeightedMMFFView`
and the ``examples/sslgraph/bench/compute_mmff_weights.py`` CLI. For
``--source qdf``, CSV rows must come from a **HOMO/LUMO** QDF model only
(``homo``, ``lumo`` columns per slot); atomization-energy checkpoints are not
used for this pipeline.

Typical flow::

    # offline (once)
    python examples/sslgraph/bench/compute_mmff_weights.py \\
        --dataset esol --source boltzmann
    #   -> dataset/esol_mmff_weights_boltzmann.pt

    # at training time
    from dig.threedgraph.dataset import PygMoleculeNet as MoleculeNet
    from dig.sslgraph.method.contrastive.views_fn.mmff_weights_io import (
        load_weights, apply_mmff_weights,
    )

    dataset = MoleculeNet(root='dataset/', name='esol')
    w = load_weights('dataset/esol_mmff_weights_boltzmann.pt')
    apply_mmff_weights(dataset, w)        # in-place: data.mmff_weights = tensor[K]

    # ... then build DataLoader as usual; WeightedMMFFView reads it.

The helper never deletes or rewrites the underlying dataset – it only adds an
attribute on each ``Data`` instance. Molecules missing from the weights dict
are silently skipped (they will fall back to Boltzmann-from-energy at view
time when ``weight_mode='auto'``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import torch


def load_weights(path: str | Path) -> dict[str, torch.Tensor]:
    """Read a ``.pt`` produced by ``compute_mmff_weights.py`` and return the
    ``{smiles: tensor[K]}`` mapping (rows already softmaxed).

    Keys are ``.strip()``-normalised on load so they match
    ``apply_mmff_weights``'s lookup, since MoleculeNet's raw SMILES list can
    carry trailing whitespace for a few molecules.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "weights" in payload:
        raw = payload["weights"]
    elif isinstance(payload, Mapping):
        raw = payload
    else:
        raise TypeError(
            f"unsupported weights file format at {path}: {type(payload)!r}"
        )
    return {str(k).strip(): v for k, v in raw.items()}


def apply_mmff_weights(
    dataset,
    weights: Mapping[str, torch.Tensor],
    *,
    attr: str = "mmff_weights",
    default: torch.Tensor | None = None,
    verbose: bool = True,
) -> tuple[int, int]:
    """Attach per-graph MMFF slot weights to ``dataset`` (in place).

    For ``torch_geometric.data.InMemoryDataset`` subclasses (e.g.
    :class:`~dig.threedgraph.dataset.MoleculeNet`) this updates the *collated*
    storage so subsequent ``DataLoader`` batches carry ``batch.mmff_weights``
    of shape ``[B, K]`` automatically. Falls back to per-Data attribute writes
    for plain sequences of :class:`~torch_geometric.data.Data`.

    Parameters
    ----------
    dataset
        InMemoryDataset or list-like of ``Data`` objects.
    weights
        ``{smiles: tensor[K]}`` mapping.
    attr
        Output attribute name on each Data (default ``mmff_weights``).
    default
        Optional fallback weight vector used for molecules missing from
        ``weights``. ``None`` ⇒ skip the molecule; ``WeightedMMFFView`` will
        fall back to Boltzmann from energies at view time when
        ``weight_mode='auto'``.

    Returns
    -------
    (n_attached, n_missing) : tuple[int, int]
    """
    has_store = hasattr(dataset, "_data") or hasattr(dataset, "data")
    is_inmem = has_store and hasattr(dataset, "slices")
    if is_inmem:
        return _apply_inmem(dataset, weights, attr=attr, default=default,
                            verbose=verbose)
    return _apply_list(dataset, weights, attr=attr, default=default,
                       verbose=verbose)


def _inmem_storage(dataset):
    """Return the underlying collated Data of an InMemoryDataset, preferring
    the private ``_data`` accessor introduced in newer PyG versions (avoids
    the public ``data`` access warning)."""
    return getattr(dataset, "_data", None) or dataset.data


def _apply_inmem(dataset, weights: Mapping[str, torch.Tensor], *,
                 attr: str, default: torch.Tensor | None,
                 verbose: bool) -> tuple[int, int]:
    store = _inmem_storage(dataset)
    smiles_list = getattr(store, "smiles", None)
    if smiles_list is None:
        raise RuntimeError("InMemoryDataset has no 'smiles' attribute; cannot key weights")

    # smiles in PygMoleculeNet is stored as a list of strings (one per graph).
    if isinstance(smiles_list, torch.Tensor):
        smiles_iter = [str(s) for s in smiles_list.tolist()]
    else:
        smiles_iter = list(smiles_list)

    N = len(smiles_iter)
    rows = []
    n_attached = 0
    n_missing = 0
    K_ref: int | None = None
    for smi in smiles_iter:
        w = weights.get(str(smi).strip())
        if w is None:
            w = default
        if w is None:
            n_missing += 1
            # placeholder, will be a zero row; never read because
            # WeightedMMFFView(weight_mode='auto') falls back to Boltzmann
            # for molecules whose weights look degenerate. Pick uniform here
            # so even 'explicit' mode produces something sane.
            if K_ref is None:
                # Infer K from the first real weight; default to 4.
                K_ref_local = 4
            else:
                K_ref_local = K_ref
            rows.append(torch.full((K_ref_local,), 1.0 / K_ref_local,
                                   dtype=torch.float32))
        else:
            w_t = w.detach().to(dtype=torch.float32).view(-1).clone()
            if K_ref is None:
                K_ref = w_t.numel()
            elif w_t.numel() != K_ref:
                raise RuntimeError(
                    f"inconsistent K: weight for {smi!r} has length "
                    f"{w_t.numel()}, expected {K_ref}"
                )
            rows.append(w_t)
            n_attached += 1

    if K_ref is None:
        K_ref = 4

    # Pad placeholder rows to match K_ref if needed (only when first hits were
    # missing). Cheap second pass.
    rows = [
        r if r.numel() == K_ref else torch.full((K_ref,), 1.0 / K_ref,
                                                dtype=torch.float32)
        for r in rows
    ]
    W = torch.stack(rows, dim=0)                  # [N, K]
    setattr(store, attr, W)

    # slices: each graph gets a single row -> arange(N + 1).
    if not hasattr(dataset, "slices") or dataset.slices is None:
        dataset.slices = {}
    dataset.slices[attr] = torch.arange(N + 1, dtype=torch.long)

    # Invalidate cached data list so __getitem__ uses the updated collated tensor.
    if hasattr(dataset, "_data_list"):
        dataset._data_list = None

    if verbose:
        print(f"[mmff_weights] (inmem) attached={n_attached}  missing={n_missing}  "
              f"(total={N}, K={K_ref})")
    return n_attached, n_missing


def _apply_list(dataset: Iterable, weights: Mapping[str, torch.Tensor], *,
                attr: str, default: torch.Tensor | None,
                verbose: bool) -> tuple[int, int]:
    n_attached = 0
    n_missing = 0
    for d in dataset:
        smi = getattr(d, "smiles", None)
        if smi is None:
            n_missing += 1
            continue
        w = weights.get(str(smi).strip())
        if w is None:
            if default is None:
                n_missing += 1
                continue
            w = default
        setattr(d, attr, w.detach().clone().float())
        n_attached += 1
    if verbose:
        print(f"[mmff_weights] (list) attached={n_attached}  missing={n_missing}")
    return n_attached, n_missing


__all__ = ["load_weights", "apply_mmff_weights"]
