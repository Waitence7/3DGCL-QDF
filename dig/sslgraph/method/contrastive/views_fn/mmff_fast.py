"""Vectorised fast path for the ``MMFFrandom`` augmentation used by GraphCL.

The original :class:`~dig.sslgraph.method.contrastive.views_fn.combination.RandomView`
+ :class:`~dig.sslgraph.method.contrastive.views_fn.translation.NodeTranslation`
pipeline calls ``batch.to_data_list()`` followed by per-graph Python clones and
``Batch.from_data_list``. With ``MoleculeNet`` batches of ~300+ small molecules
that PyG reconstruction dominates the contrastive loop wall-clock (~60% in
``profile_pretrain.py`` measurements).

This module short-circuits the same logic with one or two ``index_select``
calls on the already-batched conformer slabs (``max{1..4}pos_mmff`` /
``max{1..4}_energy``). Output is statistically identical to the original
implementation (uniform choice over a fixed pair of MMFF slots per call), but
without leaving the device tensor world.

The fast path is opt-in:

* ``args.mmffrandom_impl='fast'`` in :class:`~dig.sslgraph.method.GraphCL`
* or env ``MMFFRANDOM_FAST=1`` (read by ``graphcl.py``).

Default behaviour stays bit-for-bit identical to the original Python path.
"""
from __future__ import annotations

import os
import random
from typing import Sequence

import torch
from torch_geometric.data import Batch, Data


_MMFF_POS_ATTR = {
    "MMFF1": "max1pos_mmff",
    "MMFF2": "max2pos_mmff",
    "MMFF3": "max3pos_mmff",
    "MMFF4": "max4pos_mmff",
}
_MMFF_ENERGY_ATTR = {
    "MMFF1": "max1_energy",
    "MMFF2": "max2_energy",
    "MMFF3": "max3_energy",
    "MMFF4": "max4_energy",
}


def env_enabled() -> bool:
    """``True`` iff env ``MMFFRANDOM_FAST`` is truthy."""
    return os.environ.get("MMFFRANDOM_FAST", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


class FastRandomMMFFView:
    r"""Drop-in replacement for ``RandomView([NodeTranslation(MMFF*), NodeTranslation(MMFF*)])``.

    Picks a per-graph slot uniformly from ``slots`` (length 1 or 2) and
    returns a :class:`~torch_geometric.data.Batch` whose ``pos`` (and ``energy``,
    when present) tensors are gathered from the matching ``maxKpos_mmff`` /
    ``maxK_energy`` slabs already batched onto ``batch``.

    Parameters
    ----------
    slots
        Sequence of ``"MMFF1".."MMFF4"`` labels, length 1 or 2. Matches the
        ``random.sample(['MMFF1', ..., 'MMFF4'], 2)`` call done once at
        :class:`GraphCL` init.
    """

    def __init__(self, slots: Sequence[str]) -> None:
        if not 1 <= len(slots) <= 2:
            raise ValueError(f"slots must have length 1 or 2, got {len(slots)}")
        bad = [s for s in slots if s not in _MMFF_POS_ATTR]
        if bad:
            raise ValueError(f"unknown MMFF slot(s): {bad}")
        self.slots = list(slots)
        self._pos_attrs = [_MMFF_POS_ATTR[s] for s in self.slots]
        self._energy_attrs = [_MMFF_ENERGY_ATTR[s] for s in self.slots]

    def __call__(self, data):
        return self.views_fn(data)

    def views_fn(self, batch_data):
        if not isinstance(batch_data, Batch):
            # Single Data: cheap fall-back, pick one slot at random.
            chosen = random.choice(self.slots)
            return Data(
                pos=getattr(batch_data, _MMFF_POS_ATTR[chosen]),
                smiles=getattr(batch_data, "smiles", None),
                z=batch_data.z,
                energy=getattr(batch_data, _MMFF_ENERGY_ATTR[chosen], None),
                min_energy=getattr(batch_data, "min_energy", None),
            )

        device = batch_data.z.device
        n_g = int(batch_data.num_graphs)
        n_nodes = batch_data.z.shape[0]

        if len(self.slots) == 1:
            pos_new = getattr(batch_data, self._pos_attrs[0])
            energy_new = getattr(batch_data, self._energy_attrs[0], None)
        else:
            pos_a = getattr(batch_data, self._pos_attrs[0])
            pos_b = getattr(batch_data, self._pos_attrs[1])
            pick = torch.randint(0, 2, (n_g,), device=device)
            pick_per_node = pick.index_select(0, batch_data.batch)
            stack = torch.stack([pos_a, pos_b], dim=0)
            arange = torch.arange(n_nodes, device=device)
            pos_new = stack[pick_per_node, arange]

            e_a = getattr(batch_data, self._energy_attrs[0], None)
            e_b = getattr(batch_data, self._energy_attrs[1], None)
            if e_a is not None and e_b is not None:
                e_stack = torch.stack([e_a, e_b], dim=0)
                ar_g = torch.arange(n_g, device=device)
                energy_new = e_stack[pick, ar_g]
            else:
                energy_new = None

        new = Batch()
        new.pos = pos_new
        new.z = batch_data.z
        new.batch = batch_data.batch
        if hasattr(batch_data, "ptr"):
            new.ptr = batch_data.ptr
        if hasattr(batch_data, "smiles"):
            new.smiles = batch_data.smiles
        if hasattr(batch_data, "min_energy"):
            new.min_energy = batch_data.min_energy
        if energy_new is not None:
            new.energy = energy_new
        new._num_graphs = n_g
        return new


class WeightedMMFFView:
    r"""Per-graph weighted sampling over MMFF conformer slots.

    Generalises :class:`FastRandomMMFFView` from uniform sampling over a fixed
    pair of slots to **categorical sampling over up to K slots** with a
    per-graph weight vector :math:`w_{g,k}` derived from a *stability* /
    *existence-probability* score.

    Weight source priority (first available wins):

    1. ``batch.mmff_weights`` (tensor of shape ``[B, K]``). Treated as raw
       scores: passed through softmax / l1-normalised internally so callers can
       inject either logits or probabilities. Typically pre-populated by a
       QDF-based offline pass (e.g. ``examples/sslgraph/bench/compute_mmff_weights.py``)
       and attached on the :class:`~torch_geometric.data.Data` objects before
       batching.
    2. Boltzmann factor from ``batch.maxK_energy`` attributes:
       :math:`w_{g,k} \propto \exp(-(E_{g,k} - \min_k E_{g,k})/k_BT)`. Uses the
       MMFF energies that already ship with the molecular dataset – cheap and
       physically meaningful (lower energy ⇒ more stable conformer ⇒ higher
       sampling probability).

    Parameters
    ----------
    slots
        Sequence of MMFF slot labels (subset of ``MMFF1..MMFF4``); defaults
        to all four.
    kT
        Boltzmann temperature for the fallback weight source. Larger ``kT``
        ⇒ more uniform; ``kT → 0`` ⇒ delta on the lowest-energy slot.
    weight_mode
        ``"auto"`` (default) tries ``batch.mmff_weights`` first then falls
        back to Boltzmann. ``"explicit"`` requires ``batch.mmff_weights``
        and errors otherwise. ``"boltzmann"`` forces the energy-based path.
    weight_norm
        How to interpret the raw ``batch.mmff_weights`` rows.

        * ``"auto"`` (default) – per-row detect: if every row is non-negative
          and sums to ``1±1e-3`` (i.e. it's already a probability distribution
          produced by, say, ``compute_mmff_weights.py``), pass it through
          unchanged. Otherwise softmax it as logits. This is what you almost
          certainly want — saving softmax(z) on disk and softmaxing it again
          here would just flatten any peaked weight back toward uniform.
        * ``"softmax"`` – always softmax (legacy behaviour; only correct when
          the on-disk weights are raw **logits**, not probabilities).
        * ``"linear"`` – clamp to ``[0, ∞)`` and l1-normalise. Same as ``auto``
          for already-softmaxed inputs but skips the detection.
    pick_mode
        How to turn the per-graph weight row :math:`w_{g,\cdot}` into a single
        slot index.

        * ``"sample"`` (default) – ``multinomial(W, 1)``. Stochastic per call,
          which is what GraphCL needs when both views are *independent* draws.
        * ``"top1"`` – deterministic argmax (highest-weight slot). Combined
          with another view producing a *different* slot, this makes the
          contrastive pair "best conformer ↔ alternative", removing the
          two-i.i.d.-samples degeneracy where view1 and view2 frequently land
          on the same slot.
        * ``"top2"`` – deterministic second argmax (second-highest weight).
          Pair ``WeightedMMFFView(pick_mode='top1')`` (view1) with
          ``WeightedMMFFView(pick_mode='top2')`` (view2) to get the "1st/2nd
          place" contrastive pair.
    """

    def __init__(
        self,
        slots: Sequence[str] = ("MMFF1", "MMFF2", "MMFF3", "MMFF4"),
        kT: float = 1.0,
        weight_mode: str = "auto",
        weight_norm: str = "auto",
        pick_mode: str = "sample",
    ) -> None:
        if not 1 <= len(slots) <= 4:
            raise ValueError(f"slots must have length in [1, 4], got {len(slots)}")
        bad = [s for s in slots if s not in _MMFF_POS_ATTR]
        if bad:
            raise ValueError(f"unknown MMFF slot(s): {bad}")
        if weight_mode not in ("auto", "explicit", "boltzmann"):
            raise ValueError(f"weight_mode must be auto/explicit/boltzmann, got {weight_mode!r}")
        if weight_norm not in ("auto", "softmax", "linear"):
            raise ValueError(f"weight_norm must be auto/softmax/linear, got {weight_norm!r}")
        if pick_mode not in ("sample", "top1", "top2"):
            raise ValueError(f"pick_mode must be sample/top1/top2, got {pick_mode!r}")
        if pick_mode == "top2" and len(slots) < 2:
            raise ValueError("pick_mode='top2' requires at least 2 slots")
        self.slots = list(slots)
        self._pos_attrs = [_MMFF_POS_ATTR[s] for s in self.slots]
        self._energy_attrs = [_MMFF_ENERGY_ATTR[s] for s in self.slots]
        self._slot_idx = {s: i for i, s in enumerate(self.slots)}
        self.kT = float(kT)
        self.weight_mode = weight_mode
        self.weight_norm = weight_norm
        self.pick_mode = pick_mode

    def _pick(self, W: torch.Tensor) -> torch.Tensor:
        """Return ``[B]`` long-tensor of slot indices according to ``pick_mode``."""
        if self.pick_mode == "sample":
            return torch.multinomial(W, num_samples=1).view(-1)
        if self.pick_mode == "top1":
            return W.argmax(dim=1)
        # top2: second-largest by weight.
        k = min(2, W.size(1))
        return torch.topk(W, k=k, dim=1).indices[:, k - 1]

    def __call__(self, data):
        return self.views_fn(data)

    def _resolve_weights(self, batch_data, n_g: int, device) -> torch.Tensor:
        """Return ``W`` of shape ``[B, K]`` (rows sum to 1)."""
        explicit = getattr(batch_data, "mmff_weights", None)
        if self.weight_mode == "explicit" and explicit is None:
            raise RuntimeError(
                "WeightedMMFFView(weight_mode='explicit') but batch has no "
                "'mmff_weights' attribute. Run "
                "examples/sslgraph/bench/compute_mmff_weights.py first."
            )

        if self.weight_mode != "boltzmann" and explicit is not None:
            w = explicit.to(device=device, dtype=torch.float32)
            if w.ndim == 1:
                w = w.view(n_g, -1)
            if w.shape != (n_g, len(self.slots)):
                # Allow K_in >= len(slots): slice to requested slots.
                if w.shape[0] == n_g and w.shape[1] >= len(self.slots):
                    idx = torch.tensor(
                        [self._slot_idx[s] if s in self._slot_idx else i
                         for i, s in enumerate(self.slots)],
                        device=device, dtype=torch.long,
                    )
                    w = w.index_select(1, idx)
                else:
                    raise RuntimeError(
                        f"mmff_weights shape {tuple(w.shape)} incompatible "
                        f"with (B={n_g}, K={len(self.slots)})"
                    )
            mode = self.weight_norm
            if mode == "auto":
                # Already a probability distribution? Then don't re-softmax;
                # doing so flattens peaked weights back toward uniform (e.g.
                # softmax([0.07, 0.77, ...]) ≈ [0.20, 0.39, ...]).
                row_min = float(w.min())
                row_sums = w.sum(dim=1)
                looks_like_probs = (
                    row_min >= -1e-6
                    and bool(((row_sums - 1.0).abs() < 1e-3).all())
                )
                mode = "linear" if looks_like_probs else "softmax"
            if mode == "softmax":
                return torch.softmax(w, dim=1)
            w = torch.clamp(w, min=0.0)
            denom = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
            return w / denom

        # Boltzmann fallback from MMFF energies.
        e_cols = []
        for attr in self._energy_attrs:
            e = getattr(batch_data, attr, None)
            if e is None:
                raise RuntimeError(
                    f"WeightedMMFFView: batch missing '{attr}' for Boltzmann fallback"
                )
            e_cols.append(e.to(device=device, dtype=torch.float32).view(n_g))
        E = torch.stack(e_cols, dim=1)        # [B, K]
        E = E - E.min(dim=1, keepdim=True).values
        return torch.softmax(-E / max(self.kT, 1e-6), dim=1)

    def views_fn(self, batch_data):
        if not isinstance(batch_data, Batch):
            # Single Data: gather weights for one row, sample once.
            E = torch.stack(
                [getattr(batch_data, a).view(-1).float() for a in self._energy_attrs],
                dim=1,
            )
            E = E - E.min(dim=1, keepdim=True).values
            w = torch.softmax(-E / max(self.kT, 1e-6), dim=1).view(-1)
            if self.pick_mode == "sample":
                chosen_idx = int(torch.multinomial(w, 1).item())
            elif self.pick_mode == "top1":
                chosen_idx = int(w.argmax().item())
            else:  # top2
                k = min(2, w.numel())
                chosen_idx = int(torch.topk(w, k=k).indices[k - 1].item())
            chosen = self.slots[chosen_idx]
            return Data(
                pos=getattr(batch_data, _MMFF_POS_ATTR[chosen]),
                smiles=getattr(batch_data, "smiles", None),
                z=batch_data.z,
                energy=getattr(batch_data, _MMFF_ENERGY_ATTR[chosen], None),
                min_energy=getattr(batch_data, "min_energy", None),
            )

        device = batch_data.z.device
        n_g = int(batch_data.num_graphs)
        n_nodes = batch_data.z.shape[0]

        W = self._resolve_weights(batch_data, n_g, device)   # [B, K]
        # One slot per graph: (B,) in [0, K) — sampled or deterministic.
        pick = self._pick(W)
        pick_per_node = pick.index_select(0, batch_data.batch)

        pos_stack = torch.stack(
            [getattr(batch_data, a) for a in self._pos_attrs], dim=0
        )  # [K, N_nodes_total, 3]
        arange = torch.arange(n_nodes, device=device)
        pos_new = pos_stack[pick_per_node, arange]

        e_list = [getattr(batch_data, a, None) for a in self._energy_attrs]
        if all(e is not None for e in e_list):
            e_stack = torch.stack([e.view(n_g) for e in e_list], dim=0)  # [K, B]
            ar_g = torch.arange(n_g, device=device)
            energy_new = e_stack[pick, ar_g]
        else:
            energy_new = None

        new = Batch()
        new.pos = pos_new
        new.z = batch_data.z
        new.batch = batch_data.batch
        if hasattr(batch_data, "ptr"):
            new.ptr = batch_data.ptr
        if hasattr(batch_data, "smiles"):
            new.smiles = batch_data.smiles
        if hasattr(batch_data, "min_energy"):
            new.min_energy = batch_data.min_energy
        if energy_new is not None:
            new.energy = energy_new
        new._num_graphs = n_g
        return new


def env_weighted_enabled() -> bool:
    """``True`` iff env ``MMFFRANDOM_WEIGHTED`` is truthy."""
    return os.environ.get("MMFFRANDOM_WEIGHTED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


__all__ = [
    "FastRandomMMFFView",
    "WeightedMMFFView",
    "env_enabled",
    "env_weighted_enabled",
]
