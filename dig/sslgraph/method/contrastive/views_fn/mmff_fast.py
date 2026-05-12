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


__all__ = ["FastRandomMMFFView", "env_enabled"]
