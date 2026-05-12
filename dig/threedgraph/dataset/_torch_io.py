"""Trusted local ``torch.save`` payloads (PyG ``Data``, collated tensors, checkpoints).

PyTorch 2.6+ defaults ``torch.load(..., weights_only=True)``, which rejects
:class:`torch_geometric.data.Data` and related pickles — use ``weights_only=False``
for files produced by our own preprocessing.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch


def load_pt_trusted(fpath: str, *, map_location: Optional[str] = None) -> Any:
    kw: Dict[str, Any] = {}
    if map_location is not None:
        kw["map_location"] = map_location
    try:
        return torch.load(fpath, weights_only=False, **kw)
    except TypeError:
        # torch builds without ``weights_only`` (very old): behave like pre-2.6
        return torch.load(fpath, **kw)
