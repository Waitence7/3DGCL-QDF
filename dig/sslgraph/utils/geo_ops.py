"""PyG geometric ops that may not have native kernels for all torch devices."""
from __future__ import annotations

import torch
from torch_geometric.nn import radius_graph


def radius_graph_device_safe(
    pos: torch.Tensor,
    r: float,
    batch: torch.Tensor | None = None,
    **kwargs: object,
) -> torch.Tensor:
    """``radius_graph`` / ``torch_cluster.radius`` often ship **CPU-only** binaries.

    Passing XPU (or MPS) position tensors hits ``x must be CPU tensor`` asserts.
    Build the neighbor graph on CPU and move indices back to ``pos.device``.
    CUDA keeps the original path (GPU radius when available).
    """
    dev = pos.device
    if dev.type == "cpu":
        return radius_graph(pos, r=r, batch=batch, **kwargs)  # type: ignore[call-arg]
    if dev.type == "cuda":
        return radius_graph(pos, r=r, batch=batch, **kwargs)  # type: ignore[call-arg]

    pos_c = pos.detach().cpu().contiguous()
    b_c = None if batch is None else batch.detach().cpu().contiguous()
    edge_index = radius_graph(pos_c, r=r, batch=b_c, **kwargs)  # type: ignore[call-arg]
    return edge_index.to(device=dev, dtype=torch.long)
