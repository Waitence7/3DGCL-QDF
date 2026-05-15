"""PyG geometric ops that may not have native kernels for all torch devices."""
from __future__ import annotations

import torch
from torch_geometric.nn import radius_graph


def _torch_cluster_available() -> bool:
    try:
        import torch_cluster  # noqa: F401

        return True
    except ImportError:
        return False


_HAS_TORCH_CLUSTER = _torch_cluster_available()


def _radius_graph_pure_torch(
    pos: torch.Tensor,
    r: float,
    batch: torch.Tensor | None,
    *,
    loop: bool = False,
    max_num_neighbors: int = 32,
) -> torch.Tensor:
    """O(n²) radius graph per batch graph — correct for small molecules; no ``torch-cluster``.

    Matches the intent of ``torch_geometric.nn.radius_graph`` (neighbor pairs within ``r``).
    """
    device = pos.device
    dtype = pos.dtype
    if batch is None:
        batch = torch.zeros(pos.size(0), dtype=torch.long, device=device)

    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    r_t = torch.tensor(float(r), device=device, dtype=dtype)

    for b in batch.unique(sorted=True):
        m = batch == b
        idx = torch.nonzero(m, as_tuple=False).view(-1)
        n = int(idx.numel())
        if n == 0:
            continue
        p = pos.index_select(0, idx)
        dist = torch.cdist(p, p)
        if not loop:
            dist = dist.clone()
            dist.fill_diagonal_(torch.finfo(dtype).max)

        for i in range(n):
            drow = dist[i]
            cand = torch.nonzero(drow <= r_t, as_tuple=False).view(-1)
            if cand.numel() == 0:
                continue
            if max_num_neighbors is not None and int(cand.numel()) > int(max_num_neighbors):
                vals = drow[cand]
                _, pick = torch.topk(vals, int(max_num_neighbors), largest=False)
                cand = cand[pick]
            gi = idx[i]
            rows.append(torch.full((cand.numel(),), gi, device=device, dtype=torch.long))
            cols.append(idx[cand])

    if not rows:
        return torch.empty(2, 0, dtype=torch.long, device=device)
    row = torch.cat(rows)
    col = torch.cat(cols)
    return torch.stack([row, col], dim=0)


def _radius_graph_dispatch(
    pos: torch.Tensor,
    r: float,
    batch: torch.Tensor | None,
    **kwargs: object,
) -> torch.Tensor:
    loop = bool(kwargs.get("loop", False))
    mn = kwargs.get("max_num_neighbors", 32)
    max_num_neighbors = int(mn) if mn is not None else 32
    if _HAS_TORCH_CLUSTER:
        return radius_graph(pos, r=r, batch=batch, **kwargs)  # type: ignore[call-arg]
    return _radius_graph_pure_torch(
        pos, r, batch, loop=loop, max_num_neighbors=max_num_neighbors,
    )


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

    When ``torch-cluster`` is not installed (common on aarch64), a small-molecule
    pure PyTorch fallback is used instead of ``torch_geometric.nn.radius_graph``.
    """
    dev = pos.device
    if dev.type == "cpu":
        return _radius_graph_dispatch(pos, r=r, batch=batch, **kwargs)
    if dev.type == "cuda":
        return _radius_graph_dispatch(pos, r=r, batch=batch, **kwargs)

    pos_c = pos.detach().cpu().contiguous()
    b_c = None if batch is None else batch.detach().cpu().contiguous()
    edge_index = _radius_graph_dispatch(pos_c, r=r, batch=b_c, **kwargs)
    return edge_index.to(device=dev, dtype=torch.long)
