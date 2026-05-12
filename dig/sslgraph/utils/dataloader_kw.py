import os


def accelerator_dataloader_kw():
    r"""Keyword args for :class:`~torch_geometric.loader.DataLoader`.

    Reads **DATALOADER_NUM_WORKERS** (default ``0``). When positive, enables
    background workers and ``persistent_workers``.
    """
    try:
        nw = int(os.environ.get("DATALOADER_NUM_WORKERS", "0") or "0")
    except ValueError:
        nw = 0
    if nw <= 0:
        return {}
    return {"num_workers": nw, "persistent_workers": True}
