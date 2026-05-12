import os


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def accelerator_dataloader_kw():
    r"""Keyword args for :class:`~torch_geometric.loader.DataLoader`.

    Honoured env vars:

    * ``DATALOADER_NUM_WORKERS`` (default ``0``) — when positive, enables
      background workers and ``persistent_workers``.
    * ``PIN_MEMORY`` (default off) — set truthy to pass ``pin_memory=True``.
      Useful when copying CPU host tensors to a CUDA/XPU device.
    """
    kw: dict = {}
    try:
        nw = int(os.environ.get("DATALOADER_NUM_WORKERS", "0") or "0")
    except ValueError:
        nw = 0
    if nw > 0:
        kw["num_workers"] = nw
        kw["persistent_workers"] = True
    if _env_bool("PIN_MEMORY"):
        kw["pin_memory"] = True
    return kw
