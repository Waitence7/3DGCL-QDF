"""Optional Rust-backed reimplementations of the small CPU-side helpers used
by ``QuantumDeepField.LCAO`` (``pad`` and ``list_to_batch``).

These are *opt-in*: nothing here modifies ``train/train.py``. The caller
explicitly applies a patch to a model instance::

    from train.train import QuantumDeepField
    from train.model_patches import apply_rust_lcao

    model = QuantumDeepField(...).to(device)
    apply_rust_lcao(model, what=("pad", "list_to_batch"))

The original ``pad`` and ``list_to_batch`` methods remain on the
``QuantumDeepField`` class definition; only the instance attributes are
overridden. To revert, call ``unapply_rust_lcao(model)``.

Why this helps
--------------
The original implementations do per-molecule host->device transfers inside
a Python loop. The Rust helpers assemble the same data into a single
``numpy.ndarray`` on the host so we can do exactly one
``torch.from_numpy(...).to(device)`` per call. The model graph downstream of
these helpers is untouched -- in particular, ``distance_matrices`` is treated
as a plain input tensor (no gradient flows back through it), which matches
the original behaviour.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch


_LCAO_ORIGINALS_ATTR = "_qdf_lcao_originals"


def _ensure_numpy_f32(x) -> np.ndarray:
    """Coerce a single matrix argument of ``pad`` to a C-contiguous float32 numpy array."""
    if isinstance(x, np.ndarray):
        if x.dtype == np.float32 and x.flags.c_contiguous:
            return x
        return np.ascontiguousarray(x, dtype=np.float32)
    if torch.is_tensor(x):
        return np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
    return np.ascontiguousarray(np.asarray(x), dtype=np.float32)


def _ensure_numpy_i64(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        if x.dtype == np.int64 and x.flags.c_contiguous:
            return x
        return np.ascontiguousarray(x, dtype=np.int64)
    if torch.is_tensor(x):
        return np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.int64)
    return np.ascontiguousarray(np.asarray(x), dtype=np.int64)


def apply_rust_lcao(model, what: Iterable[str] = ("pad", "list_to_batch")) -> None:
    """Replace ``model.pad`` and/or ``model.list_to_batch`` with versions
    backed by the Rust ``qdf_io`` helpers.

    ``what`` selects which methods to patch:
      - ``"pad"``: replace ``model.pad`` only.
      - ``"list_to_batch"``: replace ``model.list_to_batch`` only.
    Pass both for the full Rust LCAO host path.
    """
    import qdf_io  # imported here so the patch module is usable even when
                   # the Rust extension is not built yet.

    what = set(what)
    if not what.issubset({"pad", "list_to_batch"}):
        raise ValueError(f"Unknown patch targets: {what}")

    device = model.device

    # Stash the originals so we can revert.
    originals = getattr(model, _LCAO_ORIGINALS_ATTR, {})

    if "pad" in what and "pad" not in originals:
        originals["pad"] = model.pad
    if "list_to_batch" in what and "list_to_batch" not in originals:
        originals["list_to_batch"] = model.list_to_batch
    setattr(model, _LCAO_ORIGINALS_ATTR, originals)

    if "pad" in what:
        def rust_pad(matrices: Sequence, pad_value):
            # Convert any non-f32 / non-contiguous inputs first; in the
            # standard QDF pipeline they are already float32 C-contiguous.
            mats = [_ensure_numpy_f32(m) for m in matrices]
            arr = qdf_io.block_diag_pad_f32(mats, float(pad_value))
            return torch.from_numpy(arr).to(device, non_blocking=True)

        model.pad = rust_pad

    if "list_to_batch" in what:
        def rust_list_to_batch(xs: Sequence, dtype=torch.FloatTensor,
                               cat=None, axis=None):
            # Mirror the original method's behaviour exactly:
            #
            #   def list_to_batch(self, xs, dtype=torch.FloatTensor,
            #                     cat=None, axis=None):
            #       xs = [dtype(x).to(self.device) for x in xs]
            #       if cat:
            #           return torch.cat(xs, axis)
            #       else:
            #           return xs
            #
            # We do the concatenation on the host (via Rust) and ship the
            # result with a single device transfer. When ``cat`` is falsy we
            # upload one concatenated tensor and split on the device, which
            # preserves the per-molecule list interface expected by the
            # downstream Python loops in LCAO / V.

            if dtype is torch.LongTensor:
                arrs = [_ensure_numpy_i64(x) for x in xs]
                merged = qdf_io.concat_i64(arrs)
                big = torch.from_numpy(merged).to(device, non_blocking=True)
                if cat:
                    return big
                lengths = [int(a.shape[0]) for a in arrs]
                return list(torch.split(big, lengths, dim=0))

            # Default: float32 path (matches dtype=torch.FloatTensor).
            arrs = [_ensure_numpy_f32(np.atleast_2d(x)) for x in xs]
            if cat:
                if axis == 1:
                    merged = qdf_io.concat_f32_axis1(arrs)
                else:
                    merged = qdf_io.concat_f32_axis0(arrs)
                return torch.from_numpy(merged).to(device, non_blocking=True)

            # No cat: keep the per-molecule list interface alive. We still
            # benefit from a single host->device transfer.
            merged = qdf_io.concat_f32_axis0(arrs)
            big = torch.from_numpy(merged).to(device, non_blocking=True)
            lengths = [int(a.shape[0]) for a in arrs]
            chunks = list(torch.split(big, lengths, dim=0))
            return chunks

        model.list_to_batch = rust_list_to_batch


def unapply_rust_lcao(model) -> None:
    """Restore the original ``pad`` / ``list_to_batch`` methods on ``model``."""
    originals = getattr(model, _LCAO_ORIGINALS_ATTR, None)
    if not originals:
        return
    for name, fn in originals.items():
        setattr(model, name, fn)
    try:
        delattr(model, _LCAO_ORIGINALS_ATTR)
    except AttributeError:
        pass


__all__ = ["apply_rust_lcao", "unapply_rust_lcao"]
