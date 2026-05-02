"""Select the best PyTorch execution device without hard-coding CUDA only."""
from __future__ import annotations

import os
from typing import Optional

import torch


def pick_torch_device(explicit: Optional[str] = None) -> torch.device:
    """Resolve training device.

    Order (first match wins):

    1. Non-empty ``explicit`` argument or ``TORCH_DEVICE`` env
    2. NVIDIA CUDA when available
    3. ``torch.npu`` (Huawei Ascend stacks) when available
    4. Apple MPS when available
    5. Intel ``torch.xpu`` (Intel Extension for PyTorch GPU path) when available
    6. ``torch-directml`` when installed and ``USE_DIRECTML`` / ``TORCH_FALLBACK_DIRECTML`` is truthy
    7. CPU

    Intel Meteor / Core Ultra *NPU* tiles are usually not surfaced as plain ``torch.Device`` backends;
    use vendor workflows (OpenVINO / ONNX EP) or, on Windows laptops, try DirectML for the *integrated GPU*
    via optional ``pip install torch-directml`` plus ``TORCH_DEVICE=directml`` or ``USE_DIRECTML=1``.
    """
    env = (os.environ.get("TORCH_DEVICE") or "").strip()
    pref = ((explicit if explicit is not None else "") or env).strip()

    if pref.lower() in ("directml", "dml"):
        try:
            import torch_directml as dml  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "TORCH_DEVICE=directml(or dml) requires the torch-directml package."
            ) from e
        idx = int(os.environ.get("DIRECTML_DEVICE_INDEX", "0"))
        return dml.device(idx)

    if pref:
        return torch.device(pref)

    if torch.cuda.is_available():
        idx = (os.environ.get("CUDA_DEVICE_INDEX", "0") or "0").strip()
        return torch.device(f"cuda:{idx}")

    npu = getattr(torch, "npu", None)
    if npu is not None and getattr(npu, "is_available", lambda: False)():
        idx = (os.environ.get("NPU_DEVICE_INDEX", "0") or "0").strip()
        return torch.device(f"npu:{idx}")

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
    except ImportError:
        pass
    xpu_mod = getattr(torch, "xpu", None)
    if xpu_mod is not None and getattr(xpu_mod, "is_available", lambda: False)():
        xi = (os.environ.get("XPU_DEVICE_INDEX", "0") or "0").strip()
        return torch.device(f"xpu:{xi}")

    def _env_bool(var: str) -> bool:
        return os.environ.get(var, "").strip().lower() in ("1", "true", "yes", "on")

    if _env_bool("USE_DIRECTML") or _env_bool("TORCH_FALLBACK_DIRECTML"):
        try:
            import torch_directml as dml  # type: ignore

            idx = int(os.environ.get("DIRECTML_DEVICE_INDEX", "0"))
            return dml.device(idx)
        except ImportError:
            pass

    return torch.device("cpu")


def empty_accel_cache() -> None:
    """Best-effort device memory caches where PyTorch exposes an API."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xmod = getattr(torch, "xpu", None)
    if xmod is not None and getattr(xmod, "is_available", lambda: False)():
        ecc = getattr(xmod, "empty_cache", None)
        if ecc is not None:
            try:
                ecc()
            except Exception:
                pass

    nmod = getattr(torch, "npu", None)
    if nmod is not None and getattr(nmod, "is_available", lambda: False)():
        ecc = getattr(nmod, "empty_cache", None)
        if ecc is not None:
            try:
                ecc()
            except Exception:
                pass
