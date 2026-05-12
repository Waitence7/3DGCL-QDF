"""Select the best PyTorch execution device without hard-coding CUDA only."""
from __future__ import annotations

import os
from typing import Optional

import torch


def _env_bool(var: str) -> bool:
    return os.environ.get(var, "").strip().lower() in ("1", "true", "yes", "on")


def _diag(msg: str) -> None:
    if _env_bool("TORCH_DEVICE_DEBUG"):
        print(f"[pick_torch_device] {msg}")


def _xpu_device_count() -> int:
    xm = getattr(torch, "xpu", None)
    if xm is None:
        return 0
    dc = getattr(xm, "device_count", None)
    if not callable(dc):
        return 0
    try:
        return int(dc())
    except Exception:
        return 0


def pick_torch_device(explicit: Optional[str] = None) -> torch.device:
    """Resolve training device.

    Order (first match wins):

    1. Non-empty ``explicit`` argument or ``TORCH_DEVICE`` env
    2. NVIDIA CUDA when available
    3. Apple MPS when available
    4. Intel ``torch.xpu`` when available (stock PyTorch ``+xpu`` wheels; optional legacy
       ``intel_extension_for_pytorch`` import is a no-op hook for older stacks)
    5. ``torch.npu`` (Huawei Ascend) when available and ``TORCH_SKIP_NPU`` is not truthy
    6. ``torch-directml`` when installed and ``USE_DIRECTML`` / ``TORCH_FALLBACK_DIRECTML`` is truthy
    7. CPU

    Intel Meteor / Core Ultra *NPU* tiles are usually not surfaced as plain ``torch.Device`` backends;
    use vendor workflows (OpenVINO / ONNX EP) or, on Windows laptops, try DirectML for the *integrated GPU*
    via optional ``pip install torch-directml`` plus ``TORCH_DEVICE=directml`` or ``USE_DIRECTML=1``.

    Set ``TORCH_DEVICE_DEBUG=1`` for a step-by-step print of why a device was (not) chosen.

    Set ``TORCH_DISABLE_XPU_DEFAULT=1`` to avoid auto-selecting XPU even for ``...+xpu`` wheels
    (e.g. you only want CUDA/CPU).
    """
    env = (os.environ.get("TORCH_DEVICE") or "").strip()
    pref = ((explicit if explicit is not None else "") or env).strip()

    _diag(f"explicit/TORCH_DEVICE={pref!r} torch={getattr(torch, '__version__', '?')}")

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
        dev = torch.device(f"cuda:{idx}")
        _diag(f"picked CUDA -> {dev}")
        return dev

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _diag("picked MPS")
        return torch.device("mps")

    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
    except ImportError:
        pass

    xm = getattr(torch, "xpu", None)
    ver_lower = (getattr(torch, "__version__", "") or "").lower()
    if xm is None:
        _diag("no torch.xpu (CPU-only wheel or very old PyTorch)")
    elif _env_bool("TORCH_DISABLE_XPU_DEFAULT"):
        _diag("TORCH_DISABLE_XPU_DEFAULT=1 — skip picking XPU even if bundled")
    else:
        xi = (os.environ.get("XPU_DEVICE_INDEX", "0") or "0").strip()
        runtime_ok = bool(getattr(xm, "is_available", lambda: False)())
        xd = _xpu_device_count()
        # Trust the ABI tag in the wheel: Intel `+xpu` builds expose XPU kernels even when
        # `device_count`/driver APIs briefly report unavailable in notebooks.
        built_for_xpu = "+xpu" in ver_lower
        if runtime_ok or xd > 0 or built_for_xpu:
            dev = torch.device(f"xpu:{xi}")
            _diag(f"picked XPU ({runtime_ok=} {xd=} built_for_xpu={built_for_xpu}) -> {dev}")
            return dev
        _diag(f"skipped XPU: runtime_ok=False device_count={xd} (+xpu not in version)")

    npu_mod = getattr(torch, "npu", None)
    if (
        not _env_bool("TORCH_SKIP_NPU")
        and npu_mod is not None
        and getattr(npu_mod, "is_available", lambda: False)()
    ):
        idx = (os.environ.get("NPU_DEVICE_INDEX", "0") or "0").strip()
        return torch.device(f"npu:{idx}")

    if _env_bool("USE_DIRECTML") or _env_bool("TORCH_FALLBACK_DIRECTML"):
        try:
            import torch_directml as dml  # type: ignore

            idx = int(os.environ.get("DIRECTML_DEVICE_INDEX", "0"))
            return dml.device(idx)
        except ImportError:
            pass

    _diag("fallback CPU — set TORCH_DEVICE_DEBUG=1 for step trace; Intel XPU needs torch+xpu (+cpu wheel never exposes xpu)")
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
