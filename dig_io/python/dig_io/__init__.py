"""Python facade for the Rust ``dig_io`` native extension.

The crate ships several optional kernels that the DGCL Python code can opt
into. Importing this package never fails on its own -- the symbols are loaded
lazily so that environments without the Rust extension still work (callers
can fall back to the original Python paths).

Pattern follows ``QuantumDeepField_molecule/qdf_io/python/qdf_io/__init__.py``:

1. The normal ``_native`` extension next to this file.
2. A sidecar copy at :func:`fresh_native_sidecar_path` (useful when
   ``maturin develop`` cannot overwrite a loaded ``.pyd`` on Windows).
3. ``dig_io/target/release/dig_io.dll`` then ``target/maturin/dig_io.dll`` if
   the user only ran ``cargo build --release``.
"""

from __future__ import annotations

import importlib.util
import sys
import sysconfig
from pathlib import Path
from typing import Any


def fresh_native_sidecar_path() -> Path:
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"
    pkg_dir = Path(__file__).resolve().parent
    if ext.endswith(".pyd"):
        return pkg_dir / f"_native{ext[:-4]}.fresh.pyd"
    if ext.endswith(".so"):
        return pkg_dir / f"_native{ext[:-3]}.fresh.so"
    return pkg_dir / f"_native{ext}.fresh"


def dev_native_dll_paths() -> tuple[Path, ...]:
    pkg_dir = Path(__file__).resolve().parent
    crate_dir = pkg_dir.parent.parent
    return (
        crate_dir / "target" / "release" / "dig_io.dll",
        crate_dir / "target" / "maturin" / "dig_io.dll",
    )


_REQUIRED_SYMBOLS = (
    "uniform_sample_subgraph",
    "rw_sample_subgraph",
    "edge_perturb",
    "scaffold_bucket_split",
    "scaffold_bucket_sort",
    "MoleculeShardWriter",
    "MoleculeShardReader",
    "format_info",
)


def _try_exec_native(native_fqname: str, path: Path):
    if not path.is_file():
        return None
    sys.modules.pop(native_fqname, None)
    spec = importlib.util.spec_from_file_location(native_fqname, path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[native_fqname] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(native_fqname, None)
        return None
    if not all(hasattr(mod, name) for name in _REQUIRED_SYMBOLS):
        sys.modules.pop(native_fqname, None)
        return None
    return mod


def _load_native_or_none():
    """Return the ``dig_io._native`` module, or ``None`` if not available."""
    native_fqname = __name__ + "._native"
    try:
        from . import _native as mod  # type: ignore[attr-defined]

        if all(hasattr(mod, name) for name in _REQUIRED_SYMBOLS):
            return mod
    except ImportError:
        pass

    sys.modules.pop(native_fqname, None)
    for path in (fresh_native_sidecar_path(), *dev_native_dll_paths()):
        mod = _try_exec_native(native_fqname, path)
        if mod is not None:
            return mod
    return None


_mod = _load_native_or_none()


def _missing(*_a: Any, **_kw: Any):
    raise ImportError(
        "dig_io: native extension not loaded. Build it with\n"
        "  cd dig_io && maturin develop --release   (or `cargo build --release`).\n"
        "Until then, callers should fall back to the original Python path "
        "(impl='python')."
    )


def is_available() -> bool:
    """Return True if the Rust extension is loaded."""
    return _mod is not None


if _mod is not None:
    uniform_sample_subgraph = _mod.uniform_sample_subgraph
    rw_sample_subgraph = _mod.rw_sample_subgraph
    edge_perturb = _mod.edge_perturb
    scaffold_bucket_split = _mod.scaffold_bucket_split
    scaffold_bucket_sort = _mod.scaffold_bucket_sort
    MoleculeShardWriter = _mod.MoleculeShardWriter
    MoleculeShardReader = _mod.MoleculeShardReader
    format_info = _mod.format_info
else:
    uniform_sample_subgraph = _missing
    rw_sample_subgraph = _missing
    edge_perturb = _missing
    scaffold_bucket_split = _missing
    scaffold_bucket_sort = _missing
    MoleculeShardWriter = _missing
    MoleculeShardReader = _missing
    format_info = _missing


__all__ = [
    "is_available",
    "uniform_sample_subgraph",
    "rw_sample_subgraph",
    "edge_perturb",
    "scaffold_bucket_split",
    "scaffold_bucket_sort",
    "MoleculeShardWriter",
    "MoleculeShardReader",
    "format_info",
    "fresh_native_sidecar_path",
    "dev_native_dll_paths",
]
