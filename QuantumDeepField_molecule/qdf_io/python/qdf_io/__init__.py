"""Python facade for the Rust ``qdf_io`` native extension.

Reader and writer for the QDF binary shard both live in Rust. The Python
helper in ``QuantumDeepField_molecule/train/dataset_shard.py`` still owns
``np.load``-ing each ``.npy``, but the binary packing is delegated to
:class:`ShardWriter` so we do not pay Python-level ``struct.pack`` and
``BufferedWriter.write_all`` overhead per field.

On Windows the installed ``_native*.pyd`` is often stale or locked (Jupyter,
``maturin develop`` cannot overwrite). This package then tries, in order:

1. The normal ``_native`` extension next to this file (must export ``ShardWriter``).
2. The sidecar file from :func:`fresh_native_sidecar_path` (copy the new DLL there).
3. ``qdf_io/target/release/qdf_io.dll`` then ``target/maturin/qdf_io.dll`` if present (after ``cargo build --release`` / maturin).
"""

from __future__ import annotations

import importlib.util
import sys
import sysconfig
from pathlib import Path


def fresh_native_sidecar_path() -> Path:
    """Path to the optional sidecar native module (e.g. ``_native.cp310-win_amd64.fresh.pyd``)."""
    ext = sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"
    pkg_dir = Path(__file__).resolve().parent
    if ext.endswith(".pyd"):
        return pkg_dir / f"_native{ext[:-4]}.fresh.pyd"
    if ext.endswith(".so"):
        return pkg_dir / f"_native{ext[:-3]}.fresh.so"
    return pkg_dir / f"_native{ext}.fresh"


def dev_native_dll_paths() -> tuple[Path, ...]:
    """In-repo cargo/maturin outputs — try in order after sidecar."""
    pkg_dir = Path(__file__).resolve().parent
    crate_dir = pkg_dir.parent.parent
    return (
        crate_dir / "target" / "release" / "libqdf_io.so",
        crate_dir / "target" / "release" / "qdf_io.dll",
        crate_dir / "target" / "maturin" / "qdf_io.dll",
    )


def _try_exec_native(native_fqname: str, path: Path):
    """Load extension from ``path``; return module or ``None`` if missing / unusable."""
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
    if not hasattr(mod, "ShardWriter"):
        sys.modules.pop(native_fqname, None)
        return None
    return mod


def _load_native():
    """Return the ``qdf_io._native`` module."""
    native_fqname = __name__ + "._native"
    first_err: BaseException | None = None
    try:
        from . import _native as mod

        if hasattr(mod, "ShardWriter"):
            # Stale in-tree ``_native`` may lack newer symbols (e.g. legacy batch).
            if hasattr(mod, "preprocess_batch_rust_legacy"):
                return mod
            sys.modules.pop(native_fqname, None)
            for path in (fresh_native_sidecar_path(), *dev_native_dll_paths()):
                mod2 = _try_exec_native(native_fqname, path)
                if mod2 is not None and hasattr(mod2, "preprocess_batch_rust_legacy"):
                    return mod2
            from . import _native as mod_fresh

            return mod_fresh
    except ImportError as e:
        first_err = e

    sys.modules.pop(native_fqname, None)

    for path in (fresh_native_sidecar_path(), *dev_native_dll_paths()):
        mod = _try_exec_native(native_fqname, path)
        if mod is not None:
            return mod

    if first_err is not None:
        raise first_err
    raise ImportError(
        "qdf_io: no usable native module (need ShardWriter). Run in "
        "``QuantumDeepField_molecule/qdf_io``: ``cargo build --release`` or "
        "``maturin develop --release``, or copy ``target/release/qdf_io.dll`` to ``"
        f"{fresh_native_sidecar_path()}``."
    )


_mod = _load_native()

ShardReader = _mod.ShardReader
ShardWriter = _mod.ShardWriter
block_diag_pad_f32 = _mod.block_diag_pad_f32
concat_f32_axis0 = _mod.concat_f32_axis0
concat_f32_axis1 = _mod.concat_f32_axis1
concat_i64 = _mod.concat_i64
format_info = _mod.format_info
preprocess_batch_rust = _mod.preprocess_batch_rust
preprocess_batch_rust_legacy = getattr(_mod, "preprocess_batch_rust_legacy", None)
preprocess_molecule_rust = _mod.preprocess_molecule_rust
parse_molecule_block_rust = _mod.parse_molecule_block_rust

__all__ = [
    "ShardReader",
    "ShardWriter",
    "block_diag_pad_f32",
    "concat_f32_axis0",
    "concat_f32_axis1",
    "concat_i64",
    "format_info",
    "preprocess_batch_rust",
    "preprocess_batch_rust_legacy",
    "preprocess_molecule_rust",
    "parse_molecule_block_rust",
    "fresh_native_sidecar_path",
    "dev_native_dll_paths",
]
