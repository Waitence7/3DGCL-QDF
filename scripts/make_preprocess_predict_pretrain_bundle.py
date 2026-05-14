#!/usr/bin/env python3
"""Build a portable source zip: QDF preprocess/predict + 3DGCL SSLGraph pretrain.

Excludes large or machine-local trees (datasets, checkpoints, Rust ``target/``,
``.venv``, ``__pycache__``). Recipients run ``uv sync`` (or pip) and build
``qdf_io`` / ``dig_io`` with maturin/cargo as in ``BUNDLE_README.md``.

Usage (repo root)::

    .venv\\Scripts\\python.exe scripts\\make_preprocess_predict_pretrain_bundle.py
    .venv\\Scripts\\python.exe scripts\\make_preprocess_predict_pretrain_bundle.py --out dist\\my.zip
"""
from __future__ import annotations

import argparse
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "target",
    }
)

# Do not walk these top-level QDF subtrees (huge user data).
SKIP_QDF_TOP = frozenset({"dataset", "output"})

# Max single file size (bytes); skip if larger (safety).
MAX_FILE_BYTES = 80 * 1024 * 1024

ROOT_FILES = ("pyproject.toml", "uv.lock", "README.md", "연구노트.md")

REL_DIRS = (
    "dig",
    "dig_io",
    "QuantumDeepField_molecule/qdf_io",
    "QuantumDeepField_molecule/train",
    "QuantumDeepField_molecule/predict",
    "QuantumDeepField_molecule/bench",
    "examples/sslgraph/bench",
)

REL_FILES = (
    "examples/sslgraph/pretrain.ipynb",
    "examples/sslgraph/verify_views_backend.py",
    "examples/sslgraph/verify_dig_io_fallback.py",
)


def iter_files_under(rel_dir: str) -> list[Path]:
    base = REPO / rel_dir
    if not base.is_dir():
        return []
    out: list[Path] = []
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(REPO)
        parts = rel.parts
        if any(n in SKIP_DIR_NAMES for n in parts):
            continue
        if parts[0] == "QuantumDeepField_molecule" and len(parts) > 1:
            if parts[1] in SKIP_QDF_TOP:
                continue
        if p.stat().st_size > MAX_FILE_BYTES:
            continue
        out.append(p)
    return out


def bundle_readme() -> str:
    return """# 3DGCL workflow bundle (preprocess · predict · pretrain)

This archive contains **source code only** (no QM9/QDF datasets, no trained checkpoints).

## 1. Environment

- Python 3.10–3.12 (see root ``pyproject.toml``).
- Install: from the extracted root, run ``uv sync`` (recommended) or install
  dependencies + editable ``dig`` per ``README.md``.
- PyTorch / PyTorch Geometric: use the **cpu** or **xpu** group from ``pyproject.toml``.

## 2. Rust extensions (optional but recommended)

- **QDF preprocess / shard:** ``QuantumDeepField_molecule/qdf_io`` — ``maturin develop --release`` (or ``cargo build --release``) from that directory.
- **DGCL views / shard:** ``dig_io`` — same pattern.

## 3. QuantumDeepField — preprocess

From repo root (example; positional args per ``train/preprocess.py``)::

    python QuantumDeepField_molecule/train/preprocess.py <dataset> 6-31G 0.75 0.3 --backend rust --output-format both

Shell helpers: ``QuantumDeepField_molecule/train/preprocess.sh``.

## 4. QuantumDeepField — predict

- ``QuantumDeepField_molecule/predict/preprocess.py`` / ``preprocess.sh``
- ``QuantumDeepField_molecule/predict/predict.py`` / ``predict.sh``

## 5. 3DGCL — SSLGraph pretrain

- Notebook: ``examples/sslgraph/pretrain.ipynb``
- Quality / MMFF-weighted comparisons: ``examples/sslgraph/bench/compare_pretrain_quality.ipynb``,
  ``compare_pretrain_ab.py``, ``pretrain_quality_core.py``

Place ESOL (or other) PyG data and any **weights ``.pt`` / QDF CSV** paths expected by
your notebooks/scripts; they are not included in this zip.

---
Generated: {ts}
""".format(
        ts=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output zip path (default: dist/3DGCL_preprocess_predict_pretrain_<date>.zip)",
    )
    args = ap.parse_args()

    dist = REPO / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    out = args.out
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        out = dist / f"3DGCL_preprocess_predict_pretrain_{stamp}.zip"

    files: list[Path] = []
    for name in ROOT_FILES:
        p = REPO / name
        if p.is_file():
            files.append(p)

    for rd in REL_DIRS:
        files.extend(iter_files_under(rd))

    for rf in REL_FILES:
        p = REPO / rf
        if p.is_file():
            files.append(p)

    seen: set[str] = set()
    unique: list[Path] = []
    for p in sorted(files, key=lambda x: str(x).lower()):
        s = str(p.relative_to(REPO)).replace("\\", "/")
        if s in seen:
            continue
        seen.add(s)
        unique.append(p)

    readme_name = "BUNDLE_README.md"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(readme_name, bundle_readme())
        for p in unique:
            arc = p.relative_to(REPO).as_posix()
            zf.write(p, arcname=arc)

    print(f"Wrote {out} ({len(unique) + 1} entries including {readme_name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
