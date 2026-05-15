# 3DGCL workflow bundle (preprocess · predict · pretrain)

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
Generated: 2026-05-14 09:57 UTC
