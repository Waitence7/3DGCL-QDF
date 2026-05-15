#!/usr/bin/env bash

# Dataset.
#dataset=QM9under7atoms_atomizationenergy_eV
# dataset=QM9under14atoms_atomizationenergy_eV
# dataset=QM9full_atomizationenergy_eV
dataset=QM9under7atoms_homolumo_eV  # Two properties (homo and lumo).
# dataset=QM9full_homolumo_eV
# dataset=yourdataset_property_unit

# Basis set.
basis_set=6-31G

# Grid field.
radius=0.75
grid_interval=0.3

# Geometry backend for the heavy per-molecule work:
#   PREPROCESS_BACKEND=numpy   (default, original SciPy path)
#   PREPROCESS_BACKEND=rust    (qdf_io Rayon parallel kernels; requires maturin build)
# Optional tuning when using rust:
#   PREPROCESS_RUST_BATCH_SIZE=64
PREPROCESS_BACKEND="${PREPROCESS_BACKEND:-numpy}"
PREPROCESS_RUST_BATCH_SIZE="${PREPROCESS_RUST_BATCH_SIZE:-64}"
# npy | shard | both  (default npy; shard needs qdf_io ShardWriter)
PREPROCESS_OUTPUT_FORMAT="${PREPROCESS_OUTPUT_FORMAT:-npy}"

python preprocess.py "$dataset" "$basis_set" "$radius" "$grid_interval" \
  --backend "$PREPROCESS_BACKEND" \
  --rust-batch-size "$PREPROCESS_RUST_BATCH_SIZE" \
  --output-format "$PREPROCESS_OUTPUT_FORMAT"
