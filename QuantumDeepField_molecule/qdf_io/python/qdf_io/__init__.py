"""Python facade for the Rust ``qdf_io`` native extension.

Only the reader is implemented in Rust; the writer lives in
``QuantumDeepField_molecule/train/dataset_shard.py`` because it is run once
per dataset and does not need to be fast. Both sides share the constants in
``format_info()``.
"""

from ._native import ShardReader, format_info

__all__ = ["ShardReader", "format_info"]
