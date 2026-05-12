"""Single-file binary shard for the preprocessed QDF dataset.

This module adds an *alternative* loader to ``train/train.py``'s ``MyDataset``.
Both loaders are kept in the code base so the user can pick at runtime
(e.g. via ``--loader npy`` vs ``--loader shard``) and compare wall-clock cost.

On-disk layout (little-endian, version 1)::

    Header  (64 bytes, fixed)
      offset  size  field
        0      8    magic = b"QDFSHRD\\0"
        8      4    version (u32 LE)         -- 1
       12      8    n_molecules (u64 LE)
       20      4    n_output (u32 LE)         -- 0 when has_property == False
       24      4    flags (u32 LE)            -- bit0 = has_property
       28      8    index_table_offset (u64 LE)
       36      8    data_section_offset (u64 LE)
       44      8    file_size (u64 LE)
       52     12    reserved (zeros)

    Index table (at index_table_offset, n_molecules * 8 bytes)
      For each molecule i: u64 LE absolute offset of its molecule record.

    Molecule record (starts 8-aligned)
      Header (16 bytes):
        u32 n_orbitals
        u32 n_field
        u32 idx_len
        u32 reserved
      [idx_len bytes]      idx UTF-8 string
      <pad to 8>
      int64[n_orbitals]    atomic_orbitals
      float32[n_field * n_orbitals]  distance_matrix (C-order)
      float32[n_orbitals]  quantum_numbers  (exposed to Python as shape (1, N))
      float32              N_electrons      (exposed as shape (1, 1))
      -- if flags has_property: --
      float32[n_output]    property_values  (exposed as shape (1, n_output))
      float32[n_field]     potential        (exposed as shape (n_field, 1))
      <pad to 8 for next record>

The Rust ``qdf_io.ShardReader`` mmaps the file and returns the same tuple
shape that ``MyDataset.__getitem__`` does, so the rest of ``train.py`` does
not need to change.
"""

from __future__ import annotations

import os
import struct
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch.utils.data

MAGIC = b"QDFSHRD\0"
VERSION = 1
HEADER_SIZE = 64
FLAG_HAS_PROPERTY = 1


def _align_up(x: int, a: int) -> int:
    return (x + a - 1) & ~(a - 1)


def _validate_against_native() -> None:
    """Best-effort runtime check that the Python writer matches the Rust reader.

    Imported lazily so writing a shard does not require the Rust extension to be
    installed (the reader is what needs it).
    """
    try:
        from qdf_io import format_info
    except Exception:  # pragma: no cover - extension may not be built yet
        return
    info = format_info()
    assert bytes(info["magic"]) == MAGIC, "magic mismatch with Rust extension"
    assert info["version"] == VERSION
    assert info["header_size"] == HEADER_SIZE
    assert info["flag_has_property"] == FLAG_HAS_PROPERTY


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #

def _write_padding(f, count: int) -> None:
    if count:
        f.write(b"\x00" * count)


def write_shard(
    src_dir: str | Path,
    dst_path: str | Path,
    *,
    has_property: bool | None = None,
    progress: bool = True,
) -> dict:
    """Convert a directory of ``np.save`` files (one ``.npy`` per molecule, as
    produced by ``train/preprocess.py``) into a single shard at ``dst_path``.

    The original ``.npy`` files are *not* touched or removed.

    Returns a small dict with ``{"n_molecules", "bytes_written", "elapsed_sec"}``.
    """
    _validate_against_native()
    src_dir = Path(src_dir)
    dst_path = Path(dst_path)
    files = sorted(p for p in src_dir.iterdir() if p.suffix == ".npy")
    if not files:
        raise FileNotFoundError(f"No .npy files in {src_dir}")

    # Peek the first record to lock the schema.
    first = np.load(files[0], allow_pickle=True)
    fields_per_record = len(first)
    if fields_per_record not in (6, 8):
        raise ValueError(
            f"Unexpected record length {fields_per_record} in {files[0]} "
            "(expected 6 without property or 8 with property)."
        )
    detected_has_property = fields_per_record == 8
    if has_property is None:
        has_property = detected_has_property
    elif has_property != detected_has_property:
        raise ValueError(
            f"has_property={has_property} but the first .npy contains "
            f"{fields_per_record} fields (expected "
            f"{8 if has_property else 6})."
        )

    n_output = int(first[6].shape[1]) if has_property else 0

    n = len(files)
    flags = FLAG_HAS_PROPERTY if has_property else 0

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # Two-pass write: we don't know offsets until we lay out the records, so
    # we (1) reserve the header + index table, (2) write each molecule and
    # record its absolute offset, (3) patch the header.
    with open(dst_path, "wb") as f:
        f.write(b"\x00" * HEADER_SIZE)
        index_table_offset = f.tell()
        f.write(b"\x00" * (n * 8))
        # Align data section to 8 bytes.
        pad = _align_up(f.tell(), 8) - f.tell()
        _write_padding(f, pad)
        data_section_offset = f.tell()

        mol_offsets: list[int] = []

        for i, path in enumerate(files):
            if progress and (i % 1000 == 0 or i == n - 1):
                pct = 100.0 * (i + 1) / n
                print(f"  shard write [{i + 1:>6}/{n}] {pct:5.1f}%  {path.name}")

            rec = np.load(path, allow_pickle=True)
            idx_str: str = str(rec[0])
            atomic_orbitals = np.ascontiguousarray(rec[1], dtype=np.int64)
            distance_matrix = np.ascontiguousarray(rec[2], dtype=np.float32)
            quantum_numbers = np.ascontiguousarray(rec[3], dtype=np.float32)
            n_electrons = np.ascontiguousarray(rec[4], dtype=np.float32)
            n_field_int = int(rec[5])

            n_orbitals = int(atomic_orbitals.shape[0])
            if distance_matrix.shape != (n_field_int, n_orbitals):
                raise ValueError(
                    f"{path}: distance_matrix shape {distance_matrix.shape} "
                    f"!= (n_field={n_field_int}, n_orbitals={n_orbitals})"
                )
            if quantum_numbers.shape != (1, n_orbitals):
                raise ValueError(
                    f"{path}: quantum_numbers shape {quantum_numbers.shape} != (1, {n_orbitals})"
                )
            if n_electrons.shape != (1, 1):
                raise ValueError(
                    f"{path}: N_electrons shape {n_electrons.shape} != (1, 1)"
                )

            # Align this record to 8 bytes.
            pad = _align_up(f.tell(), 8) - f.tell()
            _write_padding(f, pad)
            mol_offset = f.tell()
            mol_offsets.append(mol_offset)

            idx_bytes = idx_str.encode("utf-8")
            f.write(struct.pack("<IIII", n_orbitals, n_field_int, len(idx_bytes), 0))
            f.write(idx_bytes)
            pad = _align_up(f.tell(), 8) - f.tell()
            _write_padding(f, pad)

            f.write(atomic_orbitals.tobytes(order="C"))
            f.write(distance_matrix.tobytes(order="C"))
            # quantum_numbers stored flat as (n_orbitals,); reshape on the reader side
            f.write(quantum_numbers.reshape(-1).tobytes(order="C"))
            # N_electrons stored as a single float32
            f.write(np.float32(n_electrons.reshape(-1)[0]).tobytes())

            if has_property:
                property_values = np.ascontiguousarray(rec[6], dtype=np.float32)
                potential = np.ascontiguousarray(rec[7], dtype=np.float32)
                if property_values.shape != (1, n_output):
                    raise ValueError(
                        f"{path}: property_values shape {property_values.shape} "
                        f"!= (1, {n_output})"
                    )
                # preprocess.py produces potential as the result of a matmul
                # whose right operand is column-shaped, so the original layout
                # is (n_field, 1). We also accept the flat shape for safety.
                if potential.shape == (n_field_int, 1):
                    pot_flat = potential.reshape(-1)
                elif potential.shape == (n_field_int,):
                    pot_flat = potential
                else:
                    raise ValueError(
                        f"{path}: potential shape {potential.shape} "
                        f"is neither ({n_field_int}, 1) nor ({n_field_int},)"
                    )
                f.write(property_values.reshape(-1).tobytes(order="C"))
                f.write(pot_flat.tobytes(order="C"))

        file_size = f.tell()

        # Patch the index table.
        f.seek(index_table_offset)
        f.write(b"".join(struct.pack("<Q", off) for off in mol_offsets))

        # Patch the header.
        f.seek(0)
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<Q", n))
        f.write(struct.pack("<I", n_output))
        f.write(struct.pack("<I", flags))
        f.write(struct.pack("<Q", index_table_offset))
        f.write(struct.pack("<Q", data_section_offset))
        f.write(struct.pack("<Q", file_size))
        # Remaining bytes of the header are already zeroed (reserved).

    elapsed = time.perf_counter() - t0
    return {
        "n_molecules": n,
        "bytes_written": file_size,
        "elapsed_sec": elapsed,
        "has_property": has_property,
        "n_output": n_output,
    }


# --------------------------------------------------------------------------- #
# Dataset wrapper
# --------------------------------------------------------------------------- #

class MyDatasetShard(torch.utils.data.Dataset):
    """Drop-in alternative to ``train.MyDataset`` backed by a single shard file.

    The shape and dtypes of each returned tuple match those produced by
    ``MyDataset.__getitem__`` so that ``collate_fn`` and the model code in
    ``train.py`` keep working unchanged.
    """

    def __init__(self, shard_path: str | os.PathLike):
        from qdf_io import ShardReader  # imported lazily so writing a shard
                                        # does not require the Rust build.

        self.shard_path = str(shard_path)
        self._reader = ShardReader(self.shard_path)
        self._length = len(self._reader)
        self.n_output = self._reader.n_output
        self.has_property = self._reader.has_property

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple:
        return self._reader.get(int(idx))

    def __repr__(self) -> str:  # pragma: no cover - for debugging only
        return (f"MyDatasetShard(path={self.shard_path!r}, n={self._length}, "
                f"has_property={self.has_property}, n_output={self.n_output})")


# --------------------------------------------------------------------------- #
# Convenience: derive the expected shard path next to a preprocessed dir
# --------------------------------------------------------------------------- #

def default_shard_path(npy_dir: str | os.PathLike) -> Path:
    """Given a preprocessed directory like
    ``QM9under14atoms_atomizationenergy_eV/train_6-31G_0.75sphere_0.3grid``
    return the canonical shard path beside it (``..._shard.bin``).
    """
    p = Path(npy_dir)
    return p.parent / (p.name + "_shard.bin")
