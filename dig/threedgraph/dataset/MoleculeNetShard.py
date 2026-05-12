"""Drop-in alternative to ``dig.threedgraph.dataset.MoleculeNet`` backed by the
``dig_io`` Rust binary shard.

Original semantics are preserved on the ``MoleculeNet`` side. This class is an
*opt-in* loader; callers pick between the two via a ``loader='pt'|'shard'``
toggle (see :class:`dig.sslgraph.evaluation.finetune.Finetune`).

Layout on disk
--------------

For a normal ``MoleculeNet`` build the processed payload lives at
``{root}/{name}/processed/data.pt`` and is read via
:func:`dig.threedgraph.dataset._torch_io.load_pt_trusted`. The Rust shard
sibling is ``{root}/{name}/processed/data.shard`` and is read by mmapping
through :class:`dig_io.MoleculeShardReader`.

The ``data.shard`` is built once by :func:`convert_inmemory_to_shard` (CLI:
``examples/sslgraph/convert_dataset_to_shard.py``). The original ``.pt`` is
left in place so users can toggle back to the upstream loader at any time.
"""

from __future__ import annotations

import os
import os.path as osp
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset


def default_shard_path(root: Union[str, os.PathLike], name: str) -> Path:
    """``{root}/{name}/processed/data.shard``."""
    return Path(root) / name.lower() / "processed" / "data.shard"


def _maybe_load_native():
    try:
        import dig_io  # type: ignore

        if dig_io.is_available():
            return dig_io
    except Exception:
        pass
    return None


def _data_to_record(d: Data) -> dict:
    """Coerce a PyG ``Data`` (as produced by ``MoleculeNet.process``) into a
    dict of contiguous numpy arrays + a ``mmff`` dict (when present)."""
    rec: dict = {}
    rec["idx"] = str(getattr(d, "smiles", ""))[:32] or "_"
    rec["smiles"] = str(getattr(d, "smiles", ""))

    z = d.z.detach().cpu().numpy().astype(np.int64, copy=False).reshape(-1)
    pos = d.pos.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1, 3)
    if pos.shape[0] != z.shape[0]:
        raise ValueError(
            f"z/pos shape mismatch: z={z.shape}, pos={pos.shape}"
        )

    if d.edge_index is not None and d.edge_index.numel() > 0:
        edge_index = d.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
        if edge_index.shape[0] != 2:
            edge_index = edge_index.reshape(2, -1)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    if d.edge_attr is not None and d.edge_attr.numel() > 0:
        ea = d.edge_attr.detach().cpu().numpy().astype(np.int64, copy=False)
        if ea.ndim == 1:
            ea = ea.reshape(-1, 1)
    else:
        ea = np.zeros((0, 0), dtype=np.int64)

    if d.x is not None and d.x.numel() > 0:
        x = d.x.detach().cpu().numpy().astype(np.int64, copy=False)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
    else:
        x = np.zeros((0, 0), dtype=np.int64)

    y = d.y.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)

    rec["z"] = z
    rec["pos"] = pos
    rec["edge_index"] = edge_index
    rec["edge_attr"] = ea
    rec["x"] = x
    rec["y"] = y

    has_mmff = all(
        getattr(d, k, None) is not None
        for k in (
            "max1pos_mmff",
            "max2pos_mmff",
            "max3pos_mmff",
            "max4pos_mmff",
            "min_energy",
            "max1_energy",
            "max2_energy",
            "max3_energy",
            "max4_energy",
        )
    )
    if has_mmff:
        rec["mmff"] = {
            "max1pos_mmff": d.max1pos_mmff.detach().cpu().numpy().astype(np.float32),
            "max2pos_mmff": d.max2pos_mmff.detach().cpu().numpy().astype(np.float32),
            "max3pos_mmff": d.max3pos_mmff.detach().cpu().numpy().astype(np.float32),
            "max4pos_mmff": d.max4pos_mmff.detach().cpu().numpy().astype(np.float32),
            "min_energy": float(d.min_energy),
            "max1_energy": float(d.max1_energy),
            "max2_energy": float(d.max2_energy),
            "max3_energy": float(d.max3_energy),
            "max4_energy": float(d.max4_energy),
        }
    return rec


def _record_to_data(rec: dict) -> Data:
    pos = torch.from_numpy(np.ascontiguousarray(rec["pos"])).float()
    z = torch.from_numpy(np.ascontiguousarray(rec["z"])).long()
    ei = torch.from_numpy(np.ascontiguousarray(rec["edge_index"])).long()
    y = torch.from_numpy(np.ascontiguousarray(rec["y"])).float()
    smiles = rec.get("smiles", "")

    fields = dict(pos=pos, z=z, edge_index=ei, y=y, smiles=smiles)
    if "edge_attr" in rec:
        fields["edge_attr"] = torch.from_numpy(np.ascontiguousarray(rec["edge_attr"])).long()
    if "x" in rec:
        fields["x"] = torch.from_numpy(np.ascontiguousarray(rec["x"])).long()
    if "mmff" in rec:
        m = rec["mmff"]
        fields["max1pos_mmff"] = torch.from_numpy(np.ascontiguousarray(m["max1pos_mmff"])).float()
        fields["max2pos_mmff"] = torch.from_numpy(np.ascontiguousarray(m["max2pos_mmff"])).float()
        fields["max3pos_mmff"] = torch.from_numpy(np.ascontiguousarray(m["max3pos_mmff"])).float()
        fields["max4pos_mmff"] = torch.from_numpy(np.ascontiguousarray(m["max4pos_mmff"])).float()
        fields["min_energy"] = float(m["min_energy"])
        fields["max1_energy"] = float(m["max1_energy"])
        fields["max2_energy"] = float(m["max2_energy"])
        fields["max3_energy"] = float(m["max3_energy"])
        fields["max4_energy"] = float(m["max4_energy"])
    return Data(**fields)


def convert_inmemory_to_shard(
    src_dataset,
    dst_path: Union[str, os.PathLike],
    *,
    progress: bool = True,
) -> dict:
    """Build a ``data.shard`` next to an existing collated ``data.pt``.

    ``src_dataset`` must be index-able and produce PyG ``Data`` objects shaped
    like ``MoleculeNet.process``'s output.
    """
    native = _maybe_load_native()
    if native is None:
        raise RuntimeError(
            "dig_io native extension not loaded; cannot write shard. "
            "Build with `cd dig_io && maturin develop --release`."
        )
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(src_dataset)
    writer = native.MoleculeShardWriter(str(dst_path), int(n))
    written = 0
    skipped = 0
    iterator = range(n)
    if progress:
        try:
            from tqdm import tqdm  # type: ignore

            iterator = tqdm(iterator, total=n, desc="shard")
        except Exception:
            pass

    for i in iterator:
        try:
            d = src_dataset[i]
            rec = _data_to_record(d)
            writer.append_record(
                rec["idx"],
                rec["smiles"],
                rec["z"],
                rec["pos"],
                rec["edge_index"],
                rec["edge_attr"],
                rec["x"],
                rec["y"],
                rec.get("mmff"),
            )
            written += 1
        except Exception as e:  # pragma: no cover - data-defensive
            skipped += 1
            if progress:
                print(f"  skip {i}: {e}")

    writer.finalize()
    return {
        "n_written": written,
        "n_skipped": skipped,
        "path": str(dst_path),
        "bytes": os.path.getsize(dst_path),
    }


class MoleculeNetShard(Dataset):
    """Lightweight :class:`torch.utils.data.Dataset` backed by ``dig_io.MoleculeShardReader``.

    Drop-in for ``MoleculeNet`` in :class:`Finetune` flows that only need
    per-sample ``Data`` access (the original ``InMemoryDataset.data/slices``
    accessors are intentionally *not* provided).
    """

    def __init__(
        self,
        root: Union[str, os.PathLike],
        name: str,
        transform=None,
        shard_path: Optional[Union[str, os.PathLike]] = None,
    ):
        self.root = str(root)
        self.name = name.lower()
        self.transform = transform
        self.shard_path = Path(shard_path) if shard_path else default_shard_path(root, self.name)
        if not self.shard_path.is_file():
            raise FileNotFoundError(
                f"shard not found: {self.shard_path}. "
                "Build with examples/sslgraph/convert_dataset_to_shard.py "
                f"--name {self.name}."
            )
        native = _maybe_load_native()
        if native is None:
            raise RuntimeError(
                "dig_io native extension not loaded; cannot read shard. "
                "Build with `cd dig_io && maturin develop --release`."
            )
        self._reader = native.MoleculeShardReader(str(self.shard_path))
        self._length = len(self._reader)

    def __len__(self) -> int:
        return self._length

    def _get_one(self, idx: int) -> Data:
        rec = self._reader.get(int(idx))
        data = _record_to_data(rec)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __getitem__(self, idx):
        # PyG ``InMemoryDataset[list_or_slice]`` returns a sliced dataset; we
        # emulate that with :class:`torch.utils.data.Subset` so downstream code
        # such as ``scaffold_split`` and the shuffle in ``Finetune`` keeps working.
        if isinstance(idx, (list, tuple)):
            return torch.utils.data.Subset(self, [int(i) for i in idx])
        if isinstance(idx, slice):
            return torch.utils.data.Subset(self, list(range(*idx.indices(self._length))))
        if isinstance(idx, np.ndarray):
            if idx.ndim != 1:
                raise IndexError(f"ndarray index must be 1-D; got shape {idx.shape}")
            return torch.utils.data.Subset(self, idx.astype(np.int64).tolist())
        if torch.is_tensor(idx):
            if idx.dim() != 1:
                raise IndexError(f"tensor index must be 1-D; got shape {tuple(idx.shape)}")
            return torch.utils.data.Subset(self, idx.long().tolist())
        return self._get_one(int(idx))

    def __repr__(self) -> str:
        return (
            f"MoleculeNetShard(name={self.name!r}, "
            f"n={self._length}, path={str(self.shard_path)!r})"
        )
