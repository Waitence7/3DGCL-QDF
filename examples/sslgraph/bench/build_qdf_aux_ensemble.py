#!/usr/bin/env python3
"""Build per-SMILES scalar targets for QDF-assisted (ensemble) GraphCL pretrain.

Reads one or more ``qdf_mmff_predict.py`` CSVs (same schema). With multiple
files, merges per-(smiles, slot) with ``mean`` or ``median`` (same rule as
``compute_mmff_weights.py --pred-csv`` repeated).

The default scalar for each SMILES is the mean of the (merged) value over the
four MMFF slots:

* ``atomization`` schema: mean of the four ``energy`` values.
* ``homolumo`` schema: mean of the four ``homo`` values (use ``--scalar lumo``
  for mean LUMO instead).

Output ``.pt`` format::

    {\"targets\": {\"SMILES\": float, ...}, \"meta\": {...}}

If ``--pred-csv`` is passed **more than once** (distinct QDF runs / checkpoints),
also writes ``targets_members``: a list of ``K`` per-SMILES dicts (same scalar
rule as ``targets``, but **no** cross-CSV slot merge). GraphCL B/C can then use
a **K-output** auxiliary head and average MSE across members (multi-teacher
ensemble) instead of only regressing the merged scalar.

Load into MoleculeNet with::

    from dig.sslgraph.method.contrastive.views_fn.qdf_aux_io import (
        apply_qdf_aux_from_pt,
    )
    apply_qdf_aux_from_pt(dataset, \"dataset/esol_qdf_aux_ensemble.pt\")

Then set ``args.qdf_aux_lambda`` (e.g. ``0.05``) before pretrain.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

BENCH = Path(__file__).resolve().parent
REPO = BENCH.parents[2]
for _p in (REPO, BENCH):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from compute_mmff_weights import merge_qdf_slot_rows, read_qdf_slot_rows  # noqa: E402


def _scalar_per_smiles_from_merged(
    merged: dict[str, dict[int, tuple[float, ...]]],
    col_idx: int,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for smi, slots in merged.items():
        vals = [float(slots[k][col_idx]) for k in (1, 2, 3, 4)]
        out[smi] = sum(vals) / len(vals)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pred-csv", type=Path, action="append", required=True,
        help="QDF prediction CSV (repeat for multi-checkpoint ensemble).",
    )
    ap.add_argument(
        "--qdf-ensemble-reduce", choices=("mean", "median"), default="mean",
        help="Per-(smiles, slot) merge across CSVs.",
    )
    ap.add_argument(
        "--scalar", choices=("auto", "energy", "homo", "lumo"), default="auto",
        help="Column averaged over MMFF slots for the target. "
             "``auto`` = energy for atomization, homo for homolumo.",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tables = [read_qdf_slot_rows(p) for p in args.pred_csv]
    prop, spec, merged = merge_qdf_slot_rows(tables, how=args.qdf_ensemble_reduce)
    vc = spec["value_cols"]
    mode = args.scalar
    if mode == "auto":
        mode = "energy" if prop == "atomization" else "homo"
    if mode == "energy" and "energy" not in vc:
        raise SystemExit("atomization (energy column) required for --scalar energy")
    if mode == "homo" and "homo" not in vc:
        raise SystemExit("homolumo CSV required for --scalar homo")
    if mode == "lumo" and "lumo" not in vc:
        raise SystemExit("homolumo CSV required for --scalar lumo")

    col_idx = vc.index(mode)

    targets = _scalar_per_smiles_from_merged(merged, col_idx)

    meta = {
        "qdf_property": prop,
        "value_cols": list(vc),
        "scalar": mode,
        "ensemble_reduce": args.qdf_ensemble_reduce,
        "n_csv": len(args.pred_csv),
        "n_molecules": len(targets),
        "pred_csv": [str(p) for p in args.pred_csv],
    }
    out: dict = {"targets": targets, "meta": meta}

    if len(args.pred_csv) > 1:
        members: list[dict[str, float]] = []
        for p in args.pred_csv:
            one = read_qdf_slot_rows(p)
            _prop_o, _spec_o, merged_one = merge_qdf_slot_rows(
                [one], how=args.qdf_ensemble_reduce,
            )
            members.append(_scalar_per_smiles_from_merged(merged_one, col_idx))
        out["targets_members"] = members
        meta["qdf_aux_ensemble_k"] = len(members)
        print(
            f"[qdf_aux] targets_members: K={len(members)} "
            f"(per-CSV scalar; GraphCL aux = mean MSE over K heads)",
            file=sys.stderr,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(json.dumps(meta, indent=2))
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
