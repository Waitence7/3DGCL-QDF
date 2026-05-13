"""Dump a ``.pt`` MMFF-slot weights file into a human-readable CSV.

The ``.pt`` payload produced by ``compute_mmff_weights.py`` looks like::

    {
        "weights": {smiles: tensor[K]},   # K = 4 for MMFF1..4
        "slots": ("MMFF1", "MMFF2", "MMFF3", "MMFF4"),
        "kT": 1.0,
        "source": "qdf",
        "qdf_property": "homolumo",
        ...
    }

This script reads it and writes a CSV with one row per SMILES::

    smiles, w1, w2, w3, w4, top_slot, entropy_bits, source

Where ``top_slot`` is ``argmax(w) + 1`` (1-indexed MMFF slot) and
``entropy_bits`` is ``-Σ w_k log2 w_k``, capped to ``[0, log2 K]``.

Example
-------
    python examples/sslgraph/bench/dump_mmff_weights.py \\
        --in  dataset/esol_mmff_weights_qdf.pt \\
        --out dataset/esol_mmff_weights_qdf.csv \\
        --sort-by entropy
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch


def _load(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(
            f"unexpected payload type in {path}: {type(payload).__name__}; "
            "expected dict from compute_mmff_weights.py"
        )
    if "weights" not in payload:
        raise ValueError(
            f"{path} has no 'weights' key — keys: {list(payload.keys())}"
        )
    return payload


def _row(smiles: str, w: torch.Tensor, *, source: str) -> dict[str, str]:
    w = w.detach().to(torch.float64).flatten()
    K = int(w.numel())
    s = float(w.sum())
    if s <= 0 or not math.isfinite(s):
        # malformed row → fall back to uniform so the CSV still shows it
        p = [1.0 / K] * K
    else:
        p = [float(x) / s for x in w.tolist()]
    eps = 1e-12
    ent = -sum(pi * math.log2(pi + eps) for pi in p)
    top = int(max(range(K), key=lambda i: p[i])) + 1
    row = {
        "smiles": smiles.strip(),
        "top_slot": str(top),
        "entropy_bits": f"{ent:.6f}",
        "source": source,
    }
    for k in range(K):
        row[f"w{k + 1}"] = f"{p[k]:.6f}"
    return row


def _maybe_attach_top_diff(row: dict, w: torch.Tensor) -> None:
    """Add ``top_minus_uniform`` so users can sort by \"how peaked is this molecule\"."""
    K = int(w.numel())
    top_p = float(w.detach().to(torch.float64).flatten().max() / w.sum())
    row["top_minus_uniform"] = f"{(top_p - 1.0 / K):.6f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True, type=Path,
                    help=".pt file from compute_mmff_weights.py")
    ap.add_argument("--out", dest="out_path", type=Path, default=None,
                    help="output CSV (default: alongside .pt with .csv suffix)")
    ap.add_argument("--sort-by", choices=("smiles", "entropy", "top_slot",
                                          "top_minus_uniform"),
                    default="smiles", help="row ordering")
    ap.add_argument("--limit", type=int, default=0,
                    help="only write first N rows after sorting (0 = all)")
    args = ap.parse_args()

    in_path = args.in_path.resolve()
    out_path = args.out_path or in_path.with_suffix(".csv")
    out_path = out_path.resolve()

    payload = _load(in_path)
    weights: dict = payload["weights"]
    slots = payload.get("slots", ("MMFF1", "MMFF2", "MMFF3", "MMFF4"))
    source = str(payload.get("source", "?"))
    kT = payload.get("kT")
    qdf_prop = payload.get("qdf_property")
    score_expr = payload.get("score_expr")
    normalize = payload.get("normalize")

    print(f"loaded:  {in_path}")
    print(f"  source = {source}  kT = {kT}  qdf_property = {qdf_prop}")
    if source == "qdf":
        print(f"  score_expr = {score_expr!r}  normalize = {normalize!r}")
    print(f"  slots  = {tuple(slots)}  n_smiles = {len(weights)}")

    rows: list[dict[str, str]] = []
    for smi, w in weights.items():
        if not torch.is_tensor(w):
            w = torch.as_tensor(w, dtype=torch.float32)
        r = _row(smi, w, source=source)
        _maybe_attach_top_diff(r, w)
        rows.append(r)

    if args.sort_by == "smiles":
        rows.sort(key=lambda r: r["smiles"])
    elif args.sort_by == "entropy":
        rows.sort(key=lambda r: float(r["entropy_bits"]))
    elif args.sort_by == "top_slot":
        rows.sort(key=lambda r: int(r["top_slot"]))
    elif args.sort_by == "top_minus_uniform":
        rows.sort(key=lambda r: float(r["top_minus_uniform"]), reverse=True)

    if args.limit > 0:
        rows = rows[: args.limit]

    fieldnames = (
        ["smiles"]
        + [f"w{k + 1}" for k in range(len(slots))]
        + ["top_slot", "entropy_bits", "top_minus_uniform", "source"]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        wcsv = csv.DictWriter(f, fieldnames=fieldnames)
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow(r)

    print(f"wrote:   {out_path}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
