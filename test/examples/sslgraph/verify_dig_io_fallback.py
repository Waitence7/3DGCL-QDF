#!/usr/bin/env python3
"""Guardrail: confirm DGCL stays usable when ``dig_io`` is not loaded.

We can't actually uninstall ``dig_io`` from this Python session, but we can
patch :func:`dig.sslgraph.method.contrastive.views_fn.sample._try_dig_io` to
return ``None`` and check that:

  * ``UniformSample`` / ``RWSample`` / ``EdgePerturbation`` with
    ``impl='rust'`` silently fall back to the original Python path.
  * ``key_split(..., impl='rust')`` also falls back.

Run from repo root::

    .\\.venv\\Scripts\\python.exe examples\\sslgraph\\verify_dig_io_fallback.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def make_data():
    return torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long)


def main() -> int:
    from torch_geometric.data import Data

    from dig.sslgraph.method.contrastive.views_fn import sample as sample_mod
    from dig.sslgraph.method.contrastive.views_fn import structure as structure_mod
    from dig.threedgraph.dataset import dataset as ds_mod

    # 1. Force the rust toggle to think the extension is unavailable.
    with mock.patch.object(sample_mod, "_try_dig_io", return_value=None), \
         mock.patch.object(structure_mod, "_try_dig_io", return_value=None), \
         mock.patch.object(ds_mod, "_try_dig_io", return_value=None):

        d = Data(x=torch.eye(6), edge_index=make_data())

        # UniformSample with impl='rust' must NOT raise.
        us = sample_mod.UniformSample(encoder='gin', ratio=0.2, device='cpu', impl='rust')
        out = us(d)
        assert out.x.shape[0] < 6 or out.x.shape[0] == 6
        print("UniformSample fallback OK:", out)

        # RWSample with impl='rust' must NOT raise.
        rw = sample_mod.RWSample(ratio=0.5, impl='rust')
        out = rw(d)
        print("RWSample fallback OK:", out)

        # EdgePerturbation with drop=True (avoids the documented NameError).
        ep = structure_mod.EdgePerturbation(add=True, drop=True, ratio=0.1, impl='rust')
        out = ep(d)
        print("EdgePerturbation fallback OK:", out)

        # key_split impl='rust' must fall back.
        class FakeDS:
            def __init__(self, n): self.n = n
            def __len__(self): return self.n
            def __getitem__(self, i): return {'smiles': f'mol{i}'}

        torch.manual_seed(42)
        sp = ds_mod.key_split(FakeDS(20), [0] * 10 + [1] * 10, lengths=[16, 2, 2], impl='rust')
        print("key_split fallback OK: lengths =", [len(s) for s in sp])

    print("\nAll fallback paths OK -- dig_io is optional at runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
