#!/usr/bin/env python3
"""Profile the QDF preprocessing pipeline.

Re-runs the per-molecule loop that ``train/preprocess.py`` performs, but
measures each step (sphere/field, distance matrices, Gaussian potential,
disk ``np.save``) separately so we know where the time goes before we
decide what to rewrite in Rust.

Usage (from repo root):

    .\.venv\Scripts\python.exe QuantumDeepField_molecule\bench\profile_preprocess.py \
        --dataset QM9under14atoms_atomizationenergy_eV --split train --limit 500

Rust backend (requires ``qdf_io`` built via maturin):

    .\.venv\Scripts\python.exe QuantumDeepField_molecule\bench\profile_preprocess.py \
        --dataset QM9under14atoms_atomizationenergy_eV --split train --limit 500 \
        --backend rust --rust-batch-size 64

Rust **legacy** (pre-fused atom distance matrix path; same outputs, for A/B timing):

    .\.venv\Scripts\python.exe QuantumDeepField_molecule\bench\profile_preprocess.py \
        --dataset QM9under14atoms_atomizationenergy_eV --split train --limit 500 \
        --backend rust-legacy --rust-batch-size 64
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Make ``train`` importable as a package without touching its source.
REPO_ROOT = Path(__file__).resolve().parents[2]
QDF_ROOT = REPO_ROOT / "QuantumDeepField_molecule"
sys.path.insert(0, str(QDF_ROOT / "train"))

import preprocess as pp  # noqa: E402  (train/preprocess.py)


def load_dataset_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    return text.split("\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="QM9under14atoms_atomizationenergy_eV")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--basis-set", default="6-31G")
    parser.add_argument("--radius", type=float, default=0.75)
    parser.add_argument("--grid-interval", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=500,
                        help="Max molecules to process (use a small number for quick profiling).")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip np.save to isolate compute cost.")
    parser.add_argument("--cprofile", action="store_true",
                        help="Also dump a cProfile top-25 report.")
    parser.add_argument(
        "--backend",
        choices=["numpy", "rust", "rust-legacy"],
        default="numpy",
        help="Which geometry backend to benchmark. 'numpy' mirrors train/preprocess.py; "
             "'rust' uses qdf_io.preprocess_batch_rust; 'rust-legacy' uses "
             "preprocess_batch_rust_legacy (materialized atom–field distance matrix).",
    )
    parser.add_argument(
        "--rust-batch-size",
        type=int,
        default=64,
        help="Molecule batch size for --backend rust / rust-legacy.",
    )
    args = parser.parse_args()

    dataset_dir = QDF_ROOT / "dataset" / args.dataset
    src_path = dataset_dir / f"{args.split}.txt"
    if not src_path.exists():
        sys.exit(f"Missing dataset file: {src_path}")

    blocks = load_dataset_blocks(src_path)
    if args.limit:
        blocks = blocks[: args.limit]
    n_molecules = len(blocks)
    print(f"Dataset: {args.dataset}/{args.split}.txt  -> {n_molecules} molecules profiled")

    has_property = args.split in {"train", "val", "test"} and args.dataset.endswith("_eV")
    orbital_dict = defaultdict(lambda: len(orbital_dict))

    timings = defaultdict(float)
    timings_count = defaultdict(int)

    # One-shot sphere build (excluded from per-molecule timing since preprocess does it once).
    t0 = time.perf_counter()
    sphere = pp.create_sphere(args.radius, args.grid_interval)
    timings["sphere_once"] += time.perf_counter() - t0
    sphere64 = np.ascontiguousarray(sphere, dtype=np.float64)

    save_dir_ctx = tempfile.TemporaryDirectory(prefix="qdf_bench_")
    save_dir = Path(save_dir_ctx.name)
    print(f"Temp save dir: {save_dir}")

    rust_batch_fn = None
    if args.backend in ("rust", "rust-legacy"):
        try:
            import qdf_io  # noqa: WPS433
        except Exception as e:  # pragma: no cover
            sys.exit(
                "backend=rust or rust-legacy requires the ``qdf_io`` native extension. "
                "Build it from QuantumDeepField_molecule/qdf_io with:\n"
                "  maturin develop --release\n"
                f"Original import error: {e}"
            )
        if args.rust_batch_size < 1:
            sys.exit("--rust-batch-size must be >= 1")
        rust_batch_fn = (
            qdf_io.preprocess_batch_rust_legacy
            if args.backend == "rust-legacy"
            else qdf_io.preprocess_batch_rust
        )
        if args.backend == "rust-legacy" and rust_batch_fn is None:
            sys.exit(
                "rust-legacy requires ``preprocess_batch_rust_legacy`` in the qdf_io native module "
                "(your ``_native*.pyd`` is probably stale).\n\n"
                "Rebuild using the **same** Python executable that runs this script, from\n"
                "  QuantumDeepField_molecule/qdf_io\n"
                f"  {sys.executable} -m maturin develop --release\n\n"
                "If maturin picks a different venv, set PYO3_PYTHON to the interpreter above, or run\n"
                "  cargo build --release\n"
                "and rely on ``qdf_io`` loading ``target/release/qdf_io.dll`` (see package __init__). "
                "Close Jupyter while rebuilding if Windows locks the .pyd."
            )

    def run_loop():
        rust_buf: list[dict] = []

        def flush_rust_chunk(buf: list[dict]) -> None:
            t = time.perf_counter()
            ac_list = [np.ascontiguousarray(m["atomic_coords"]) for m in buf]
            oc_list = [np.ascontiguousarray(m["orbital_coords"]) for m in buf]
            an_list = [np.ascontiguousarray(m["atomic_numbers"]) for m in buf]
            timings["rust_pack_inputs"] += time.perf_counter() - t

            t = time.perf_counter()
            outs = rust_batch_fn(ac_list, oc_list, an_list, sphere64)
            timings["rust_preprocess_batch"] += time.perf_counter() - t
            timings_count["rust_preprocess_batch"] += 1

            for mol, (dm_orb, pot, n_field) in zip(buf, outs, strict=True):
                idx = mol["idx"]
                atomic_orbitals = mol["atomic_orbitals"]
                quantum_numbers = mol["quantum_numbers"]
                n_electrons = mol["N_electrons"]
                property_values = mol["property_values"]

                t = time.perf_counter()
                data = [
                    idx,
                    atomic_orbitals,
                    np.asarray(dm_orb, dtype=np.float32),
                    quantum_numbers.astype(np.float32),
                    n_electrons.astype(np.float32),
                    int(n_field),
                ]
                if has_property and property_values is not None:
                    data += [property_values.astype(np.float32), np.asarray(pot, dtype=np.float32)]
                data = np.array(data, dtype=object)
                timings["assemble"] += time.perf_counter() - t

                if not args.no_save:
                    t = time.perf_counter()
                    np.save(save_dir / f"{idx}.npy", data)
                    timings["np_save"] += time.perf_counter() - t

        for block in blocks:
            t = time.perf_counter()
            mol = pp._parse_molecule_block(
                block, args.basis_set, orbital_dict, property=has_property
            )
            timings["parse_molecule"] += time.perf_counter() - t
            timings_count["parse_molecule"] += 1

            if args.backend == "numpy":
                idx = mol["idx"]
                atomic_coords = mol["atomic_coords"]
                orbital_coords = mol["orbital_coords"]
                atomic_numbers = mol["atomic_numbers"]
                atomic_orbitals = mol["atomic_orbitals"]
                quantum_numbers = mol["quantum_numbers"]
                n_electrons = mol["N_electrons"]
                property_values = mol["property_values"]

                t = time.perf_counter()
                field_coords = pp.create_field(sphere, atomic_coords)
                timings["create_field"] += time.perf_counter() - t

                t = time.perf_counter()
                distance_matrix_atoms = pp.create_distancematrix(field_coords, atomic_coords)
                timings["distmat_atoms"] += time.perf_counter() - t

                t = time.perf_counter()
                potential = pp.create_potential(distance_matrix_atoms, atomic_numbers)
                timings["potential"] += time.perf_counter() - t

                t = time.perf_counter()
                distance_matrix_orbs = pp.create_distancematrix(field_coords, orbital_coords)
                timings["distmat_orbitals"] += time.perf_counter() - t

                t = time.perf_counter()
                n_field = len(field_coords)
                data = [
                    idx,
                    atomic_orbitals,
                    distance_matrix_orbs.astype(np.float32),
                    quantum_numbers.astype(np.float32),
                    n_electrons.astype(np.float32),
                    n_field,
                ]
                if has_property and property_values is not None:
                    data += [property_values.astype(np.float32), potential.astype(np.float32)]
                data = np.array(data, dtype=object)
                timings["assemble"] += time.perf_counter() - t

                if not args.no_save:
                    t = time.perf_counter()
                    np.save(save_dir / f"{idx}.npy", data)
                    timings["np_save"] += time.perf_counter() - t
                continue

            rust_buf.append(mol)
            if len(rust_buf) >= args.rust_batch_size:
                flush_rust_chunk(rust_buf)
                rust_buf.clear()

        if args.backend in ("rust", "rust-legacy") and rust_buf:
            flush_rust_chunk(rust_buf)

    if args.cprofile:
        prof = cProfile.Profile()
        prof.enable()

    total_t0 = time.perf_counter()
    run_loop()
    total_wall = time.perf_counter() - total_t0

    if args.cprofile:
        prof.disable()
        print("\n--- cProfile (top 25 by cumulative) ---")
        pstats.Stats(prof).strip_dirs().sort_stats("cumulative").print_stats(25)

    print("\n--- Per-step totals (seconds) ---")
    sphere_once = timings.pop("sphere_once")
    print(f"sphere_once           : {sphere_once*1000:8.3f} ms (once per dataset)")
    print(f"backend               : {args.backend}")
    if args.backend in ("rust", "rust-legacy"):
        print(f"rust_batch_size       : {args.rust_batch_size}")
    rows = sorted(timings.items(), key=lambda kv: -kv[1])
    sum_steps = sum(timings.values())
    for name, secs in rows:
        share = 100.0 * secs / total_wall
        per_mol_ms = 1000.0 * secs / n_molecules
        print(f"{name:22s}: {secs:8.3f} s  ({share:5.1f}%)  {per_mol_ms:7.3f} ms/molecule")
    print("-" * 60)
    print(f"sum of steps          : {sum_steps:8.3f} s")
    print(f"wall time             : {total_wall:8.3f} s  ({1000*total_wall/n_molecules:.3f} ms/molecule)")
    print(f"molecules             : {n_molecules}")
    if not args.no_save:
        print(f"files written         : {n_molecules}  -> {save_dir}")

    print(
        "\nExtrapolation to QM9full (~130k molecules) at this rate:"
        f" {total_wall * 130_000 / n_molecules / 60:.1f} minutes (single thread)"
    )

    # Temp directory cleans itself up via the context manager; no separate
    # shutil.rmtree call is needed and we never touch the real dataset dir.
    save_dir_ctx.cleanup()


if __name__ == "__main__":
    main()
