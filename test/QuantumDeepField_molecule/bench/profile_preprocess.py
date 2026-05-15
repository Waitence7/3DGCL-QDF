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

Side-by-side wall time (numpy vs rust; ``--compare`` skips ``np.save`` by default
so geometry dominates — use ``--compare-with-save`` to include disk writes):

    .venv/bin/python QuantumDeepField_molecule/bench/profile_preprocess.py \
        --dataset QM9full_homolumo_eV --split train --limit 2000 --compare \
        --rust-batch-size 64
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


def _print_step_table(
    timings: dict[str, float],
    *,
    backend: str,
    rust_batch_size: int,
    total_wall: float,
    n_molecules: int,
    no_save: bool,
    save_dir: Path | None,
) -> None:
    tcopy = dict(timings)
    sphere_once = tcopy.pop("sphere_once", 0.0)
    print(f"sphere_once           : {sphere_once*1000:8.3f} ms (once per dataset)")
    print(f"backend               : {backend}")
    if backend == "rust":
        print(f"rust_batch_size       : {rust_batch_size}")
    rows = sorted(tcopy.items(), key=lambda kv: -kv[1])
    sum_steps = sum(tcopy.values())
    for name, secs in rows:
        share = 100.0 * secs / total_wall if total_wall else 0.0
        per_mol_ms = 1000.0 * secs / n_molecules if n_molecules else 0.0
        print(f"{name:22s}: {secs:8.3f} s  ({share:5.1f}%)  {per_mol_ms:7.3f} ms/molecule")
    print("-" * 60)
    print(f"sum of steps          : {sum_steps:8.3f} s")
    print(f"wall time             : {total_wall:8.3f} s  ({1000*total_wall/n_molecules:.3f} ms/molecule)")
    print(f"molecules             : {n_molecules}")
    if not no_save and save_dir is not None:
        print(f"files written         : {n_molecules}  -> {save_dir}")
    print(
        "\nExtrapolation to QM9full (~130k molecules) at this rate:"
        f" {total_wall * 130_000 / n_molecules / 60:.1f} minutes (single thread)"
    )


def time_preprocess_pipeline(
    blocks: list[str],
    *,
    backend: str,
    rust_batch_size: int,
    basis_set: str,
    radius: float,
    grid_interval: float,
    has_property: bool,
    no_save: bool,
) -> tuple[float, dict[str, float], int, Path | None]:
    """Run the preprocess loop once; return (wall_s, timings, n_mol, save_dir or None)."""
    n_molecules = len(blocks)
    timings: defaultdict[str, float] = defaultdict(float)
    timings_count: defaultdict[str, int] = defaultdict(int)

    t0 = time.perf_counter()
    sphere = pp.create_sphere(radius, grid_interval)
    timings["sphere_once"] += time.perf_counter() - t0
    sphere64 = np.ascontiguousarray(sphere, dtype=np.float64)

    save_dir_ctx = tempfile.TemporaryDirectory(prefix="qdf_bench_")
    save_dir = Path(save_dir_ctx.name)
    print(f"Temp save dir: {save_dir}")

    if backend == "rust":
        try:
            import qdf_io  # noqa: WPS433
        except Exception as e:  # pragma: no cover
            save_dir_ctx.cleanup()
            raise RuntimeError(
                "backend=rust requires the qdf_io native extension. "
                "Build it from QuantumDeepField_molecule/qdf_io with:\n"
                "  maturin develop --release\n"
                f"Original import error: {e}"
            ) from e
        if rust_batch_size < 1:
            save_dir_ctx.cleanup()
            raise ValueError("--rust-batch-size must be >= 1")

    orbital_dict: defaultdict = defaultdict(lambda: len(orbital_dict))

    def run_loop() -> None:
        rust_buf: list[dict] = []

        def flush_rust_chunk(buf: list[dict]) -> None:
            t = time.perf_counter()
            ac_list = [np.ascontiguousarray(m["atomic_coords"]) for m in buf]
            oc_list = [np.ascontiguousarray(m["orbital_coords"]) for m in buf]
            an_list = [np.ascontiguousarray(m["atomic_numbers"]) for m in buf]
            timings["rust_pack_inputs"] += time.perf_counter() - t

            t = time.perf_counter()
            outs = qdf_io.preprocess_batch_rust(ac_list, oc_list, an_list, sphere64)
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

                if not no_save:
                    t = time.perf_counter()
                    np.save(save_dir / f"{idx}.npy", data)
                    timings["np_save"] += time.perf_counter() - t

        for block in blocks:
            t = time.perf_counter()
            mol = pp._parse_molecule_block(
                block, basis_set, orbital_dict, property=has_property
            )
            timings["parse_molecule"] += time.perf_counter() - t
            timings_count["parse_molecule"] += 1

            if backend == "numpy":
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

                if not no_save:
                    t = time.perf_counter()
                    np.save(save_dir / f"{idx}.npy", data)
                    timings["np_save"] += time.perf_counter() - t
                continue

            rust_buf.append(mol)
            if len(rust_buf) >= rust_batch_size:
                flush_rust_chunk(rust_buf)
                rust_buf.clear()

        if backend == "rust" and rust_buf:
            flush_rust_chunk(rust_buf)

    total_t0 = time.perf_counter()
    run_loop()
    total_wall = time.perf_counter() - total_t0

    save_dir_out: Path | None = save_dir if not no_save else None
    save_dir_ctx.cleanup()
    return total_wall, dict(timings), n_molecules, save_dir_out


def load_dataset_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    return text.split("\n\n")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Visible numpy vs rust gap: use many molecules with heavy fields, e.g.\n"
            "  --dataset QM9full_homolumo_eV --split train --limit 3000 --compare\n"
            "and tune --rust-batch-size (often 32–128). Under7 train (~100 mols) is too small."
        ),
    )
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
        choices=["numpy", "rust"],
        default="numpy",
        help="Which geometry backend to benchmark. 'numpy' mirrors train/preprocess.py; "
             "'rust' benchmarks qdf_io.preprocess_batch_rust in Rayon batches.",
    )
    parser.add_argument(
        "--rust-batch-size",
        type=int,
        default=64,
        help="Molecule batch size for --backend rust.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run numpy then rust on the same molecule blocks and print wall-time speedup. "
             "Implies skipping np.save unless --compare-with-save (so geometry dominates).",
    )
    parser.add_argument(
        "--compare-with-save",
        action="store_true",
        help="With --compare, write .npy for both runs (slower; disk I/O may hide the gap).",
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

    no_save_effective = args.no_save or (args.compare and not args.compare_with_save)
    do_cprofile = args.cprofile and not args.compare
    if args.cprofile and args.compare:
        print("Note: --cprofile is ignored when using --compare.", file=sys.stderr)

    if args.compare:
        print(
            f"compare mode: no_save={no_save_effective} "
            f"(use --no-save / --compare-with-save to override)\n"
            "Tip: use QM9full_homolumo_eV --limit 2000+ for a clear wall-time gap.\n"
        )

        print("--- numpy ---")
        try:
            wall_np, timings_np, n_molecules, save_dir_np = time_preprocess_pipeline(
                blocks,
                backend="numpy",
                rust_batch_size=args.rust_batch_size,
                basis_set=args.basis_set,
                radius=args.radius,
                grid_interval=args.grid_interval,
                has_property=has_property,
                no_save=no_save_effective,
            )
        except (RuntimeError, ValueError) as e:
            sys.exit(str(e))
        print("\n--- Per-step totals (seconds) ---")
        _print_step_table(
            timings_np,
            backend="numpy",
            rust_batch_size=args.rust_batch_size,
            total_wall=wall_np,
            n_molecules=n_molecules,
            no_save=no_save_effective,
            save_dir=save_dir_np,
        )

        print("\n--- rust ---")
        try:
            wall_rs, timings_rs, _, save_dir_rs = time_preprocess_pipeline(
                blocks,
                backend="rust",
                rust_batch_size=args.rust_batch_size,
                basis_set=args.basis_set,
                radius=args.radius,
                grid_interval=args.grid_interval,
                has_property=has_property,
                no_save=no_save_effective,
            )
        except (RuntimeError, ValueError) as e:
            sys.exit(str(e))
        print("\n--- Per-step totals (seconds) ---")
        _print_step_table(
            timings_rs,
            backend="rust",
            rust_batch_size=args.rust_batch_size,
            total_wall=wall_rs,
            n_molecules=n_molecules,
            no_save=no_save_effective,
            save_dir=save_dir_rs,
        )

        print("\n--- wall-time summary ---")
        print(f"numpy wall: {wall_np:.4f} s  ({1000 * wall_np / n_molecules:.3f} ms/molecule)")
        print(f"rust wall:  {wall_rs:.4f} s  ({1000 * wall_rs / n_molecules:.3f} ms/molecule)")
        if wall_rs > 0:
            print(f"speedup (numpy wall ÷ rust wall): {wall_np / wall_rs:.2f}x")
        return

    if args.backend == "rust" and args.rust_batch_size < 1:
        sys.exit("--rust-batch-size must be >= 1")

    if do_cprofile:
        prof = cProfile.Profile()
        prof.enable()

    try:
        wall, timings, n_molecules, save_dir_out = time_preprocess_pipeline(
            blocks,
            backend=args.backend,
            rust_batch_size=args.rust_batch_size,
            basis_set=args.basis_set,
            radius=args.radius,
            grid_interval=args.grid_interval,
            has_property=has_property,
            no_save=args.no_save,
        )
    except (RuntimeError, ValueError) as e:
        sys.exit(str(e))

    if do_cprofile:
        prof.disable()
        print("\n--- cProfile (top 25 by cumulative) ---")
        pstats.Stats(prof).strip_dirs().sort_stats("cumulative").print_stats(25)

    print("\n--- Per-step totals (seconds) ---")
    _print_step_table(
        timings,
        backend=args.backend,
        rust_batch_size=args.rust_batch_size,
        total_wall=wall,
        n_molecules=n_molecules,
        no_save=args.no_save,
        save_dir=save_dir_out,
    )


if __name__ == "__main__":
    main()
