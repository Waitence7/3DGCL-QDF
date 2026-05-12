#!/usr/bin/env python3
"""Profile the QDF preprocessing pipeline.

Re-runs the per-molecule loop that ``train/preprocess.py`` performs, but
measures each step (sphere/field, distance matrices, Gaussian potential,
disk ``np.save``) separately so we know where the time goes before we
decide what to rewrite in Rust.

Usage (from repo root):

    .\.venv\Scripts\python.exe QuantumDeepField_molecule\bench\profile_preprocess.py \
        --dataset QM9under14atoms_atomizationenergy_eV --split train --limit 500
"""

from __future__ import annotations

import argparse
import cProfile
import os
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


def parse_molecule(block: str, with_property: bool) -> tuple[str, list[str], np.ndarray | None]:
    lines = block.strip().split("\n")
    idx = lines[0]
    if with_property:
        atom_xyzs = lines[1:-1]
        prop = np.array([[float(p) for p in lines[-1].split()]], dtype=np.float32)
    else:
        atom_xyzs = lines[1:]
        prop = None
    return idx, atom_xyzs, prop


def build_orbitals(atom_xyzs: list[str], inner: int, outer: int):
    atoms: list[str] = []
    atomic_numbers: list[list[int]] = []
    atomic_coords: list[list[float]] = []
    atomic_orbital_names: list[str] = []
    orbital_coords: list[list[float]] = []
    quantum_numbers: list[int] = []
    n_electrons = 0
    for atom_xyz in atom_xyzs:
        atom, x, y, z = atom_xyz.split()
        atoms.append(atom)
        atomic_number = pp.atomicnumber_dict[atom]
        atomic_numbers.append([atomic_number])
        n_electrons += atomic_number
        xyz = [float(v) for v in (x, y, z)]
        atomic_coords.append(xyz)
        if atomic_number <= 2:
            aqs = [(atom + "1s" + str(i), 1) for i in range(outer)]
        else:
            aqs = (
                [(atom + "1s" + str(i), 1) for i in range(inner)]
                + [(atom + "2s" + str(i), 2) for i in range(outer)]
                + [(atom + "2p" + str(i), 2) for i in range(outer)]
            )
        for name, q in aqs:
            atomic_orbital_names.append(name)
            orbital_coords.append(xyz)
            quantum_numbers.append(q)
    return (
        atoms,
        np.array(atomic_numbers),
        np.array(atomic_coords),
        atomic_orbital_names,
        orbital_coords,
        quantum_numbers,
        n_electrons,
    )


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

    inner_outer = [int(b) for b in args.basis_set[:-1].replace("-", "")]
    inner, outer = inner_outer[0], sum(inner_outer[1:])

    has_property = args.split in {"train", "val", "test"} and args.dataset.endswith("_eV")
    orbital_dict = defaultdict(lambda: len(orbital_dict))

    timings = defaultdict(float)
    timings_count = defaultdict(int)

    # One-shot sphere build (excluded from per-molecule timing since preprocess does it once).
    t0 = time.perf_counter()
    sphere = pp.create_sphere(args.radius, args.grid_interval)
    timings["sphere_once"] += time.perf_counter() - t0

    save_dir = Path(tempfile.mkdtemp(prefix="qdf_bench_"))
    print(f"Temp save dir: {save_dir}")

    def run_loop():
        for block in blocks:
            idx, atom_xyzs, prop = parse_molecule(block, has_property)

            t = time.perf_counter()
            (
                _atoms,
                atomic_numbers,
                atomic_coords,
                atomic_orbital_names,
                orbital_coords,
                quantum_numbers,
                n_electrons,
            ) = build_orbitals(atom_xyzs, inner, outer)
            timings["parse_and_orbitals"] += time.perf_counter() - t
            timings_count["parse_and_orbitals"] += 1

            t = time.perf_counter()
            atomic_orbitals = pp.create_orbitals(atomic_orbital_names, orbital_dict)
            timings["create_orbitals"] += time.perf_counter() - t

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
            quantum_numbers_arr = np.array([quantum_numbers], dtype=np.float32)
            n_electrons_arr = np.array([[n_electrons]], dtype=np.float32)
            n_field = len(field_coords)
            data = [
                idx,
                atomic_orbitals.astype(np.int64),
                distance_matrix_orbs.astype(np.float32),
                quantum_numbers_arr,
                n_electrons_arr,
                n_field,
            ]
            if has_property and prop is not None:
                data += [prop, potential.astype(np.float32)]
            data = np.array(data, dtype=object)
            timings["assemble"] += time.perf_counter() - t

            if not args.no_save:
                t = time.perf_counter()
                np.save(save_dir / f"{idx}.npy", data)
                timings["np_save"] += time.perf_counter() - t

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

    if not args.no_save:
        import shutil
        try:
            shutil.rmtree(save_dir)
        except OSError:
            pass


if __name__ == "__main__":
    main()
