#!/usr/bin/env python3

import argparse
from collections import defaultdict
import os
import pickle
from pathlib import Path

import numpy as np

from scipy import spatial


"""Dictionary of atomic numbers."""
all_atoms = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
             'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
             'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
             'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
             'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
             'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
             'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
             'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
             'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
             'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
             'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
             'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
atomicnumber_dict = dict(zip(all_atoms, range(1, len(all_atoms)+1)))


def create_sphere(radius, grid_interval):
    """Create the sphere to be placed on each atom of a molecule."""
    xyz = np.arange(-radius, radius+1e-3, grid_interval)
    sphere = [[x, y, z] for x in xyz for y in xyz for z in xyz
              if (x**2 + y**2 + z**2 <= radius**2) and [x, y, z] != [0, 0, 0]]
    return np.array(sphere)


def create_field(sphere, coords):
    """Create the grid field of a molecule."""
    field = [f for coord in coords for f in sphere+coord]
    return np.array(field)


def create_orbitals(orbitals, orbital_dict):
    """Transform the atomic orbital types (e.g., H1s, C1s, N2s, and O2p)
    into the indices (e.g., H1s=0, C1s=1, N2s=2, and O2p=3) using orbital_dict.
    """
    orbitals = [orbital_dict[o] for o in orbitals]
    return np.array(orbitals)   


def create_distancematrix(coords1, coords2):
    """Create the distance matrix from coords1 and coords2,
    where coords = [[x_1, y_1, z_1], [x_2, y_2, z_2], ...].
    For example, when coords1 is field_coords and coords2 is atomic_coords
    of a molecule, each element of the matrix is the distance
    between a field point and an atomic position in the molecule.
    Note that we transform all 0 elements in the distance matrix
    into a large value (e.g., 1e6) because we use the Gaussian:
    exp(-d^2), where d is the distance, and exp(-1e6^2) becomes 0.
    """
    distance_matrix = spatial.distance_matrix(coords1, coords2)
    return np.where(distance_matrix == 0.0, 1e6, distance_matrix)


def create_potential(distance_matrix, atomic_numbers):
    """Create the Gaussian external potential used in Brockherde et al., 2017,
    Bypassing the Kohn-Sham equations with machine learning.
    """
    Gaussians = np.exp(-distance_matrix**2)
    return -np.matmul(Gaussians, atomic_numbers)


def _parse_molecule_block(block, basis_set, orbital_dict, property=True):
    """Parse one molecule block from ``train.txt`` / ``val.txt`` / ``test.txt``.

    Returns a dict with everything needed to build the saved ``.npy`` record.
    The heavy geometry math is *not* performed here; callers choose either the
    original NumPy/SciPy path or the Rust+Rayon path.

    When ``qdf_io`` exposes ``parse_molecule_block_rust`` and the environment
    variable ``QDF_FORCE_PYTHON_PARSE`` is not set to ``1``/``true``, parsing
    is delegated to Rust (AO labels are still mapped through ``orbital_dict``
    in Python so indices stay dataset-consistent).
    """
    force_py = os.environ.get("QDF_FORCE_PYTHON_PARSE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not force_py:
        try:
            import qdf_io  # noqa: WPS433

            rust_parse = getattr(qdf_io, "parse_molecule_block_rust", None)
            if rust_parse is not None:
                mol = dict(rust_parse(block, basis_set, property))
                labels = mol.pop("atomic_orbital_labels")
                mol["atomic_orbitals"] = create_orbitals(
                    list(labels), orbital_dict
                ).astype(np.int64)
                return mol
        except Exception:
            pass

    inner_outer = [int(b) for b in basis_set[:-1].replace('-', '')]
    inner, outer = inner_outer[0], sum(inner_outer[1:])

    lines = block.strip().split('\n')
    idx = lines[0]

    if property:
        atom_xyzs = lines[1:-1]
        property_values = lines[-1].strip().split()
        property_values = np.array([[float(p) for p in property_values]], dtype=np.float32)
    else:
        atom_xyzs = lines[1:]
        property_values = None

    atomic_numbers = []
    N_electrons = 0
    atomic_coords = []
    atomic_orbitals = []
    orbital_coords = []
    quantum_numbers = []

    for atom_xyz in atom_xyzs:
        atom, x, y, z = atom_xyz.split()
        atomic_number = atomicnumber_dict[atom]
        atomic_numbers.append([atomic_number])
        N_electrons += atomic_number
        xyz = [float(v) for v in [x, y, z]]
        atomic_coords.append(xyz)

        if atomic_number <= 2:
            aqs = [(atom + '1s' + str(i), 1) for i in range(outer)]
        elif atomic_number >= 3:
            aqs = ([(atom + '1s' + str(i), 1) for i in range(inner)] +
                   [(atom + '2s' + str(i), 2) for i in range(outer)] +
                   [(atom + '2p' + str(i), 2) for i in range(outer)])
        for a, q in aqs:
            atomic_orbitals.append(a)
            orbital_coords.append(xyz)
            quantum_numbers.append(q)

    atomic_coords = np.asarray(atomic_coords, dtype=np.float64)
    orbital_coords = np.asarray(orbital_coords, dtype=np.float64)
    atomic_numbers = np.asarray(atomic_numbers, dtype=np.int64)

    atomic_orbitals = create_orbitals(atomic_orbitals, orbital_dict)
    quantum_numbers = np.array([quantum_numbers], dtype=np.float32)
    N_electrons = np.array([[N_electrons]], dtype=np.float32)

    return {
        'idx': idx,
        'atomic_coords': atomic_coords,
        'orbital_coords': orbital_coords,
        'atomic_numbers': atomic_numbers,
        'atomic_orbitals': atomic_orbitals.astype(np.int64),
        'quantum_numbers': quantum_numbers,
        'N_electrons': N_electrons,
        'property_values': property_values,
    }


def _append_shard_record(
    writer,
    idx,
    atomic_orbitals,
    dm_orb,
    quantum_numbers,
    n_electrons,
    n_field_int: int,
    property_values,
    potential,
    *,
    has_property: bool,
) -> None:
    """Pack one molecule into ``ShardWriter`` (same layout as ``np.save``)."""
    ao = np.ascontiguousarray(atomic_orbitals, dtype=np.int64)
    dm = np.ascontiguousarray(dm_orb, dtype=np.float32)
    qn = np.ascontiguousarray(quantum_numbers, dtype=np.float32)
    if qn.ndim == 1:
        qn = qn.reshape(1, -1)
    ne = np.ascontiguousarray(n_electrons, dtype=np.float32)
    if has_property and property_values is not None:
        pv = np.ascontiguousarray(property_values, dtype=np.float32)
        pot = np.ascontiguousarray(potential, dtype=np.float32)
        if pot.ndim == 1:
            pot = pot.reshape(int(n_field_int), 1)
        writer.append_molecule(
            str(idx), ao, dm, qn, ne, int(n_field_int), pv, pot,
        )
    else:
        writer.append_molecule(
            str(idx), ao, dm, qn, ne, int(n_field_int), None, None,
        )


def create_dataset(dir_dataset, filename, basis_set,
                   radius, grid_interval, orbital_dict, property=True,
                   backend='numpy', rust_batch_size=64,
                   output_format='npy'):

    """Directory of a preprocessed dataset."""
    if property:
        dir_preprocess = (dir_dataset + filename + '_' + basis_set + '_' +
                          str(radius) + 'sphere_' +
                          str(grid_interval) + 'grid/')
    else:  # For demo.
        dir_preprocess = filename + '/'

    if output_format not in ('npy', 'shard', 'both'):
        raise ValueError(f"Unknown output_format: {output_format!r}")

    write_npy = output_format in ('npy', 'both')
    write_shard = output_format in ('shard', 'both')

    if write_npy:
        os.makedirs(dir_preprocess, exist_ok=True)

    """Load a dataset."""
    with open(dir_dataset + filename + '.txt', 'r') as f:
        dataset = f.read().strip().split('\n\n')

    N = len(dataset)
    percent = 10

    shard_writer = None
    if write_shard:
        try:
            import qdf_io  # noqa: WPS433
            from dataset_shard import default_shard_path
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "output_format includes 'shard' which requires ``qdf_io`` and "
                "``dataset_shard``. Build qdf_io with maturin."
            ) from e
        if N == 0:
            raise ValueError("Cannot write shard: empty split.")
        n_out = 0
        if property:
            first_mol = _parse_molecule_block(
                dataset[0], basis_set, orbital_dict, property=True,
            )
            if first_mol['property_values'] is not None:
                n_out = int(first_mol['property_values'].shape[1])
        shard_path = default_shard_path(Path(dir_preprocess.rstrip('/\\')))
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        shard_writer = qdf_io.ShardWriter(str(shard_path), N, bool(property), int(n_out))

    """A sphere for creating the grid field of a molecule."""
    sphere = create_sphere(radius, grid_interval)
    sphere64 = np.ascontiguousarray(sphere, dtype=np.float64)

    if backend not in ('numpy', 'rust'):
        raise ValueError(f"Unknown backend: {backend!r} (expected 'numpy' or 'rust')")

    if backend == 'rust':
        try:
            import qdf_io  # noqa: WPS433 (runtime optional dependency)
        except Exception as e:  # pragma: no cover - user environment specific
            raise RuntimeError(
                "backend='rust' requires the ``qdf_io`` native extension. "
                "Build it from ``QuantumDeepField_molecule/qdf_io`` with:\n"
                "  maturin develop --release\n"
                "If Windows locks ``_native*.pyd``, run ``cargo build --release`` in that folder; "
                "``qdf_io`` can load ``target/release/qdf_io.dll`` when present.\n"
            ) from e
    else:
        qdf_io = None  # type: ignore

    if backend == 'rust' and rust_batch_size < 1:
        raise ValueError('rust_batch_size must be >= 1')

    def flush_rust_batch(buf: list[dict]) -> None:
        assert qdf_io is not None
        ac_list = [np.ascontiguousarray(m['atomic_coords']) for m in buf]
        oc_list = [np.ascontiguousarray(m['orbital_coords']) for m in buf]
        an_list = [np.ascontiguousarray(m['atomic_numbers']) for m in buf]

        outs = qdf_io.preprocess_batch_rust(ac_list, oc_list, an_list, sphere64)

        for mol, (dm_orb, pot, n_field) in zip(buf, outs, strict=True):
            idx = mol['idx']
            atomic_orbitals = mol['atomic_orbitals']
            quantum_numbers = mol['quantum_numbers']
            N_electrons = mol['N_electrons']
            property_values = mol['property_values']

            if write_npy:
                data = [idx,
                        atomic_orbitals,
                        np.asarray(dm_orb, dtype=np.float32),
                        quantum_numbers.astype(np.float32),
                        N_electrons.astype(np.float32),
                        int(n_field)]

                if property and property_values is not None:
                    data += [property_values.astype(np.float32),
                             np.asarray(pot, dtype=np.float32)]

                data = np.array(data, dtype=object)
                np.save(dir_preprocess + idx, data)

            if write_shard:
                assert shard_writer is not None
                pot_arr = np.asarray(pot, dtype=np.float32)
                _append_shard_record(
                    shard_writer, idx, atomic_orbitals,
                    np.asarray(dm_orb, dtype=np.float32),
                    quantum_numbers.astype(np.float32),
                    N_electrons.astype(np.float32),
                    int(n_field),
                    property_values,
                    pot_arr,
                    has_property=bool(property),
                )

    rust_buf: list[dict] = []

    for n, block in enumerate(dataset):
        if N and 100 * n / N >= percent:
            print(str(percent) + '％ has finished.')
            percent += 40

        mol = _parse_molecule_block(block, basis_set, orbital_dict, property=property)

        if backend == 'numpy':
            idx = mol['idx']
            atomic_coords = mol['atomic_coords']
            orbital_coords = mol['orbital_coords']
            atomic_numbers = mol['atomic_numbers']
            atomic_orbitals = mol['atomic_orbitals']
            quantum_numbers = mol['quantum_numbers']
            N_electrons = mol['N_electrons']
            property_values = mol['property_values']

            field_coords = create_field(sphere, atomic_coords)
            distance_matrix = create_distancematrix(field_coords, atomic_coords)
            potential = create_potential(distance_matrix, atomic_numbers)
            distance_matrix = create_distancematrix(field_coords, orbital_coords)
            N_field = len(field_coords)

            data = [idx,
                    atomic_orbitals,
                    distance_matrix.astype(np.float32),
                    quantum_numbers.astype(np.float32),
                    N_electrons.astype(np.float32),
                    N_field]

            if property and property_values is not None:
                data += [property_values.astype(np.float32),
                         potential.astype(np.float32)]

            if write_npy:
                data = np.array(data, dtype=object)
                np.save(dir_preprocess + idx, data)
            if write_shard:
                assert shard_writer is not None
                _append_shard_record(
                    shard_writer, idx, atomic_orbitals,
                    distance_matrix.astype(np.float32),
                    quantum_numbers.astype(np.float32),
                    N_electrons.astype(np.float32),
                    int(N_field),
                    property_values,
                    potential.astype(np.float32),
                    has_property=bool(property),
                )
            continue

        # backend == 'rust'
        assert qdf_io is not None
        rust_buf.append(mol)
        if len(rust_buf) >= rust_batch_size:
            flush_rust_batch(rust_buf)
            rust_buf.clear()

    if backend == 'rust' and rust_buf:
        assert qdf_io is not None
        flush_rust_batch(rust_buf)

    if shard_writer is not None:
        shard_writer.finalize()


if __name__ == "__main__":

    """Args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('basis_set')
    parser.add_argument('radius', type=float)
    parser.add_argument('grid_interval', type=float)
    parser.add_argument(
        '--backend',
        choices=['numpy', 'rust'],
        default='numpy',
        help="Geometry backend for the heavy per-molecule work. "
             "'numpy' keeps the original SciPy distance_matrix implementation; "
             "'rust' uses the qdf_io Rayon parallel kernels (build with maturin). "
             "Both paths write identical .npy layouts when numerically aligned.",
    )
    parser.add_argument(
        '--rust-batch-size',
        type=int,
        default=64,
        help="When --backend rust, how many molecules to process per Rayon batch.",
    )
    parser.add_argument(
        '--output-format',
        choices=['npy', 'shard', 'both'],
        default='npy',
        help="Where to store preprocessed molecules. 'npy' (default) writes one "
             ".npy per molecule under train_<field>/; 'shard' writes a single "
             "train_<field>_shard.bin next to that directory (no per-molecule files); "
             "'both' writes both.",
    )
    args = parser.parse_args()
    dataset = args.dataset
    basis_set = args.basis_set
    radius = args.radius
    grid_interval = args.grid_interval
    backend = args.backend
    rust_batch_size = args.rust_batch_size
    output_format = args.output_format

    """Dataset directory (absolute path relative to project root)."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dir_dataset = os.path.join(project_root, 'dataset', dataset) + os.sep

    """Initialize orbital_dict, in which
    each key is an orbital type and each value is its index.
    """
    orbital_dict = defaultdict(lambda: len(orbital_dict))

    print('Preprocess', dataset, 'dataset.\n'
          'The preprocessed dataset is saved in', dir_dataset, 'directory.\n'
          'If the dataset size is large, '
          'it takes a long time and consume storage.\n'
          'Wait for a while...')
    rust_info = f"(rust_batch_size={rust_batch_size})" if backend == "rust" else ""
    print('Backend:', backend, rust_info, '  output:', output_format)
    print('-'*50)

    print('Training dataset...')
    create_dataset(dir_dataset, 'train',
                   basis_set, radius, grid_interval, orbital_dict,
                   backend=backend, rust_batch_size=rust_batch_size,
                   output_format=output_format)
    print('-'*50)

    print('Validation dataset...')
    create_dataset(dir_dataset, 'val',
                   basis_set, radius, grid_interval, orbital_dict,
                   backend=backend, rust_batch_size=rust_batch_size,
                   output_format=output_format)
    print('-'*50)

    print('Test dataset...')
    create_dataset(dir_dataset, 'test',
                   basis_set, radius, grid_interval, orbital_dict,
                   backend=backend, rust_batch_size=rust_batch_size,
                   output_format=output_format)
    print('-'*50)

    with open(dir_dataset + 'orbitaldict_' + basis_set + '.pickle', 'wb') as f:
        pickle.dump(dict(orbital_dict), f)

    print('The preprocess has finished.')
