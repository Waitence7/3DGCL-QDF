//! QDF binary shard reader.
//!
//! See `python/qdf_io/__init__.py` and `QuantumDeepField_molecule/train/dataset_shard.py`
//! for the Python-side counterpart. The on-disk layout is documented at the top of
//! that Python file.

use memmap2::Mmap;
use ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIOError, PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::time::Instant;

const MAGIC: &[u8; 8] = b"QDFSHRD\0";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 64;
const FLAG_HAS_PROPERTY: u32 = 1;

#[inline]
fn read_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

#[inline]
fn read_u64(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        buf[off], buf[off + 1], buf[off + 2], buf[off + 3],
        buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7],
    ])
}

#[inline]
fn read_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

#[inline]
fn align_up(x: usize, a: usize) -> usize {
    (x + a - 1) & !(a - 1)
}

/// Copy `n` little-endian f32 values starting at `bytes` into a fresh `Vec<f32>`.
///
/// On x86/x86_64 little-endian is native so this is just a `memcpy`. We do an
/// unaligned load (mmap pointer + offset may not be 4-aligned in general) via
/// `read_unaligned`, but our writer pads everything to 8-byte boundaries so
/// in practice the loads end up aligned.
#[inline]
fn copy_f32_slice(bytes: &[u8], n: usize) -> Vec<f32> {
    debug_assert_eq!(bytes.len(), n * 4);
    let mut out: Vec<f32> = Vec::with_capacity(n);
    unsafe {
        let src = bytes.as_ptr() as *const f32;
        let dst = out.as_mut_ptr();
        // SAFETY: out has capacity n; src points to n*4 bytes of valid mmapped data.
        // We use copy_nonoverlapping which is equivalent to memcpy and does not
        // require src alignment.
        std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, n * 4);
        out.set_len(n);
    }
    out
}

#[inline]
fn copy_i64_slice(bytes: &[u8], n: usize) -> Vec<i64> {
    debug_assert_eq!(bytes.len(), n * 8);
    let mut out: Vec<i64> = Vec::with_capacity(n);
    unsafe {
        let src = bytes.as_ptr();
        let dst = out.as_mut_ptr();
        std::ptr::copy_nonoverlapping(src, dst as *mut u8, n * 8);
        out.set_len(n);
    }
    out
}

/// A read-only handle over a memory-mapped QDF shard.
#[pyclass(module = "qdf_io._native")]
struct ShardReader {
    mmap: Mmap,
    n_molecules: usize,
    n_output: usize,
    has_property: bool,
    index_table_offset: usize,
    path: String,
}

#[pymethods]
impl ShardReader {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let file = File::open(path)
            .map_err(|e| PyIOError::new_err(format!("open '{}': {}", path, e)))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| PyIOError::new_err(format!("mmap '{}': {}", path, e)))?;

        if mmap.len() < HEADER_SIZE {
            return Err(PyValueError::new_err(format!(
                "Shard file '{}' too small for header ({} bytes)",
                path,
                mmap.len()
            )));
        }
        if &mmap[0..8] != MAGIC {
            return Err(PyValueError::new_err(format!(
                "Bad magic in '{}'; not a QDF shard",
                path
            )));
        }
        let version = read_u32(&mmap, 8);
        if version != VERSION {
            return Err(PyValueError::new_err(format!(
                "Unsupported shard version {} (expected {})",
                version, VERSION
            )));
        }
        let n_molecules = read_u64(&mmap, 12) as usize;
        let n_output = read_u32(&mmap, 20) as usize;
        let flags = read_u32(&mmap, 24);
        let has_property = (flags & FLAG_HAS_PROPERTY) != 0;
        let index_table_offset = read_u64(&mmap, 28) as usize;

        if index_table_offset + n_molecules * 8 > mmap.len() {
            return Err(PyValueError::new_err(
                "Shard index table extends past file end",
            ));
        }

        Ok(ShardReader {
            mmap,
            n_molecules,
            n_output,
            has_property,
            index_table_offset,
            path: path.to_string(),
        })
    }

    fn __len__(&self) -> usize {
        self.n_molecules
    }

    fn __repr__(&self) -> String {
        format!(
            "ShardReader(path='{}', n_molecules={}, n_output={}, has_property={})",
            self.path, self.n_molecules, self.n_output, self.has_property
        )
    }

    #[getter]
    fn n_molecules(&self) -> usize {
        self.n_molecules
    }

    #[getter]
    fn n_output(&self) -> usize {
        self.n_output
    }

    #[getter]
    fn has_property(&self) -> bool {
        self.has_property
    }

    /// Read one molecule record by index and return a tuple in the same order
    /// produced by ``train/train.py``'s ``MyDataset.__getitem__``:
    ///
    ///     (idx_str, atomic_orbitals[N_orb] int64,
    ///      distance_matrix[N_field, N_orb] float32,
    ///      quantum_numbers[1, N_orb] float32,
    ///      N_electrons[1, 1] float32,
    ///      N_field python int,
    ///      [property_values[1, n_output] float32,
    ///       potential[N_field] float32])  # last two only if has_property
    fn get<'py>(&self, py: Python<'py>, idx: usize) -> PyResult<Bound<'py, PyTuple>> {
        if idx >= self.n_molecules {
            return Err(PyIndexError::new_err(format!(
                "Index {} out of range (n={})",
                idx, self.n_molecules
            )));
        }
        let entry = self.index_table_offset + idx * 8;
        let mol_offset = read_u64(&self.mmap, entry) as usize;
        let mmap = &self.mmap[..];
        let len = mmap.len();

        let mut cur = mol_offset;
        if cur + 16 > len {
            return Err(PyValueError::new_err("Molecule header out of bounds"));
        }
        let n_orbitals = read_u32(mmap, cur) as usize;
        let n_field = read_u32(mmap, cur + 4) as usize;
        let idx_len = read_u32(mmap, cur + 8) as usize;
        // 4 bytes reserved at cur+12
        cur += 16;

        if cur + idx_len > len {
            return Err(PyValueError::new_err("Molecule idx string out of bounds"));
        }
        let idx_str = std::str::from_utf8(&mmap[cur..cur + idx_len])
            .map_err(|_| PyValueError::new_err("Invalid UTF-8 in molecule idx"))?
            .to_string();
        cur += idx_len;
        cur = align_up(cur, 8);

        // atomic_orbitals: int64[n_orbitals]
        let ao_bytes_len = n_orbitals * 8;
        if cur + ao_bytes_len > len {
            return Err(PyValueError::new_err("atomic_orbitals out of bounds"));
        }
        let ao_vec = copy_i64_slice(&mmap[cur..cur + ao_bytes_len], n_orbitals);
        let atomic_orbitals = PyArray1::<i64>::from_vec_bound(py, ao_vec);
        cur += ao_bytes_len;

        // distance_matrix: float32[n_field * n_orbitals], reshape to (n_field, n_orbitals)
        let dm_elems = n_field * n_orbitals;
        let dm_bytes_len = dm_elems * 4;
        if cur + dm_bytes_len > len {
            return Err(PyValueError::new_err("distance_matrix out of bounds"));
        }
        let dm_vec = copy_f32_slice(&mmap[cur..cur + dm_bytes_len], dm_elems);
        let dm_arr = Array2::<f32>::from_shape_vec((n_field, n_orbitals), dm_vec)
            .map_err(|e| PyValueError::new_err(format!("distance_matrix reshape: {}", e)))?;
        let distance_matrix = dm_arr.into_pyarray_bound(py);
        cur += dm_bytes_len;

        // quantum_numbers: float32[n_orbitals] stored, exposed as shape (1, n_orbitals)
        let qn_bytes_len = n_orbitals * 4;
        if cur + qn_bytes_len > len {
            return Err(PyValueError::new_err("quantum_numbers out of bounds"));
        }
        let qn_vec = copy_f32_slice(&mmap[cur..cur + qn_bytes_len], n_orbitals);
        let qn_arr = Array2::<f32>::from_shape_vec((1, n_orbitals), qn_vec)
            .map_err(|e| PyValueError::new_err(format!("quantum_numbers reshape: {}", e)))?;
        let quantum_numbers = qn_arr.into_pyarray_bound(py);
        cur += qn_bytes_len;

        // N_electrons: float32 scalar, exposed as shape (1, 1)
        if cur + 4 > len {
            return Err(PyValueError::new_err("N_electrons out of bounds"));
        }
        let ne_val = read_f32(mmap, cur);
        let ne_arr = Array2::<f32>::from_shape_vec((1, 1), vec![ne_val])
            .map_err(|e| PyValueError::new_err(format!("N_electrons reshape: {}", e)))?;
        let n_electrons = ne_arr.into_pyarray_bound(py);
        cur += 4;

        // N_field as a Python int
        let n_field_py = n_field.into_py(py);

        if self.has_property {
            // property_values: float32[n_output] stored, exposed as (1, n_output)
            let pv_bytes_len = self.n_output * 4;
            if cur + pv_bytes_len > len {
                return Err(PyValueError::new_err("property_values out of bounds"));
            }
            let pv_vec = copy_f32_slice(&mmap[cur..cur + pv_bytes_len], self.n_output);
            let pv_arr = Array2::<f32>::from_shape_vec((1, self.n_output), pv_vec)
                .map_err(|e| PyValueError::new_err(format!("property_values reshape: {}", e)))?;
            let property_values = pv_arr.into_pyarray_bound(py);
            cur += pv_bytes_len;

            // potential: float32[n_field] stored, exposed as (n_field, 1) to
            // match the shape produced by ``preprocess.py``'s np.matmul.
            let pot_bytes_len = n_field * 4;
            if cur + pot_bytes_len > len {
                return Err(PyValueError::new_err("potential out of bounds"));
            }
            let pot_vec = copy_f32_slice(&mmap[cur..cur + pot_bytes_len], n_field);
            let pot_arr = Array2::<f32>::from_shape_vec((n_field, 1), pot_vec)
                .map_err(|e| PyValueError::new_err(format!("potential reshape: {}", e)))?;
            let potential = pot_arr.into_pyarray_bound(py);

            let items: [PyObject; 8] = [
                idx_str.into_py(py),
                atomic_orbitals.into_py(py),
                distance_matrix.into_py(py),
                quantum_numbers.into_py(py),
                n_electrons.into_py(py),
                n_field_py,
                property_values.into_py(py),
                potential.into_py(py),
            ];
            Ok(PyTuple::new_bound(py, items))
        } else {
            let items: [PyObject; 6] = [
                idx_str.into_py(py),
                atomic_orbitals.into_py(py),
                distance_matrix.into_py(py),
                quantum_numbers.into_py(py),
                n_electrons.into_py(py),
                n_field_py,
            ];
            Ok(PyTuple::new_bound(py, items))
        }
    }
}

// --------------------------------------------------------------------------- //
// LCAO helper kernels
//
// These are tiny CPU-side functions that replace per-molecule Python loops
// inside ``QuantumDeepField.list_to_batch`` and ``QuantumDeepField.pad``.
// They never touch the device: the Python wrapper does a single
// ``torch.from_numpy(...).to(device)`` afterwards, so the goal is to replace
// N small host->device transfers with 1 large one.
// --------------------------------------------------------------------------- //

/// Build the block-diagonal matrix used by ``QuantumDeepField.pad``.
///
/// Given a sequence of float32 2-D arrays ``[M_0, M_1, ...]`` with shapes
/// ``[(r_0, c_0), (r_1, c_1), ...]``, returns a single ``(sum r_i, sum c_i)``
/// float32 array where each ``M_k`` is placed at row-offset ``sum_{i<k} r_i``
/// and column-offset ``sum_{i<k} c_i`` and every other entry is ``pad_value``.
#[pyfunction]
fn block_diag_pad_f32<'py>(
    py: Python<'py>,
    matrices: Vec<PyReadonlyArray2<'py, f32>>,
    pad_value: f32,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let n = matrices.len();
    if n == 0 {
        return Err(PyValueError::new_err(
            "block_diag_pad_f32: empty matrix list",
        ));
    }
    let mut shapes: Vec<(usize, usize)> = Vec::with_capacity(n);
    let mut total_rows: usize = 0;
    let mut total_cols: usize = 0;
    for m in matrices.iter() {
        let s = m.shape();
        if s.len() != 2 {
            return Err(PyValueError::new_err(
                "block_diag_pad_f32: each matrix must be 2-D",
            ));
        }
        shapes.push((s[0], s[1]));
        total_rows += s[0];
        total_cols += s[1];
    }

    let mut buf: Vec<f32> = vec![pad_value; total_rows * total_cols];

    let mut row_off: usize = 0;
    let mut col_off: usize = 0;
    for (k, m) in matrices.iter().enumerate() {
        let (r, c) = shapes[k];
        let src = m.as_slice().map_err(|_| {
            PyValueError::new_err(
                "block_diag_pad_f32: matrices must be C-contiguous",
            )
        })?;
        for ri in 0..r {
            let dst_start = (row_off + ri) * total_cols + col_off;
            let src_start = ri * c;
            buf[dst_start..dst_start + c]
                .copy_from_slice(&src[src_start..src_start + c]);
        }
        row_off += r;
        col_off += c;
    }

    let arr = Array2::from_shape_vec((total_rows, total_cols), buf)
        .map_err(|e| PyValueError::new_err(format!("reshape: {}", e)))?;
    Ok(arr.into_pyarray_bound(py))
}

/// Concatenate a sequence of 1-D int64 arrays into a single 1-D int64 array.
///
/// Equivalent to ``np.concatenate(xs)`` for the case used by ``list_to_batch``
/// with ``dtype=torch.LongTensor``.
#[pyfunction]
fn concat_i64<'py>(
    py: Python<'py>,
    xs: Vec<PyReadonlyArray1<'py, i64>>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let total: usize = xs.iter().map(|a| a.len()).sum();
    let mut out: Vec<i64> = Vec::with_capacity(total);
    for a in xs.iter() {
        let s = a.as_slice().map_err(|_| {
            PyValueError::new_err("concat_i64: arrays must be C-contiguous")
        })?;
        out.extend_from_slice(s);
    }
    let arr = Array1::from(out);
    Ok(arr.into_pyarray_bound(py))
}

/// Concatenate a sequence of float32 2-D arrays along axis 0.
///
/// All inputs must share the same number of columns. Mirrors
/// ``np.concatenate(xs, axis=0)`` and matches the behaviour of
/// ``list_to_batch(..., cat=True, axis=0)`` used for property values and
/// the potential.
#[pyfunction]
fn concat_f32_axis0<'py>(
    py: Python<'py>,
    xs: Vec<PyReadonlyArray2<'py, f32>>,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    if xs.is_empty() {
        return Err(PyValueError::new_err("concat_f32_axis0: empty list"));
    }
    let cols = xs[0].shape()[1];
    let mut total_rows: usize = 0;
    for a in xs.iter() {
        let s = a.shape();
        if s.len() != 2 {
            return Err(PyValueError::new_err(
                "concat_f32_axis0: each input must be 2-D",
            ));
        }
        if s[1] != cols {
            return Err(PyValueError::new_err(format!(
                "concat_f32_axis0: column mismatch ({} vs {})",
                s[1], cols
            )));
        }
        total_rows += s[0];
    }

    let mut out: Vec<f32> = Vec::with_capacity(total_rows * cols);
    for a in xs.iter() {
        let s = a.as_slice().map_err(|_| {
            PyValueError::new_err("concat_f32_axis0: arrays must be C-contiguous")
        })?;
        out.extend_from_slice(s);
    }
    let arr = Array2::from_shape_vec((total_rows, cols), out)
        .map_err(|e| PyValueError::new_err(format!("reshape: {}", e)))?;
    Ok(arr.into_pyarray_bound(py))
}

/// Concatenate a sequence of float32 2-D arrays along axis 1.
///
/// All inputs must share the same number of rows. Mirrors
/// ``np.concatenate(xs, axis=1)`` and matches the behaviour of
/// ``list_to_batch(..., cat=True, axis=1)`` used for quantum_numbers.
#[pyfunction]
fn concat_f32_axis1<'py>(
    py: Python<'py>,
    xs: Vec<PyReadonlyArray2<'py, f32>>,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    if xs.is_empty() {
        return Err(PyValueError::new_err("concat_f32_axis1: empty list"));
    }
    let rows = xs[0].shape()[0];
    let mut total_cols: usize = 0;
    let mut col_sizes: Vec<usize> = Vec::with_capacity(xs.len());
    for a in xs.iter() {
        let s = a.shape();
        if s.len() != 2 {
            return Err(PyValueError::new_err(
                "concat_f32_axis1: each input must be 2-D",
            ));
        }
        if s[0] != rows {
            return Err(PyValueError::new_err(format!(
                "concat_f32_axis1: row mismatch ({} vs {})",
                s[0], rows
            )));
        }
        col_sizes.push(s[1]);
        total_cols += s[1];
    }

    let mut out: Vec<f32> = vec![0.0; rows * total_cols];
    let mut col_off = 0usize;
    for (i, a) in xs.iter().enumerate() {
        let c = col_sizes[i];
        let src = a.as_slice().map_err(|_| {
            PyValueError::new_err("concat_f32_axis1: arrays must be C-contiguous")
        })?;
        for ri in 0..rows {
            let dst_start = ri * total_cols + col_off;
            let src_start = ri * c;
            out[dst_start..dst_start + c]
                .copy_from_slice(&src[src_start..src_start + c]);
        }
        col_off += c;
    }
    let arr = Array2::from_shape_vec((rows, total_cols), out)
        .map_err(|e| PyValueError::new_err(format!("reshape: {}", e)))?;
    Ok(arr.into_pyarray_bound(py))
}

// --------------------------------------------------------------------------- //
// Preprocess pipeline (parallel)
//
// These functions reproduce the heavy NumPy/SciPy operations performed inside
// ``train/preprocess.py`` per molecule (create_field, distance matrices,
// Gaussian potential), but written in plain Rust so we can drop the GIL and
// run a whole batch of molecules in parallel via Rayon.
//
// Numerical contract: all intermediates are computed in f64 to match what the
// original NumPy/SciPy implementation does, and results are cast to f32 at
// the boundary, matching the dtype that ``preprocess.py`` ultimately stores
// in the per-molecule .npy file.
// --------------------------------------------------------------------------- //

mod preprocess_core {
    use ndarray::Array2;

    /// Pairwise Euclidean distance matrix between ``coords1`` (n1, 3) and
    /// ``coords2`` (n2, 3). Zero entries are replaced with 1e6 to match the
    /// original ``create_distancematrix`` behaviour (avoiding division by
    /// zero or huge Gaussians at self-pairs).
    pub fn distance_matrix(coords1: &Array2<f64>, coords2: &Array2<f64>) -> Array2<f64> {
        let n1 = coords1.nrows();
        let n2 = coords2.nrows();
        let mut out = Array2::<f64>::zeros((n1, n2));
        let s1 = coords1.as_slice().expect("coords1 contiguous");
        let s2 = coords2.as_slice().expect("coords2 contiguous");
        let so = out.as_slice_mut().expect("out contiguous");
        for i in 0..n1 {
            let ix = i * 3;
            let x1 = s1[ix];
            let y1 = s1[ix + 1];
            let z1 = s1[ix + 2];
            let row = i * n2;
            for j in 0..n2 {
                let jx = j * 3;
                let dx = x1 - s2[jx];
                let dy = y1 - s2[jx + 1];
                let dz = z1 - s2[jx + 2];
                let d2 = dx * dx + dy * dy + dz * dz;
                let d = d2.sqrt();
                so[row + j] = if d == 0.0 { 1e6 } else { d };
            }
        }
        out
    }

    /// ``-matmul(exp(-d^2), atomic_numbers)`` where ``distance_matrix`` is
    /// shape (n_field, n_atoms) and ``atomic_numbers`` is (n_atoms, 1).
    /// Returns (n_field, 1).
    ///
    /// Used by [`process_one_legacy`] and unit tests; the optimized hot path
    /// prefers [`potential_from_field_atoms`].
    pub fn potential(
        distance_matrix: &Array2<f64>,
        atomic_numbers: &Array2<i64>,
    ) -> Array2<f64> {
        let n_field = distance_matrix.nrows();
        let n_atoms = distance_matrix.ncols();
        assert_eq!(atomic_numbers.shape(), &[n_atoms, 1]);
        let dm = distance_matrix.as_slice().expect("dm contiguous");
        let an_slice = atomic_numbers.as_slice().expect("an contiguous");
        let mut out = Array2::<f64>::zeros((n_field, 1));
        let so = out.as_slice_mut().expect("out contiguous");
        for i in 0..n_field {
            let row = i * n_atoms;
            let mut sum = 0.0f64;
            for j in 0..n_atoms {
                let d = dm[row + j];
                let g = (-d * d).exp();
                sum += g * (an_slice[j] as f64);
            }
            so[i] = -sum;
        }
        out
    }

    /// Same result as ``potential(distance_matrix(field, atomic_coords), atomic_numbers)``
    /// without allocating the (n_field × n_atoms) atom–field distance matrix.
    ///
    /// Matches ``distance_matrix`` zero handling: ``d == 0`` is treated like ``d = 1e6``
    /// for the Gaussian weight (``exp(-d²)`` → effectively zero).
    pub fn potential_from_field_atoms(
        field: &Array2<f64>,
        atomic_coords: &Array2<f64>,
        atomic_numbers: &Array2<i64>,
    ) -> Array2<f64> {
        let n_field = field.nrows();
        let n_atoms = atomic_coords.nrows();
        assert_eq!(atomic_numbers.shape(), &[n_atoms, 1]);
        let f = field.as_slice().expect("field contiguous");
        let ac = atomic_coords.as_slice().expect("atomic_coords contiguous");
        let an_slice = atomic_numbers.as_slice().expect("an contiguous");
        let mut out = Array2::<f64>::zeros((n_field, 1));
        let so = out.as_slice_mut().expect("out contiguous");
        for i in 0..n_field {
            let ix = i * 3;
            let x1 = f[ix];
            let y1 = f[ix + 1];
            let z1 = f[ix + 2];
            let mut sum = 0.0f64;
            for j in 0..n_atoms {
                let jx = j * 3;
                let dx = x1 - ac[jx];
                let dy = y1 - ac[jx + 1];
                let dz = z1 - ac[jx + 2];
                let d2 = dx * dx + dy * dy + dz * dz;
                let d = d2.sqrt();
                let g = if d == 0.0 {
                    (-(1e6_f64 * 1e6_f64)).exp()
                } else {
                    (-d2).exp()
                };
                sum += g * (an_slice[j] as f64);
            }
            so[i] = -sum;
        }
        out
    }

    /// For each atomic position, shift the shared sphere offsets to produce
    /// the molecule's grid field. Matches ``create_field`` in
    /// ``preprocess.py``.
    ///
    /// Output shape: (n_atoms * n_sphere, 3) C-contiguous.
    pub fn create_field(
        sphere: &Array2<f64>,
        atomic_coords: &Array2<f64>,
    ) -> Array2<f64> {
        let n_atoms = atomic_coords.nrows();
        let n_sphere = sphere.nrows();
        let n_field = n_atoms * n_sphere;
        let mut out = Array2::<f64>::zeros((n_field, 3));
        let sph = sphere.as_slice().expect("sphere contiguous");
        let ac = atomic_coords.as_slice().expect("atomic_coords contiguous");
        let so = out.as_slice_mut().expect("out contiguous");
        for i in 0..n_atoms {
            let ax = ac[i * 3];
            let ay = ac[i * 3 + 1];
            let az = ac[i * 3 + 2];
            for j in 0..n_sphere {
                let row = (i * n_sphere + j) * 3;
                so[row] = sph[j * 3] + ax;
                so[row + 1] = sph[j * 3 + 1] + ay;
                so[row + 2] = sph[j * 3 + 2] + az;
            }
        }
        out
    }

    /// Cast f64 (n, m) → f32 (n, m).
    pub fn to_f32(arr: &Array2<f64>) -> Array2<f32> {
        arr.mapv(|x| x as f32)
    }

    pub struct MolOut {
        pub dm_orbital: Array2<f32>,
        pub potential: Array2<f32>,
        pub n_field: usize,
    }

    /// Full per-molecule pipeline: create_field, then both distance matrices,
    /// then the Gaussian potential. Returns the three outputs already cast
    /// to the f32 dtypes used by the downstream pickle/.npy/shard format.
    pub fn process_one(
        atomic_coords: &Array2<f64>,
        orbital_coords: &Array2<f64>,
        atomic_numbers: &Array2<i64>,
        sphere: &Array2<f64>,
    ) -> MolOut {
        let field = create_field(sphere, atomic_coords);
        let pot_f64 = potential_from_field_atoms(&field, atomic_coords, atomic_numbers);
        let dm_orb_f64 = distance_matrix(&field, orbital_coords);
        MolOut {
            n_field: field.nrows(),
            dm_orbital: to_f32(&dm_orb_f64),
            potential: to_f32(&pot_f64),
        }
    }

    /// Same outputs as [`process_one`], using the pre-optimization path that
    /// materializes the full atom–field distance matrix before the potential.
    /// Exposed for A/B benchmarks against [`process_one`].
    pub fn process_one_legacy(
        atomic_coords: &Array2<f64>,
        orbital_coords: &Array2<f64>,
        atomic_numbers: &Array2<i64>,
        sphere: &Array2<f64>,
    ) -> MolOut {
        let field = create_field(sphere, atomic_coords);
        let dm_atoms = distance_matrix(&field, atomic_coords);
        let pot_f64 = potential(&dm_atoms, atomic_numbers);
        let dm_orb_f64 = distance_matrix(&field, orbital_coords);
        MolOut {
            n_field: field.nrows(),
            dm_orbital: to_f32(&dm_orb_f64),
            potential: to_f32(&pot_f64),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use ndarray::arr2;

        #[test]
        fn potential_from_field_matches_distance_matrix_path() {
            let atomic_coords = arr2(&[
                [0.0_f64, 0.0, 0.0],
                [1.5, 0.0, 0.0],
            ]);
            let sphere = arr2(&[
                [0.1, 0.0, 0.0],
                [0.0, 0.2, 0.0],
            ]);
            let field = create_field(&sphere, &atomic_coords);
            let an = arr2(&[[1_i64], [6]]);
            let dm = distance_matrix(&field, &atomic_coords);
            let pot_dm = potential(&dm, &an);
            let pot_fused = potential_from_field_atoms(&field, &atomic_coords, &an);
            assert_eq!(pot_dm.dim(), pot_fused.dim());
            let a = pot_dm.as_slice().expect("c");
            let b = pot_fused.as_slice().expect("c");
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-12, "pot mismatch: {} vs {}", x, y);
            }
        }

        #[test]
        fn process_one_matches_legacy_outputs() {
            let atomic_coords = arr2(&[
                [0.0_f64, 0.0, 0.0],
                [1.5, 0.0, 0.0],
            ]);
            let sphere = arr2(&[
                [0.1, 0.0, 0.0],
                [0.0, 0.2, 0.0],
            ]);
            let orbital_coords = arr2(&[
                [0.0_f64, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]);
            let an = arr2(&[[1_i64], [6]]);
            let a = process_one(&atomic_coords, &orbital_coords, &an, &sphere);
            let b = process_one_legacy(&atomic_coords, &orbital_coords, &an, &sphere);
            assert_eq!(a.n_field, b.n_field);
            let da = a.dm_orbital.as_slice().expect("c");
            let db = b.dm_orbital.as_slice().expect("c");
            for (x, y) in da.iter().zip(db.iter()) {
                assert!(((*x as f64) - (*y as f64)).abs() < 1e-5f64, "dm {}", x);
            }
            let pa = a.potential.as_slice().expect("c");
            let pb = b.potential.as_slice().expect("c");
            for (x, y) in pa.iter().zip(pb.iter()) {
                assert!(((*x as f64) - (*y as f64)).abs() < 1e-5f64, "pot {}", x);
            }
        }
    }
}

/// Process a single molecule (release GIL not necessary here since the
/// caller is already holding it; useful for unit-style tests or when the
/// caller wants to interleave Python work).
#[pyfunction]
fn preprocess_molecule_rust<'py>(
    py: Python<'py>,
    atomic_coords: PyReadonlyArray2<'py, f64>,
    orbital_coords: PyReadonlyArray2<'py, f64>,
    atomic_numbers: PyReadonlyArray2<'py, i64>,
    sphere: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, numpy::PyArray2<f32>>, Bound<'py, numpy::PyArray2<f32>>, usize)> {
    let ac = atomic_coords.as_array().to_owned();
    let oc = orbital_coords.as_array().to_owned();
    let an = atomic_numbers.as_array().to_owned();
    let sp = sphere.as_array().to_owned();
    let out = preprocess_core::process_one(&ac, &oc, &an, &sp);
    Ok((
        out.dm_orbital.into_pyarray_bound(py),
        out.potential.into_pyarray_bound(py),
        out.n_field,
    ))
}

/// Process a batch of molecules in parallel via Rayon. The GIL is released
/// while the heavy compute runs, so this scales with the number of physical
/// CPU cores (saturating well above 4 cores in practice for the QM9 dataset).
///
/// All three input lists must have the same length; each entry is a
/// (n_atoms, 3) f64 / (n_orbitals, 3) f64 / (n_atoms, 1) i64 array for one
/// molecule. ``sphere`` is the shared (n_sphere, 3) f64 grid produced once
/// per dataset by ``train/preprocess.py``'s ``create_sphere``.
///
/// Returns a Python list of ``(distance_matrix_to_orbitals, potential,
/// n_field)`` triples in the same order as the inputs.
#[pyfunction]
fn preprocess_batch_rust<'py>(
    py: Python<'py>,
    atomic_coords_list: Vec<PyReadonlyArray2<'py, f64>>,
    orbital_coords_list: Vec<PyReadonlyArray2<'py, f64>>,
    atomic_numbers_list: Vec<PyReadonlyArray2<'py, i64>>,
    sphere: PyReadonlyArray2<'py, f64>,
) -> PyResult<Vec<(Bound<'py, numpy::PyArray2<f32>>, Bound<'py, numpy::PyArray2<f32>>, usize)>> {
    use rayon::prelude::*;

    let n = atomic_coords_list.len();
    if n != orbital_coords_list.len() || n != atomic_numbers_list.len() {
        return Err(PyValueError::new_err(format!(
            "input list lengths differ: atomic_coords={}, orbital_coords={}, atomic_numbers={}",
            n,
            orbital_coords_list.len(),
            atomic_numbers_list.len(),
        )));
    }

    // Materialize all inputs to owned ndarray::Array2 so we can drop the GIL
    // and hand them to a Rayon worker pool.
    struct Inp {
        ac: ndarray::Array2<f64>,
        oc: ndarray::Array2<f64>,
        an: ndarray::Array2<i64>,
    }

    let mut inputs: Vec<Inp> = Vec::with_capacity(n);
    for i in 0..n {
        inputs.push(Inp {
            ac: atomic_coords_list[i].as_array().to_owned(),
            oc: orbital_coords_list[i].as_array().to_owned(),
            an: atomic_numbers_list[i].as_array().to_owned(),
        });
    }
    let sphere_owned = sphere.as_array().to_owned();

    // Heavy work without the GIL.
    let outputs: Vec<preprocess_core::MolOut> = py.allow_threads(|| {
        inputs
            .par_iter()
            .map(|m| preprocess_core::process_one(&m.ac, &m.oc, &m.an, &sphere_owned))
            .collect()
    });

    // Repack into Python arrays.
    let mut results = Vec::with_capacity(outputs.len());
    for out in outputs.into_iter() {
        results.push((
            out.dm_orbital.into_pyarray_bound(py),
            out.potential.into_pyarray_bound(py),
            out.n_field,
        ));
    }
    Ok(results)
}

/// Same as [`preprocess_batch_rust`] but runs [`preprocess_core::process_one_legacy`]
/// per molecule (atom–field distance matrix fully materialized). For benchmarks only.
#[pyfunction]
fn preprocess_batch_rust_legacy<'py>(
    py: Python<'py>,
    atomic_coords_list: Vec<PyReadonlyArray2<'py, f64>>,
    orbital_coords_list: Vec<PyReadonlyArray2<'py, f64>>,
    atomic_numbers_list: Vec<PyReadonlyArray2<'py, i64>>,
    sphere: PyReadonlyArray2<'py, f64>,
) -> PyResult<Vec<(Bound<'py, numpy::PyArray2<f32>>, Bound<'py, numpy::PyArray2<f32>>, usize)>> {
    use rayon::prelude::*;

    let n = atomic_coords_list.len();
    if n != orbital_coords_list.len() || n != atomic_numbers_list.len() {
        return Err(PyValueError::new_err(format!(
            "input list lengths differ: atomic_coords={}, orbital_coords={}, atomic_numbers={}",
            n,
            orbital_coords_list.len(),
            atomic_numbers_list.len(),
        )));
    }

    struct Inp {
        ac: ndarray::Array2<f64>,
        oc: ndarray::Array2<f64>,
        an: ndarray::Array2<i64>,
    }

    let mut inputs: Vec<Inp> = Vec::with_capacity(n);
    for i in 0..n {
        inputs.push(Inp {
            ac: atomic_coords_list[i].as_array().to_owned(),
            oc: orbital_coords_list[i].as_array().to_owned(),
            an: atomic_numbers_list[i].as_array().to_owned(),
        });
    }
    let sphere_owned = sphere.as_array().to_owned();

    let outputs: Vec<preprocess_core::MolOut> = py.allow_threads(|| {
        inputs
            .par_iter()
            .map(|m| preprocess_core::process_one_legacy(&m.ac, &m.oc, &m.an, &sphere_owned))
            .collect()
    });

    let mut results = Vec::with_capacity(outputs.len());
    for out in outputs.into_iter() {
        results.push((
            out.dm_orbital.into_pyarray_bound(py),
            out.potential.into_pyarray_bound(py),
            out.n_field,
        ));
    }
    Ok(results)
}

/// Returns the on-disk constants the Python writer must match.
#[pyfunction]
fn format_info(py: Python<'_>) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("magic", &MAGIC[..])?;
    d.set_item("version", VERSION)?;
    d.set_item("header_size", HEADER_SIZE)?;
    d.set_item("flag_has_property", FLAG_HAS_PROPERTY)?;
    Ok(d.into())
}

// ---------------------------------------------------------------------------
// ShardWriter — same on-disk layout as ``train/dataset_shard.write_shard``.
// ---------------------------------------------------------------------------

fn write_zeros<W: Write>(w: &mut W, n: usize) -> std::io::Result<()> {
    const Z: [u8; 4096] = [0u8; 4096];
    let mut rem = n;
    while rem > 0 {
        let chunk = rem.min(Z.len());
        w.write_all(&Z[..chunk])?;
        rem -= chunk;
    }
    Ok(())
}

/// Incremental shard builder: reserve header + index table up front, append
/// aligned molecule records, then ``finalize`` patches the index + header.
/// Python still ``np.load``\s each ``.npy`` (object-array format); Rust owns
/// the binary packing and file layout so we do not duplicate a second delete
/// / repack pass in pure Python.
#[pyclass(module = "qdf_io._native")]
struct ShardWriter {
    path: String,
    file: BufWriter<File>,
    mol_offsets: Vec<u64>,
    index_table_offset: u64,
    data_section_offset: u64,
    n_expected: usize,
    appended: usize,
    has_property: bool,
    n_output: u32,
    t0: Instant,
}

#[pymethods]
impl ShardWriter {
    #[new]
    #[pyo3(signature = (path, n_molecules, has_property, n_output))]
    fn new(path: String, n_molecules: usize, has_property: bool, n_output: u32) -> PyResult<Self> {
        if n_molecules == 0 {
            return Err(PyValueError::new_err("n_molecules must be > 0"));
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| PyIOError::new_err(format!("open '{}': {}", path, e)))?;
        let mut file = BufWriter::new(file);
        write_zeros(&mut file, HEADER_SIZE)?;
        let index_table_offset = HEADER_SIZE as u64;
        write_zeros(&mut file, n_molecules * 8)?;
        let cur = file.stream_position().map_err(|e| PyIOError::new_err(e.to_string()))?;
        let pad = align_up(cur as usize, 8) - cur as usize;
        write_zeros(&mut file, pad)?;
        let data_section_offset = file
            .stream_position()
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self {
            path,
            file,
            mol_offsets: Vec::with_capacity(n_molecules),
            index_table_offset,
            data_section_offset,
            n_expected: n_molecules,
            appended: 0,
            has_property,
            n_output,
            t0: Instant::now(),
        })
    }

    /// Append one molecule record (must be called exactly ``n_molecules`` times
    /// before ``finalize``).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        idx,
        atomic_orbitals,
        distance_matrix,
        quantum_numbers,
        n_electrons,
        n_field,
        property_values,
        potential,
    ))]
    fn append_molecule(
        &mut self,
        idx: &str,
        atomic_orbitals: PyReadonlyArray1<'_, i64>,
        distance_matrix: PyReadonlyArray2<'_, f32>,
        quantum_numbers: PyReadonlyArray2<'_, f32>,
        n_electrons: PyReadonlyArray2<'_, f32>,
        n_field: u32,
        property_values: Option<PyReadonlyArray2<'_, f32>>,
        potential: Option<PyReadonlyArray2<'_, f32>>,
    ) -> PyResult<()> {
        if self.appended >= self.n_expected {
            return Err(PyValueError::new_err(format!(
                "append_molecule called too many times (expected {})",
                self.n_expected
            )));
        }
        let n_field = n_field as usize;
        let ao = atomic_orbitals.as_slice().map_err(|_| {
            PyValueError::new_err("atomic_orbitals must be C-contiguous int64 vector")
        })?;
        let n_orb = ao.len();
        let dm = distance_matrix.as_slice().map_err(|_| {
            PyValueError::new_err("distance_matrix must be C-contiguous float32")
        })?;
        if distance_matrix.shape() != [n_field, n_orb] {
            return Err(PyValueError::new_err(format!(
                "distance_matrix shape {:?} != (n_field={}, n_orbitals={})",
                distance_matrix.shape(),
                n_field,
                n_orb
            )));
        }
        if dm.len() != n_field * n_orb {
            return Err(PyValueError::new_err("distance_matrix size mismatch"));
        }
        let qn = quantum_numbers.as_slice().map_err(|_| {
            PyValueError::new_err("quantum_numbers must be C-contiguous float32")
        })?;
        if quantum_numbers.shape() != [1, n_orb] {
            return Err(PyValueError::new_err(format!(
                "quantum_numbers shape {:?} != (1, {})",
                quantum_numbers.shape(),
                n_orb
            )));
        }
        let ne = n_electrons.as_slice().map_err(|_| {
            PyValueError::new_err("N_electrons must be C-contiguous float32")
        })?;
        if n_electrons.shape() != [1, 1] || ne.len() != 1 {
            return Err(PyValueError::new_err(format!(
                "N_electrons shape {:?} != (1, 1)",
                n_electrons.shape()
            )));
        }
        let ne_val = ne[0];

        let (prop_vec, pot_vec): (Option<Vec<f32>>, Option<Vec<f32>>) = if self.has_property {
            let pv = property_values.ok_or_else(|| {
                PyValueError::new_err("has_property=True but property_values is None")
            })?;
            let pot = potential.ok_or_else(|| {
                PyValueError::new_err("has_property=True but potential is None")
            })?;
            let ps = pv.as_slice().map_err(|_| {
                PyValueError::new_err("property_values must be C-contiguous float32")
            })?;
            if pv.shape() != [1, self.n_output as usize] {
                return Err(PyValueError::new_err(format!(
                    "property_values shape {:?} != (1, {})",
                    pv.shape(),
                    self.n_output
                )));
            }
            let pots = pot.as_slice().map_err(|_| {
                PyValueError::new_err("potential must be C-contiguous float32")
            })?;
            let pot_flat_len = match pot.shape() {
                [a, 1] if *a == n_field => n_field,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "potential shape {:?} must be ({}, 1)",
                        pot.shape(),
                        n_field
                    )));
                }
            };
            if pots.len() < pot_flat_len {
                return Err(PyValueError::new_err("potential data too short"));
            }
            (
                Some(ps.iter().copied().collect()),
                Some(pots[..pot_flat_len].iter().copied().collect()),
            )
        } else {
            if property_values.is_some() || potential.is_some() {
                return Err(PyValueError::new_err(
                    "has_property=False but property_values/potential were passed",
                ));
            }
            (None, None)
        };

        let cur = self
            .file
            .stream_position()
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let pad = align_up(cur as usize, 8) - cur as usize;
        write_zeros(&mut self.file, pad).map_err(|e| PyIOError::new_err(e.to_string()))?;
        let mol_off = self
            .file
            .stream_position()
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        self.mol_offsets.push(mol_off);

        let idx_bytes = idx.as_bytes();
        self
            .file
            .write_all(&(n_orb as u32).to_le_bytes())
            .and_then(|_| self.file.write_all(&(n_field as u32).to_le_bytes()))
            .and_then(|_| self.file.write_all(&(idx_bytes.len() as u32).to_le_bytes()))
            .and_then(|_| self.file.write_all(&0u32.to_le_bytes()))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        self.file.write_all(idx_bytes).map_err(|e| PyIOError::new_err(e.to_string()))?;

        let pos = self
            .file
            .stream_position()
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let pad2 = align_up(pos as usize, 8) - pos as usize;
        write_zeros(&mut self.file, pad2).map_err(|e| PyIOError::new_err(e.to_string()))?;

        for v in ao {
            self.file
                .write_all(&v.to_le_bytes())
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }
        for v in dm {
            self.file
                .write_all(&v.to_le_bytes())
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }
        for v in qn {
            self.file
                .write_all(&v.to_le_bytes())
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }
        self.file
            .write_all(&ne_val.to_le_bytes())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        if let (Some(pv_v), Some(pot_v)) = (prop_vec.as_ref(), pot_vec.as_ref()) {
            for v in pv_v {
                self.file
                    .write_all(&v.to_le_bytes())
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
            }
            for v in pot_v {
                self.file
                    .write_all(&v.to_le_bytes())
                    .map_err(|e| PyIOError::new_err(e.to_string()))?;
            }
        }

        self.appended += 1;
        Ok(())
    }

    fn finalize(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        if self.appended != self.n_expected {
            return Err(PyValueError::new_err(format!(
                "finalize: expected {} molecules, got {}",
                self.n_expected, self.appended
            )));
        }
        self.file
            .flush()
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let file_size = self
            .file
            .get_mut()
            .seek(SeekFrom::End(0))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        self.file
            .seek(SeekFrom::Start(self.index_table_offset))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        for off in &self.mol_offsets {
            self.file
                .write_all(&off.to_le_bytes())
                .map_err(|e| PyIOError::new_err(e.to_string()))?;
        }

        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        self.file.write_all(MAGIC).map_err(|e| PyIOError::new_err(e.to_string()))?;
        self.file
            .write_all(&VERSION.to_le_bytes())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let n = self.n_expected as u64;
        self.file
            .write_all(&n.to_le_bytes())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        self.file
            .write_all(&self.n_output.to_le_bytes())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let flags = if self.has_property {
            FLAG_HAS_PROPERTY
        } else {
            0
        };
        self.file
            .write_all(&flags.to_le_bytes())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        self.file
            .write_all(&self.index_table_offset.to_le_bytes())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        self.file
            .write_all(&self.data_section_offset.to_le_bytes())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        self.file
            .write_all(&file_size.to_le_bytes())
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        write_zeros(&mut self.file, 12).map_err(|e| PyIOError::new_err(e.to_string()))?;

        self.file
            .flush()
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        let elapsed = self.t0.elapsed().as_secs_f64();
        let d = PyDict::new_bound(py);
        d.set_item("n_molecules", self.n_expected)?;
        d.set_item("bytes_written", file_size)?;
        d.set_item("elapsed_sec", elapsed)?;
        d.set_item("has_property", self.has_property)?;
        d.set_item("n_output", self.n_output)?;
        d.set_item("path", &self.path)?;
        Ok(d.into())
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ShardReader>()?;
    m.add_class::<ShardWriter>()?;
    m.add_function(wrap_pyfunction!(format_info, m)?)?;
    m.add_function(wrap_pyfunction!(block_diag_pad_f32, m)?)?;
    m.add_function(wrap_pyfunction!(concat_i64, m)?)?;
    m.add_function(wrap_pyfunction!(concat_f32_axis0, m)?)?;
    m.add_function(wrap_pyfunction!(concat_f32_axis1, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_molecule_rust, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_batch_rust, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_batch_rust_legacy, m)?)?;
    Ok(())
}
