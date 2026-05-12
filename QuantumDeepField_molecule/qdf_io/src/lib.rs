//! QDF binary shard reader.
//!
//! See `python/qdf_io/__init__.py` and `QuantumDeepField_molecule/train/dataset_shard.py`
//! for the Python-side counterpart. The on-disk layout is documented at the top of
//! that Python file.

use memmap2::Mmap;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyIOError, PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::fs::File;

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

/// Returns the on-disk constants the Python writer must match.
#[pyfunction]
fn format_info(py: Python<'_>) -> PyResult<PyObject> {
    use pyo3::types::PyDict;
    let d = PyDict::new_bound(py);
    d.set_item("magic", &MAGIC[..])?;
    d.set_item("version", VERSION)?;
    d.set_item("header_size", HEADER_SIZE)?;
    d.set_item("flag_has_property", FLAG_HAS_PROPERTY)?;
    Ok(d.into())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ShardReader>()?;
    m.add_function(wrap_pyfunction!(format_info, m)?)?;
    Ok(())
}
