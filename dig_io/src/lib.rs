//! DGCL native helpers.
//!
//! Mirrors the `qdf_io` crate's PyO3 + numpy + rayon + memmap2 pattern, but exposes
//! kernels that DGCL's molecular pipeline can opt into:
//!
//! * Contrastive view kernels — `uniform_sample_subgraph`, `rw_sample_subgraph`,
//!   `edge_perturb`. Each accepts a deterministic `seed`, runs on CPU with Rayon
//!   when called in batch, and returns numpy arrays compatible with
//!   `torch.from_numpy(...)`.
//!
//! * Scaffold split helper — `scaffold_bucket_split`. Given per-sample scaffold
//!   ids and target lengths, returns `(indices, offsets)` for building
//!   `torch.utils.data.Subset` without rerunning RDKit.
//!
//! * `MoleculeShardWriter` / `MoleculeShardReader` — single-file binary shard
//!   that holds enough fields to rebuild the PyG `Data` records produced by
//!   `dig/threedgraph/dataset/PygMoleculeNet.py` (`z, pos, edge_index, edge_attr,
//!   x, y, smiles, max1..4pos_mmff, min/max1..4_energy`). The reader returns a
//!   dict of numpy arrays; the Python wrapper converts to PyG `Data`.
//!
//! On-disk shard format (little-endian, version 1)::
//!
//!   HEADER (64 bytes)
//!     0   8   magic = b"DIGSHRD\0"
//!     8   4   version  u32  = 1
//!    12   8   n_records  u64
//!    20   4   flags  u32  (reserved)
//!    24   8   index_table_offset  u64
//!    32   8   data_section_offset u64
//!    40   8   file_size  u64
//!    48  16   reserved
//!
//!   INDEX TABLE: n_records * u64 absolute offsets.
//!
//!   RECORD (8-aligned)
//!     RecordHeader (40 bytes)
//!       u32 n_atoms
//!       u32 n_edges
//!       u32 n_edge_attr     (per-edge attribute width; may be 0)
//!       u32 n_x_feat        (per-atom feature width; may be 0)
//!       u32 n_y
//!       u32 idx_len
//!       u32 smiles_len
//!       u32 flags_local     (bit0 = mmff conformers/energies present)
//!       u64 reserved
//!     idx     [idx_len bytes]      utf8 (pad to 8)
//!     smiles  [smiles_len bytes]   utf8 (pad to 8)
//!     z           i64[n_atoms]
//!     pos         f32[n_atoms*3]
//!     edge_index  i64[2 * n_edges]
//!     edge_attr   i64[n_edges * n_edge_attr]
//!     x           i64[n_atoms * n_x_feat]
//!     y           f32[n_y]
//!     if mmff:
//!       max1pos..max4pos    f32[n_atoms*3] each
//!       min_energy..max4_energy  f64 each (5 values, in order min/max1/max2/max3/max4)
//!     <pad to 8>

use memmap2::Mmap;
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIOError, PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};

const MAGIC: &[u8; 8] = b"DIGSHRD\0";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 64;
const RECORD_HEADER_SIZE: usize = 40;
const FLAG_HAS_MMFF: u32 = 1;

#[inline]
fn align_up(x: usize, a: usize) -> usize {
    (x + a - 1) & !(a - 1)
}

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
fn copy_f32_slice(bytes: &[u8], n: usize) -> Vec<f32> {
    debug_assert_eq!(bytes.len(), n * 4);
    let mut out: Vec<f32> = Vec::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            out.as_mut_ptr() as *mut u8,
            n * 4,
        );
        out.set_len(n);
    }
    out
}

#[inline]
fn copy_i64_slice(bytes: &[u8], n: usize) -> Vec<i64> {
    debug_assert_eq!(bytes.len(), n * 8);
    let mut out: Vec<i64> = Vec::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            out.as_mut_ptr() as *mut u8,
            n * 8,
        );
        out.set_len(n);
    }
    out
}

#[inline]
fn copy_f64_slice(bytes: &[u8], n: usize) -> Vec<f64> {
    debug_assert_eq!(bytes.len(), n * 8);
    let mut out: Vec<f64> = Vec::with_capacity(n);
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            out.as_mut_ptr() as *mut u8,
            n * 8,
        );
        out.set_len(n);
    }
    out
}

#[inline]
fn write_pad8(w: &mut impl Write, count_so_far: usize) -> std::io::Result<usize> {
    let pad = align_up(count_so_far, 8) - count_so_far;
    if pad > 0 {
        w.write_all(&[0u8; 8][..pad])?;
    }
    Ok(pad)
}

fn seeded_rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

// --------------------------------------------------------------------------- //
// view fn kernels
// --------------------------------------------------------------------------- //

/// Sample `keep_num` unique node ids in ``[0, num_nodes)`` deterministically.
///
/// Returns the kept ids in **sorted ascending** order so that the relabeling
/// matches the original PyTorch path
/// (``torch.zeros_like(...).scatter_(0, idx, 1).bool()``).
fn fisher_yates_sample(num_nodes: usize, keep_num: usize, rng: &mut ChaCha8Rng) -> Vec<i64> {
    assert!(keep_num <= num_nodes);
    let mut pool: Vec<i64> = (0..num_nodes as i64).collect();
    for i in 0..keep_num {
        let j = (rng.next_u64() as usize) % (num_nodes - i) + i;
        pool.swap(i, j);
    }
    let mut keep: Vec<i64> = pool[..keep_num].to_vec();
    keep.sort_unstable();
    keep
}

/// Drop nodes uniformly at random; return the kept ids (sorted) and the
/// relabeled ``edge_index`` containing only edges where both endpoints survived.
#[pyfunction]
#[pyo3(name = "uniform_sample_subgraph")]
fn py_uniform_sample_subgraph<'py>(
    py: Python<'py>,
    edge_index: PyReadonlyArray2<'py, i64>,
    num_nodes: usize,
    keep_num: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<i64>>)> {
    let ei = edge_index.as_array();
    if ei.shape()[0] != 2 {
        return Err(PyValueError::new_err(format!(
            "edge_index must have shape (2, E); got {:?}",
            ei.shape()
        )));
    }
    if keep_num > num_nodes {
        return Err(PyValueError::new_err(
            "keep_num must be <= num_nodes",
        ));
    }

    // Collect edges and constants up front so we can drop the GIL during the heavy work.
    let n_edges = ei.shape()[1];
    let src: Vec<i64> = ei.row(0).to_vec();
    let dst: Vec<i64> = ei.row(1).to_vec();

    let (keep, new_edges): (Vec<i64>, (Vec<i64>, Vec<i64>)) = py.allow_threads(move || {
        let mut rng = seeded_rng(seed);
        let keep = fisher_yates_sample(num_nodes, keep_num, &mut rng);

        // Build relabel map: old_id -> new_id (or -1).
        let mut relabel: Vec<i64> = vec![-1; num_nodes];
        for (new_id, &old_id) in keep.iter().enumerate() {
            relabel[old_id as usize] = new_id as i64;
        }

        let mut new_src: Vec<i64> = Vec::new();
        let mut new_dst: Vec<i64> = Vec::new();
        for i in 0..n_edges {
            let u = src[i];
            let v = dst[i];
            if u < 0 || v < 0 || (u as usize) >= num_nodes || (v as usize) >= num_nodes {
                continue;
            }
            let nu = relabel[u as usize];
            let nv = relabel[v as usize];
            if nu >= 0 && nv >= 0 {
                new_src.push(nu);
                new_dst.push(nv);
            }
        }
        (keep, (new_src, new_dst))
    });

    let (new_src, new_dst) = new_edges;
    let m = new_src.len();
    let mut flat = Vec::with_capacity(2 * m);
    flat.extend_from_slice(&new_src);
    flat.extend_from_slice(&new_dst);
    let arr = Array2::from_shape_vec((2, m), flat)
        .map_err(|e| PyValueError::new_err(format!("shape: {e}")))?;
    Ok((arr.into_pyarray_bound(py), Array1::from(keep).into_pyarray_bound(py)))
}

/// Random-walk sub-sampling. Mirrors the *intended* behaviour of
/// ``dig.sslgraph.method.contrastive.views_fn.sample.RWSample.do_trans`` with
/// a bug fix: the neighbor frontier actually accumulates (the original Python
/// uses ``set.union`` and discards the return value).
///
/// Returns ``(new_edge_index, kept_indices_sorted)`` where ``kept_indices_sorted``
/// is the sorted list of original node ids; the new edge_index is relabeled.
#[pyfunction]
#[pyo3(name = "rw_sample_subgraph")]
fn py_rw_sample_subgraph<'py>(
    py: Python<'py>,
    edge_index: PyReadonlyArray2<'py, i64>,
    num_nodes: usize,
    sub_num: usize,
    seed: u64,
    add_self_loop: bool,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray1<i64>>)> {
    let ei = edge_index.as_array();
    if ei.shape()[0] != 2 {
        return Err(PyValueError::new_err(format!(
            "edge_index must have shape (2, E); got {:?}",
            ei.shape()
        )));
    }
    if sub_num == 0 || num_nodes == 0 {
        let arr = Array2::<i64>::zeros((2, 0));
        let keep = Array1::<i64>::zeros(0);
        return Ok((arr.into_pyarray_bound(py), keep.into_pyarray_bound(py)));
    }

    let n_edges = ei.shape()[1];
    let src: Vec<i64> = ei.row(0).to_vec();
    let dst: Vec<i64> = ei.row(1).to_vec();
    let sub_num = sub_num.min(num_nodes);

    let (keep_sorted, new_src, new_dst): (Vec<i64>, Vec<i64>, Vec<i64>) =
        py.allow_threads(move || {
            let mut rng = seeded_rng(seed);

            // CSR-style adjacency (outgoing). Optionally include self-loops.
            let extra = if add_self_loop { num_nodes } else { 0 };
            let mut adj: Vec<Vec<i64>> = vec![Vec::new(); num_nodes];
            for i in 0..n_edges {
                let u = src[i];
                let v = dst[i];
                if u >= 0 && (u as usize) < num_nodes && v >= 0 && (v as usize) < num_nodes {
                    adj[u as usize].push(v);
                }
            }
            if add_self_loop {
                for u in 0..num_nodes {
                    adj[u].push(u as i64);
                }
            }
            let _ = extra; // capacity hint only

            // Pick a uniform random seed node, grow the visited set via random
            // moves through the accumulated neighbor frontier.
            let start = (rng.next_u64() as usize) % num_nodes;
            let mut visited: HashSet<i64> = HashSet::new();
            visited.insert(start as i64);
            let mut order: Vec<i64> = vec![start as i64];
            let mut frontier: Vec<i64> = adj[start]
                .iter()
                .copied()
                .filter(|n| !visited.contains(n))
                .collect();
            let mut frontier_set: HashSet<i64> = frontier.iter().copied().collect();

            let mut guard = 0usize;
            while order.len() < sub_num {
                guard += 1;
                if guard > num_nodes * 4 {
                    break;
                }
                if frontier.is_empty() {
                    break;
                }
                let idx = (rng.next_u64() as usize) % frontier.len();
                // O(1) remove by swap_remove
                let pick = frontier.swap_remove(idx);
                frontier_set.remove(&pick);
                if visited.contains(&pick) {
                    continue;
                }
                visited.insert(pick);
                order.push(pick);
                for &n in &adj[pick as usize] {
                    if !visited.contains(&n) && !frontier_set.contains(&n) {
                        frontier.push(n);
                        frontier_set.insert(n);
                    }
                }
            }

            let mut keep_sorted: Vec<i64> = order.iter().copied().collect();
            keep_sorted.sort_unstable();

            let mut relabel: Vec<i64> = vec![-1; num_nodes];
            for (new_id, &old_id) in keep_sorted.iter().enumerate() {
                relabel[old_id as usize] = new_id as i64;
            }
            // Note: subgraph relabel uses original edge_index (without the
            // optionally-added self-loops) so the result matches PyG
            // ``subgraph(mask, edge_index, relabel_nodes=True)``.
            let mut new_src: Vec<i64> = Vec::new();
            let mut new_dst: Vec<i64> = Vec::new();
            for i in 0..n_edges {
                let u = src[i];
                let v = dst[i];
                if u < 0 || v < 0 || (u as usize) >= num_nodes || (v as usize) >= num_nodes {
                    continue;
                }
                let nu = relabel[u as usize];
                let nv = relabel[v as usize];
                if nu >= 0 && nv >= 0 {
                    new_src.push(nu);
                    new_dst.push(nv);
                }
            }
            (keep_sorted, new_src, new_dst)
        });

    let m = new_src.len();
    let mut flat = Vec::with_capacity(2 * m);
    flat.extend_from_slice(&new_src);
    flat.extend_from_slice(&new_dst);
    let arr = Array2::from_shape_vec((2, m), flat)
        .map_err(|e| PyValueError::new_err(format!("shape: {e}")))?;
    Ok((arr.into_pyarray_bound(py), Array1::from(keep_sorted).into_pyarray_bound(py)))
}

/// Drop ``floor(ratio*E)`` random edges and/or add the same count of random
/// edges, then dedupe (col-wise unique). Returns the new ``edge_index``.
#[pyfunction]
#[pyo3(name = "edge_perturb")]
fn py_edge_perturb<'py>(
    py: Python<'py>,
    edge_index: PyReadonlyArray2<'py, i64>,
    num_nodes: usize,
    ratio: f64,
    add: bool,
    drop: bool,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let ei = edge_index.as_array();
    if ei.shape()[0] != 2 {
        return Err(PyValueError::new_err(format!(
            "edge_index must have shape (2, E); got {:?}",
            ei.shape()
        )));
    }
    if !(0.0..=1.0).contains(&ratio) {
        return Err(PyValueError::new_err("ratio must be in [0, 1]"));
    }
    let n_edges = ei.shape()[1];
    let src: Vec<i64> = ei.row(0).to_vec();
    let dst: Vec<i64> = ei.row(1).to_vec();
    let perturb_num = (n_edges as f64 * ratio).floor() as usize;

    let (mut new_src, mut new_dst): (Vec<i64>, Vec<i64>) = py.allow_threads(move || {
        let mut rng = seeded_rng(seed);

        // Drop: build a Bernoulli mask over edge ids (matches PyG dropout_adj semantics
        // closely enough for benchmarking; deterministic per-seed).
        let kept_mask: Vec<bool> = if drop {
            (0..n_edges).map(|_| rng.gen::<f64>() >= ratio).collect()
        } else {
            vec![true; n_edges]
        };

        let mut s: Vec<i64> = Vec::with_capacity(n_edges);
        let mut d: Vec<i64> = Vec::with_capacity(n_edges);
        for i in 0..n_edges {
            if kept_mask[i] {
                s.push(src[i]);
                d.push(dst[i]);
            }
        }

        if add && num_nodes > 0 && perturb_num > 0 {
            for _ in 0..perturb_num {
                let u = (rng.next_u64() as usize) % num_nodes;
                let v = (rng.next_u64() as usize) % num_nodes;
                s.push(u as i64);
                d.push(v as i64);
            }
        }
        (s, d)
    });

    // Column-wise unique (matches torch.unique(dim=1) which sorts lex by (src,dst)).
    if !new_src.is_empty() {
        let mut idx: Vec<usize> = (0..new_src.len()).collect();
        idx.sort_by(|&a, &b| {
            new_src[a]
                .cmp(&new_src[b])
                .then_with(|| new_dst[a].cmp(&new_dst[b]))
        });
        let mut out_s: Vec<i64> = Vec::with_capacity(idx.len());
        let mut out_d: Vec<i64> = Vec::with_capacity(idx.len());
        let mut last: Option<(i64, i64)> = None;
        for i in idx {
            let cur = (new_src[i], new_dst[i]);
            if last != Some(cur) {
                out_s.push(cur.0);
                out_d.push(cur.1);
                last = Some(cur);
            }
        }
        new_src = out_s;
        new_dst = out_d;
    }

    let m = new_src.len();
    let mut flat = Vec::with_capacity(2 * m);
    flat.extend_from_slice(&new_src);
    flat.extend_from_slice(&new_dst);
    let arr = Array2::from_shape_vec((2, m), flat)
        .map_err(|e| PyValueError::new_err(format!("shape: {e}")))?;
    Ok(arr.into_pyarray_bound(py))
}

// --------------------------------------------------------------------------- //
// scaffold bucket split
// --------------------------------------------------------------------------- //

/// Replicates the index/offset math of
/// ``dig.threedgraph.dataset.dataset.key_split`` (sans the ``Subset`` build).
///
/// Self-contained variant: takes raw scaffold ids and an internal Rust ChaCha8
/// seed. Use this when reproducibility against the upstream PyTorch RNG is not
/// required. For *exact* equality to ``key_split`` use
/// :func:`scaffold_bucket_sort` and pass a torch-permuted key array from Python.
///
/// Inputs
///   * ``scaffold_ids`` — per-sample integer id (must be in ``[0, n_keys)``).
///   * ``lengths`` — three integers (train/valid/test target counts).
///   * ``seed`` — RNG seed for the key permutation.
///
/// Returns ``(indices, offsets)`` where ``indices`` is a length-N permutation
/// (sorted by permuted scaffold id) and ``offsets`` is length 4 specifying the
/// split boundaries (``[0, train, train+valid, N]`` after key-boundary rounding).
#[pyfunction]
#[pyo3(name = "scaffold_bucket_split")]
fn py_scaffold_bucket_split<'py>(
    py: Python<'py>,
    scaffold_ids: PyReadonlyArray1<'py, i64>,
    lengths: Vec<i64>,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    if lengths.len() != 3 {
        return Err(PyValueError::new_err(
            "lengths must have length 3 (train/valid/test)",
        ));
    }
    let ids = scaffold_ids.as_array();
    let keys: Vec<i64> = ids.iter().copied().collect();

    let (indices, offsets): (Vec<i64>, Vec<i64>) = py.allow_threads(move || {
        let mut rng = seeded_rng(seed);

        let n_keys: usize = keys.iter().max().copied().unwrap_or(-1).max(-1) as usize + 1;
        let mut perm: Vec<i64> = (0..n_keys as i64).collect();
        for i in (1..n_keys).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            perm.swap(i, j);
        }
        let permuted_keys: Vec<i64> = keys.iter().map(|&k| perm[k as usize]).collect();

        bucket_sort_and_split(&permuted_keys, &lengths)
    });

    Ok((
        Array1::from(indices).into_pyarray_bound(py),
        Array1::from(offsets).into_pyarray_bound(py),
    ))
}

/// Sort + boundary-rounding kernel for ``key_split``. ``permuted_keys`` is
/// expected to be the result of permuting ``scaffold_ids`` through a
/// caller-supplied key permutation (typically ``torch.randperm`` so the result
/// matches the upstream Python bit-for-bit).
///
/// Returns ``(indices, offsets)`` identical in layout to
/// :func:`scaffold_bucket_split`.
#[pyfunction]
#[pyo3(name = "scaffold_bucket_sort")]
fn py_scaffold_bucket_sort<'py>(
    py: Python<'py>,
    permuted_keys: PyReadonlyArray1<'py, i64>,
    lengths: Vec<i64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    if lengths.len() != 3 {
        return Err(PyValueError::new_err(
            "lengths must have length 3 (train/valid/test)",
        ));
    }
    let permuted_keys: Vec<i64> = permuted_keys.as_array().iter().copied().collect();
    let (indices, offsets) =
        py.allow_threads(move || bucket_sort_and_split(&permuted_keys, &lengths));
    Ok((
        Array1::from(indices).into_pyarray_bound(py),
        Array1::from(offsets).into_pyarray_bound(py),
    ))
}

fn bucket_sort_and_split(permuted_keys: &[i64], lengths: &[i64]) -> (Vec<i64>, Vec<i64>) {
    let n = permuted_keys.len();
    // Stable sort indices by permuted key (matches torch.argsort default).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| permuted_keys[a].cmp(&permuted_keys[b]));

    let round = |off: usize| -> usize {
        if n == 0 {
            return 0;
        }
        let max_j = off.min(n.saturating_sub(off));
        for j in 0..max_j {
            if off > j {
                let a = permuted_keys[order[off - j]];
                let b = permuted_keys[order[off - j - 1]];
                if a != b {
                    return off - j;
                }
            }
            if off + j < n && off + j > 0 {
                let a = permuted_keys[order[off + j]];
                let b = permuted_keys[order[off + j - 1]];
                if a != b {
                    return off + j;
                }
            }
        }
        if off < n - off { 0 } else { n }
    };

    let mut offsets: Vec<usize> = Vec::with_capacity(4);
    let mut off = 0usize;
    offsets.push(off);
    for &len in &lengths[..lengths.len() - 1] {
        off = round(off + len as usize);
        offsets.push(off);
    }
    offsets.push(n);

    let indices: Vec<i64> = order.into_iter().map(|x| x as i64).collect();
    let offsets_i64: Vec<i64> = offsets.into_iter().map(|x| x as i64).collect();
    (indices, offsets_i64)
}

// --------------------------------------------------------------------------- //
// MoleculeShardWriter / Reader
// --------------------------------------------------------------------------- //

#[pyclass]
struct MoleculeShardWriter {
    path: String,
    file: Option<BufWriter<File>>,
    n_records: u64,
    expected_records: u64,
    index_table_offset: u64,
    data_section_offset: u64,
    record_offsets: Vec<u64>,
    cursor: u64,
    finalized: bool,
}

#[pymethods]
impl MoleculeShardWriter {
    #[new]
    fn new(path: &str, expected_records: u64) -> PyResult<Self> {
        let mut f = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| PyIOError::new_err(format!("open {path}: {e}")))?;

        // Reserve header.
        f.write_all(&[0u8; HEADER_SIZE])
            .map_err(|e| PyIOError::new_err(format!("write header: {e}")))?;
        // Reserve index table.
        let index_table_offset = HEADER_SIZE as u64;
        let index_table_size = expected_records.saturating_mul(8);
        for _ in 0..expected_records {
            f.write_all(&[0u8; 8])
                .map_err(|e| PyIOError::new_err(format!("write idx slot: {e}")))?;
        }
        let mut cursor = index_table_offset + index_table_size;
        let pad = align_up(cursor as usize, 8) as u64 - cursor;
        if pad > 0 {
            f.write_all(&[0u8; 8][..pad as usize])
                .map_err(|e| PyIOError::new_err(format!("pad after idx: {e}")))?;
            cursor += pad;
        }
        let data_section_offset = cursor;

        Ok(Self {
            path: path.to_string(),
            file: Some(BufWriter::new(f)),
            n_records: 0,
            expected_records,
            index_table_offset,
            data_section_offset,
            record_offsets: Vec::with_capacity(expected_records as usize),
            cursor,
            finalized: false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (idx, smiles, z, pos, edge_index, edge_attr, x, y, mmff=None))]
    fn append_record(
        &mut self,
        py: Python<'_>,
        idx: &str,
        smiles: &str,
        z: PyReadonlyArray1<'_, i64>,
        pos: PyReadonlyArray2<'_, f32>,
        edge_index: PyReadonlyArray2<'_, i64>,
        edge_attr: PyReadonlyArray2<'_, i64>,
        x: PyReadonlyArray2<'_, i64>,
        y: PyReadonlyArray1<'_, f32>,
        mmff: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        if self.finalized {
            return Err(PyValueError::new_err("ShardWriter already finalized"));
        }
        if self.n_records >= self.expected_records {
            return Err(PyValueError::new_err(format!(
                "too many records: expected_records={}",
                self.expected_records
            )));
        }
        let n_atoms = z.as_array().len();
        let pos_a = pos.as_array();
        if pos_a.shape() != [n_atoms, 3] {
            return Err(PyValueError::new_err(format!(
                "pos shape must be (n_atoms, 3); got {:?}",
                pos_a.shape()
            )));
        }
        let ei_a = edge_index.as_array();
        if ei_a.shape()[0] != 2 {
            return Err(PyValueError::new_err(format!(
                "edge_index must be (2, E); got {:?}",
                ei_a.shape()
            )));
        }
        let n_edges = ei_a.shape()[1];
        let ea_a = edge_attr.as_array();
        let n_edge_attr = if ea_a.shape().is_empty() {
            0
        } else if ea_a.shape()[0] == n_edges {
            ea_a.shape()[1]
        } else if ea_a.shape() == [0usize, 0usize] {
            0
        } else {
            return Err(PyValueError::new_err(format!(
                "edge_attr must be (E, n_edge_attr) or empty; got {:?}",
                ea_a.shape()
            )));
        };
        let x_a = x.as_array();
        let n_x_feat = if x_a.shape().is_empty() {
            0
        } else if x_a.shape()[0] == n_atoms {
            x_a.shape()[1]
        } else if x_a.shape() == [0usize, 0usize] {
            0
        } else {
            return Err(PyValueError::new_err(format!(
                "x must be (n_atoms, n_x_feat) or empty; got {:?}",
                x_a.shape()
            )));
        };
        let n_y = y.as_array().len();

        let (mmff_block, flags_local): (Option<MmffBlock>, u32) = if let Some(d) = mmff {
            let block = MmffBlock::from_dict(py, d, n_atoms)?;
            (Some(block), FLAG_HAS_MMFF)
        } else {
            (None, 0)
        };

        let idx_bytes = idx.as_bytes();
        let smiles_bytes = smiles.as_bytes();
        if idx_bytes.len() > u32::MAX as usize || smiles_bytes.len() > u32::MAX as usize {
            return Err(PyValueError::new_err("idx/smiles too long"));
        }

        // Align record start to 8.
        let pad = align_up(self.cursor as usize, 8) as u64 - self.cursor;
        let w = self.file.as_mut().unwrap();
        if pad > 0 {
            w.write_all(&[0u8; 8][..pad as usize])
                .map_err(|e| PyIOError::new_err(format!("pad: {e}")))?;
            self.cursor += pad;
        }
        let rec_off = self.cursor;
        self.record_offsets.push(rec_off);

        let mut written = 0usize;
        let header = [
            n_atoms as u32,
            n_edges as u32,
            n_edge_attr as u32,
            n_x_feat as u32,
            n_y as u32,
            idx_bytes.len() as u32,
            smiles_bytes.len() as u32,
            flags_local,
        ];
        for v in header {
            w.write_all(&v.to_le_bytes())
                .map_err(|e| PyIOError::new_err(format!("hdr: {e}")))?;
            written += 4;
        }
        w.write_all(&0u64.to_le_bytes())
            .map_err(|e| PyIOError::new_err(format!("hdr reserved: {e}")))?;
        written += 8;
        debug_assert_eq!(written, RECORD_HEADER_SIZE);

        // idx + pad8
        w.write_all(idx_bytes)
            .map_err(|e| PyIOError::new_err(format!("idx: {e}")))?;
        written += idx_bytes.len();
        let p = write_pad8(w, written).map_err(|e| PyIOError::new_err(format!("pad idx: {e}")))?;
        written += p;

        // smiles + pad8
        w.write_all(smiles_bytes)
            .map_err(|e| PyIOError::new_err(format!("smiles: {e}")))?;
        written += smiles_bytes.len();
        let p = write_pad8(w, written).map_err(|e| PyIOError::new_err(format!("pad smiles: {e}")))?;
        written += p;

        // z
        let z_a = z.as_array();
        for v in z_a.iter() {
            w.write_all(&v.to_le_bytes()).map_err(|e| PyIOError::new_err(format!("z: {e}")))?;
        }
        written += n_atoms * 8;

        // pos
        for v in pos_a.iter() {
            w.write_all(&v.to_le_bytes()).map_err(|e| PyIOError::new_err(format!("pos: {e}")))?;
        }
        written += n_atoms * 3 * 4;

        // edge_index
        for v in ei_a.iter() {
            w.write_all(&v.to_le_bytes()).map_err(|e| PyIOError::new_err(format!("edge_index: {e}")))?;
        }
        written += 2 * n_edges * 8;

        // edge_attr
        for v in ea_a.iter() {
            w.write_all(&v.to_le_bytes()).map_err(|e| PyIOError::new_err(format!("edge_attr: {e}")))?;
        }
        written += n_edges * n_edge_attr * 8;

        // x
        for v in x_a.iter() {
            w.write_all(&v.to_le_bytes()).map_err(|e| PyIOError::new_err(format!("x: {e}")))?;
        }
        written += n_atoms * n_x_feat * 8;

        // y
        let y_a = y.as_array();
        for v in y_a.iter() {
            w.write_all(&v.to_le_bytes()).map_err(|e| PyIOError::new_err(format!("y: {e}")))?;
        }
        written += n_y * 4;

        // mmff
        if let Some(b) = mmff_block.as_ref() {
            for v in &b.pos1 {
                w.write_all(&v.to_le_bytes()).map_err(|e| PyIOError::new_err(format!("pos1: {e}")))?;
            }
            for v in &b.pos2 {
                w.write_all(&v.to_le_bytes()).map_err(|e| PyIOError::new_err(format!("pos2: {e}")))?;
            }
            for v in &b.pos3 {
                w.write_all(&v.to_le_bytes()).map_err(|e| PyIOError::new_err(format!("pos3: {e}")))?;
            }
            for v in &b.pos4 {
                w.write_all(&v.to_le_bytes()).map_err(|e| PyIOError::new_err(format!("pos4: {e}")))?;
            }
            for v in &b.energies {
                w.write_all(&v.to_le_bytes()).map_err(|e| PyIOError::new_err(format!("energies: {e}")))?;
            }
            written += b.byte_len(n_atoms);
        }

        let p = write_pad8(w, written).map_err(|e| PyIOError::new_err(format!("pad rec: {e}")))?;
        written += p;

        self.cursor += written as u64;
        self.n_records += 1;
        Ok(())
    }

    fn finalize(&mut self) -> PyResult<()> {
        if self.finalized {
            return Ok(());
        }
        let w = self
            .file
            .take()
            .ok_or_else(|| PyIOError::new_err("writer already closed"))?;
        let file_size = self.cursor;
        let mut inner = w
            .into_inner()
            .map_err(|e| PyIOError::new_err(format!("flush: {e}")))?;

        inner
            .seek(SeekFrom::Start(self.index_table_offset))
            .map_err(|e| PyIOError::new_err(format!("seek idx: {e}")))?;
        for off in &self.record_offsets {
            inner
                .write_all(&off.to_le_bytes())
                .map_err(|e| PyIOError::new_err(format!("write idx entry: {e}")))?;
        }

        inner
            .seek(SeekFrom::Start(0))
            .map_err(|e| PyIOError::new_err(format!("seek hdr: {e}")))?;
        inner.write_all(MAGIC).map_err(|e| PyIOError::new_err(format!("magic: {e}")))?;
        inner
            .write_all(&VERSION.to_le_bytes())
            .map_err(|e| PyIOError::new_err(format!("ver: {e}")))?;
        inner
            .write_all(&self.n_records.to_le_bytes())
            .map_err(|e| PyIOError::new_err(format!("n: {e}")))?;
        inner
            .write_all(&0u32.to_le_bytes())
            .map_err(|e| PyIOError::new_err(format!("flags: {e}")))?;
        inner
            .write_all(&self.index_table_offset.to_le_bytes())
            .map_err(|e| PyIOError::new_err(format!("idx off: {e}")))?;
        inner
            .write_all(&self.data_section_offset.to_le_bytes())
            .map_err(|e| PyIOError::new_err(format!("data off: {e}")))?;
        inner
            .write_all(&file_size.to_le_bytes())
            .map_err(|e| PyIOError::new_err(format!("size: {e}")))?;
        inner
            .sync_data()
            .map_err(|e| PyIOError::new_err(format!("sync_data: {e}")))?;

        self.finalized = true;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "MoleculeShardWriter(path={:?}, n_records={}, finalized={})",
            self.path, self.n_records, self.finalized
        )
    }
}

struct MmffBlock {
    pos1: Vec<f32>,
    pos2: Vec<f32>,
    pos3: Vec<f32>,
    pos4: Vec<f32>,
    energies: [f64; 5],
}

impl MmffBlock {
    fn byte_len(&self, n_atoms: usize) -> usize {
        n_atoms * 3 * 4 * 4 + 8 * 5
    }

    fn from_dict(_py: Python<'_>, d: &Bound<'_, PyDict>, n_atoms: usize) -> PyResult<Self> {
        let get_pos = |key: &str| -> PyResult<Vec<f32>> {
            let v = d.get_item(key)?.ok_or_else(|| {
                PyValueError::new_err(format!("mmff missing key: {key}"))
            })?;
            let arr: PyReadonlyArray2<f32> = v.extract()?;
            let a = arr.as_array();
            if a.shape() != [n_atoms, 3] {
                return Err(PyValueError::new_err(format!(
                    "mmff {key} shape mismatch: {:?} vs ({}, 3)",
                    a.shape(),
                    n_atoms
                )));
            }
            Ok(a.iter().copied().collect())
        };
        let get_energy = |key: &str| -> PyResult<f64> {
            let v = d.get_item(key)?.ok_or_else(|| {
                PyValueError::new_err(format!("mmff missing key: {key}"))
            })?;
            let f: f64 = v.extract()?;
            Ok(f)
        };
        let pos1 = get_pos("max1pos_mmff")?;
        let pos2 = get_pos("max2pos_mmff")?;
        let pos3 = get_pos("max3pos_mmff")?;
        let pos4 = get_pos("max4pos_mmff")?;
        let energies = [
            get_energy("min_energy")?,
            get_energy("max1_energy")?,
            get_energy("max2_energy")?,
            get_energy("max3_energy")?,
            get_energy("max4_energy")?,
        ];
        Ok(Self { pos1, pos2, pos3, pos4, energies })
    }
}

#[pyclass]
struct MoleculeShardReader {
    path: String,
    mmap: Mmap,
    n_records: u64,
    index_table_offset: u64,
    #[allow(dead_code)]
    data_section_offset: u64,
    file_size: u64,
}

#[pymethods]
impl MoleculeShardReader {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let f = File::open(path).map_err(|e| PyIOError::new_err(format!("open {path}: {e}")))?;
        let mmap = unsafe { Mmap::map(&f) }
            .map_err(|e| PyIOError::new_err(format!("mmap {path}: {e}")))?;
        if mmap.len() < HEADER_SIZE {
            return Err(PyIOError::new_err(format!(
                "{path}: file too small ({} bytes)",
                mmap.len()
            )));
        }
        if &mmap[..8] != MAGIC {
            return Err(PyIOError::new_err(format!(
                "{path}: bad magic"
            )));
        }
        let version = read_u32(&mmap, 8);
        if version != VERSION {
            return Err(PyIOError::new_err(format!(
                "{path}: unsupported version {version}"
            )));
        }
        let n_records = read_u64(&mmap, 12);
        let index_table_offset = read_u64(&mmap, 24);
        let data_section_offset = read_u64(&mmap, 32);
        let file_size = read_u64(&mmap, 40);
        if file_size as usize > mmap.len() {
            return Err(PyIOError::new_err(format!(
                "{path}: header says file_size={} but mmap len={}",
                file_size,
                mmap.len()
            )));
        }
        Ok(Self {
            path: path.to_string(),
            mmap,
            n_records,
            index_table_offset,
            data_section_offset,
            file_size,
        })
    }

    fn __len__(&self) -> usize {
        self.n_records as usize
    }

    #[getter]
    fn n_records(&self) -> u64 {
        self.n_records
    }

    fn get<'py>(&self, py: Python<'py>, idx: usize) -> PyResult<Bound<'py, PyDict>> {
        if (idx as u64) >= self.n_records {
            return Err(PyIndexError::new_err(format!(
                "index {idx} >= n_records {}",
                self.n_records
            )));
        }
        let entry_off = self.index_table_offset as usize + idx * 8;
        let rec_off = read_u64(&self.mmap, entry_off) as usize;
        if rec_off + RECORD_HEADER_SIZE > self.mmap.len() {
            return Err(PyIOError::new_err("record header out of bounds"));
        }
        let n_atoms = read_u32(&self.mmap, rec_off) as usize;
        let n_edges = read_u32(&self.mmap, rec_off + 4) as usize;
        let n_edge_attr = read_u32(&self.mmap, rec_off + 8) as usize;
        let n_x_feat = read_u32(&self.mmap, rec_off + 12) as usize;
        let n_y = read_u32(&self.mmap, rec_off + 16) as usize;
        let idx_len = read_u32(&self.mmap, rec_off + 20) as usize;
        let smiles_len = read_u32(&self.mmap, rec_off + 24) as usize;
        let flags_local = read_u32(&self.mmap, rec_off + 28);

        let mut cur = rec_off + RECORD_HEADER_SIZE;
        let idx_str = std::str::from_utf8(&self.mmap[cur..cur + idx_len])
            .map_err(|e| PyValueError::new_err(format!("idx utf8: {e}")))?
            .to_string();
        cur += idx_len;
        cur = align_up(cur, 8);

        let smiles_str = std::str::from_utf8(&self.mmap[cur..cur + smiles_len])
            .map_err(|e| PyValueError::new_err(format!("smiles utf8: {e}")))?
            .to_string();
        cur += smiles_len;
        cur = align_up(cur, 8);

        let z_bytes = &self.mmap[cur..cur + n_atoms * 8];
        let z_vec = copy_i64_slice(z_bytes, n_atoms);
        cur += n_atoms * 8;

        let pos_bytes = &self.mmap[cur..cur + n_atoms * 3 * 4];
        let pos_vec = copy_f32_slice(pos_bytes, n_atoms * 3);
        cur += n_atoms * 3 * 4;

        let ei_bytes = &self.mmap[cur..cur + 2 * n_edges * 8];
        let ei_vec = copy_i64_slice(ei_bytes, 2 * n_edges);
        cur += 2 * n_edges * 8;

        let ea_vec = if n_edge_attr > 0 && n_edges > 0 {
            let bytes = &self.mmap[cur..cur + n_edges * n_edge_attr * 8];
            let v = copy_i64_slice(bytes, n_edges * n_edge_attr);
            cur += n_edges * n_edge_attr * 8;
            v
        } else {
            Vec::new()
        };

        let x_vec = if n_x_feat > 0 && n_atoms > 0 {
            let bytes = &self.mmap[cur..cur + n_atoms * n_x_feat * 8];
            let v = copy_i64_slice(bytes, n_atoms * n_x_feat);
            cur += n_atoms * n_x_feat * 8;
            v
        } else {
            Vec::new()
        };

        let y_bytes = &self.mmap[cur..cur + n_y * 4];
        let y_vec = copy_f32_slice(y_bytes, n_y);
        cur += n_y * 4;

        let mut mmff_dict: Option<Bound<'py, PyDict>> = None;
        if flags_local & FLAG_HAS_MMFF != 0 {
            let mut posses: [Vec<f32>; 4] = Default::default();
            for slot in posses.iter_mut() {
                let bytes = &self.mmap[cur..cur + n_atoms * 3 * 4];
                *slot = copy_f32_slice(bytes, n_atoms * 3);
                cur += n_atoms * 3 * 4;
            }
            let e_bytes = &self.mmap[cur..cur + 5 * 8];
            let energies = copy_f64_slice(e_bytes, 5);
            cur += 5 * 8;

            let d = PyDict::new_bound(py);
            let [p1, p2, p3, p4] = posses;
            for (key, v) in [
                ("max1pos_mmff", p1),
                ("max2pos_mmff", p2),
                ("max3pos_mmff", p3),
                ("max4pos_mmff", p4),
            ] {
                let arr = Array2::from_shape_vec((n_atoms, 3), v)
                    .map_err(|e| PyValueError::new_err(format!("mmff {key}: {e}")))?;
                d.set_item(key, arr.into_pyarray_bound(py))?;
            }
            for (key, v) in [
                ("min_energy", energies[0]),
                ("max1_energy", energies[1]),
                ("max2_energy", energies[2]),
                ("max3_energy", energies[3]),
                ("max4_energy", energies[4]),
            ] {
                d.set_item(key, v)?;
            }
            mmff_dict = Some(d);
        }
        let _ = cur; // record may have trailing padding

        let out = PyDict::new_bound(py);
        out.set_item("idx", idx_str)?;
        out.set_item("smiles", smiles_str)?;
        out.set_item("z", Array1::from(z_vec).into_pyarray_bound(py))?;
        let pos_arr = Array2::from_shape_vec((n_atoms, 3), pos_vec)
            .map_err(|e| PyValueError::new_err(format!("pos: {e}")))?;
        out.set_item("pos", pos_arr.into_pyarray_bound(py))?;
        let ei_arr = Array2::from_shape_vec((2, n_edges), ei_vec)
            .map_err(|e| PyValueError::new_err(format!("edge_index: {e}")))?;
        out.set_item("edge_index", ei_arr.into_pyarray_bound(py))?;
        if n_edges > 0 && n_edge_attr > 0 {
            let arr = Array2::from_shape_vec((n_edges, n_edge_attr), ea_vec)
                .map_err(|e| PyValueError::new_err(format!("edge_attr: {e}")))?;
            out.set_item("edge_attr", arr.into_pyarray_bound(py))?;
        }
        if n_atoms > 0 && n_x_feat > 0 {
            let arr = Array2::from_shape_vec((n_atoms, n_x_feat), x_vec)
                .map_err(|e| PyValueError::new_err(format!("x: {e}")))?;
            out.set_item("x", arr.into_pyarray_bound(py))?;
        }
        out.set_item("y", Array1::from(y_vec).into_pyarray_bound(py))?;
        if let Some(d) = mmff_dict {
            out.set_item("mmff", d)?;
        }
        Ok(out)
    }

    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    #[getter]
    fn file_size(&self) -> u64 {
        self.file_size
    }
}

// --------------------------------------------------------------------------- //
// format_info()
// --------------------------------------------------------------------------- //

#[pyfunction]
fn format_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("magic", PyBytes::new_bound(py, MAGIC))?;
    d.set_item("version", VERSION)?;
    d.set_item("header_size", HEADER_SIZE)?;
    d.set_item("record_header_size", RECORD_HEADER_SIZE)?;
    d.set_item("flag_has_mmff", FLAG_HAS_MMFF)?;
    Ok(d)
}

// --------------------------------------------------------------------------- //
// module init
// --------------------------------------------------------------------------- //

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_uniform_sample_subgraph, m)?)?;
    m.add_function(wrap_pyfunction!(py_rw_sample_subgraph, m)?)?;
    m.add_function(wrap_pyfunction!(py_edge_perturb, m)?)?;
    m.add_function(wrap_pyfunction!(py_scaffold_bucket_split, m)?)?;
    m.add_function(wrap_pyfunction!(py_scaffold_bucket_sort, m)?)?;
    m.add_function(wrap_pyfunction!(format_info, m)?)?;
    m.add_class::<MoleculeShardWriter>()?;
    m.add_class::<MoleculeShardReader>()?;
    Ok(())
}
