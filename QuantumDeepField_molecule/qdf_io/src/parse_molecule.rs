//! Parse a single ``train.txt`` / ``val.txt`` molecule block (same contract as
//! ``train/preprocess.py::_parse_molecule_block``) without NumPy / SciPy.
//!
//! Atomic orbital **names** are returned as strings; Python maps them through
//! ``create_orbitals(..., orbital_dict)`` so dataset-global AO indices match.

use ndarray::Array2;

/// Same element order as ``preprocess.py::all_atoms`` (Z = index + 1).
const ELEMENTS: &[&str] = &[
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
    "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
    "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
    "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg",
    "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

#[inline]
fn sym_to_z(sym: &str) -> Option<i64> {
    ELEMENTS
        .iter()
        .position(|&s| s == sym)
        .map(|i| (i + 1) as i64)
}

/// Reproduce ``inner_outer = [int(b) for b in basis_set[:-1].replace('-', '')]`` then
/// ``inner, outer = inner_outer[0], sum(inner_outer[1:])``.
fn basis_primitives(basis_set: &str) -> Result<(usize, usize), String> {
    let t = basis_set.trim();
    if t.len() < 2 {
        return Err("basis_set too short".into());
    }
    let no_last = &t[..t.len() - 1];
    let mut digits: Vec<usize> = Vec::new();
    for ch in no_last.chars().filter(|c| *c != '-') {
        let d = ch
            .to_digit(10)
            .ok_or_else(|| format!("non-digit in basis_set {basis_set:?}"))? as usize;
        digits.push(d);
    }
    if digits.is_empty() {
        return Err(format!("no digits in basis_set {basis_set:?}"));
    }
    let inner = digits[0];
    let outer: usize = digits[1..].iter().sum();
    Ok((inner, outer))
}

pub struct ParsedMolecule {
    pub idx: String,
    pub atomic_coords: Array2<f64>,
    pub orbital_coords: Array2<f64>,
    pub atomic_numbers: Array2<i64>,
    pub ao_labels: Vec<String>,
    pub quantum_numbers: Vec<f32>,
    pub n_electrons: f32,
    pub property_values: Option<Array2<f32>>,
}

pub fn parse_block(block: &str, basis_set: &str, property: bool) -> Result<ParsedMolecule, String> {
    let (inner, outer) = basis_primitives(basis_set)?;
    let lines: Vec<&str> = block.trim().split('\n').collect();
    if lines.is_empty() {
        return Err("empty block".into());
    }
    let idx = lines[0].to_string();

    let (atom_lines, prop_row) = if property {
        if lines.len() < 3 {
            return Err("property=True but block has fewer than 3 lines".into());
        }
        let prop = *lines.last().unwrap();
        (&lines[1..lines.len() - 1], Some(prop))
    } else {
        (&lines[1..], None)
    };

    let mut ac_list: Vec<[f64; 3]> = Vec::new();
    let mut an_list: Vec<i64> = Vec::new();
    let mut ao_labels: Vec<String> = Vec::new();
    let mut oc_list: Vec<[f64; 3]> = Vec::new();
    let mut qn: Vec<f32> = Vec::new();
    let mut n_electrons: i64 = 0;

    for line in atom_lines {
        let mut it = line.split_whitespace();
        let sym = it.next().ok_or_else(|| format!("bad atom line: {line}"))?;
        let x: f64 = it
            .next()
            .ok_or_else(|| format!("bad atom line: {line}"))?
            .parse()
            .map_err(|e| format!("float x: {e}"))?;
        let y: f64 = it
            .next()
            .ok_or_else(|| format!("bad atom line: {line}"))?
            .parse()
            .map_err(|e| format!("float y: {e}"))?;
        let z: f64 = it
            .next()
            .ok_or_else(|| format!("bad atom line: {line}"))?
            .parse()
            .map_err(|e| format!("float z: {e}"))?;
        if it.next().is_some() {
            return Err(format!("extra tokens in atom line: {line}"));
        }

        let atomic_number = sym_to_z(sym).ok_or_else(|| format!("unknown element {sym}"))?;
        n_electrons += atomic_number;
        ac_list.push([x, y, z]);
        an_list.push(atomic_number);

        let xyz = [x, y, z];
        if atomic_number <= 2 {
            for i in 0..outer {
                ao_labels.push(format!("{sym}1s{i}"));
                oc_list.push(xyz);
                qn.push(1.0);
            }
        } else {
            for i in 0..inner {
                ao_labels.push(format!("{sym}1s{i}"));
                oc_list.push(xyz);
                qn.push(1.0);
            }
            for i in 0..outer {
                ao_labels.push(format!("{sym}2s{i}"));
                oc_list.push(xyz);
                qn.push(2.0);
            }
            for i in 0..outer {
                ao_labels.push(format!("{sym}2p{i}"));
                oc_list.push(xyz);
                qn.push(2.0);
            }
        }
    }

    let n_atoms = ac_list.len();
    if n_atoms == 0 {
        return Err("no atoms in block".into());
    }

    let mut atomic_coords = Array2::<f64>::zeros((n_atoms, 3));
    for (i, p) in ac_list.iter().enumerate() {
        atomic_coords[[i, 0]] = p[0];
        atomic_coords[[i, 1]] = p[1];
        atomic_coords[[i, 2]] = p[2];
    }

    let n_orb = oc_list.len();
    let mut orbital_coords = Array2::<f64>::zeros((n_orb, 3));
    for (i, p) in oc_list.iter().enumerate() {
        orbital_coords[[i, 0]] = p[0];
        orbital_coords[[i, 1]] = p[1];
        orbital_coords[[i, 2]] = p[2];
    }

    let mut atomic_numbers = Array2::<i64>::zeros((n_atoms, 1));
    for (i, z) in an_list.iter().enumerate() {
        atomic_numbers[[i, 0]] = *z;
    }

    let property_values = if let Some(pr) = prop_row {
        let vals: Result<Vec<f32>, _> = pr.split_whitespace().map(|s| s.parse::<f32>()).collect();
        let vals = vals.map_err(|e| format!("property float: {e}"))?;
        if vals.is_empty() {
            return Err("empty property row".into());
        }
        let arr = Array2::from_shape_vec((1, vals.len()), vals)
            .map_err(|e| format!("property reshape: {e}"))?;
        Some(arr)
    } else {
        None
    };

    Ok(ParsedMolecule {
        idx,
        atomic_coords,
        orbital_coords,
        atomic_numbers,
        ao_labels,
        quantum_numbers: qn,
        n_electrons: n_electrons as f32,
        property_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_631g() {
        assert_eq!(basis_primitives("6-31G").unwrap(), (6, 4));
    }

    #[test]
    fn water_like_block() {
        let block = "testidx\n\
            O 0.0 0.0 0.0\n\
            H 0.0 0.0 1.0\n\
            H 1.0 0.0 0.0\n\
            -1.0 2.0";
        let m = parse_block(block, "6-31G", true).unwrap();
        assert_eq!(m.idx, "testidx");
        assert_eq!(m.atomic_coords.nrows(), 3);
        assert!(m.ao_labels.iter().any(|s| s.starts_with("O")));
        assert_eq!(m.property_values.as_ref().unwrap().shape(), &[1, 2]);
    }
}
