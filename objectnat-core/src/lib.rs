//! Python bindings for the ObjectNat native core.

pub mod visibility;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use visibility::{Obstacles, Scratch, Seg, Wall};

/// Closed rings flattened to ``(k, 2)`` coordinates plus per-observer offsets.
type RaggedRings<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>);

/// Batch visibility polygons.
///
/// Args:
///     observers: ``(n, 2)`` float64 observer coordinates (projected CRS).
///     radii: ``(n,)`` float64 view radii.
///     segments: ``(m, 4)`` float64 obstacle segments ``x1, y1, x2, y2``.
///     one_sided: optional ``(m,)`` bool; ``True`` marks walls of valid polygon rings oriented with
///         the interior on the left, which are skipped when seen from the interior side.
///     quad_segs: segments per quarter circle of the view disc (GEOS ``quad_segs``).
///     n_threads: worker threads; ``None`` or ``0`` uses all cores.
///
/// Returns:
///     ``(coords, offsets)``: ``coords`` is a ``(k, 2)`` array of closed counter-clockwise rings,
///     ring ``i`` spans ``coords[offsets[i]:offsets[i + 1]]``; an empty span means "no polygon".
#[pyfunction]
#[pyo3(signature = (observers, radii, segments, quad_segs=16, n_threads=None, one_sided=None))]
fn visibility_polygons<'py>(
    py: Python<'py>,
    observers: PyReadonlyArray2<'py, f64>,
    radii: PyReadonlyArray1<'py, f64>,
    segments: PyReadonlyArray2<'py, f64>,
    quad_segs: usize,
    n_threads: Option<usize>,
    one_sided: Option<PyReadonlyArray1<'py, bool>>,
) -> PyResult<RaggedRings<'py>> {
    let obs = observers.as_array();
    let rad = radii.as_array();
    let seg = segments.as_array();
    if obs.ncols() != 2 {
        return Err(PyValueError::new_err("observers must have shape (n, 2)"));
    }
    if seg.ncols() != 4 && seg.nrows() > 0 {
        return Err(PyValueError::new_err("segments must have shape (m, 4)"));
    }
    if rad.len() != obs.nrows() {
        return Err(PyValueError::new_err(
            "radii must have one value per observer",
        ));
    }
    if quad_segs == 0 {
        return Err(PyValueError::new_err("quad_segs must be >= 1"));
    }
    let points: Vec<[f64; 2]> = obs.rows().into_iter().map(|r| [r[0], r[1]]).collect();
    let radii: Vec<f64> = rad.iter().copied().collect();
    let flags: Vec<bool> = match &one_sided {
        Some(f) => f.as_array().iter().copied().collect(),
        None => vec![false; seg.nrows()],
    };
    if flags.len() != seg.nrows() {
        return Err(PyValueError::new_err(
            "one_sided must have one value per segment",
        ));
    }
    let walls: Vec<Wall> = seg
        .rows()
        .into_iter()
        .zip(flags)
        .map(|(r, one_sided)| Wall {
            seg: Seg {
                a: [r[0], r[1]],
                b: [r[2], r[3]],
            },
            one_sided,
        })
        .filter(|w| w.seg.a.iter().chain(w.seg.b.iter()).all(|v| v.is_finite()))
        .collect();

    let rings: Vec<Vec<[f64; 2]>> = py.detach(|| {
        let mut sorted = radii.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let typical = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
        let index = Obstacles::from_walls(walls, typical);
        let run = || {
            points
                .par_iter()
                .zip(radii.par_iter())
                .map_init(Scratch::default, |sc, (p, r)| {
                    if !p[0].is_finite() || !p[1].is_finite() || !r.is_finite() {
                        return Vec::new();
                    }
                    index.visibility(*p, *r, quad_segs, sc)
                })
                .collect()
        };
        match n_threads {
            Some(n) if n > 0 => rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map(|pool| pool.install(run))
                .unwrap_or_else(|_| run()),
            _ => run(),
        }
    });

    let mut offsets: Vec<i64> = Vec::with_capacity(rings.len() + 1);
    offsets.push(0);
    let total: usize = rings
        .iter()
        .map(|r| if r.len() >= 3 { r.len() + 1 } else { 0 })
        .sum();
    let mut flat: Vec<f64> = Vec::with_capacity(total * 2);
    for ring in &rings {
        if ring.len() >= 3 {
            for p in ring {
                flat.extend_from_slice(p);
            }
            flat.extend_from_slice(&ring[0]);
        }
        offsets.push((flat.len() / 2) as i64);
    }
    let coords = numpy::ndarray::Array2::from_shape_vec((flat.len() / 2, 2), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((coords.into_pyarray(py), offsets.into_pyarray(py)))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(visibility_polygons, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
