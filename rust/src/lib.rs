//! Python bindings: compiled into the `objectnat._rust` extension module.

pub mod catchment;
pub mod visibility;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use catchment::{ring_runs, trace_label_rings, CountGrid, GridSpec};
use visibility::{Obstacles, Scratch, Seg, Wall};

/// Closed rings flattened to ``(k, 2)`` coordinates plus per-observer offsets.
type RaggedRings<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>);

/// ``(ny, nx)`` uint32 counts and optional float64 weight sums.
type CountArrays<'py> = (Bound<'py, PyArray2<u32>>, Option<Bound<'py, PyArray2<f64>>>);

/// Closed rings, their offsets and labels.
type LabelRings<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<u8>>,
);

type Scene = (Vec<[f64; 2]>, Vec<f64>, Vec<Wall>);

/// Validate and convert the observer / obstacle arrays shared by the visibility functions.
fn parse_scene(
    observers: &PyReadonlyArray2<'_, f64>,
    radii: &PyReadonlyArray1<'_, f64>,
    segments: &PyReadonlyArray2<'_, f64>,
    quad_segs: usize,
    one_sided: Option<&PyReadonlyArray1<'_, bool>>,
) -> PyResult<Scene> {
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
    let flags: Vec<bool> = match one_sided {
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
    Ok((points, radii, walls))
}

/// Median view radius, used to size the obstacle index.
fn median(radii: &[f64]) -> f64 {
    let mut sorted = radii.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted.get(sorted.len() / 2).copied().unwrap_or(0.0)
}

fn observer_ring(
    index: &Obstacles,
    p: [f64; 2],
    r: f64,
    quad_segs: usize,
    sc: &mut Scratch,
) -> Vec<[f64; 2]> {
    if !p[0].is_finite() || !p[1].is_finite() || !r.is_finite() {
        return Vec::new();
    }
    index.visibility(p, r, quad_segs, sc)
}

/// Run `f` on a dedicated pool of `n_threads` workers (`None` or `0`: the global pool).
fn in_pool<T: Send>(n_threads: Option<usize>, f: impl FnOnce() -> T + Send) -> T {
    match n_threads {
        Some(n) if n > 0 => match rayon::ThreadPoolBuilder::new().num_threads(n).build() {
            Ok(pool) => pool.install(f),
            Err(_) => f(),
        },
        _ => f(),
    }
}

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
    let (points, radii, walls) =
        parse_scene(&observers, &radii, &segments, quad_segs, one_sided.as_ref())?;

    let rings: Vec<Vec<[f64; 2]>> = py.detach(|| {
        let index = Obstacles::from_walls(walls, median(&radii));
        in_pool(n_threads, || {
            points
                .par_iter()
                .zip(radii.par_iter())
                .map_init(Scratch::default, |sc, (p, r)| {
                    observer_ring(&index, *p, *r, quad_segs, sc)
                })
                .collect()
        })
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

/// Count, for every cell of a regular grid, the observers whose visibility polygon covers its centre.
///
/// Args:
///     observers, radii, segments, quad_segs, n_threads, one_sided: as in :func:`visibility_polygons`.
///     origin: ``(x0, y0)`` lower-left corner of the grid.
///     cell_size: cell side length (CRS units).
///     shape: ``(ny, nx)`` number of rows and columns.
///     weights: optional ``(n,)`` float64 observer weights; their per-cell sums are returned too.
///
/// Returns:
///     ``(counts, weight_sums)``: ``(ny, nx)`` uint32 observer counts and ``(ny, nx)`` float64 weight
///     sums (``None`` without ``weights``); row ``0`` is the bottom row (``y0``).
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (observers, radii, segments, origin, cell_size, shape, weights=None, quad_segs=16, n_threads=None, one_sided=None))]
fn visibility_count_grid<'py>(
    py: Python<'py>,
    observers: PyReadonlyArray2<'py, f64>,
    radii: PyReadonlyArray1<'py, f64>,
    segments: PyReadonlyArray2<'py, f64>,
    origin: (f64, f64),
    cell_size: f64,
    shape: (usize, usize),
    weights: Option<PyReadonlyArray1<'py, f64>>,
    quad_segs: usize,
    n_threads: Option<usize>,
    one_sided: Option<PyReadonlyArray1<'py, bool>>,
) -> PyResult<CountArrays<'py>> {
    let (points, radii, walls) =
        parse_scene(&observers, &radii, &segments, quad_segs, one_sided.as_ref())?;
    if !(cell_size.is_finite() && cell_size > 0.0) {
        return Err(PyValueError::new_err("cell_size must be a positive number"));
    }
    if !(origin.0.is_finite() && origin.1.is_finite()) {
        return Err(PyValueError::new_err("origin must be finite"));
    }
    let weights: Option<Vec<f64>> = weights.map(|w| w.as_array().iter().copied().collect());
    if let Some(w) = &weights {
        if w.len() != points.len() {
            return Err(PyValueError::new_err(
                "weights must have one value per observer",
            ));
        }
    }
    let (ny, nx) = shape;
    let spec = GridSpec {
        x0: origin.0,
        y0: origin.1,
        cell: cell_size,
        nx,
        ny,
    };

    let (counts, sums) = py.detach(|| {
        let index = Obstacles::from_walls(walls, median(&radii));
        let grid = CountGrid::new(spec, weights.is_some());
        in_pool(n_threads, || {
            points.par_iter().enumerate().for_each_init(
                || (Scratch::default(), Vec::new(), Vec::new()),
                |(sc, xs, runs), (k, p)| {
                    let w = weights.as_ref().map_or(1.0, |w| w[k]);
                    if weights.is_some() && !(w.is_finite() && w > 0.0) {
                        return;
                    }
                    let ring = observer_ring(&index, *p, radii[k], quad_segs, sc);
                    ring_runs(&ring, grid.spec(), xs, runs);
                    grid.add_runs(runs, w);
                },
            );
            grid.finish()
        })
    });
    let counts = numpy::ndarray::Array2::from_shape_vec((ny, nx), counts)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let sums = sums
        .map(|v| numpy::ndarray::Array2::from_shape_vec((ny, nx), v))
        .transpose()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((counts.into_pyarray(py), sums.map(|a| a.into_pyarray(py))))
}

/// Outlines of the 4-connected regions of every non-zero label of a grid.
///
/// Args:
///     labels: ``(ny, nx)`` uint8 labels, ``0`` is background; row ``0`` is the bottom row.
///     origin: ``(x0, y0)`` lower-left corner of the grid.
///     cell_size: cell side length.
///
/// Returns:
///     ``(coords, offsets, ring_labels)``: closed rings flattened to ``(k, 2)`` coordinates, ring ``i``
///     spans ``coords[offsets[i]:offsets[i + 1]]`` and outlines a region of label ``ring_labels[i]``.
///     Exteriors are counter-clockwise, holes clockwise.
#[pyfunction]
fn label_rings<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray2<'py, u8>,
    origin: (f64, f64),
    cell_size: f64,
) -> PyResult<LabelRings<'py>> {
    let arr = labels.as_array();
    let (ny, nx) = arr.dim();
    let flat: Vec<u8> = arr.iter().copied().collect();
    let rings = py.detach(|| trace_label_rings(&flat, nx, ny));
    let mut coords: Vec<f64> = Vec::new();
    let mut offsets: Vec<i64> = vec![0];
    let mut ring_labels: Vec<u8> = Vec::with_capacity(rings.len());
    for (label, ring) in &rings {
        for &(i, j) in ring.iter().chain(ring.first()) {
            coords.push(origin.0 + i as f64 * cell_size);
            coords.push(origin.1 + j as f64 * cell_size);
        }
        offsets.push((coords.len() / 2) as i64);
        ring_labels.push(*label);
    }
    let coords = numpy::ndarray::Array2::from_shape_vec((coords.len() / 2, 2), coords)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        coords.into_pyarray(py),
        offsets.into_pyarray(py),
        ring_labels.into_pyarray(py),
    ))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(visibility_polygons, m)?)?;
    m.add_function(wrap_pyfunction!(visibility_count_grid, m)?)?;
    m.add_function(wrap_pyfunction!(label_rings, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
