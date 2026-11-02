//! Python bindings: compiled into the `objectnat._rust` extension module.

pub mod noise;
pub mod provision;
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

/// Sparse building-service flows as ``(rows, cols, units)``.
type Flows<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
);

/// Gravity-probabilistic service provision on sparse OD entries.
///
/// Args:
///     rows, cols: ``(k,)`` int64 building / service positions of the OD entries.
///     cost: ``(k,)`` float64 OD costs (non-finite entries are unreachable).
///     demand: ``(n_buildings,)`` int64 demand per building.
///     capacity: ``(n_services,)`` int64 capacity per service.
///     threshold: normative cost; links above ``3 * threshold`` are never used.
///     seed: random seed; the result does not depend on ``n_threads``.
///     n_threads: worker threads; ``None`` or ``0`` uses all cores.
///
/// Returns:
///     ``(rows, cols, units)`` of allocated flows; a pair can repeat (sum duplicates).
#[pyfunction]
#[pyo3(signature = (rows, cols, cost, demand, capacity, threshold, seed=0, n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn provision_allocate<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<'py, i64>,
    cols: PyReadonlyArray1<'py, i64>,
    cost: PyReadonlyArray1<'py, f64>,
    demand: PyReadonlyArray1<'py, i64>,
    capacity: PyReadonlyArray1<'py, i64>,
    threshold: f64,
    seed: u64,
    n_threads: Option<usize>,
) -> PyResult<Flows<'py>> {
    let (rows, cols, cost) = (rows.as_array(), cols.as_array(), cost.as_array());
    if rows.len() != cols.len() || rows.len() != cost.len() {
        return Err(PyValueError::new_err(
            "rows, cols and cost must have the same length",
        ));
    }
    let demand: Vec<i64> = demand.as_array().iter().copied().collect();
    let capacity: Vec<i64> = capacity.as_array().iter().copied().collect();
    let to_index =
        |values: numpy::ndarray::ArrayView1<i64>, n: usize, name: &str| -> PyResult<Vec<u32>> {
            values
                .iter()
                .map(|&v| {
                    if v < 0 || v as usize >= n {
                        Err(PyValueError::new_err(format!(
                            "{name} index {v} is out of range"
                        )))
                    } else {
                        Ok(v as u32)
                    }
                })
                .collect()
        };
    let rows = to_index(rows, demand.len(), "building")?;
    let cols = to_index(cols, capacity.len(), "service")?;
    let cost: Vec<f64> = cost.iter().copied().collect();
    if threshold.is_nan() || threshold < 0.0 {
        return Err(PyValueError::new_err("threshold must be >= 0"));
    }

    let (b, s, v) = py.detach(|| {
        let run = || {
            provision::allocate(&provision::ProvisionInput {
                rows: &rows,
                cols: &cols,
                cost: &cost,
                demand: &demand,
                capacity: &capacity,
                threshold,
                seed,
            })
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
    let b: Vec<i64> = b.into_iter().map(i64::from).collect();
    let s: Vec<i64> = s.into_iter().map(i64::from).collect();
    Ok((b.into_pyarray(py), s.into_pyarray(py), v.into_pyarray(py)))
}

/// Sound energy per receiver of a regular grid (direct paths + first-order reflections).
///
/// Args:
///     sources: ``(n, 2)`` source coordinates; ``source_db``, ``air_db_per_m``, ``veg_db_per_m``,
///         ``max_dist``: ``(n,)`` level at 1 m, air absorption, vegetation rate and reach per source.
///     segments: ``(m, 4)`` walls; ``one_sided``: ``(m,)`` back-face flags; ``wall_loss_db``: ``(m,)``.
///     tree_segments: ``(t, 4)`` boundaries of non-overlapping vegetation polygons (may be empty).
///     grid: ``(x0, y0, step, nx, ny)``; ``active``: ``(ny * nx,)`` bool, receivers to compute.
///     reflections: include first-order reflections; floor_db: paths below it are skipped.
///
/// Returns:
///     ``(ny, nx)`` float64 energy, ``sum(10 ** (L / 10))``; 0 where nothing arrives.
#[pyfunction]
#[pyo3(signature = (sources, source_db, air_db_per_m, veg_db_per_m, max_dist, segments, one_sided, wall_loss_db,
    tree_segments, grid, active, reflections=true, floor_db=0.0, n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn noise_grid_energy<'py>(
    py: Python<'py>,
    sources: PyReadonlyArray2<'py, f64>,
    source_db: PyReadonlyArray1<'py, f64>,
    air_db_per_m: PyReadonlyArray1<'py, f64>,
    veg_db_per_m: PyReadonlyArray1<'py, f64>,
    max_dist: PyReadonlyArray1<'py, f64>,
    segments: PyReadonlyArray2<'py, f64>,
    one_sided: PyReadonlyArray1<'py, bool>,
    wall_loss_db: PyReadonlyArray1<'py, f64>,
    tree_segments: PyReadonlyArray2<'py, f64>,
    grid: (f64, f64, f64, usize, usize),
    active: PyReadonlyArray1<'py, bool>,
    reflections: bool,
    floor_db: f64,
    n_threads: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let src = sources.as_array();
    let n = src.nrows();
    let (level, air, veg, reach) = (
        source_db.as_array(),
        air_db_per_m.as_array(),
        veg_db_per_m.as_array(),
        max_dist.as_array(),
    );
    if src.ncols() != 2 || level.len() != n || air.len() != n || veg.len() != n || reach.len() != n
    {
        return Err(PyValueError::new_err(
            "sources must be (n, 2) with one level/air/veg/reach value each",
        ));
    }
    let segs = segments.as_array();
    let flags = one_sided.as_array();
    let losses = wall_loss_db.as_array();
    if (segs.nrows() > 0 && segs.ncols() != 4)
        || flags.len() != segs.nrows()
        || losses.len() != segs.nrows()
    {
        return Err(PyValueError::new_err(
            "segments must be (m, 4) with one flag and one loss per wall",
        ));
    }
    let trees = tree_segments.as_array();
    if trees.nrows() > 0 && trees.ncols() != 4 {
        return Err(PyValueError::new_err("tree_segments must be (t, 4)"));
    }
    let (x0, y0, step, nx, ny) = grid;
    let active: Vec<bool> = active.as_array().iter().copied().collect();
    if active.len() != nx * ny || step.is_nan() || step <= 0.0 {
        return Err(PyValueError::new_err(
            "active must have nx * ny values and step must be > 0",
        ));
    }

    let sources: Vec<noise::Source> = (0..n)
        .map(|k| noise::Source {
            at: [src[[k, 0]], src[[k, 1]]],
            level_db: level[k],
            air_db_per_m: air[k],
            veg_db_per_m: veg[k],
            max_dist: reach[k],
        })
        .collect();
    let typical = reach.iter().copied().fold(0.0, f64::max);
    let mut walls = Vec::with_capacity(segs.nrows());
    let mut wall_losses = Vec::with_capacity(segs.nrows());
    for k in 0..segs.nrows() {
        let seg = Seg {
            a: [segs[[k, 0]], segs[[k, 1]]],
            b: [segs[[k, 2]], segs[[k, 3]]],
        };
        if seg.a != seg.b && seg.a.iter().chain(seg.b.iter()).all(|v| v.is_finite()) {
            walls.push(Wall {
                seg,
                one_sided: flags[k],
            });
            wall_losses.push(losses[k]);
        }
    }
    let tree_segs: Vec<Seg> = trees
        .rows()
        .into_iter()
        .map(|r| Seg {
            a: [r[0], r[1]],
            b: [r[2], r[3]],
        })
        .filter(|s| s.a != s.b)
        .collect();

    let energy = py.detach(|| {
        let walls = Obstacles::from_walls(walls, typical);
        let trees = (!tree_segs.is_empty()).then(|| Obstacles::new(tree_segs, typical));
        let input = noise::NoiseInput {
            walls: &walls,
            wall_loss_db: &wall_losses,
            trees: trees.as_ref(),
            sources: &sources,
            grid: noise::GridSpec {
                x0,
                y0,
                step,
                nx,
                ny,
            },
            active: &active,
            reflections,
            floor_db,
        };
        let run = || noise::receiver_energy(&input);
        match n_threads {
            Some(t) if t > 0 => rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .map(|pool| pool.install(run))
                .unwrap_or_else(|_| run()),
            _ => run(),
        }
    });
    let energy = numpy::ndarray::Array2::from_shape_vec((ny, nx), energy)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(energy.into_pyarray(py))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(noise_grid_energy, m)?)?;
    m.add_function(wrap_pyfunction!(visibility_polygons, m)?)?;
    m.add_function(wrap_pyfunction!(provision_allocate, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
