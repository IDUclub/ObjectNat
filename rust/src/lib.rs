//! Python bindings: compiled into the `objectnat._rust` extension module.

pub mod catchment;
pub mod insolation;
pub mod noise;
pub mod provision;
pub mod visibility;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use catchment::{ring_runs, trace_label_rings, CountGrid, GridSpec};
use visibility::{Obstacles, Scratch, Seg, Wall};

/// Closed rings flattened to ``(k, 2)`` coordinates plus per-observer offsets.
type RaggedRings<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>);

/// Sunlit counts and longest sunlit runs, ``(n, days)`` uint32 each.
type DayStats<'py> = (Bound<'py, PyArray2<u32>>, Bound<'py, PyArray2<u32>>);

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
///     reflections: include first-order reflections from walls within ``reflection_max_dist`` of a source;
///     floor_db: paths below it are skipped; reflection_reach: ``(n,)`` reach of each source's image
///     sources (defaults to ``max_dist``).
///
/// Returns:
///     ``(ny, nx)`` float64 energy, ``sum(10 ** (L / 10))``; 0 where nothing arrives.
#[pyfunction]
#[pyo3(signature = (sources, source_db, air_db_per_m, veg_db_per_m, max_dist, segments, one_sided, wall_loss_db,
    tree_segments, grid, active, reflections=true, reflection_max_dist=50.0, floor_db=0.0, n_threads=None,
    reflection_reach=None))]
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
    reflection_max_dist: f64,
    floor_db: f64,
    n_threads: Option<usize>,
    reflection_reach: Option<PyReadonlyArray1<'py, f64>>,
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

    let image_reach: Vec<f64> = match &reflection_reach {
        Some(r) => r.as_array().iter().copied().collect(),
        None => reach.iter().copied().collect(),
    };
    if image_reach.len() != n {
        return Err(PyValueError::new_err(
            "reflection_reach must have one value per source",
        ));
    }
    let sources: Vec<noise::Source> = (0..n)
        .map(|k| noise::Source {
            at: [src[[k, 0]], src[[k, 1]]],
            level_db: level[k],
            air_db_per_m: air[k],
            veg_db_per_m: veg[k],
            max_dist: reach[k],
            reflection_reach: image_reach[k],
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
            reflection_max_dist,
            floor_energy: 10f64.powf(floor_db / 10.0),
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

/// Per receiver and day: sunlit sun positions and the longest run of consecutive sunlit positions, both
/// ``(n, days)`` uint32, for receivers ``(n, 3)`` [x, y, z] and sun positions ``(t, 3)``
/// [dir_x, dir_y, tan(elevation)] in time order, day ``d`` being ``suns[day_offsets[d]:day_offsets[d+1]]``,
/// among walls ``(m, 4)`` with absolute roof heights ``tops`` and an optional terrain grid ``(ny, nx)``
/// placed by ``terrain_grid = (x0, y0, sx, sy)``. Facade receivers pass their outward unit ``normals``
/// ``(n, 2)``: the sun then counts only while ``normal . sun_dir > min_facing``.
#[pyfunction]
#[pyo3(signature = (receivers, suns, day_offsets, segments, tops, terrain=None, terrain_grid=None, normals=None,
    min_facing=0.0, n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn insolation_day_stats<'py>(
    py: Python<'py>,
    receivers: PyReadonlyArray2<'py, f64>,
    suns: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
    segments: PyReadonlyArray2<'py, f64>,
    tops: PyReadonlyArray1<'py, f64>,
    terrain: Option<PyReadonlyArray2<'py, f64>>,
    terrain_grid: Option<(f64, f64, f64, f64)>,
    normals: Option<PyReadonlyArray2<'py, f64>>,
    min_facing: f64,
    n_threads: Option<usize>,
) -> PyResult<DayStats<'py>> {
    let rec = receivers.as_array();
    let sun = suns.as_array();
    let segs = segments.as_array();
    let tops = tops.as_array();
    if (rec.nrows() > 0 && rec.ncols() != 3)
        || (sun.nrows() > 0 && sun.ncols() != 3)
        || (segs.nrows() > 0 && segs.ncols() != 4)
        || tops.len() != segs.nrows()
    {
        return Err(PyValueError::new_err(
            "receivers must be (n, 3), suns (t, 3), segments (m, 4) with one top height each",
        ));
    }
    let points: Vec<[f64; 3]> = rec.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
    let sun_list: Vec<insolation::Sun> = sun
        .rows()
        .into_iter()
        .map(|r| insolation::Sun {
            dir: [r[0], r[1]],
            tan_el: r[2],
        })
        .collect();
    let walls: Vec<Seg> = segs
        .rows()
        .into_iter()
        .map(|r| Seg {
            a: [r[0], r[1]],
            b: [r[2], r[3]],
        })
        .collect();
    let tops: Vec<f64> = tops.iter().copied().collect();
    let terrain = match (terrain, terrain_grid) {
        (Some(z), Some((x0, y0, sx, sy))) => {
            let z = z.as_array();
            let (ny, nx) = z.dim();
            if nx < 2 || ny < 2 || !(sx > 0.0 && sy > 0.0) {
                return Err(PyValueError::new_err(
                    "terrain must be at least 2 x 2 with positive cell sizes",
                ));
            }
            Some(insolation::Terrain {
                x0,
                y0,
                sx,
                sy,
                nx,
                ny,
                z: z.iter().copied().collect(),
            })
        }
        (None, _) => None,
        (Some(_), None) => {
            return Err(PyValueError::new_err("terrain needs terrain_grid"));
        }
    };
    let offsets: Vec<usize> = day_offsets
        .as_array()
        .iter()
        .map(|&v| usize::try_from(v).unwrap_or(usize::MAX))
        .collect();
    if offsets.is_empty()
        || offsets.windows(2).any(|w| w[0] > w[1])
        || *offsets.last().unwrap_or(&0) > sun_list.len()
    {
        return Err(PyValueError::new_err(
            "day_offsets must be non-decreasing positions within suns",
        ));
    }
    let normals: Option<Vec<[f64; 2]>> = match normals {
        Some(a) => {
            let a = a.as_array();
            if a.nrows() != points.len() || (a.nrows() > 0 && a.ncols() != 2) {
                return Err(PyValueError::new_err(
                    "normals must be (n, 2), one per receiver",
                ));
            }
            Some(a.rows().into_iter().map(|r| [r[0], r[1]]).collect())
        }
        None => None,
    };
    let n = points.len();
    let days = offsets.len() - 1;
    let (counts, longest) = py.detach(|| {
        let scene = insolation::Scene::new(walls, tops, terrain);
        let facing = normals.as_deref().map(|v| (v, min_facing));
        let run = || insolation::day_stats(&scene, &points, &sun_list, &offsets, facing);
        match n_threads {
            Some(k) if k > 0 => rayon::ThreadPoolBuilder::new()
                .num_threads(k)
                .build()
                .map(|pool| pool.install(run))
                .unwrap_or_else(|_| run()),
            _ => run(),
        }
    });
    let shape = |v| {
        numpy::ndarray::Array2::from_shape_vec((n, days), v)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    };
    Ok((
        shape(counts)?.into_pyarray(py),
        shape(longest)?.into_pyarray(py),
    ))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(insolation_day_stats, m)?)?;
    m.add_function(wrap_pyfunction!(noise_grid_energy, m)?)?;
    m.add_function(wrap_pyfunction!(visibility_polygons, m)?)?;
    m.add_function(wrap_pyfunction!(visibility_count_grid, m)?)?;
    m.add_function(wrap_pyfunction!(label_rings, m)?)?;
    m.add_function(wrap_pyfunction!(provision_allocate, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
