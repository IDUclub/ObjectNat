//! Python bindings: compiled into the `objectnat._rust` extension module.

mod buffers;
pub mod catchment;
mod grid;
pub mod insolation;
pub mod noise;
pub mod provision;
pub mod roofs;
pub mod viewshed;
pub mod visibility;

use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use catchment::{ring_runs, trace_label_rings, CountGrid, GridSpec};
use visibility::{Obstacles, Scratch, Seg, Wall};

/// Closed rings flattened to ``(k, 2)`` coordinates plus per-observer offsets.
type RaggedRings<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>);

/// Flat cell indices of all observers and ``(k + 1)`` offsets.
type RaggedCells<'py> = (Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<i64>>);

/// Sunlit counts and longest sunlit runs, ``(n, days)`` uint32 each.
type DayStats<'py> = (Bound<'py, PyArray2<u32>>, Bound<'py, PyArray2<u32>>);

type TreeDayStats<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<u32>>,
    Bound<'py, PyArray2<u32>>,
);

/// ``(ny, nx)`` uint32 counts and optional float64 weight sums.
type CountArrays<'py> = (Bound<'py, PyArray2<u32>>, Option<Bound<'py, PyArray2<f64>>>);

/// Closed rings, their offsets and labels.
type LabelRings<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<u8>>,
);

type PolygonRings = Vec<Vec<Vec<[f64; 2]>>>;
type RoofArrays<'py> = (
    (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>),
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<u8>>,
);

/// Rings are indexed by vertex offsets; polygons by ring offsets. Empty batches use offsets [0].
fn polygon_arrays(
    xy: &PyReadonlyArray2<'_, f64>,
    ring_offsets: &PyReadonlyArray1<'_, i64>,
    poly_offsets: &PyReadonlyArray1<'_, i64>,
) -> PyResult<PolygonRings> {
    let xy = xy.as_array();
    if xy.ncols() != 2 || xy.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "xy must have shape (n, 2) with finite coordinates",
        ));
    }
    let offsets =
        |values: &PyReadonlyArray1<'_, i64>, end: usize, name: &str| -> PyResult<Vec<usize>> {
            let values: Vec<usize> = values
                .as_array()
                .iter()
                .map(|&v| usize::try_from(v).unwrap_or(usize::MAX))
                .collect();
            if values.first() != Some(&0)
                || values.last() != Some(&end)
                || values.windows(2).any(|w| w[0] > w[1])
            {
                return Err(PyValueError::new_err(format!(
                    "{name} must start at 0, be non-decreasing and end at {end}"
                )));
            }
            Ok(values)
        };
    let ro = offsets(ring_offsets, xy.nrows(), "ring_offsets")?;
    let po = offsets(poly_offsets, ro.len() - 1, "poly_offsets")?;
    if ro.windows(2).any(|w| w[1] - w[0] < 3) || po.windows(2).any(|w| w[0] == w[1]) {
        return Err(PyValueError::new_err(
            "each polygon needs an exterior ring; each ring needs at least 3 coordinates",
        ));
    }
    Ok(po
        .windows(2)
        .map(|p| {
            (p[0]..p[1])
                .map(|r| {
                    (ro[r]..ro[r + 1])
                        .map(|k| [xy[[k, 0]], xy[[k, 1]]])
                        .collect()
                })
                .collect()
        })
        .collect())
}

type PointPairs<'py> = (Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<i64>>);
type GridClassification<'py> = (Bound<'py, PyArray1<u8>>, PointPairs<'py>);

fn hit_arrays<'py>(py: Python<'py>, hits: grid::Hits) -> PointPairs<'py> {
    let pairs = |items: Vec<(usize, usize)>| {
        numpy::ndarray::Array2::from_shape_fn((2, items.len()), |(row, col)| {
            if row == 0 {
                items[col].0 as i64
            } else {
                items[col].1 as i64
            }
        })
        .into_pyarray(py)
    };
    (pairs(hits.inside), pairs(hits.uncertain))
}

/// Conservative regular-grid classifier: 0 outside, 1 inside, 2 uncertain (resolve in GEOS).
/// With union=true return union states; otherwise return (point, polygon) inside/uncertain pairs.
/// Inputs use the roof polygon-array contract, in the original projected coordinates.
#[pyfunction]
#[pyo3(signature = (xy, ring_offsets, poly_offsets, gx, gy, *, union=false, n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn classify_grid<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray2<'py, f64>,
    ring_offsets: PyReadonlyArray1<'py, i64>,
    poly_offsets: PyReadonlyArray1<'py, i64>,
    gx: PyReadonlyArray1<'py, f64>,
    gy: PyReadonlyArray1<'py, f64>,
    union: bool,
    n_threads: Option<usize>,
) -> PyResult<GridClassification<'py>> {
    let polygons = polygon_arrays(&xy, &ring_offsets, &poly_offsets)?;
    let gx = gx.as_array().to_vec();
    let gy = gy.as_array().to_vec();
    for axis in [&gx, &gy] {
        if axis.iter().any(|v| !v.is_finite()) || axis.windows(2).any(|w| w[0] > w[1]) {
            return Err(PyValueError::new_err(
                "grid axes must be finite and non-decreasing",
            ));
        }
    }
    let count = gx
        .len()
        .checked_mul(gy.len())
        .ok_or_else(|| PyValueError::new_err("grid is too large"))?;
    let (states, hits) = py.detach(move || {
        in_pool(n_threads, || {
            let hits = grid::grid_pairs(&polygons, &gx, &gy);
            if union {
                (grid::area_states(&hits, count), grid::Hits::default())
            } else {
                (Vec::new(), hits)
            }
        })
    });
    Ok((states.into_pyarray(py), hit_arrays(py, hits)))
}

/// Conservative arbitrary-point classifier, indexed by polygon bounding boxes.
/// Return inside/uncertain (point, polygon) pairs; uncertain pairs must be resolved in GEOS.
#[pyfunction]
#[pyo3(signature = (xy, ring_offsets, poly_offsets, points, *, n_threads=None))]
fn classify_points<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray2<'py, f64>,
    ring_offsets: PyReadonlyArray1<'py, i64>,
    poly_offsets: PyReadonlyArray1<'py, i64>,
    points: PyReadonlyArray2<'py, f64>,
    n_threads: Option<usize>,
) -> PyResult<PointPairs<'py>> {
    let polygons = polygon_arrays(&xy, &ring_offsets, &poly_offsets)?;
    let points = points.as_array();
    if points.ncols() != 2 || points.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(
            "points must have shape (n, 2) with finite coordinates",
        ));
    }
    let points: Vec<_> = points.rows().into_iter().map(|p| [p[0], p[1]]).collect();
    let hits = py.detach(move || in_pool(n_threads, || grid::point_pairs(&polygons, &points)));
    Ok(hit_arrays(py, hits))
}

/// Batch piecewise-planar roofs over projected XY footprints.
///
/// Args:
///     xy: (n, 2) float64, finite projected coordinates. Rings may be open or closed, either winding.
///     ring_offsets: (r+1,) int64 vertex offsets, starting at 0 and ending at n.
///     poly_offsets: (p+1,) int64 ring offsets, starting at 0 and ending at r;
///         the exterior precedes the holes of each polygon. Empty batches use [0] for both offsets.
///     shape: (p,) uint8 codes: 0 flat, 1 skillion, 2 gabled, 3 hipped, 4 pyramidal,
///         5 cone, 6 dome, 7 onion, 8 half-dome, 9 round, 10 half-hipped, 11 side_hipped,
///         12 quadruple_saltbox, 13 mansard, 14 gambrel, 15 saltbox, 16 many.
///     eave, ridge: (p,) float64 finite absolute roof-start and roof-top heights.
///     direction: optional (p,) float64 downslope compass bearings in radians clockwise from
///         grid north; NaN means absent. Overrides across. Without a bearing, the ridge follows
///         the longer side of the minimum rotated rectangle (shorter side if across is true).
///     across: optional (p,) bool, default false.
///     n_threads: worker threads; None or 0 uses the global pool. Work releases the GIL.
///
/// Returns:
///     ((face_xyz, face_offsets), face_owner, edges, edge_owner, status, used_shape).
///     face_xyz: (k, 3) float64; face_offsets: (f+1,) int64. Each slice is an open CCW
///     planar ring without holes (implicit closing edge). Holed planar faces are triangulated;
///     concave footprints may also have extra coplanar subdivisions. face_owner: (f,) int64.
///     edges: (e, 6) float64 [x1,y1,z1,x2,y2,z2], unique undirected subdivision edges including
///     boundary edges split at every facet junction; edge_owner: (e,) int64. Owners are zero-based
///     input polygon indices. Faces and edges are grouped in input order; no ordering within a roof
///     is promised. status: (p,) uint8, 0 requested shape, 1 approximation or fallback.
///     used_shape: (p,) uint8 gives the actual base shape (0..9).
///
/// Hipped roofs use geo-straight-skeleton 0.1.1 (pure Rust f64, MIT/Apache-2.0, supports holes),
/// validated for planarity, coverage, overlap and height bounds; failures fall back to gabled.
/// Non-star-shaped or holed pyramidal/cone/dome/onion/half-dome roofs fall back to hipped,
/// then gabled if necessary. Dome variants use the same four-tier quarter-ellipse approximation;
/// round uses four strips per side. Codes 10..13 approximate hipped; 14..15 approximate gabled;
/// many and unknown codes use flat. Zero-area inputs or ridge <= eave use flat at max(eave,ridge).
#[pyfunction]
#[pyo3(signature = (xy, ring_offsets, poly_offsets, shape, eave, ridge, direction=None, across=None, n_threads=None))]
#[allow(clippy::too_many_arguments)]
fn roof_mesh<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray2<'py, f64>,
    ring_offsets: PyReadonlyArray1<'py, i64>,
    poly_offsets: PyReadonlyArray1<'py, i64>,
    shape: PyReadonlyArray1<'py, u8>,
    eave: PyReadonlyArray1<'py, f64>,
    ridge: PyReadonlyArray1<'py, f64>,
    direction: Option<PyReadonlyArray1<'py, f64>>,
    across: Option<PyReadonlyArray1<'py, bool>>,
    n_threads: Option<usize>,
) -> PyResult<RoofArrays<'py>> {
    let polygons = polygon_arrays(&xy, &ring_offsets, &poly_offsets)?;
    let p = polygons.len();
    let (shape, eave, ridge) = (shape.as_array(), eave.as_array(), ridge.as_array());
    if shape.len() != p
        || eave.len() != p
        || ridge.len() != p
        || eave.iter().chain(ridge.iter()).any(|z| !z.is_finite())
    {
        return Err(PyValueError::new_err(
            "shape, eave and ridge must have shape (p,); heights must be finite",
        ));
    }
    let direction = direction
        .map(|a| a.as_array().iter().copied().collect::<Vec<_>>())
        .unwrap_or_else(|| vec![f64::NAN; p]);
    let across = across
        .map(|a| a.as_array().iter().copied().collect::<Vec<_>>())
        .unwrap_or_else(|| vec![false; p]);
    if direction.len() != p || direction.iter().any(|v| v.is_infinite()) || across.len() != p {
        return Err(PyValueError::new_err(
            "direction and across must have shape (p,); direction must be finite or NaN",
        ));
    }
    let specs: Vec<_> = (0..p)
        .map(|k| roofs::RoofSpec {
            shape: shape[k],
            eave: eave[k],
            ridge: ridge[k],
            direction: direction[k],
            across: across[k],
        })
        .collect();
    let meshes = py.detach(|| run_in_pool(n_threads, || roofs::roof_meshes(&polygons, &specs)));
    let (mut xyz, mut offsets, mut owners, mut edges, mut edge_owners, mut status, mut used) = (
        Vec::new(),
        vec![0i64],
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    );
    for (owner, mesh) in meshes.into_iter().enumerate() {
        for face in mesh.faces {
            xyz.extend(face.iter().flatten().copied());
            offsets.push((xyz.len() / 3) as i64);
            owners.push(owner as i64);
        }
        for edge in mesh.edges {
            edges.extend(edge);
            edge_owners.push(owner as i64);
        }
        status.push(mesh.status);
        used.push(mesh.used_shape);
    }
    let xyz = numpy::ndarray::Array2::from_shape_vec((xyz.len() / 3, 3), xyz)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let edges = numpy::ndarray::Array2::from_shape_vec((edges.len() / 6, 6), edges)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        (xyz.into_pyarray(py), offsets.into_pyarray(py)),
        owners.into_pyarray(py),
        edges.into_pyarray(py),
        edge_owners.into_pyarray(py),
        status.into_pyarray(py),
        used.into_pyarray(py),
    ))
}

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
///     sources and diffracted paths (defaults to ``max_dist``).
///     corners: ``(c, 8)`` edges for diffraction, ``x, y`` of the corner, of its previous and next
///         vertex and of a point just outside it (``None``: no diffraction); corner_links: ``(c, 2)``
///         int64, the corner rows of the previous and next vertex when they are corners of the same
///         outline, ``-1`` otherwise; wavelength: ``(n,)`` wavelength per source, metres, required with
///         ``corners``.
///
/// Returns:
///     ``(ny, nx)`` float64 energy, ``sum(10 ** (L / 10))``; 0 where nothing arrives.
#[pyfunction]
#[pyo3(signature = (sources, source_db, air_db_per_m, veg_db_per_m, max_dist, segments, one_sided, wall_loss_db,
    tree_segments, grid, active, reflections=true, reflection_max_dist=50.0, floor_db=0.0, n_threads=None,
    reflection_reach=None, corners=None, corner_links=None, wavelength=None))]
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
    corners: Option<PyReadonlyArray2<'py, f64>>,
    corner_links: Option<PyReadonlyArray2<'py, i64>>,
    wavelength: Option<PyReadonlyArray1<'py, f64>>,
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
    let edges: Vec<noise::Corner> = match &corners {
        Some(c) => {
            let c = c.as_array();
            let nc = c.nrows();
            if nc > 0 && c.ncols() != 8 || c.iter().any(|v| !v.is_finite()) {
                return Err(PyValueError::new_err(
                    "corners must be (c, 8) with finite values",
                ));
            }
            let links = corner_links.as_ref().map(|l| l.as_array());
            if links
                .as_ref()
                .is_some_and(|l| l.nrows() != nc || (nc > 0 && l.ncols() != 2))
            {
                return Err(PyValueError::new_err(
                    "corner_links must be (c, 2), one row per corner",
                ));
            }
            let link = |k: usize, side: usize| {
                links.as_ref().and_then(|l| {
                    let v = l[[k, side]];
                    (v >= 0 && (v as usize) < nc).then_some(v as u32)
                })
            };
            (0..nc)
                .map(|k| noise::Corner {
                    at: [c[[k, 0]], c[[k, 1]]],
                    prev: [c[[k, 2]], c[[k, 3]]],
                    next: [c[[k, 4]], c[[k, 5]]],
                    outside: [c[[k, 6]], c[[k, 7]]],
                    links: [link(k, 0), link(k, 1)],
                })
                .collect()
        }
        None => Vec::new(),
    };
    let lambda: Vec<f64> = match &wavelength {
        Some(w) => w.as_array().iter().copied().collect(),
        None if edges.is_empty() => vec![f64::NAN; n],
        None => {
            return Err(PyValueError::new_err(
                "corners need a wavelength per source",
            ))
        }
    };
    if lambda.len() != n {
        return Err(PyValueError::new_err(
            "wavelength must have one value per source",
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
            wavelength: lambda[k],
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
            corners: &edges,
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
/// ``tops`` may be ``(m,)`` or ``(m, 2)`` endpoint heights. Optional keyword-only ``bottoms``
/// (``(m,)``, finite or -inf) exclude wall crossings at or below the bottom. Undersides are supplied
/// together as ``under_xy`` (``(k, 2)``), ``under_ring_offsets`` (``(r+1,)`` int64 vertex offsets),
/// ``under_poly_offsets`` (``(p+1,)`` int64 ring offsets), and finite ``under_z`` (``(p,)``).
/// Rings use even-odd filling, arbitrary winding and optional closing vertices.
#[pyfunction]
#[pyo3(signature = (receivers, suns, day_offsets, segments, tops, terrain=None, terrain_grid=None, normals=None,
    min_facing=0.0, n_threads=None, *, bottoms=None, under_xy=None, under_ring_offsets=None, under_poly_offsets=None, under_z=None))]
#[allow(clippy::too_many_arguments)]
fn insolation_day_stats<'py>(
    py: Python<'py>,
    receivers: PyReadonlyArray2<'py, f64>,
    suns: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
    segments: PyReadonlyArray2<'py, f64>,
    tops: PyReadonlyArrayDyn<'py, f64>,
    terrain: Option<PyReadonlyArray2<'py, f64>>,
    terrain_grid: Option<(f64, f64, f64, f64)>,
    normals: Option<PyReadonlyArray2<'py, f64>>,
    min_facing: f64,
    n_threads: Option<usize>,
    bottoms: Option<PyReadonlyArray1<'py, f64>>,
    under_xy: Option<PyReadonlyArray2<'py, f64>>,
    under_ring_offsets: Option<PyReadonlyArray1<'py, i64>>,
    under_poly_offsets: Option<PyReadonlyArray1<'py, i64>>,
    under_z: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<DayStats<'py>> {
    let rec = receivers.as_array();
    let sun = suns.as_array();
    if (rec.nrows() > 0 && rec.ncols() != 3) || (sun.nrows() > 0 && sun.ncols() != 3) {
        return Err(PyValueError::new_err(
            "receivers must be (n, 3) and suns (t, 3)",
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
    let scene = scene_from_arrays(
        segments,
        tops,
        terrain,
        terrain_grid,
        bottoms,
        under_xy,
        under_ring_offsets,
        under_poly_offsets,
        under_z,
        None,
    )?;
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

/// Tree-weighted direct sun: (weighted f64 counts, opaque-only u32 counts, longest u32 runs),
/// all (n, days). Crown records (k, 8) are [x, y, z, r, h, mu_on, mu_off, shape]: shape 0
/// is a spheroid centred at z with vertical semi-axis h; shape 1 is a cone based at z with
/// full height h. All coordinates are absolute in the scene CRS; extinction is per metre.
/// leaf_on is one uint8 0/1 per simulated day; evergreen records use mu_off = mu_on.
/// Opaque arrays, terrain and facade normals use the insolation_day_stats contract unchanged.
#[pyfunction]
#[pyo3(signature = (receivers, suns, day_offsets, segments, tops, trees, leaf_on, sunlit_threshold=0.5, terrain=None, terrain_grid=None, normals=None,
    min_facing=0.0, n_threads=None, *, bottoms=None, under_xy=None, under_ring_offsets=None, under_poly_offsets=None, under_z=None))]
#[allow(clippy::too_many_arguments)]
fn insolation_day_stats_trees<'py>(
    py: Python<'py>,
    receivers: PyReadonlyArray2<'py, f64>,
    suns: PyReadonlyArray2<'py, f64>,
    day_offsets: PyReadonlyArray1<'py, i64>,
    segments: PyReadonlyArray2<'py, f64>,
    tops: PyReadonlyArrayDyn<'py, f64>,
    trees: PyReadonlyArray2<'py, f64>,
    leaf_on: PyReadonlyArray1<'py, u8>,
    sunlit_threshold: f64,
    terrain: Option<PyReadonlyArray2<'py, f64>>,
    terrain_grid: Option<(f64, f64, f64, f64)>,
    normals: Option<PyReadonlyArray2<'py, f64>>,
    min_facing: f64,
    n_threads: Option<usize>,
    bottoms: Option<PyReadonlyArray1<'py, f64>>,
    under_xy: Option<PyReadonlyArray2<'py, f64>>,
    under_ring_offsets: Option<PyReadonlyArray1<'py, i64>>,
    under_poly_offsets: Option<PyReadonlyArray1<'py, i64>>,
    under_z: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<TreeDayStats<'py>> {
    let rec = receivers.as_array();
    let sun = suns.as_array();
    if (rec.nrows() > 0 && rec.ncols() != 3) || (sun.nrows() > 0 && sun.ncols() != 3) {
        return Err(PyValueError::new_err(
            "receivers must be (n, 3) and suns (t, 3)",
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
    let tree_array = trees.as_array();
    if tree_array.ncols() != 8
        || tree_array.rows().into_iter().any(|r| {
            r.iter().any(|v| !v.is_finite())
                || r[3] <= 0.0
                || r[4] <= 0.0
                || r[5] < 0.0
                || r[6] < 0.0
                || (r[7] != 0.0 && r[7] != 1.0)
        })
    {
        return Err(PyValueError::new_err("trees must be finite (k, 8) crown records with positive r/h, nonnegative mu and shape 0 or 1"));
    }
    if !sunlit_threshold.is_finite() || !(0.0..=1.0).contains(&sunlit_threshold) {
        return Err(PyValueError::new_err(
            "sunlit_threshold must be finite and in [0, 1]",
        ));
    }
    if rec.ncols() != 3
        || sun.ncols() != 3
        || rec.iter().chain(sun.iter()).any(|v| !v.is_finite())
        || sun
            .rows()
            .into_iter()
            .any(|r| (r[0].hypot(r[1]) - 1.0).abs() > 1e-8)
    {
        return Err(PyValueError::new_err(
            "receivers/suns must be finite (n, 3)/(t, 3), with unit horizontal sun directions",
        ));
    }
    let crowns = tree_array
        .rows()
        .into_iter()
        .map(|r| insolation::Crown {
            x: r[0],
            y: r[1],
            z: r[2],
            r: r[3],
            h: r[4],
            mu_on: r[5],
            mu_off: r[6],
            shape: r[7] as u8,
        })
        .collect();
    let scene = scene_from_arrays(
        segments,
        tops,
        terrain,
        terrain_grid,
        bottoms,
        under_xy,
        under_ring_offsets,
        under_poly_offsets,
        under_z,
        Some(crowns),
    )?;
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
    let leaves: Vec<u8> = leaf_on.as_array().iter().copied().collect();
    if leaves.len() != offsets.len() - 1 || leaves.iter().any(|&v| v > 1) {
        return Err(PyValueError::new_err(
            "leaf_on must contain one 0/1 value per day",
        ));
    }
    if !min_facing.is_finite()
        || normals
            .as_ref()
            .is_some_and(|v| v.iter().flatten().any(|x| !x.is_finite()))
    {
        return Err(PyValueError::new_err(
            "normals and min_facing must be finite",
        ));
    }
    let n = points.len();
    let days = offsets.len() - 1;
    let (weighted, counts, longest) = py.detach(|| {
        let facing = normals.as_deref().map(|v| (v, min_facing));
        run_in_pool(n_threads, || {
            insolation::day_stats_trees(
                &scene,
                &points,
                &sun_list,
                &offsets,
                &leaves,
                sunlit_threshold,
                facing,
            )
        })
    });
    let shape = |v| {
        numpy::ndarray::Array2::from_shape_vec((n, days), v)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    };
    let weighted = numpy::ndarray::Array2::from_shape_vec((n, days), weighted)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        weighted.into_pyarray(py),
        shape(counts)?.into_pyarray(py),
        shape(longest)?.into_pyarray(py),
    ))
}

/// Lines of sight: 1 for every target ``(m, 3)`` visible from its eye ``eyes[owner]`` (``(k, 3)``, x y z)
/// among walls ``(w, 4)`` with absolute roof heights ``tops`` and an optional terrain grid.
/// ``tops`` accepts ``(w,)`` or ``(w, 2)`` endpoint heights. Keyword-only ``bottoms`` and the four
/// ``under_*`` arrays describe raised volumes; their encoding is documented by ``insolation_day_stats``.
#[pyfunction]
#[pyo3(signature = (eyes, targets, owner, segments, tops, terrain=None, terrain_grid=None, n_threads=None, *, bottoms=None, under_xy=None, under_ring_offsets=None, under_poly_offsets=None, under_z=None))]
#[allow(clippy::too_many_arguments)]
fn line_of_sight<'py>(
    py: Python<'py>,
    eyes: PyReadonlyArray2<'py, f64>,
    targets: PyReadonlyArray2<'py, f64>,
    owner: PyReadonlyArray1<'py, i64>,
    segments: PyReadonlyArray2<'py, f64>,
    tops: PyReadonlyArrayDyn<'py, f64>,
    terrain: Option<PyReadonlyArray2<'py, f64>>,
    terrain_grid: Option<(f64, f64, f64, f64)>,
    n_threads: Option<usize>,
    bottoms: Option<PyReadonlyArray1<'py, f64>>,
    under_xy: Option<PyReadonlyArray2<'py, f64>>,
    under_ring_offsets: Option<PyReadonlyArray1<'py, i64>>,
    under_poly_offsets: Option<PyReadonlyArray1<'py, i64>>,
    under_z: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let (eyes, targets, owner) = (eyes.as_array(), targets.as_array(), owner.as_array());
    if (eyes.nrows() > 0 && eyes.ncols() != 3)
        || (targets.nrows() > 0 && targets.ncols() != 3)
        || owner.len() != targets.nrows()
        || owner.iter().any(|&k| k < 0 || k as usize >= eyes.nrows())
    {
        return Err(PyValueError::new_err(
            "eyes must be (k, 3), targets (m, 3) with one valid eye index each",
        ));
    }
    let eye_pts: Vec<[f64; 3]> = eyes
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();
    let target_pts: Vec<[f64; 3]> = targets
        .rows()
        .into_iter()
        .map(|r| [r[0], r[1], r[2]])
        .collect();
    let owner: Vec<usize> = owner.iter().map(|&k| k as usize).collect();
    let scene = scene_from_arrays(
        segments,
        tops,
        terrain,
        terrain_grid,
        bottoms,
        under_xy,
        under_ring_offsets,
        under_poly_offsets,
        under_z,
        None,
    )?;
    let seen = py.detach(|| {
        let run = || insolation::line_of_sight(&scene, &eye_pts, &target_pts, &owner);
        match n_threads {
            Some(k) if k > 0 => rayon::ThreadPoolBuilder::new()
                .num_threads(k)
                .build()
                .map(|pool| pool.install(run))
                .unwrap_or_else(|_| run()),
            _ => run(),
        }
    });
    Ok(seen.into_pyarray(py))
}

/// Walls and optional raised-volume details as an indexed 2.5D scene.
#[allow(clippy::too_many_arguments)]
fn scene_from_arrays(
    segments: PyReadonlyArray2<'_, f64>,
    tops: PyReadonlyArrayDyn<'_, f64>,
    terrain: Option<PyReadonlyArray2<'_, f64>>,
    terrain_grid: Option<(f64, f64, f64, f64)>,
    bottoms: Option<PyReadonlyArray1<'_, f64>>,
    under_xy: Option<PyReadonlyArray2<'_, f64>>,
    under_ring_offsets: Option<PyReadonlyArray1<'_, i64>>,
    under_poly_offsets: Option<PyReadonlyArray1<'_, i64>>,
    under_z: Option<PyReadonlyArray1<'_, f64>>,
    trees: Option<Vec<insolation::Crown>>,
) -> PyResult<insolation::Scene> {
    let segs = segments.as_array();
    let tops = tops.as_array();
    if segs.ncols() != 4 {
        return Err(PyValueError::new_err("segments must have shape (m, 4)"));
    }
    let m = segs.nrows();
    if tops.shape() != [m] && tops.shape() != [m, 2] {
        return Err(PyValueError::new_err("tops must have shape (m,) or (m, 2)"));
    }
    let bottoms = bottoms.map(|v| v.as_array().iter().copied().collect::<Vec<_>>());
    if bottoms
        .as_ref()
        .is_some_and(|v| v.len() != m || v.iter().any(|z| z.is_nan() || *z == f64::INFINITY))
    {
        return Err(PyValueError::new_err(
            "bottoms must have shape (m,) and contain finite heights or -inf",
        ));
    }
    let undersides = match (under_xy, under_ring_offsets, under_poly_offsets, under_z) {
        (None, None, None, None) => Vec::new(),
        (Some(xy), Some(ro), Some(po), Some(z)) => {
            let rings = polygon_arrays(&xy, &ro, &po)?;
            let z = z.as_array();
            if z.len() != rings.len() || z.iter().any(|v| !v.is_finite()) {
                return Err(PyValueError::new_err("under_z must have shape (p,) with finite heights"));
            }
            rings.into_iter().zip(z).map(|(rings, &z)| insolation::Underside { rings, z }).collect()
        }
        _ => return Err(PyValueError::new_err("under_xy, under_ring_offsets, under_poly_offsets and under_z must be supplied together")),
    };
    let walls: Vec<Seg> = segs
        .rows()
        .into_iter()
        .map(|r| Seg {
            a: [r[0], r[1]],
            b: [r[2], r[3]],
        })
        .collect();
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
        (Some(_), None) => return Err(PyValueError::new_err("terrain needs terrain_grid")),
    };
    if trees.is_none() && tops.ndim() == 1 && bottoms.is_none() && undersides.is_empty() {
        return Ok(insolation::Scene::new(
            walls,
            tops.iter().copied().collect(),
            terrain,
        ));
    }
    let heights = if tops.ndim() == 1 {
        tops.iter().map(|&z| [z, z]).collect()
    } else {
        (0..m).map(|k| [tops[[k, 0]], tops[[k, 1]]]).collect()
    };
    Ok(match trees {
        Some(trees) => insolation::Scene::with_details_and_trees(
            walls, heights, bottoms, undersides, terrain, trees,
        ),
        None => insolation::Scene::with_details(walls, heights, bottoms, undersides, terrain),
    })
}

fn run_in_pool<T: Send>(n_threads: Option<usize>, run: impl FnOnce() -> T + Send) -> T {
    match n_threads {
        Some(k) if k > 0 => match rayon::ThreadPoolBuilder::new().num_threads(k).build() {
            Ok(pool) => pool.install(run),
            Err(_) => run(),
        },
        _ => run(),
    }
}

fn cell_grid(grid: (f64, f64, f64, usize, usize)) -> PyResult<viewshed::CellGrid> {
    let (x0, y0, cell, nx, ny) = grid;
    if cell.is_nan() || cell <= 0.0 || nx == 0 || ny == 0 {
        return Err(PyValueError::new_err(
            "grid must be (x0, y0, cell > 0, nx > 0, ny > 0)",
        ));
    }
    Ok(viewshed::CellGrid {
        x0,
        y0,
        cell,
        nx,
        ny,
    })
}

fn eyes_and_radii(
    eyes: &PyReadonlyArray2<'_, f64>,
    radii: &PyReadonlyArray1<'_, f64>,
) -> PyResult<(Vec<[f64; 3]>, Vec<f64>)> {
    let (e, r) = (eyes.as_array(), radii.as_array());
    if (e.nrows() > 0 && e.ncols() != 3) || r.len() != e.nrows() {
        return Err(PyValueError::new_err(
            "eyes must be (k, 3) with one radius each",
        ));
    }
    Ok((
        e.rows().into_iter().map(|v| [v[0], v[1], v[2]]).collect(),
        r.iter().copied().collect(),
    ))
}

fn target_heights(
    target_z: &PyReadonlyArray2<'_, f64>,
    grid: &viewshed::CellGrid,
) -> PyResult<Vec<f64>> {
    let z = target_z.as_array();
    if z.dim() != (grid.ny, grid.nx) {
        return Err(PyValueError::new_err("target_z must be (ny, nx)"));
    }
    Ok(z.iter().copied().collect())
}

/// Index of the footprint covering every cell centre of ``grid = (x0, y0, cell, nx, ny)`` (-1 for none):
/// ring edges ``(m, 4)`` with their polygon ``owner``; overlaps go to the highest ``priority``.
#[pyfunction]
fn rasterize_footprints<'py>(
    py: Python<'py>,
    segments: PyReadonlyArray2<'py, f64>,
    owner: PyReadonlyArray1<'py, i64>,
    priority: PyReadonlyArray1<'py, f64>,
    grid: (f64, f64, f64, usize, usize),
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let grid = cell_grid(grid)?;
    let (segs, owner, priority) = (segments.as_array(), owner.as_array(), priority.as_array());
    if (segs.nrows() > 0 && segs.ncols() != 4) || owner.len() != segs.nrows() {
        return Err(PyValueError::new_err(
            "segments must be (m, 4) with one owner each",
        ));
    }
    let walls: Vec<Seg> = segs
        .rows()
        .into_iter()
        .map(|r| Seg {
            a: [r[0], r[1]],
            b: [r[2], r[3]],
        })
        .collect();
    let owner: Vec<usize> = owner
        .iter()
        .map(|&o| usize::try_from(o).unwrap_or(usize::MAX))
        .collect();
    let priority: Vec<f64> = priority.iter().copied().collect();
    let out = py.detach(|| viewshed::rasterize(&walls, &owner, &priority, &grid));
    let out = numpy::ndarray::Array2::from_shape_vec((grid.ny, grid.nx), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray(py))
}

/// Visible cells of every eye ``(k, 3)`` within its radius on ``grid`` with target heights ``target_z``
/// ``(ny, nx)`` (NaN = not a target): flat cell indices of all eyes concatenated, and ``(k + 1)`` offsets.
/// ``tops`` accepts one or two endpoint heights per wall; keyword-only ``bottoms`` and ``under_*``
/// have the same raised-volume encoding as ``insolation_day_stats``.
#[pyfunction]
#[pyo3(signature = (eyes, radii, grid, target_z, segments, tops, terrain=None, terrain_grid=None, n_threads=None, *, bottoms=None, under_xy=None, under_ring_offsets=None, under_poly_offsets=None, under_z=None))]
#[allow(clippy::too_many_arguments)]
fn viewshed_cells<'py>(
    py: Python<'py>,
    eyes: PyReadonlyArray2<'py, f64>,
    radii: PyReadonlyArray1<'py, f64>,
    grid: (f64, f64, f64, usize, usize),
    target_z: PyReadonlyArray2<'py, f64>,
    segments: PyReadonlyArray2<'py, f64>,
    tops: PyReadonlyArrayDyn<'py, f64>,
    terrain: Option<PyReadonlyArray2<'py, f64>>,
    terrain_grid: Option<(f64, f64, f64, f64)>,
    n_threads: Option<usize>,
    bottoms: Option<PyReadonlyArray1<'py, f64>>,
    under_xy: Option<PyReadonlyArray2<'py, f64>>,
    under_ring_offsets: Option<PyReadonlyArray1<'py, i64>>,
    under_poly_offsets: Option<PyReadonlyArray1<'py, i64>>,
    under_z: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<RaggedCells<'py>> {
    let grid = cell_grid(grid)?;
    let (eyes, radii) = eyes_and_radii(&eyes, &radii)?;
    let target_z = target_heights(&target_z, &grid)?;
    let scene = scene_from_arrays(
        segments,
        tops,
        terrain,
        terrain_grid,
        bottoms,
        under_xy,
        under_ring_offsets,
        under_poly_offsets,
        under_z,
        None,
    )?;
    let seen = py.detach(|| {
        run_in_pool(n_threads, || {
            viewshed::visible_cells_many(&scene, &eyes, &radii, &grid, &target_z)
        })
    });
    let mut offsets = vec![0i64];
    let mut flat = Vec::with_capacity(seen.iter().map(Vec::len).sum());
    for cells in seen {
        flat.extend(cells);
        offsets.push(flat.len() as i64);
    }
    Ok((flat.into_pyarray(py), offsets.into_pyarray(py)))
}

/// Per cell of ``grid``: the number of eyes that see it (uint32 ``(ny, nx)``) and, with ``weights``, the sum
/// of their weights (float64 ``(ny, nx)``).
/// ``tops`` accepts one or two endpoint heights per wall; keyword-only ``bottoms`` and ``under_*``
/// have the same raised-volume encoding as ``insolation_day_stats``.
#[pyfunction]
#[pyo3(signature = (eyes, radii, grid, target_z, segments, tops, weights=None, terrain=None, terrain_grid=None,
    n_threads=None, *, bottoms=None, under_xy=None, under_ring_offsets=None, under_poly_offsets=None, under_z=None))]
#[allow(clippy::too_many_arguments)]
fn viewshed_count_grid<'py>(
    py: Python<'py>,
    eyes: PyReadonlyArray2<'py, f64>,
    radii: PyReadonlyArray1<'py, f64>,
    grid: (f64, f64, f64, usize, usize),
    target_z: PyReadonlyArray2<'py, f64>,
    segments: PyReadonlyArray2<'py, f64>,
    tops: PyReadonlyArrayDyn<'py, f64>,
    weights: Option<PyReadonlyArray1<'py, f64>>,
    terrain: Option<PyReadonlyArray2<'py, f64>>,
    terrain_grid: Option<(f64, f64, f64, f64)>,
    n_threads: Option<usize>,
    bottoms: Option<PyReadonlyArray1<'py, f64>>,
    under_xy: Option<PyReadonlyArray2<'py, f64>>,
    under_ring_offsets: Option<PyReadonlyArray1<'py, i64>>,
    under_poly_offsets: Option<PyReadonlyArray1<'py, i64>>,
    under_z: Option<PyReadonlyArray1<'py, f64>>,
) -> PyResult<CountArrays<'py>> {
    let grid = cell_grid(grid)?;
    let (eyes, radii) = eyes_and_radii(&eyes, &radii)?;
    let target_z = target_heights(&target_z, &grid)?;
    let weights: Option<Vec<f64>> = weights.map(|w| w.as_array().iter().copied().collect());
    if weights.as_ref().is_some_and(|w| w.len() != eyes.len()) {
        return Err(PyValueError::new_err("weights must have one value per eye"));
    }
    let scene = scene_from_arrays(
        segments,
        tops,
        terrain,
        terrain_grid,
        bottoms,
        under_xy,
        under_ring_offsets,
        under_poly_offsets,
        under_z,
        None,
    )?;
    let (counts, sums) = py.detach(|| {
        run_in_pool(n_threads, || {
            viewshed::count_grid(&scene, &eyes, &radii, weights.as_deref(), &grid, &target_z)
        })
    });
    let shape = (grid.ny, grid.nx);
    let counts = numpy::ndarray::Array2::from_shape_vec(shape, counts)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let sums = match sums {
        Some(s) => Some(
            numpy::ndarray::Array2::from_shape_vec(shape, s)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_pyarray(py),
        ),
        None => None,
    };
    Ok((counts.into_pyarray(py), sums))
}

/// Mark point buffers that can contribute to their union.
#[pyfunction]
fn keep_point_buffers<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'_, f64>,
    radii: PyReadonlyArray1<'_, f64>,
    quad_segs: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let points = points.as_array();
    if points.ncols() != 2 || points.nrows() != radii.len()? || quad_segs == 0 {
        return Err(PyValueError::new_err(
            "expected (n, 2) points, n radii and positive quad_segs",
        ));
    }
    let points = points
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1]])
        .collect();
    let radii = radii.as_array().iter().copied().collect();
    Ok(py
        .detach(|| buffers::keep_buffers(points, radii, quad_segs))
        .into_pyarray(py))
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(keep_point_buffers, m)?)?;
    m.add_function(wrap_pyfunction!(classify_grid, m)?)?;
    m.add_function(wrap_pyfunction!(classify_points, m)?)?;
    m.add_function(wrap_pyfunction!(roof_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(insolation_day_stats, m)?)?;
    m.add_function(wrap_pyfunction!(insolation_day_stats_trees, m)?)?;
    m.add_function(wrap_pyfunction!(line_of_sight, m)?)?;
    m.add_function(wrap_pyfunction!(rasterize_footprints, m)?)?;
    m.add_function(wrap_pyfunction!(viewshed_cells, m)?)?;
    m.add_function(wrap_pyfunction!(viewshed_count_grid, m)?)?;
    m.add_function(wrap_pyfunction!(noise_grid_energy, m)?)?;
    m.add_function(wrap_pyfunction!(visibility_polygons, m)?)?;
    m.add_function(wrap_pyfunction!(visibility_count_grid, m)?)?;
    m.add_function(wrap_pyfunction!(label_rings, m)?)?;
    m.add_function(wrap_pyfunction!(provision_allocate, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
