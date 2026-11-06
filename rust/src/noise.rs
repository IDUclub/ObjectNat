//! Receiver-grid noise model.
//!
//! For every receiver of a regular grid the sound energy from every point source is summed over the
//! propagation paths:
//!
//! * the direct path, when no wall blocks the line between source and receiver;
//! * first-order reflections (image-source method): the source is mirrored in every wall it sees
//!   (visible parts of walls come from the visibility kernel), and a receiver gets the reflection when
//!   the line from the image to it crosses that visible part and nothing blocks the way back;
//! * diffraction around vertical edges: a convex corner of an obstacle outline that the source sees at
//!   the edge of the obstacle (a silhouette corner) sends sound into the obstacle's shadow, to the
//!   receivers that are hidden from the source but seen from the corner; along the shadowed side of
//!   the obstacle's convex hull the sound also goes on to the next corner and bends around it, which
//!   reaches the receivers right behind the obstacle.
//!
//! Level along a path of length `r` (metres): `L = L0 - 20 lg r - k r - A_veg - A_refl`, the same
//! distance law as `simulate_noise` (`L0` is the level at 1 m, `k` the air absorption in dB/m);
//! `A_veg` is a dB/m rate times the length of the path inside vegetation, `A_refl` the reflection
//! loss of the wall. A diffracted path has the level of the direct distance `d` minus the barrier
//! attenuation of ISO 9613-2 without ground, `A_dif = 10 lg(3 + 40 C3 δ / λ)` (Maekawa's
//! `10 lg(3 + 20 N)` for one edge, `C3 = 1`), where `δ` is how much longer the way through the corners
//! is than `d` and `λ` the wavelength; around two corners `e` metres apart
//! `C3 = (1 + (5 λ / e)²) / (1/3 + (5 λ / e)²)`. The attenuation is at most 20 dB around one corner and
//! 25 dB around two. Levels of all paths and sources add up as energies.

use std::sync::Arc;

use rayon::prelude::*;

use crate::visibility::{disc_polygon, Grid, Obstacles, Pt, Scratch, Seg};

/// Regular receiver grid: receiver `(i, j)` sits at `(x0 + i * step, y0 + j * step)`.
#[derive(Clone, Copy, Debug)]
pub struct GridSpec {
    pub x0: f64,
    pub y0: f64,
    pub step: f64,
    pub nx: usize,
    pub ny: usize,
}

pub struct Source {
    pub at: Pt,
    pub level_db: f64,
    pub air_db_per_m: f64,
    pub veg_db_per_m: f64,
    pub max_dist: f64,
    /// Reach of the image sources (reflections) and of the diffracted paths of this source, metres.
    pub reflection_reach: f64,
    /// Wavelength at the characteristic frequency, metres.
    pub wavelength: f64,
}

/// A vertical edge sound bends around: in plan, a convex corner of an obstacle outline or the end of
/// a wall line, with its neighbouring vertices.
#[derive(Clone, Copy, Debug)]
pub struct Corner {
    pub at: Pt,
    pub prev: Pt,
    pub next: Pt,
    /// A point just outside the corner, where what the corner sees is computed from.
    pub outside: Pt,
    /// The neighbouring corners on the convex hull of the obstacle (on the `prev` and the `next` side),
    /// for the way around two corners.
    pub links: [Option<u32>; 2],
}

/// `10^(-x/10)` for an attenuation of `x` dB per metre over `m` metres, as `exp(-K x m)`.
const DB_TO_NEPER: f64 = std::f64::consts::LN_10 / 10.0;
/// Diffraction takes at least `10 lg 3` dB (Maekawa, at the shadow boundary) ...
const SHADOW_BOUNDARY_FACTOR: f64 = 1.0 / 3.0;
/// ... and at most 20 dB around one edge and 25 dB around two (ISO 9613-2).
const MIN_DIFFRACTION_FACTOR: f64 = 0.01;
const MIN_DOUBLE_DIFFRACTION_FACTOR: f64 = 0.003_162_277_660_168_379_5;

pub struct NoiseInput<'a> {
    pub walls: &'a Obstacles,
    /// Reflection loss per wall, in dB (same order as the walls of `walls`).
    pub wall_loss_db: &'a [f64],
    /// Vegetation boundaries (rings of non-overlapping polygons), if any.
    pub trees: Option<&'a Obstacles>,
    pub sources: &'a [Source],
    pub grid: GridSpec,
    /// `true` for receivers to compute (`nx * ny`, row-major by `j`).
    pub active: &'a [bool],
    pub reflections: bool,
    /// Only walls closer than this to a source reflect it (metres).
    pub reflection_max_dist: f64,
    /// Edges sound bends around; empty for no diffraction.
    pub corners: &'a [Corner],
    /// Paths quieter than this energy (`10^(floor_db/10)`) are skipped.
    pub floor_energy: f64,
}

/// Monotonic stand-in for the angle of `d` in [0, 4): cheaper than `atan2`, same order.
#[inline]
fn pseudo_angle(d: Pt) -> f64 {
    let p = d[0] / (d[0].abs() + d[1].abs());
    if d[1] < 0.0 {
        3.0 + p
    } else {
        1.0 - p
    }
}

/// A visibility polygon, star-shaped around its source: point-in test by angular binary search.
struct Star {
    center: Pt,
    angles: Vec<f64>,
    verts: Vec<Pt>,
}

impl Star {
    fn new(center: Pt, ring: Vec<Pt>) -> Option<Star> {
        if ring.len() < 3 {
            return None;
        }
        let raw: Vec<f64> = ring.iter().map(|v| pseudo_angle(sub(*v, center))).collect();
        let start = (0..raw.len())
            .min_by(|&a, &b| {
                raw[a]
                    .partial_cmp(&raw[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        let n = ring.len();
        let mut angles = Vec::with_capacity(n);
        let mut verts = Vec::with_capacity(n);
        let mut prev = raw[start];
        for k in 0..n {
            let i = (start + k) % n;
            let mut a = raw[i];
            // unwrap against the previous vertex: a radial edge at the starting angle closes the
            // ring with a vertex whose angle equals the first one and must come after it
            while a < prev - 2.0 {
                a += 4.0;
            }
            angles.push(a);
            verts.push(ring[i]);
            prev = a;
        }
        Some(Star {
            center,
            angles,
            verts,
        })
    }

    fn contains(&self, p: Pt) -> bool {
        let d = sub(p, self.center);
        if d[0] == 0.0 && d[1] == 0.0 {
            return true;
        }
        let mut a = pseudo_angle(d);
        if a < self.angles[0] {
            a += 4.0;
        }
        let n = self.verts.len();
        let k = self.angles.partition_point(|&x| x <= a).max(1) - 1;
        let (v0, v1) = (self.verts[k], self.verts[(k + 1) % n]);
        let e = sub(v1, v0);
        let scale = e[0].hypot(e[1]) * sub(p, v0)[0].hypot(sub(p, v0)[1]);
        cross(e, sub(p, v0)) >= -1e-9 * scale.max(1e-12)
    }
}

#[inline]
fn sub(p: Pt, q: Pt) -> Pt {
    [p[0] - q[0], p[1] - q[1]]
}

#[inline]
fn cross(p: Pt, q: Pt) -> f64 {
    p[0] * q[1] - p[1] * q[0]
}

#[inline]
fn dot(p: Pt, q: Pt) -> f64 {
    p[0] * q[0] + p[1] * q[1]
}

#[inline]
fn dist(p: Pt, q: Pt) -> f64 {
    (p[0] - q[0]).hypot(p[1] - q[1])
}

fn point_segment_dist(p: Pt, a: Pt, b: Pt) -> f64 {
    let ab = sub(b, a);
    let len2 = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if len2 > 0.0 {
        ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / len2
    } else {
        0.0
    };
    let t = t.clamp(0.0, 1.0);
    dist(p, [a[0] + t * ab[0], a[1] + t * ab[1]])
}

/// Reach of the reflections and diffraction paths of a source.
#[inline]
fn secondary_reach(src: &Source) -> f64 {
    src.reflection_reach.min(src.max_dist)
}

/// How the sound of an emitter gets to a receiver.
enum Path {
    /// Straight from the source, where it is visible (`None`: everywhere within reach).
    Direct(Option<Arc<Star>>),
    /// Mirrored in a wall: the visible part `(a, b)` of the wall and its id.
    Reflection(Pt, Pt, u32),
    /// Bent around a corner into the shadow of its obstacle.
    Diffraction(Box<Edge>),
}

/// The corner a diffracted path reaches its receivers from, as one source sees it.
struct Edge {
    corner: Pt,
    /// Where the last leg to the corner comes from: the source, or the first corner of two.
    from: Pt,
    /// The way from the source to the corner: length and vegetation loss (dB).
    to_corner: f64,
    veg_db: f64,
    /// Side of the line `from` -> corner the obstacle lies on (sign of the cross product): the
    /// diffracted path bends towards it.
    side: f64,
    /// `C3` of the attenuation and its least factor (the largest attenuation).
    c3: f64,
    min_factor: f64,
    /// Where the corner is seen from; where the source is seen from (those receivers hear it
    /// directly) and, around two corners, where the first corner is seen from (those receivers get
    /// the sound around the first corner alone).
    corner_star: Arc<Star>,
    source_star: Arc<Star>,
    first_star: Option<Arc<Star>>,
}

/// Something that sends sound into a region: a source, an image source or a diffracting corner.
struct Emitter {
    source: usize,
    /// Source power `10^(L0/10)`, minus the reflection loss for image sources.
    power: f64,
    /// Point the path length is measured from: the source (direct and diffracted paths) or its image.
    origin: Pt,
    path: Path,
    bbox: [f64; 4],
}

/// The direct path and the reflections of one source, its visibility polygon and the corners it sees at
/// the edges of obstacles.
struct SourcePaths {
    emitters: Vec<Emitter>,
    star: Option<Arc<Star>>,
    sights: Vec<Sight>,
}

/// A corner that a source sees at the edge of an obstacle, or the next corner of that obstacle along
/// the shadowed side of its convex hull.
struct Sight {
    corner: usize,
    /// For the next corner: the corner the source sees.
    first: Option<usize>,
    /// Where the last leg to the corner comes from: the source, or the first corner.
    from: Pt,
    to_corner: f64,
    side: f64,
}

const TILE: usize = 32;

/// Energy (sum of `10^(L/10)`) per receiver; 0 where nothing arrives.
pub fn receiver_energy(input: &NoiseInput) -> Vec<f64> {
    let g = input.grid;
    let emitters = build_emitters(input);

    // register emitters in receiver tiles
    let ntx = g.nx.div_ceil(TILE);
    let nty = g.ny.div_ceil(TILE);
    let mut tiles: Vec<Vec<u32>> = vec![Vec::new(); ntx * nty];
    for (k, e) in emitters.iter().enumerate() {
        let i0 = (((e.bbox[0] - g.x0) / g.step).floor().max(0.0) as usize) / TILE;
        let j0 = (((e.bbox[1] - g.y0) / g.step).floor().max(0.0) as usize) / TILE;
        let i1 = ((((e.bbox[2] - g.x0) / g.step).ceil().max(0.0) as usize) / TILE)
            .min(ntx.saturating_sub(1));
        let j1 = ((((e.bbox[3] - g.y0) / g.step).ceil().max(0.0) as usize) / TILE)
            .min(nty.saturating_sub(1));
        if e.bbox[2] < g.x0 || e.bbox[3] < g.y0 || i0 >= ntx || j0 >= nty {
            continue;
        }
        for tj in j0..=j1 {
            for ti in i0..=i1 {
                tiles[tj * ntx + ti].push(k as u32);
            }
        }
    }

    let tile_results: Vec<(usize, Vec<(usize, f64)>)> = (0..ntx * nty)
        .into_par_iter()
        .filter(|&t| !tiles[t].is_empty())
        .map_init(Vec::new, |crossings, t| {
            let (ti, tj) = (t % ntx, t / ntx);
            let mut out = Vec::new();
            for j in tj * TILE..((tj + 1) * TILE).min(g.ny) {
                for i in ti * TILE..((ti + 1) * TILE).min(g.nx) {
                    let idx = j * g.nx + i;
                    if !input.active[idx] {
                        continue;
                    }
                    let r = [g.x0 + i as f64 * g.step, g.y0 + j as f64 * g.step];
                    let mut energy = 0.0;
                    for &k in &tiles[t] {
                        energy += path_energy(input, &emitters[k as usize], r, crossings);
                    }
                    if energy > 0.0 {
                        out.push((idx, energy));
                    }
                }
            }
            (t, out)
        })
        .collect();

    let mut energy = vec![0.0; g.nx * g.ny];
    for (_, cells) in tile_results {
        for (idx, e) in cells {
            energy[idx] += e;
        }
    }
    energy
}

fn build_emitters(input: &NoiseInput) -> Vec<Emitter> {
    let corner_index = corner_grid(input.corners);
    // 1. per source: the direct path, the reflections and the corners at the edges of what it sees
    let per_source: Vec<SourcePaths> = input
        .sources
        .par_iter()
        .enumerate()
        .map_init(Scratch::default, |sc, (s, src)| {
            source_paths(input, corner_index.as_ref(), s, src, sc)
        })
        .collect();

    // 2. what every corner in use sees, once for all the sources that see it
    let mut radius = vec![0.0f64; input.corners.len()];
    for (src, paths) in input.sources.iter().zip(&per_source) {
        for sight in &paths.sights {
            radius[sight.corner] = radius[sight.corner].max(secondary_reach(src) - sight.to_corner);
        }
    }
    let corner_stars: Vec<Option<Arc<Star>>> = radius
        .par_iter()
        .enumerate()
        .map_init(Scratch::default, |sc, (c, &r)| {
            if r <= 0.0 {
                return None;
            }
            let o = input.corners[c].outside;
            let ring = input
                .walls
                .visible_runs(o, r, 64, sc)
                .map_or_else(|| disc_polygon(o, r, 64), |runs| runs.ring(r));
            Star::new(o, ring).map(Arc::new)
        })
        .collect();

    // 3. one diffraction emitter per source and corner
    let per_source: Vec<Vec<Emitter>> = per_source
        .into_par_iter()
        .enumerate()
        .map_init(Vec::new, |crossings, (s, paths)| {
            let mut out = paths.emitters;
            if let Some(source_star) = paths.star {
                for sight in &paths.sights {
                    out.extend(diffraction_emitter(
                        input,
                        s,
                        sight,
                        &source_star,
                        &corner_stars,
                        crossings,
                    ));
                }
            }
            out
        })
        .collect();
    per_source.into_iter().flatten().collect()
}

fn source_paths(
    input: &NoiseInput,
    corner_index: Option<&Grid>,
    s: usize,
    src: &Source,
    sc: &mut Scratch,
) -> SourcePaths {
    let r = src.max_dist;
    let runs = if r > 0.0 {
        input.walls.visible_runs(src.at, r, 64, sc)
    } else {
        None
    };
    let star = runs
        .as_ref()
        .and_then(|runs| Star::new(src.at, runs.ring(r)))
        .map(Arc::new);
    let power = 10f64.powf(src.level_db / 10.0);
    let mut out = vec![Emitter {
        source: s,
        power,
        origin: src.at,
        path: Path::Direct(star.clone()),
        bbox: [src.at[0] - r, src.at[1] - r, src.at[0] + r, src.at[1] + r],
    }];
    if let (Some(runs), true) = (&runs, input.reflections) {
        for &(wall, a, b) in &runs.runs {
            let Some(w) = wall else { continue };
            if point_segment_dist(src.at, a, b) > input.reflection_max_dist {
                continue;
            }
            let seg = &input.walls.segs[w as usize];
            let e = sub(seg.b, seg.a);
            let len2 = e[0] * e[0] + e[1] * e[1];
            if len2 == 0.0 || dist(a, b) < 1e-6 {
                continue;
            }
            // mirror the source in the wall's supporting line
            let ap = sub(src.at, seg.a);
            let t = (ap[0] * e[0] + ap[1] * e[1]) / len2;
            let foot = [seg.a[0] + t * e[0], seg.a[1] + t * e[1]];
            let image = [2.0 * foot[0] - src.at[0], 2.0 * foot[1] - src.at[1]];
            if dist(image, src.at) < 1e-9 {
                continue; // source on the wall line
            }
            let loss = input.wall_loss_db.get(w as usize).copied().unwrap_or(0.0);
            // wedge from the image through the visible part, cut at the reach of the image
            let reach = secondary_reach(src);
            let mut xs = vec![a[0], b[0]];
            let mut ys = vec![a[1], b[1]];
            for k in 0..=8 {
                let f = k as f64 / 8.0;
                let p = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
                let d = sub(p, image);
                let n = d[0].hypot(d[1]);
                if n > 0.0 {
                    xs.push(image[0] + d[0] / n * reach);
                    ys.push(image[1] + d[1] / n * reach);
                }
            }
            let bbox = [
                xs.iter().copied().fold(f64::INFINITY, f64::min),
                ys.iter().copied().fold(f64::INFINITY, f64::min),
                xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                ys.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ];
            out.push(Emitter {
                source: s,
                power: power * 10f64.powf(-loss / 10.0),
                origin: image,
                path: Path::Reflection(a, b, w),
                bbox,
            });
        }
    }
    let sights = match (&star, corner_index) {
        (Some(star), Some(index)) => edge_sights(input, index, src, star),
        _ => Vec::new(),
    };
    SourcePaths {
        emitters: out,
        star,
        sights,
    }
}

/// Uniform grid over the corners (as zero-length segments), to find the corners near a source.
fn corner_grid(corners: &[Corner]) -> Option<Grid> {
    if corners.is_empty() {
        return None;
    }
    let segs: Vec<Seg> = corners.iter().map(|c| Seg { a: c.at, b: c.at }).collect();
    let mut bbox = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for c in corners {
        bbox = [
            bbox[0].min(c.at[0]),
            bbox[1].min(c.at[1]),
            bbox[2].max(c.at[0]),
            bbox[3].max(c.at[1]),
        ];
    }
    let cells = ((corners.len() as f64).sqrt().ceil() as usize).clamp(1, 1024);
    Some(Grid::build(&segs, bbox, cells))
}

/// Corners that a source sees at the edges of obstacles (silhouette corners), within the reach of its
/// diffracted paths.
fn edge_sights(input: &NoiseInput, index: &Grid, src: &Source, star: &Star) -> Vec<Sight> {
    let reach = secondary_reach(src);
    let mut sights = Vec::new();
    let around = [
        src.at[0] - reach,
        src.at[1] - reach,
        src.at[0] + reach,
        src.at[1] + reach,
    ];
    index.query_box(around, |id| {
        let c = &input.corners[id as usize];
        let to_corner = dist(src.at, c.at);
        if to_corner >= reach || to_corner < 1e-6 {
            return;
        }
        let u = sub(c.at, src.at);
        let (side_prev, side_next) = (cross(u, sub(c.prev, c.at)), cross(u, sub(c.next, c.at)));
        // both neighbours on one side of the line of sight: it grazes the obstacle instead of entering it
        if side_prev * side_next <= 0.0 || !star.contains(c.outside) {
            return;
        }
        sights.push(Sight {
            corner: id as usize,
            first: None,
            from: src.at,
            to_corner,
            side: side_prev.signum(),
        });
        // on the convex hull, the side going on into the shadow is the one closer to straight ahead
        let ahead = |link: Option<u32>| {
            link.map(|k| {
                let d = sub(input.corners[k as usize].at, c.at);
                dot(u, d) / d[0].hypot(d[1])
            })
        };
        let link = match (ahead(c.links[0]), ahead(c.links[1])) {
            (Some(a), Some(b)) if a < b => c.links[1],
            (Some(_), _) => c.links[0],
            _ => c.links[1],
        };
        let Some(k) = link else { return };
        let next = &input.corners[k as usize];
        let face = dist(c.at, next.at);
        // around the next corner the path bends towards the hull corner after it
        let Some(&after) = next.links.iter().flatten().find(|&&j| j != id) else {
            return;
        };
        let side = cross(
            sub(next.at, c.at),
            sub(input.corners[after as usize].at, next.at),
        );
        if face > 1e-6 && to_corner + face < reach && side != 0.0 && !star.contains(next.outside) {
            sights.push(Sight {
                corner: k as usize,
                first: Some(id as usize),
                from: c.at,
                to_corner: to_corner + face,
                side: side.signum(),
            });
        }
    });
    sights
}

/// The diffraction emitter of a source at a corner; `None` when the corner sees nothing on the shadow
/// side.
fn diffraction_emitter(
    input: &NoiseInput,
    s: usize,
    sight: &Sight,
    source_star: &Arc<Star>,
    corner_stars: &[Option<Arc<Star>>],
    crossings: &mut Vec<f64>,
) -> Option<Emitter> {
    let src = &input.sources[s];
    let corner = input.corners[sight.corner].at;
    let corner_star = corner_stars[sight.corner].as_ref()?;
    let (c3, min_factor, first_star, veg_to) = match sight.first {
        None => (1.0, MIN_DIFFRACTION_FACTOR, None, corner),
        Some(k) => {
            // ISO 9613-2's C3 for a thick barrier whose two edges are `face` metres apart
            let face = dist(sight.from, corner);
            let q = (5.0 * src.wavelength / face).powi(2);
            let first = corner_stars[k].as_ref()?;
            let c3 = (1.0 + q) / (1.0 / 3.0 + q);
            (
                c3,
                MIN_DOUBLE_DIFFRACTION_FACTOR,
                Some(Arc::clone(first)),
                sight.from,
            )
        }
    };
    let u = sub(corner, sight.from);
    let shadow_side = |p: Pt| cross(u, sub(p, corner)) * sight.side > 0.0;
    // bounding box of what the corner sees on the shadow side; the edges that cross the line of
    // sight keep both ends, so the far crossing is inside
    let verts = &corner_star.verts;
    let mut bbox = [corner[0], corner[1], corner[0], corner[1]];
    let mut any = false;
    for k in 0..verts.len() {
        let (p, q) = (verts[k], verts[(k + 1) % verts.len()]);
        if shadow_side(p) || shadow_side(q) {
            for v in [p, q] {
                bbox = [
                    bbox[0].min(v[0]),
                    bbox[1].min(v[1]),
                    bbox[2].max(v[0]),
                    bbox[3].max(v[1]),
                ];
            }
            any = true;
        }
    }
    if !any {
        return None;
    }
    let reach = secondary_reach(src);
    Some(Emitter {
        source: s,
        power: 10f64.powf(src.level_db / 10.0),
        origin: src.at,
        path: Path::Diffraction(Box::new(Edge {
            corner,
            from: sight.from,
            to_corner: sight.to_corner,
            veg_db: vegetation_db(input, src, src.at, veg_to, crossings),
            side: sight.side,
            c3,
            min_factor,
            corner_star: Arc::clone(corner_star),
            source_star: Arc::clone(source_star),
            first_star,
        })),
        bbox: [
            bbox[0].max(src.at[0] - reach),
            bbox[1].max(src.at[1] - reach),
            bbox[2].min(src.at[0] + reach),
            bbox[3].min(src.at[1] + reach),
        ],
    })
}

/// Energy (`10^(L/10)`) delivered by an emitter to receiver `r`; 0 when the path does not exist.
fn path_energy(input: &NoiseInput, e: &Emitter, r: Pt, crossings: &mut Vec<f64>) -> f64 {
    let src = &input.sources[e.source];
    let length = dist(e.origin, r);
    let reach = match e.path {
        Path::Direct(_) => src.max_dist,
        _ => secondary_reach(src),
    };
    if length > reach {
        return 0.0;
    }
    let l = length.max(1.0);
    let mut energy = e.power / (l * l) * (-DB_TO_NEPER * src.air_db_per_m * length).exp();
    if energy < input.floor_energy {
        return 0.0;
    }
    match &e.path {
        Path::Direct(star) => {
            if let Some(star) = star {
                if !star.contains(r) {
                    return 0.0;
                }
            }
            let veg = vegetation_db(input, src, src.at, r, crossings);
            if veg > 0.0 {
                energy *= (-DB_TO_NEPER * veg).exp();
            }
            energy
        }
        &Path::Reflection(a, b, wall) => {
            // receiver on the source's side of the wall, path image -> receiver through the visible part
            let side_src = cross(sub(b, a), sub(src.at, a));
            let side_rcv = cross(sub(b, a), sub(r, a));
            if side_src * side_rcv <= 0.0 {
                return 0.0;
            }
            let d = sub(r, e.origin);
            let ab = sub(b, a);
            let den = cross(d, ab);
            if den == 0.0 {
                return 0.0;
            }
            let w = sub(a, e.origin);
            let t = cross(w, ab) / den;
            let u = cross(w, d) / den;
            if !(0.0..=1.0).contains(&u) || t <= 0.0 || t >= 1.0 {
                return 0.0;
            }
            let p = [a[0] + u * ab[0], a[1] + u * ab[1]];
            if input.walls.segment_blocked(p, r, Some(wall)) {
                return 0.0;
            }
            let veg = vegetation_db(input, src, src.at, p, crossings)
                + vegetation_db(input, src, p, r, crossings);
            if veg > 0.0 {
                energy *= (-DB_TO_NEPER * veg).exp();
            }
            energy
        }
        Path::Diffraction(edge) => {
            if energy * SHADOW_BOUNDARY_FACTOR < input.floor_energy {
                return 0.0;
            }
            // in the obstacle's shadow, seen from the corner but not from the source (nor from the
            // first corner of two)
            if cross(sub(edge.corner, edge.from), sub(r, edge.corner)) * edge.side <= 0.0
                || edge.source_star.contains(r)
                || edge
                    .first_star
                    .as_ref()
                    .is_some_and(|star| star.contains(r))
                || !edge.corner_star.contains(r)
            {
                return 0.0;
            }
            let detour = (edge.to_corner + dist(edge.corner, r) - length).max(0.0);
            energy *= (1.0 / (3.0 + 40.0 * edge.c3 * detour / src.wavelength)).max(edge.min_factor);
            if energy < input.floor_energy {
                return 0.0;
            }
            let veg = edge.veg_db + vegetation_db(input, src, edge.corner, r, crossings);
            if veg > 0.0 {
                energy *= (-DB_TO_NEPER * veg).exp();
            }
            energy
        }
    }
}

/// Attenuation by vegetation along `p -> q`: rate times the length inside tree polygons.
fn vegetation_db(input: &NoiseInput, src: &Source, p: Pt, q: Pt, crossings: &mut Vec<f64>) -> f64 {
    let Some(trees) = input.trees else {
        return 0.0;
    };
    if src.veg_db_per_m <= 0.0 {
        return 0.0;
    }
    trees.segment_crossings(p, q, crossings);
    if crossings.is_empty() {
        // either fully inside or fully outside one polygon
        return if inside(trees, p, crossings) {
            src.veg_db_per_m * dist(p, q)
        } else {
            0.0
        };
    }
    let ts: Vec<f64> = crossings.clone();
    let mut is_in = inside(trees, p, crossings);
    let mut prev = 0.0;
    let mut length = 0.0;
    for t in ts.into_iter().chain(std::iter::once(1.0)) {
        if is_in {
            length += t - prev;
        }
        is_in = !is_in;
        prev = t;
    }
    src.veg_db_per_m * length * dist(p, q)
}

/// Point-in-union test by crossing parity of a horizontal ray to the right.
fn inside(trees: &Obstacles, p: Pt, crossings: &mut Vec<f64>) -> bool {
    let far = [p[0] + 1e7, p[1] + 1e-7];
    trees.segment_crossings(p, far, crossings);
    crossings.len() % 2 == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visibility::{Seg, Wall};

    fn square(cx: f64, cy: f64, h: f64) -> Vec<Wall> {
        let p = [
            [cx - h, cy - h],
            [cx + h, cy - h],
            [cx + h, cy + h],
            [cx - h, cy + h],
        ];
        (0..4)
            .map(|k| Wall {
                seg: Seg {
                    a: p[k],
                    b: p[(k + 1) % 4],
                },
                one_sided: true,
            })
            .collect()
    }

    /// The corners of `square`, each with a point 5 cm outside it on the diagonal.
    fn square_corners(cx: f64, cy: f64, h: f64) -> Vec<Corner> {
        let p = [
            [cx - h, cy - h],
            [cx + h, cy - h],
            [cx + h, cy + h],
            [cx - h, cy + h],
        ];
        (0..4)
            .map(|k| Corner {
                at: p[k],
                prev: p[(k + 3) % 4],
                next: p[(k + 1) % 4],
                outside: [
                    p[k][0] + (p[k][0] - cx).signum() * 0.035,
                    p[k][1] + (p[k][1] - cy).signum() * 0.035,
                ],
                links: [Some(((k + 3) % 4) as u32), Some(((k + 1) % 4) as u32)],
            })
            .collect()
    }

    fn run(walls: Vec<Wall>, trees: Option<Vec<Seg>>, reflections: bool) -> (Vec<f64>, GridSpec) {
        run_with(walls, trees, reflections, &[])
    }

    fn run_with(
        walls: Vec<Wall>,
        trees: Option<Vec<Seg>>,
        reflections: bool,
        corners: &[Corner],
    ) -> (Vec<f64>, GridSpec) {
        let n = walls.len();
        let obstacles = Obstacles::from_walls(walls, 50.0);
        let trees = trees.map(|t| Obstacles::new(t, 50.0));
        let grid = GridSpec {
            x0: -50.0,
            y0: -50.0,
            step: 1.0,
            nx: 101,
            ny: 101,
        };
        let source = Source {
            at: [0.0, 0.0],
            level_db: 90.0,
            air_db_per_m: 0.0,
            veg_db_per_m: 0.1,
            max_dist: 70.0,
            reflection_reach: 70.0,
            wavelength: 0.343,
        };
        let input = NoiseInput {
            walls: &obstacles,
            wall_loss_db: &vec![0.0; n],
            trees: trees.as_ref(),
            sources: std::slice::from_ref(&source),
            grid,
            active: &vec![true; grid.nx * grid.ny],
            reflections,
            reflection_max_dist: 1e9,
            corners,
            floor_energy: 1.0,
        };
        (receiver_energy(&input), grid)
    }

    fn level(e: &[f64], g: GridSpec, x: f64, y: f64) -> f64 {
        let i = ((x - g.x0) / g.step).round() as usize;
        let j = ((y - g.y0) / g.step).round() as usize;
        10.0 * e[j * g.nx + i].log10()
    }

    #[test]
    fn star_test_matches_segment_blocking() {
        // a few buildings around the source; compare the star polygon with direct blocking tests
        let mut walls = square(20.0, 0.0, 5.0);
        walls.extend(square(-15.0, 12.0, 4.0));
        walls.extend(square(3.0, -25.0, 6.0));
        walls.extend(square(30.0, 30.0, 3.0));
        let obstacles = Obstacles::from_walls(walls, 80.0);
        let mut sc = Scratch::default();
        let src = [0.5, 0.25];
        let runs = obstacles.visible_runs(src, 80.0, 64, &mut sc).unwrap();
        let star = Star::new(src, runs.ring(80.0)).unwrap();
        let mut mismatches = 0;
        let mut total = 0;
        for i in 0..121 {
            for j in 0..121 {
                let p = [-60.0 + i as f64 * 1.0 + 0.3, -60.0 + j as f64 * 1.0 + 0.7];
                if dist(p, src) > 75.0 {
                    continue;
                }
                let inside_any = |c: Pt, h: f64| (p[0] - c[0]).abs() < h && (p[1] - c[1]).abs() < h;
                if inside_any([20.0, 0.0], 5.0)
                    || inside_any([-15.0, 12.0], 4.0)
                    || inside_any([3.0, -25.0], 6.0)
                    || inside_any([30.0, 30.0], 3.0)
                {
                    continue;
                }
                total += 1;
                if star.contains(p) == obstacles.segment_blocked(src, p, None) {
                    mismatches += 1;
                }
            }
        }
        assert!(total > 1000);
        assert_eq!(mismatches, 0, "{mismatches} of {total}");
    }

    #[test]
    fn free_field_follows_the_distance_law() {
        let (e, g) = run(Vec::new(), None, true);
        assert!((level(&e, g, 10.0, 0.0) - (90.0 - 20.0)).abs() < 1e-9);
        assert!((level(&e, g, 0.0, -40.0) - (90.0 - 20.0 * 40f64.log10())).abs() < 1e-9);
    }

    #[test]
    fn building_casts_a_shadow_and_reflects() {
        let (e, g) = run(square(20.0, 0.0, 5.0), None, true);
        assert!(
            e[(50) * g.nx + 50 + 40] == 0.0,
            "behind the building nothing arrives"
        );
        // in front of the building the reflection adds energy: louder than free field
        let free = 90.0 - 20.0 * 10f64.log10();
        assert!(level(&e, g, 10.0, 0.0) > free + 0.5);
        let (e0, _) = run(square(20.0, 0.0, 5.0), None, false);
        assert!((level(&e0, g, 10.0, 0.0) - free).abs() < 1e-9);
    }

    #[test]
    fn diffraction_bends_sound_around_a_corner() {
        // the source at the origin sees the two left corners of the building at its edges
        let (e, g) = run_with(
            square(20.0, 0.0, 5.0),
            None,
            false,
            &square_corners(20.0, 0.0, 5.0),
        );
        let (e0, _) = run(square(20.0, 0.0, 5.0), None, false);
        let at = |e: &[f64], x: f64, y: f64| {
            e[((y - g.y0) / g.step).round() as usize * g.nx
                + ((x - g.x0) / g.step).round() as usize]
        };
        // receivers that see the source keep the direct sound only
        assert_eq!(at(&e, 10.0, 0.0), at(&e0, 10.0, 0.0));
        assert_eq!(at(&e, 30.0, 20.0), at(&e0, 30.0, 20.0));
        // in the shadow beside the top-left corner: the direct distance minus Maekawa's attenuation,
        // plus the way around the two lower corners at the 25 dB limit
        let (s, v, r) = ([0.0, 0.0], [15.0, 5.0], [40.0, 8.0]);
        assert_eq!(at(&e0, r[0], r[1]), 0.0);
        let detour = dist(s, v) + dist(v, r) - dist(s, r);
        let factor = 1.0 / (3.0 + 40.0 * detour / 0.343) + 10f64.powf(-2.5);
        let expected = 90.0 - 20.0 * dist(s, r).log10() + 10.0 * factor.log10();
        assert!((level(&e, g, r[0], r[1]) - expected).abs() < 1e-9);
        // right behind the building it bends around two corners, above and below: 25 dB each way
        assert_eq!(at(&e0, 40.0, 0.0), 0.0);
        let expected = 90.0 - 20.0 * 40f64.log10() - 25.0 + 10.0 * 2f64.log10();
        assert!((level(&e, g, 40.0, 0.0) - expected).abs() < 1e-9);
    }

    #[test]
    fn vegetation_attenuates_by_length_inside() {
        let ring = [[10.0, -5.0], [20.0, -5.0], [20.0, 5.0], [10.0, 5.0]];
        let trees: Vec<Seg> = (0..4)
            .map(|k| Seg {
                a: ring[k],
                b: ring[(k + 1) % 4],
            })
            .collect();
        let (e, g) = run(Vec::new(), Some(trees), false);
        let free = 90.0 - 20.0 * 30f64.log10();
        assert!((level(&e, g, 30.0, 0.0) - (free - 0.1 * 10.0)).abs() < 1e-6);
        // inside the trees: 5 m of foliage
        let free15 = 90.0 - 20.0 * 15f64.log10();
        assert!((level(&e, g, 15.0, 0.0) - (free15 - 0.1 * 5.0)).abs() < 1e-6);
    }
}
