//! Receiver-grid noise model.
//!
//! For every receiver of a regular grid the sound energy from every point source is summed over the
//! propagation paths:
//!
//! * the direct path, when no wall blocks the line between source and receiver;
//! * first-order reflections (image-source method): the source is mirrored in every wall it sees
//!   (visible parts of walls come from the visibility kernel), and a receiver gets the reflection when
//!   the line from the image to it crosses that visible part and nothing blocks the way back.
//!
//! Level along a path of length `r` (metres): `L = L0 - 20 lg r - k r - A_veg - A_refl`, the same
//! distance law as `simulate_noise` (`L0` is the level at 1 m, `k` the air absorption in dB/m);
//! `A_veg` is a dB/m rate times the length of the path inside vegetation, `A_refl` the reflection
//! loss of the wall. Levels of all paths and sources add up as energies.

use rayon::prelude::*;

use crate::visibility::{Obstacles, Pt, Scratch};

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
}

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
    /// Paths quieter than this are skipped (they cannot matter for the output).
    pub floor_db: f64,
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
fn dist(p: Pt, q: Pt) -> f64 {
    (p[0] - q[0]).hypot(p[1] - q[1])
}

/// Something that sends sound into a region: a source (direct path) or an image source (reflection).
struct Emitter {
    source: usize,
    /// Point the path length is measured from (source or image).
    origin: Pt,
    /// Reflection: visible wall part `(a, b)`, wall id, loss; `None` for the direct path.
    reflection: Option<(Pt, Pt, u32, f64)>,
    bbox: [f64; 4],
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
                        if let Some(level) = path_level(input, &emitters[k as usize], r, crossings)
                        {
                            energy += 10f64.powf(level / 10.0);
                        }
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
    let per_source: Vec<Vec<Emitter>> = input
        .sources
        .par_iter()
        .enumerate()
        .map_init(Scratch::default, |sc, (s, src)| {
            let r = src.max_dist;
            let mut out = vec![Emitter {
                source: s,
                origin: src.at,
                reflection: None,
                bbox: [src.at[0] - r, src.at[1] - r, src.at[0] + r, src.at[1] + r],
            }];
            if !input.reflections || r <= 0.0 {
                return out;
            }
            let Some(runs) = input.walls.visible_runs(src.at, r, 16, sc) else {
                return out;
            };
            for (wall, a, b) in runs.runs {
                let Some(w) = wall else { continue };
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
                let reach = r;
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
                    origin: image,
                    reflection: Some((a, b, w, loss)),
                    bbox,
                });
            }
            out
        })
        .collect();
    per_source.into_iter().flatten().collect()
}

/// Level (dB) delivered by an emitter to receiver `r`, if the path exists.
fn path_level(input: &NoiseInput, e: &Emitter, r: Pt, crossings: &mut Vec<f64>) -> Option<f64> {
    let src = &input.sources[e.source];
    let length = dist(e.origin, r);
    if length > src.max_dist {
        return None;
    }
    let spread = 20.0 * length.max(1.0).log10() + src.air_db_per_m * length;
    let base = src.level_db - spread;
    match e.reflection {
        None => {
            if base < input.floor_db || input.walls.segment_blocked(src.at, r, None) {
                return None;
            }
            Some(base - vegetation_db(input, src, src.at, r, crossings))
        }
        Some((a, b, wall, loss)) => {
            let level = base - loss;
            if level < input.floor_db {
                return None;
            }
            // receiver on the source's side of the wall, path image -> receiver through the visible part
            let side_src = cross(sub(b, a), sub(src.at, a));
            let side_rcv = cross(sub(b, a), sub(r, a));
            if side_src * side_rcv <= 0.0 {
                return None;
            }
            let d = sub(r, e.origin);
            let ab = sub(b, a);
            let den = cross(d, ab);
            if den == 0.0 {
                return None;
            }
            let w = sub(a, e.origin);
            let t = cross(w, ab) / den;
            let u = cross(w, d) / den;
            if !(0.0..=1.0).contains(&u) || t <= 0.0 || t >= 1.0 {
                return None;
            }
            let p = [a[0] + u * ab[0], a[1] + u * ab[1]];
            if input.walls.segment_blocked(p, r, Some(wall)) {
                return None;
            }
            let veg = vegetation_db(input, src, src.at, p, crossings)
                + vegetation_db(input, src, p, r, crossings);
            Some(level - veg)
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

    fn run(walls: Vec<Wall>, trees: Option<Vec<Seg>>, reflections: bool) -> (Vec<f64>, GridSpec) {
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
        };
        let input = NoiseInput {
            walls: &obstacles,
            wall_loss_db: &vec![0.0; n],
            trees: trees.as_ref(),
            sources: std::slice::from_ref(&source),
            grid,
            active: &vec![true; grid.nx * grid.ny],
            reflections,
            floor_db: 0.0,
        };
        (receiver_energy(&input), grid)
    }

    fn level(e: &[f64], g: GridSpec, x: f64, y: f64) -> f64 {
        let i = ((x - g.x0) / g.step).round() as usize;
        let j = ((y - g.y0) / g.step).round() as usize;
        10.0 * e[j * g.nx + i].log10()
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
