//! Direct sunlight and lines of sight among buildings and terrain (2.5D scene).
//!
//! Buildings are extruded footprints with flat roofs: every footprint wall carries the absolute height
//! of its roof top. Terrain is an optional height grid. A receiver at height `z` sees the sun when the
//! ray towards it — horizontal direction `dir`, rising `tan_el` metres per metre — passes above every
//! wall it crosses and above the terrain. The ray only needs to be followed while it is still below the
//! highest point of the scene, so high sun means short rays.

use rayon::prelude::*;

use crate::visibility::{segment_params, Grid, Seg};

/// Height grid: `z[j * nx + i]` is the ground height at `(x0 + i * sx, y0 + j * sy)`; NaN is no data.
pub struct Terrain {
    pub x0: f64,
    pub y0: f64,
    pub sx: f64,
    pub sy: f64,
    pub nx: usize,
    pub ny: usize,
    pub z: Vec<f64>,
}

impl Terrain {
    /// Bilinear ground height at `(x, y)`: `None` outside the grid, NaN next to missing data.
    pub fn at(&self, x: f64, y: f64) -> Option<f64> {
        let fx = (x - self.x0) / self.sx;
        let fy = (y - self.y0) / self.sy;
        let (mx, my) = ((self.nx - 1) as f64, (self.ny - 1) as f64);
        if !(0.0..=mx).contains(&fx) || !(0.0..=my).contains(&fy) {
            return None;
        }
        let i = (fx.floor() as usize).min(self.nx.saturating_sub(2));
        let j = (fy.floor() as usize).min(self.ny.saturating_sub(2));
        let (i1, j1) = ((i + 1).min(self.nx - 1), (j + 1).min(self.ny - 1));
        let (tx, ty) = (fx - i as f64, fy - j as f64);
        let z = |a: usize, b: usize| self.z[b * self.nx + a];
        let v = (1.0 - tx) * (1.0 - ty) * z(i, j)
            + tx * (1.0 - ty) * z(i1, j)
            + (1.0 - tx) * ty * z(i, j1)
            + tx * ty * z(i1, j1);
        Some(v)
    }

    fn max(&self) -> f64 {
        self.z
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// One sun position: unit horizontal direction towards the sun in the scene's coordinates and the
/// tangent of its elevation.
#[derive(Clone, Copy, Debug)]
pub struct Sun {
    pub dir: [f64; 2],
    pub tan_el: f64,
}

/// Walls with roof heights and optional terrain, indexed for ray walks.
pub struct Scene {
    segs: Vec<Seg>,
    tops: Vec<f64>,
    grid: Grid,
    terrain: Option<Terrain>,
    z_max: f64,
}

impl Scene {
    pub fn new(segs: Vec<Seg>, tops: Vec<f64>, terrain: Option<Terrain>) -> Scene {
        let (segs, tops): (Vec<Seg>, Vec<f64>) = segs
            .into_iter()
            .zip(tops)
            .filter(|(s, top)| {
                s.a != s.b && top.is_finite() && s.a.iter().chain(s.b.iter()).all(|v| v.is_finite())
            })
            .unzip();
        let mut bbox = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for s in &segs {
            bbox[0] = bbox[0].min(s.a[0].min(s.b[0]));
            bbox[1] = bbox[1].min(s.a[1].min(s.b[1]));
            bbox[2] = bbox[2].max(s.a[0].max(s.b[0]));
            bbox[3] = bbox[3].max(s.a[1].max(s.b[1]));
        }
        if segs.is_empty() {
            bbox = [0.0, 0.0, 1.0, 1.0];
        }
        let cells = (4.0 * segs.len() as f64).sqrt().clamp(1.0, 4096.0) as usize;
        let grid = Grid::build(&segs, bbox, cells);
        let z_max = tops
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(terrain.as_ref().map_or(f64::NEG_INFINITY, Terrain::max));
        Scene {
            segs,
            tops,
            grid,
            terrain,
            z_max,
        }
    }

    /// True when the receiver at `(x, y)` and height `z` gets direct sunlight from `sun`.
    pub fn sunlit(&self, p: [f64; 3], sun: &Sun) -> bool {
        if sun.tan_el <= 0.0 {
            return false;
        }
        let reach = (self.z_max - p[2]) / sun.tan_el;
        if reach.is_nan() || reach <= 0.0 {
            return true;
        }
        let o = [p[0], p[1]];
        let q = [o[0] + sun.dir[0] * reach, o[1] + sun.dir[1] * reach];
        let min_t = 1e-6 / reach; // a receiver on a wall is not shaded by that wall
        let mut blocked = false;
        self.grid.walk_segment(o, q, |items| {
            for &id in items {
                if let Some((t, u)) = segment_params(o, q, &self.segs[id as usize]) {
                    if t > min_t
                        && t <= 1.0
                        && (0.0..=1.0).contains(&u)
                        && self.tops[id as usize] > p[2] + t * reach * sun.tan_el
                    {
                        blocked = true;
                        return true;
                    }
                }
            }
            false
        });
        if blocked {
            return false;
        }
        if let Some(terrain) = &self.terrain {
            let step = terrain.sx.min(terrain.sy).abs();
            let mut d = step;
            while d <= reach {
                match terrain.at(o[0] + sun.dir[0] * d, o[1] + sun.dir[1] * d) {
                    Some(h) if h > p[2] + d * sun.tan_el + 1e-6 => return false,
                    Some(_) => {}
                    None => break,
                }
                d += step;
            }
        }
        true
    }
}

impl Scene {
    /// True when nothing blocks the straight line between the 3D points `e` (eye) and `t` (target): every
    /// wall it crosses has its roof below the line there, and so does the terrain. Walls touching either
    /// end are ignored, so targets on a roof or right at a wall can be seen.
    pub fn line_clear(&self, e: [f64; 3], t: [f64; 3]) -> bool {
        let (o, q) = ([e[0], e[1]], [t[0], t[1]]);
        let len = (q[0] - o[0]).hypot(q[1] - o[1]);
        if len == 0.0 {
            return true;
        }
        let eps = 1e-6 / len;
        let rise = t[2] - e[2];
        let mut blocked = false;
        self.grid.walk_segment(o, q, |items| {
            for &id in items {
                if let Some((s, u)) = segment_params(o, q, &self.segs[id as usize]) {
                    if s > eps
                        && s < 1.0 - eps
                        && (0.0..=1.0).contains(&u)
                        && self.tops[id as usize] > e[2] + s * rise + 1e-9
                    {
                        blocked = true;
                        return true;
                    }
                }
            }
            false
        });
        if blocked {
            return false;
        }
        if let Some(terrain) = &self.terrain {
            let step = terrain.sx.min(terrain.sy).abs();
            let (dx, dy) = ((q[0] - o[0]) / len, (q[1] - o[1]) / len);
            let mut d = step;
            while d < len - step * 0.5 {
                if let Some(h) = terrain.at(o[0] + dx * d, o[1] + dy * d) {
                    if h > e[2] + d / len * rise + 1e-6 {
                        return false;
                    }
                }
                d += step;
            }
        }
        true
    }
}

/// For every target `targets[k]`, 1 when it is visible from `eyes[owner[k]]`.
pub fn line_of_sight(
    scene: &Scene,
    eyes: &[[f64; 3]],
    targets: &[[f64; 3]],
    owner: &[usize],
) -> Vec<u8> {
    targets
        .par_iter()
        .zip(owner.par_iter())
        .map(|(t, &k)| u8::from(scene.line_clear(eyes[k], *t)))
        .collect()
}

/// Per receiver and day: the number of sunlit sun positions and the longest run of consecutive sunlit
/// positions. `suns[day_offsets[d]..day_offsets[d + 1]]` are the positions of day `d` in time order.
/// Both results are `receivers.len() x days` matrices (row-major).
///
/// Receivers on a facade pass `facing`: the facade's outward unit normal per receiver and the least dot
/// product between it and the horizontal direction to the sun; the sun counts only in front of the facade.
pub fn day_stats(
    scene: &Scene,
    receivers: &[[f64; 3]],
    suns: &[Sun],
    day_offsets: &[usize],
    facing: Option<(&[[f64; 2]], f64)>,
) -> (Vec<u32>, Vec<u32>) {
    let days = day_offsets.len().saturating_sub(1);
    let mut counts = vec![0u32; receivers.len() * days];
    let mut longest = vec![0u32; receivers.len() * days];
    if days == 0 {
        return (counts, longest);
    }
    counts
        .par_chunks_mut(days)
        .zip(longest.par_chunks_mut(days))
        .zip(receivers.par_iter().enumerate())
        .for_each(|((count_row, run_row), (k, p))| {
            let in_front = |sun: &Sun| {
                facing.is_none_or(|(normals, min_dot)| {
                    let n = normals[k];
                    n[0] * sun.dir[0] + n[1] * sun.dir[1] > min_dot
                })
            };
            for d in 0..days {
                let (mut count, mut run, mut best) = (0u32, 0u32, 0u32);
                for sun in &suns[day_offsets[d]..day_offsets[d + 1]] {
                    if in_front(sun) && scene.sunlit(*p, sun) {
                        count += 1;
                        run += 1;
                        best = best.max(run);
                    } else {
                        run = 0;
                    }
                }
                count_row[d] = count;
                run_row[d] = best;
            }
        });
    (counts, longest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<Seg> {
        let c = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        (0..4)
            .map(|k| Seg {
                a: c[k],
                b: c[(k + 1) % 4],
            })
            .collect()
    }

    fn sun(azimuth_deg: f64, elevation_deg: f64) -> Sun {
        let a = azimuth_deg.to_radians();
        Sun {
            dir: [a.sin(), a.cos()],
            tan_el: elevation_deg.to_radians().tan(),
        }
    }

    #[test]
    fn building_shadow_length_follows_the_elevation() {
        // 10 m high building, x 0..10, y 0..10; sun from the south at 45 degrees -> 10 m shadow north
        let scene = Scene::new(square(0.0, 0.0, 10.0, 10.0), vec![10.0; 4], None);
        let s = sun(180.0, 45.0);
        assert!(!scene.sunlit([5.0, 15.0, 0.0], &s));
        assert!(!scene.sunlit([5.0, 19.9, 0.0], &s));
        assert!(scene.sunlit([5.0, 20.1, 0.0], &s));
        assert!(scene.sunlit([5.0, -5.0, 0.0], &s)); // south side faces the sun
        assert!(scene.sunlit([5.0, 15.0, 6.0], &s)); // a receiver higher up sees over the roof
        assert!(!scene.sunlit([5.0, 15.0, 0.0], &sun(180.0, -1.0))); // below the horizon
    }

    #[test]
    fn terrain_ridge_casts_a_shadow() {
        // flat ground at 0 with a 20 m ridge along y = 50 (rows 4..6 of a 10 m grid)
        let (nx, ny) = (11, 11);
        let z: Vec<f64> = (0..nx * ny)
            .map(|k| {
                if (4..=6).contains(&(k / nx)) {
                    20.0
                } else {
                    0.0
                }
            })
            .collect();
        let terrain = Terrain {
            x0: 0.0,
            y0: 0.0,
            sx: 10.0,
            sy: 10.0,
            nx,
            ny,
            z,
        };
        let scene = Scene::new(Vec::new(), Vec::new(), Some(terrain));
        let s = sun(0.0, 30.0); // sun in the north: the ridge shades the ground south of it
        assert!(!scene.sunlit([50.0, 30.0, 0.0], &s));
        assert!(scene.sunlit([50.0, 80.0, 0.0], &s));
        assert_eq!(scene.terrain.as_ref().unwrap().at(50.0, 35.0), Some(10.0));
    }

    #[test]
    fn line_of_sight_over_and_behind_a_building() {
        // 10 m high building x 0..10; eye at 1.6 m west of it
        let scene = Scene::new(square(0.0, -5.0, 10.0, 5.0), vec![10.0; 4], None);
        let eye = [-10.0, 0.0, 1.6];
        assert!(!scene.line_clear(eye, [20.0, 0.0, 0.0])); // ground behind the building
        assert!(scene.line_clear(eye, [20.0, 0.0, 40.0])); // high enough to show over it
        assert!(scene.line_clear(eye, [-20.0, 0.0, 0.0])); // nothing in between
        assert!(scene.line_clear([-10.0, 0.0, 20.0], [5.0, 0.0, 10.0])); // its roof seen from above
        assert!(!scene.line_clear(eye, [5.0, 0.0, 10.0])); // but not from below
        let eyes = [eye];
        let seen = line_of_sight(
            &scene,
            &eyes,
            &[[20.0, 0.0, 0.0], [-20.0, 0.0, 0.0]],
            &[0, 0],
        );
        assert_eq!(seen, vec![0, 1]);
    }

    #[test]
    fn day_stats_count_and_longest_run() {
        // a wall east of the receiver blocks the morning sun only
        let wall = vec![Seg {
            a: [10.0, -50.0],
            b: [10.0, 50.0],
        }];
        let scene = Scene::new(wall, vec![30.0], None);
        let suns = [
            sun(90.0, 20.0),  // east, low: blocked
            sun(135.0, 30.0), // south-east: blocked
            sun(180.0, 40.0), // south: free
            sun(225.0, 30.0), // south-west: free
            sun(270.0, 20.0), // west: free
        ];
        let (counts, longest) = day_stats(&scene, &[[0.0, 0.0, 0.0]], &suns, &[0, 5, 5], None);
        assert_eq!(counts, vec![3, 0]);
        assert_eq!(longest, vec![3, 0]);
        let (counts, _) = day_stats(&scene, &[[0.0, 0.0, 0.0]], &suns[..], &[0, 2, 5], None);
        assert_eq!(counts, vec![0, 3]);
        // a facade facing west sees only the south-west and west sun
        let west: &[[f64; 2]] = &[[-1.0, 0.0]];
        let (counts, _) = day_stats(
            &scene,
            &[[0.0, 0.0, 0.0]],
            &suns,
            &[0, 5],
            Some((west, 0.0)),
        );
        assert_eq!(counts, vec![2]);
    }
}
