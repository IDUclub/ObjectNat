//! Direct sunlight on receivers among buildings and terrain.
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
    /// Bilinear ground height at `(x, y)`, `None` outside the grid or on missing data.
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
        v.is_finite().then_some(v)
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

/// `receivers.len() x suns.len()` matrix (row-major) of 1 where the receiver is sunlit.
pub fn sunlit_matrix(scene: &Scene, receivers: &[[f64; 3]], suns: &[Sun]) -> Vec<u8> {
    let t = suns.len();
    let mut out = vec![0u8; receivers.len() * t];
    if t == 0 {
        return out;
    }
    out.par_chunks_mut(t)
        .zip(receivers.par_iter())
        .for_each(|(row, p)| {
            for (cell, sun) in row.iter_mut().zip(suns) {
                *cell = u8::from(scene.sunlit(*p, sun));
            }
        });
    out
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
}
