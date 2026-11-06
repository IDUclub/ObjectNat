//! 2.5D viewsheds on a regular grid: footprint rasterization and per-observer visible cells.
//!
//! Cell `(i, j)` of a [`CellGrid`] has its centre at `(x0 + (i + 0.5) * cell, y0 + (j + 0.5) * cell)`.
//! Targets are given as one absolute height per cell (NaN = not a target); an observer sees a cell
//! within its radius when the line of sight from its eye to the target is clear in the 2.5D [`Scene`].

use rayon::prelude::*;

use crate::insolation::Scene;
use crate::visibility::Seg;

#[derive(Clone, Copy, Debug)]
pub struct CellGrid {
    pub x0: f64,
    pub y0: f64,
    pub cell: f64,
    pub nx: usize,
    pub ny: usize,
}

impl CellGrid {
    #[inline]
    fn centre(&self, i: usize, j: usize) -> [f64; 2] {
        [
            self.x0 + (i as f64 + 0.5) * self.cell,
            self.y0 + (j as f64 + 0.5) * self.cell,
        ]
    }

    /// Range of cell indices whose centres lie in `[lo, hi]` along one axis (`origin`, `n` cells).
    fn span(&self, origin: f64, n: usize, lo: f64, hi: f64) -> Option<(usize, usize)> {
        let first = ((lo - origin) / self.cell - 0.5).ceil().max(0.0);
        let last = ((hi - origin) / self.cell - 0.5)
            .floor()
            .min(n as f64 - 1.0);
        (first <= last).then_some((first as usize, last as usize))
    }
}

/// Index of the polygon covering every cell centre (-1 for none); where polygons overlap the one with the
/// highest `priority` wins. `segments` are the ring edges of all polygons, `owner[k]` the polygon of edge
/// `k`; each polygon is filled with the even-odd rule, so holes stay empty.
pub fn rasterize(segs: &[Seg], owner: &[usize], priority: &[f64], grid: &CellGrid) -> Vec<i64> {
    let mut out = vec![-1i64; grid.nx * grid.ny];
    let mut by_owner: Vec<Vec<usize>> = vec![Vec::new(); priority.len()];
    for (k, &o) in owner.iter().enumerate() {
        if o < by_owner.len() {
            by_owner[o].push(k);
        }
    }
    let mut xs: Vec<f64> = Vec::new();
    for (poly, edges) in by_owner.iter().enumerate() {
        if edges.is_empty() {
            continue;
        }
        let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
        for &k in edges {
            ymin = ymin.min(segs[k].a[1].min(segs[k].b[1]));
            ymax = ymax.max(segs[k].a[1].max(segs[k].b[1]));
        }
        let Some((j0, j1)) = grid.span(grid.y0, grid.ny, ymin, ymax) else {
            continue;
        };
        for j in j0..=j1 {
            let yc = grid.y0 + (j as f64 + 0.5) * grid.cell;
            xs.clear();
            for &k in edges {
                let (a, b) = (segs[k].a, segs[k].b);
                if (a[1] <= yc) != (b[1] <= yc) {
                    xs.push(a[0] + (yc - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
                }
            }
            xs.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
            for pair in xs.chunks_exact(2) {
                if let Some((i0, i1)) = grid.span(grid.x0, grid.nx, pair[0], pair[1]) {
                    for i in i0..=i1 {
                        let c = j * grid.nx + i;
                        let cur = out[c];
                        if cur < 0 || priority[poly] > priority[cur as usize] {
                            out[c] = poly as i64;
                        }
                    }
                }
            }
        }
    }
    out
}

/// Flat indices of the cells visible from `eye` within horizontal radius `r`.
pub fn visible_cells(
    scene: &Scene,
    eye: [f64; 3],
    r: f64,
    grid: &CellGrid,
    target_z: &[f64],
) -> Vec<u32> {
    let mut out = Vec::new();
    let Some((j0, j1)) = grid.span(grid.y0, grid.ny, eye[1] - r, eye[1] + r) else {
        return out;
    };
    let r2 = r * r;
    for j in j0..=j1 {
        let dy = grid.y0 + (j as f64 + 0.5) * grid.cell - eye[1];
        let half = (r2 - dy * dy).max(0.0).sqrt();
        let Some((i0, i1)) = grid.span(grid.x0, grid.nx, eye[0] - half, eye[0] + half) else {
            continue;
        };
        for i in i0..=i1 {
            let c = j * grid.nx + i;
            let z = target_z[c];
            if !z.is_finite() {
                continue;
            }
            let p = grid.centre(i, j);
            if scene.line_clear(eye, [p[0], p[1], z]) {
                out.push(c as u32);
            }
        }
    }
    out
}

/// Visible cells of every observer, in parallel.
pub fn visible_cells_many(
    scene: &Scene,
    eyes: &[[f64; 3]],
    radii: &[f64],
    grid: &CellGrid,
    target_z: &[f64],
) -> Vec<Vec<u32>> {
    eyes.par_iter()
        .zip(radii.par_iter())
        .map(|(e, &r)| visible_cells(scene, *e, r, grid, target_z))
        .collect()
}

/// Per cell: the number of observers that see it and the sum of their weights.
pub fn count_grid(
    scene: &Scene,
    eyes: &[[f64; 3]],
    radii: &[f64],
    weights: Option<&[f64]>,
    grid: &CellGrid,
    target_z: &[f64],
) -> (Vec<u32>, Option<Vec<f64>>) {
    let mut counts = vec![0u32; grid.nx * grid.ny];
    let mut sums = weights.map(|_| vec![0.0f64; grid.nx * grid.ny]);
    const CHUNK: usize = 512; // bounds the memory of the collected index lists
    for start in (0..eyes.len()).step_by(CHUNK) {
        let end = (start + CHUNK).min(eyes.len());
        let seen = visible_cells_many(scene, &eyes[start..end], &radii[start..end], grid, target_z);
        for (k, cells) in seen.iter().enumerate() {
            let w = weights.map(|w| w[start + k]);
            for &c in cells {
                counts[c as usize] += 1;
                if let (Some(s), Some(w)) = (sums.as_mut(), w) {
                    s[c as usize] += w;
                }
            }
        }
    }
    (counts, sums)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(c: &[[f64; 2]]) -> Vec<Seg> {
        (0..c.len())
            .map(|k| Seg {
                a: c[k],
                b: c[(k + 1) % c.len()],
            })
            .collect()
    }

    #[test]
    fn rasterize_fills_polygons_and_keeps_holes() {
        let grid = CellGrid {
            x0: 0.0,
            y0: 0.0,
            cell: 1.0,
            nx: 10,
            ny: 10,
        };
        // a 2..8 square with a 4..6 courtyard, and a taller 7..9 square overlapping it
        let mut segs = ring(&[[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]]);
        segs.extend(ring(&[[4.0, 4.0], [4.0, 6.0], [6.0, 6.0], [6.0, 4.0]]));
        segs.extend(ring(&[[7.0, 7.0], [9.0, 7.0], [9.0, 9.0], [7.0, 9.0]]));
        let owner = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1];
        let out = rasterize(&segs, &owner, &[10.0, 20.0], &grid);
        let at = |i: usize, j: usize| out[j * 10 + i];
        assert_eq!(at(2, 2), 0);
        assert_eq!(at(4, 4), -1); // courtyard
        assert_eq!(at(7, 7), 1); // overlap goes to the taller one
        assert_eq!(at(8, 8), 1);
        assert_eq!(at(0, 0), -1);
        assert_eq!(out.iter().filter(|&&v| v == 0).count(), 36 - 4 - 1);
    }

    #[test]
    fn visible_cells_stop_behind_a_wall() {
        let grid = CellGrid {
            x0: -10.0,
            y0: -10.0,
            cell: 1.0,
            nx: 20,
            ny: 20,
        };
        let wall = vec![Seg {
            a: [3.0, -10.0],
            b: [3.0, 10.0],
        }];
        let scene = Scene::new(wall, vec![5.0], None);
        let target_z = vec![0.0; 400];
        let seen = visible_cells(&scene, [0.0, 0.0, 1.6], 8.0, &grid, &target_z);
        let has = |x: f64, y: f64| {
            let (i, j) = ((x + 10.0) as usize, (y + 10.0) as usize);
            seen.contains(&((j * 20 + i) as u32))
        };
        assert!(has(-5.5, 0.5) && has(2.5, 0.5));
        assert!(!has(5.5, 0.5)); // behind the 5 m wall
        assert!(!has(-9.5, 0.5)); // outside the radius
        let (counts, sums) = count_grid(
            &scene,
            &[[0.0, 0.0, 1.6]; 2],
            &[8.0; 2],
            Some(&[1.0, 2.5]),
            &grid,
            &target_z,
        );
        assert_eq!(counts[10 * 20 + 7], 2);
        assert_eq!(sums.unwrap()[10 * 20 + 7], 3.5);
    }
}
