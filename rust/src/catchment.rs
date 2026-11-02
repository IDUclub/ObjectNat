//! Visibility catchment: how many observers see each cell of a regular grid, and the outlines of
//! classified cell regions.
//!
//! * [`CountGrid`] accumulates visibility polygons into a cell grid. A cell counts as visible from an
//!   observer when its centre lies inside the observer's visibility polygon. Each polygon is filled
//!   row by row (scanline, even-odd rule): every covered run of cells adds its weight at the run's
//!   first cell and subtracts it after the last one, so the work per observer is proportional to the
//!   polygon boundary, not to its area. A prefix sum along each row turns the differences into values.
//!   Rows are shared between threads through atomics, so no per-thread copy of the grid is needed.
//! * [`trace_label_rings`] turns a label grid into closed rings around 4-connected regions of equal
//!   label: exteriors counter-clockwise, holes clockwise, only corner vertices kept.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use rayon::prelude::*;

use crate::visibility::Pt;

/// Regular grid: cell `(i, j)` spans `x0 + i * cell .. x0 + (i + 1) * cell` (same for `y`, row `j`).
#[derive(Clone, Copy, Debug)]
pub struct GridSpec {
    pub x0: f64,
    pub y0: f64,
    pub cell: f64,
    pub nx: usize,
    pub ny: usize,
}

/// Covered runs of cells, as `(row, first, last_exclusive)`, for a ring under the even-odd rule.
pub fn ring_runs(
    ring: &[Pt],
    spec: &GridSpec,
    xs: &mut Vec<(u32, f64)>,
    out: &mut Vec<(u32, u32, u32)>,
) {
    xs.clear();
    out.clear();
    let n = ring.len();
    if n < 3 {
        return;
    }
    let inv = 1.0 / spec.cell;
    for k in 0..n {
        let a = ring[k];
        let b = ring[(k + 1) % n];
        if a[1] == b[1] {
            continue;
        }
        let (lo, hi) = if a[1] < b[1] { (a, b) } else { (b, a) };
        // rows whose centre y_c satisfies lo.y <= y_c < hi.y (half-open, so shared vertices count once)
        let j_first = ((lo[1] - spec.y0) * inv - 0.5).ceil().max(0.0);
        let j_end = ((hi[1] - spec.y0) * inv - 0.5).ceil().min(spec.ny as f64);
        if j_first >= j_end || j_first.is_nan() || j_end.is_nan() {
            continue;
        }
        let slope = (hi[0] - lo[0]) / (hi[1] - lo[1]);
        let mut j = j_first as usize;
        let j_end = j_end as usize;
        while j < j_end {
            let yc = spec.y0 + (j as f64 + 0.5) * spec.cell;
            xs.push((j as u32, lo[0] + (yc - lo[1]) * slope));
            j += 1;
        }
    }
    xs.sort_unstable_by(|p, q| p.0.cmp(&q.0).then(p.1.total_cmp(&q.1)));
    let nx = spec.nx as f64;
    for pair in xs.chunks_exact(2) {
        let (j, xa) = pair[0];
        let (j2, xb) = pair[1];
        debug_assert_eq!(j, j2);
        if j != j2 {
            continue;
        }
        // cells whose centre x_c satisfies xa <= x_c < xb
        let i0 = ((xa - spec.x0) * inv - 0.5).ceil().clamp(0.0, nx);
        let i1 = ((xb - spec.x0) * inv - 0.5).ceil().clamp(0.0, nx);
        if i0 < i1 {
            out.push((j, i0 as u32, i1 as u32));
        }
    }
}

/// Shared accumulator of observer counts (and optional weights) over a grid.
pub struct CountGrid {
    spec: GridSpec,
    counts: Vec<AtomicI64>,
    weights: Option<Vec<AtomicU64>>,
}

impl CountGrid {
    pub fn new(spec: GridSpec, weighted: bool) -> CountGrid {
        let len = spec.ny * (spec.nx + 1);
        CountGrid {
            spec,
            counts: (0..len).map(|_| AtomicI64::new(0)).collect(),
            weights: weighted.then(|| (0..len).map(|_| AtomicU64::new(0f64.to_bits())).collect()),
        }
    }

    pub fn spec(&self) -> &GridSpec {
        &self.spec
    }

    fn add_weight(cell: &AtomicU64, w: f64) {
        let mut cur = cell.load(Ordering::Relaxed);
        loop {
            let new = (f64::from_bits(cur) + w).to_bits();
            match cell.compare_exchange_weak(cur, new, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return,
                Err(v) => cur = v,
            }
        }
    }

    /// Add one observer's covered runs with weight `w`.
    pub fn add_runs(&self, runs: &[(u32, u32, u32)], w: f64) {
        let stride = self.spec.nx + 1;
        for &(j, i0, i1) in runs {
            let row = j as usize * stride;
            self.counts[row + i0 as usize].fetch_add(1, Ordering::Relaxed);
            self.counts[row + i1 as usize].fetch_sub(1, Ordering::Relaxed);
            if let Some(ws) = &self.weights {
                Self::add_weight(&ws[row + i0 as usize], w);
                Self::add_weight(&ws[row + i1 as usize], -w);
            }
        }
    }

    /// Row-major `(ny, nx)` counts and optional weight sums (row 0 is the bottom row, `y0`).
    pub fn finish(self) -> (Vec<u32>, Option<Vec<f64>>) {
        let (nx, ny) = (self.spec.nx, self.spec.ny);
        let stride = nx + 1;
        let mut counts = vec![0u32; nx * ny];
        counts
            .par_chunks_mut(nx.max(1))
            .enumerate()
            .for_each(|(j, row)| {
                let mut acc = 0i64;
                for (i, v) in row.iter_mut().enumerate() {
                    acc += self.counts[j * stride + i].load(Ordering::Relaxed);
                    *v = acc.max(0) as u32;
                }
            });
        let weights = self.weights.as_ref().map(|ws| {
            let mut out = vec![0f64; nx * ny];
            out.par_chunks_mut(nx.max(1))
                .enumerate()
                .for_each(|(j, row)| {
                    let mut acc = 0f64;
                    for (i, v) in row.iter_mut().enumerate() {
                        acc += f64::from_bits(ws[j * stride + i].load(Ordering::Relaxed));
                        *v = acc;
                    }
                });
            out
        });
        // a weight sum where no observer sees the cell is exactly zero (no rounding residue)
        let weights = weights.map(|mut w| {
            w.iter_mut().zip(&counts).for_each(|(v, &c)| {
                if c == 0 {
                    *v = 0.0;
                }
            });
            w
        });
        (counts, weights)
    }
}

const E: u8 = 0;
const N: u8 = 1;
const W: u8 = 2;
const S: u8 = 3;

#[inline]
fn step(v: (usize, usize), d: u8) -> (usize, usize) {
    match d {
        E => (v.0 + 1, v.1),
        N => (v.0, v.1 + 1),
        W => (v.0 - 1, v.1),
        _ => (v.0, v.1 - 1),
    }
}

/// Rings around the 4-connected regions of every non-zero label of a row-major `(ny, nx)` grid.
///
/// Returns `(label, ring)` pairs; ring vertices are grid-vertex indices `(i, j)` (vertex `(i, j)` is
/// the lower-left corner of cell `(i, j)`), not closed, with the region on the left: exteriors are
/// counter-clockwise and holes clockwise. Collinear vertices are dropped.
pub fn trace_label_rings(labels: &[u8], nx: usize, ny: usize) -> Vec<(u8, Vec<(usize, usize)>)> {
    let vx = nx + 1;
    let lab = |i: isize, j: isize| -> u8 {
        if i < 0 || j < 0 || i >= nx as isize || j >= ny as isize {
            0
        } else {
            labels[j as usize * nx + i as usize]
        }
    };
    let mut present = [false; 256];
    labels.iter().for_each(|&l| present[l as usize] = true);

    let mut rings = Vec::new();
    let mut out_dirs = vec![0u8; vx * (ny + 1)];
    for label in 1..=255u8 {
        if !present[label as usize] {
            continue;
        }
        // directed boundary edges with the region on the left, as outgoing-direction bits per vertex
        let mut starts: Vec<usize> = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                if labels[j * nx + i] != label {
                    continue;
                }
                let (ii, jj) = (i as isize, j as isize);
                let mut mark = |v: usize, d: u8| {
                    if out_dirs[v] == 0 {
                        starts.push(v);
                    }
                    out_dirs[v] |= 1 << d;
                };
                if lab(ii, jj - 1) != label {
                    mark(j * vx + i, E);
                }
                if lab(ii + 1, jj) != label {
                    mark(j * vx + i + 1, N);
                }
                if lab(ii, jj + 1) != label {
                    mark((j + 1) * vx + i + 1, W);
                }
                if lab(ii - 1, jj) != label {
                    mark((j + 1) * vx + i, S);
                }
            }
        }
        for &s in &starts {
            while out_dirs[s] != 0 {
                let start = (s % vx, s / vx);
                let d0 = out_dirs[s].trailing_zeros() as u8;
                out_dirs[s] &= !(1 << d0);
                let mut ring = vec![start];
                let mut d = d0;
                let mut v = step(start, d);
                loop {
                    let idx = v.1 * vx + v.0;
                    let mut avail = out_dirs[idx];
                    if v == start {
                        avail |= 1 << d0;
                    }
                    // turn left first: keeps diagonal-only neighbours in separate regions
                    let next = [(d + 1) % 4, d, (d + 3) % 4]
                        .into_iter()
                        .find(|nd| avail & (1 << nd) != 0)
                        .expect("boundary edges form closed rings");
                    if v == start && next == d0 {
                        // the start vertex is a corner only if the direction changes there
                        if next == d {
                            ring.remove(0);
                        }
                        break;
                    }
                    out_dirs[idx] &= !(1 << next);
                    if next != d {
                        ring.push(v);
                    }
                    d = next;
                    v = step(v, d);
                }
                for simple in split_at_touches(ring) {
                    rings.push((label, simple));
                }
            }
        }
    }
    rings
}

/// Split a closed orthogonal ring at vertices it passes more than once (two cells of the region, or two
/// of its outside, meeting only at a corner) into simple rings, dropping collinear vertices. Orientation
/// is kept, so a self-touching exterior becomes an exterior plus a hole touching it at one point.
fn split_at_touches(ring: Vec<(usize, usize)>) -> Vec<Vec<(usize, usize)>> {
    let mut seen: HashMap<(usize, usize), usize> = HashMap::new();
    let mut path: Vec<(usize, usize)> = Vec::with_capacity(ring.len());
    let mut out = Vec::new();
    for v in ring {
        if let Some(&pos) = seen.get(&v) {
            let lp: Vec<(usize, usize)> = path.drain(pos + 1..).collect();
            for p in &lp {
                seen.remove(p);
            }
            let mut sub = Vec::with_capacity(lp.len() + 1);
            sub.push(v);
            sub.extend(lp);
            push_corners(sub, &mut out);
        } else {
            seen.insert(v, path.len());
            path.push(v);
        }
    }
    push_corners(path, &mut out);
    out
}

fn push_corners(ring: Vec<(usize, usize)>, out: &mut Vec<Vec<(usize, usize)>>) {
    let n = ring.len();
    let corners: Vec<(usize, usize)> = (0..n)
        .filter(|&k| {
            let (a, b, c) = (ring[(k + n - 1) % n], ring[k], ring[(k + 1) % n]);
            // orthogonal edges: b is a corner unless a, b, c share a row or a column
            !((a.0 == b.0 && b.0 == c.0) || (a.1 == b.1 && b.1 == c.1))
        })
        .map(|k| ring[k])
        .collect();
    if corners.len() >= 4 {
        out.push(corners);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(nx: usize, ny: usize) -> GridSpec {
        GridSpec {
            x0: 0.0,
            y0: 0.0,
            cell: 1.0,
            nx,
            ny,
        }
    }

    fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<Pt> {
        vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    }

    fn grid_of(
        rings: &[(Vec<Pt>, f64)],
        s: GridSpec,
        weighted: bool,
    ) -> (Vec<u32>, Option<Vec<f64>>) {
        let grid = CountGrid::new(s, weighted);
        let (mut xs, mut runs) = (Vec::new(), Vec::new());
        for (ring, w) in rings {
            ring_runs(ring, &s, &mut xs, &mut runs);
            grid.add_runs(&runs, *w);
        }
        grid.finish()
    }

    #[test]
    fn square_covers_cells_by_centre() {
        // centres at 0.5, 1.5, ...; [0.6, 3.4] covers centres 1.5 and 2.5 on both axes
        let (counts, _) = grid_of(&[(square(0.6, 0.6, 3.4, 3.4), 1.0)], spec(5, 5), false);
        let covered: Vec<usize> = (0..25).filter(|&k| counts[k] == 1).collect();
        assert_eq!(covered, vec![6, 7, 11, 12]);
        assert!(counts.iter().all(|&c| c <= 1));
    }

    #[test]
    fn overlaps_add_up_with_weights() {
        let rings = [
            (square(0.0, 0.0, 3.0, 1.0), 2.0),
            (square(2.0, 0.0, 5.0, 1.0), 0.5),
        ];
        let (counts, weights) = grid_of(&rings, spec(5, 1), true);
        assert_eq!(counts, vec![1, 1, 2, 1, 1]);
        assert_eq!(weights.unwrap(), vec![2.0, 2.0, 2.5, 0.5, 0.5]);
    }

    #[test]
    fn ring_outside_the_grid_is_clipped() {
        let (counts, _) = grid_of(
            &[(square(-10.0, -10.0, 10.0, 10.0), 1.0)],
            spec(3, 2),
            false,
        );
        assert!(counts.iter().all(|&c| c == 1));
    }

    #[test]
    fn traces_exterior_and_hole() {
        // 3x3 block of label 1 with a hole in the middle
        let mut labels = vec![1u8; 9];
        labels[4] = 0;
        let rings = trace_label_rings(&labels, 3, 3);
        assert_eq!(rings.len(), 2);
        let area = |r: &Vec<(usize, usize)>| -> f64 {
            let n = r.len();
            (0..n)
                .map(|k| {
                    let (a, b) = (r[k], r[(k + 1) % n]);
                    a.0 as f64 * b.1 as f64 - b.0 as f64 * a.1 as f64
                })
                .sum::<f64>()
                / 2.0
        };
        let mut areas: Vec<f64> = rings.iter().map(|(_, r)| area(r)).collect();
        areas.sort_by(f64::total_cmp);
        assert_eq!(areas, vec![-1.0, 9.0]);
        assert!(rings.iter().all(|(_, r)| r.len() == 4));
    }

    #[test]
    fn diagonal_cells_are_separate_regions() {
        let labels = vec![1u8, 0, 0, 1];
        let rings = trace_label_rings(&labels, 2, 2);
        assert_eq!(rings.len(), 2);
        assert!(rings.iter().all(|(_, r)| r.len() == 4));
    }

    #[test]
    fn self_touching_rings_are_split() {
        // a ring of label 1 closed only by a diagonal touch (top-left corner cell missing,
        // its neighbours touch at a corner): the outline splits into an exterior and a hole
        #[rustfmt::skip]
        let labels = vec![
            1u8, 1, 1,
            1, 0, 1,
            1, 1, 0,
        ];
        let rings = trace_label_rings(&labels, 3, 3);
        for (_, r) in &rings {
            let mut uniq = r.clone();
            uniq.sort();
            uniq.dedup();
            assert_eq!(uniq.len(), r.len(), "ring {r:?} repeats a vertex");
        }
        // a hole of label 0 touching its own outline: two holes or exterior + hole
        let mut holes = vec![1u8; 16];
        holes[5] = 0;
        holes[10] = 0;
        let rings = trace_label_rings(&holes, 4, 4);
        assert_eq!(rings.len(), 3);
    }

    #[test]
    fn adjacent_labels_get_their_own_rings() {
        let labels = vec![1u8, 1, 2, 2];
        let rings = trace_label_rings(&labels, 4, 1);
        let mut found: Vec<(u8, usize)> = rings.iter().map(|(l, r)| (*l, r.len())).collect();
        found.sort();
        assert_eq!(found, vec![(1, 4), (2, 4)]);
    }
}
