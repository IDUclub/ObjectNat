//! Conservative point/polygon classification. GEOS resolves every uncertain result in Python.
//!
//! Coordinates are translated before crossing arithmetic. Outward-rounded intervals enclose
//! the exact crossings of the original f64 coordinates, including translation error. The eps
//! halo is additional protection, not an assumption about a nearly horizontal edge's condition.

use rayon::prelude::*;

use crate::visibility::{Grid, Seg};

pub const OUTSIDE: u8 = 0;
pub const INSIDE: u8 = 1;
pub const UNCERTAIN: u8 = 2;

fn up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        x
    } else if x == 0.0 {
        f64::from_bits(1)
    } else {
        f64::from_bits(if x > 0.0 {
            x.to_bits() + 1
        } else {
            x.to_bits() - 1
        })
    }
}

fn down(x: f64) -> f64 {
    -up(-x)
}

#[derive(Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn shifted(x: f64, origin: f64) -> Self {
        let v = x - origin;
        Self {
            lo: down(v),
            hi: up(v),
        }
    }

    fn sub(self, b: Self) -> Self {
        Self {
            lo: down(self.lo - b.hi),
            hi: up(self.hi - b.lo),
        }
    }

    fn add(self, b: Self) -> Self {
        Self {
            lo: down(self.lo + b.lo),
            hi: up(self.hi + b.hi),
        }
    }

    fn mul(self, b: Self) -> Self {
        let v = [
            self.lo * b.lo,
            self.lo * b.hi,
            self.hi * b.lo,
            self.hi * b.hi,
        ];
        Self {
            lo: down(v.iter().copied().fold(f64::INFINITY, f64::min)),
            hi: up(v.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
        }
    }

    fn div(self, b: Self) -> Self {
        if b.lo <= 0.0 && b.hi >= 0.0 {
            return Self {
                lo: f64::NEG_INFINITY,
                hi: f64::INFINITY,
            };
        }
        self.mul(Self {
            lo: down(1.0 / b.hi),
            hi: up(1.0 / b.lo),
        })
    }
}

struct Polygon {
    rings: Vec<Vec<[Interval; 2]>>,
    bbox: [f64; 4],
    origin: [f64; 2],
    eps: f64,
}

impl Polygon {
    fn new(rings: &[Vec<[f64; 2]>]) -> Self {
        let mut bbox = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        let mut scale = 1.0_f64;
        for p in rings.iter().flatten() {
            for k in 0..2 {
                bbox[k] = bbox[k].min(p[k]);
                bbox[k + 2] = bbox[k + 2].max(p[k]);
                scale = scale.max(p[k].abs());
            }
        }
        let origin = [bbox[0], bbox[1]];
        // At ~1e6 m this is 0.1 micrometre; larger coordinates get a scale-aware halo.
        let eps = 1e-7_f64.max(128.0 * f64::EPSILON * scale);
        let rings = rings
            .iter()
            .map(|ring| {
                ring.iter()
                    .map(|p| {
                        [
                            Interval::shifted(p[0], origin[0]),
                            Interval::shifted(p[1], origin[1]),
                        ]
                    })
                    .collect()
            })
            .collect();
        for k in 0..2 {
            bbox[k] = down(bbox[k] - eps);
            bbox[k + 2] = up(bbox[k + 2] + eps);
        }
        Self {
            rings,
            bbox,
            origin,
            eps,
        }
    }

    /// False means a row near a vertex/horizontal edge, or arithmetic too ill-conditioned to decide.
    fn crossings(&self, y: f64, out: &mut Vec<Interval>) -> bool {
        out.clear();
        let y = Interval::shifted(y, self.origin[1]);
        for ring in &self.rings {
            for i in 0..ring.len() {
                let a = ring[i];
                let b = ring[(i + 1) % ring.len()];
                if y.lo <= up(a[1].hi + self.eps) && y.hi >= down(a[1].lo - self.eps) {
                    return false;
                }
                if (a[1].hi < y.lo && b[1].lo > y.hi) || (b[1].hi < y.lo && a[1].lo > y.hi) {
                    let x = a[0].add(y.sub(a[1]).div(b[1].sub(a[1])).mul(b[0].sub(a[0])));
                    // Non-finite arithmetic, including overflow, always goes back to GEOS.
                    if !x.lo.is_finite() || !x.hi.is_finite() {
                        return false;
                    }
                    out.push(x);
                }
            }
        }
        out.sort_unstable_by(|a, b| a.lo.total_cmp(&b.lo));
        out.len() % 2 == 0
    }

    fn contains_bbox(&self, p: [f64; 2]) -> bool {
        p[0] >= self.bbox[0] && p[0] <= self.bbox[2] && p[1] >= self.bbox[1] && p[1] <= self.bbox[3]
    }

    fn point(&self, p: [f64; 2], crossings: &mut Vec<Interval>) -> u8 {
        if !self.contains_bbox(p) {
            return OUTSIDE;
        }
        if !self.crossings(p[1], crossings) {
            return UNCERTAIN;
        }
        let x = Interval::shifted(p[0], self.origin[0]);
        let mut inside = false;
        for c in crossings {
            if x.hi >= down(c.lo - self.eps) && x.lo <= up(c.hi + self.eps) {
                return UNCERTAIN;
            }
            if c.hi < x.lo {
                inside = !inside;
            }
        }
        if inside {
            INSIDE
        } else {
            OUTSIDE
        }
    }

    fn grid(&self, gx: &[f64], gy: &[f64]) -> Hits {
        let mut result = Hits::default();
        let x0 = gx.partition_point(|&x| x < self.bbox[0]);
        let x1 = gx.partition_point(|&x| x <= self.bbox[2]);
        let y0 = gy.partition_point(|&y| y < self.bbox[1]);
        let y1 = gy.partition_point(|&y| y <= self.bbox[3]);
        if x0 == x1 || y0 == y1 {
            return result;
        }
        let xs: Vec<_> = gx[x0..x1]
            .iter()
            .map(|&x| Interval::shifted(x, self.origin[0]))
            .collect();
        let mut row = vec![OUTSIDE; xs.len()];
        let mut crossings = Vec::new();
        for (iy, &y) in gy.iter().enumerate().take(y1).skip(y0) {
            row.fill(OUTSIDE);
            if self.crossings(y, &mut crossings) {
                // Only emit certainly interior intervals. Overlapping crossing intervals remain
                // uncertain, so sorting rounded endpoints cannot change parity outside the halo.
                for pair in crossings.chunks_exact(2) {
                    let lo = xs.partition_point(|x| x.lo <= pair[0].hi);
                    let hi = xs.partition_point(|x| x.hi < pair[1].lo);
                    if lo < hi {
                        row[lo..hi].fill(INSIDE);
                    }
                }
                for c in &crossings {
                    let lo = xs.partition_point(|x| x.hi < down(c.lo - self.eps));
                    let hi = xs.partition_point(|x| x.lo <= up(c.hi + self.eps));
                    row[lo..hi].fill(UNCERTAIN);
                }
            } else {
                row.fill(UNCERTAIN);
            }
            for (ix, &state) in row.iter().enumerate() {
                result.push(iy * gx.len() + x0 + ix, 0, state);
            }
        }
        result
    }
}

#[derive(Default)]
pub struct Hits {
    pub inside: Vec<(usize, usize)>,
    pub uncertain: Vec<(usize, usize)>,
}

impl Hits {
    fn push(&mut self, point: usize, polygon: usize, state: u8) {
        match state {
            INSIDE => self.inside.push((point, polygon)),
            UNCERTAIN => self.uncertain.push((point, polygon)),
            _ => (),
        }
    }
}

pub fn grid_pairs(polygons: &[Vec<Vec<[f64; 2]>>], gx: &[f64], gy: &[f64]) -> Hits {
    let batches: Vec<_> = polygons
        .par_iter()
        .map(|p| Polygon::new(p).grid(gx, gy))
        .collect();
    let mut result = Hits::default();
    for (polygon, batch) in batches.into_iter().enumerate() {
        result
            .inside
            .extend(batch.inside.into_iter().map(|(p, _)| (p, polygon)));
        result
            .uncertain
            .extend(batch.uncertain.into_iter().map(|(p, _)| (p, polygon)));
    }
    result
}

pub fn area_states(hits: &Hits, count: usize) -> Vec<u8> {
    let mut states = vec![OUTSIDE; count];
    for &(p, _) in &hits.uncertain {
        states[p] = UNCERTAIN;
    }
    for &(p, _) in &hits.inside {
        states[p] = INSIDE;
    }
    states
}

pub fn point_pairs(polygons: &[Vec<Vec<[f64; 2]>>], points: &[[f64; 2]]) -> Hits {
    if polygons.is_empty() || points.is_empty() {
        return Hits::default();
    }
    let polygons: Vec<_> = polygons.iter().map(|p| Polygon::new(p)).collect();
    let boxes: Vec<_> = polygons
        .iter()
        .map(|p| Seg {
            a: [p.bbox[0], p.bbox[1]],
            b: [p.bbox[2], p.bbox[3]],
        })
        .collect();
    let bbox = boxes.iter().fold(
        [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ],
        |mut b, s| {
            for k in 0..2 {
                b[k] = b[k].min(s.a[k]);
                b[k + 2] = b[k + 2].max(s.b[k]);
            }
            b
        },
    );
    let axis = ((polygons.len() as f64).sqrt().ceil() as usize).clamp(1, 128);
    let index = Grid::build(&boxes, bbox, axis);
    let batches: Vec<_> = points
        .par_chunks(256)
        .enumerate()
        .map(|(chunk, points)| {
            let mut hits = Hits::default();
            let mut crossings = Vec::new();
            for (i, &p) in points.iter().enumerate() {
                for &polygon in index.ids(index.cell_at(p)) {
                    hits.push(
                        chunk * 256 + i,
                        polygon as usize,
                        polygons[polygon as usize].point(p, &mut crossings),
                    );
                }
            }
            hits
        })
        .collect();
    let mut result = Hits::default();
    for batch in batches {
        result.inside.extend(batch.inside);
        result.uncertain.extend(batch.uncertain);
    }
    result
}

#[cfg(test)]
#[path = "grid_tests.rs"]
mod tests;
