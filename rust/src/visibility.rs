//! Exact 2D visibility polygons inside a regular polygonal view disc.
//!
//! The view disc is the same regular polygon GEOS builds for `Point.buffer(r, quad_segs=q)`
//! (`4 * q` vertices at angles `k * 2π / (4q)`), so results are directly comparable with the
//! shapely-based reference implementation in ObjectNat.
//!
//! Algorithm (per observer):
//! 1. take the obstacle segments near the observer from a global uniform grid;
//! 2. clip them to the (convex) view polygon and add the polygon's own edges as walls;
//! 3. collect *critical directions*: every segment endpoint and every crossing of two segments;
//! 4. between two consecutive critical directions the nearest wall cannot change, so one ray cast
//!    through the middle of the interval (grid DDA traversal) identifies that wall, and intersecting
//!    its supporting line with both bounding directions yields the exact boundary vertices.
//!
//! The result is the exact visible region (up to floating point), a single star-shaped ring.

pub type Pt = [f64; 2];

#[derive(Clone, Copy, Debug)]
pub struct Seg {
    pub a: Pt,
    pub b: Pt,
}

/// An obstacle wall. `one_sided` walls belong to a valid polygon ring oriented with the
/// obstacle's interior on the left; seen from that side they are always hidden behind another
/// wall of the same polygon, so they can be skipped (back-face culling).
#[derive(Clone, Copy, Debug)]
pub struct Wall {
    pub seg: Seg,
    pub one_sided: bool,
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
fn norm(p: Pt) -> f64 {
    p[0].hypot(p[1])
}

/// Squared distance from point `p` to segment `s`.
fn dist2_point_segment(p: Pt, s: &Seg) -> f64 {
    let ab = sub(s.b, s.a);
    let ap = sub(p, s.a);
    let len2 = dot(ab, ab);
    let t = if len2 > 0.0 {
        (dot(ap, ab) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let c = [s.a[0] + t * ab[0] - p[0], s.a[1] + t * ab[1] - p[1]];
    dot(c, c)
}

/// Uniform grid over segment bounding boxes, stored in CSR form.
pub struct Grid {
    x0: f64,
    y0: f64,
    cell: f64,
    nx: usize,
    ny: usize,
    starts: Vec<u32>,
    items: Vec<u32>,
}

impl Grid {
    pub fn build(segs: &[Seg], bbox: [f64; 4], cells_per_axis: usize) -> Grid {
        let w = (bbox[2] - bbox[0]).max(1e-9);
        let h = (bbox[3] - bbox[1]).max(1e-9);
        let cell = (w.max(h) / cells_per_axis.max(1) as f64).max(1e-9);
        let nx = ((w / cell).ceil() as usize).max(1);
        let ny = ((h / cell).ceil() as usize).max(1);
        let mut grid = Grid {
            x0: bbox[0],
            y0: bbox[1],
            cell,
            nx,
            ny,
            starts: vec![0; nx * ny + 1],
            items: Vec::new(),
        };
        for s in segs {
            let (i0, j0, i1, j1) = grid.cell_range(s);
            for j in j0..=j1 {
                for i in i0..=i1 {
                    grid.starts[j * nx + i + 1] += 1;
                }
            }
        }
        for k in 0..nx * ny {
            grid.starts[k + 1] += grid.starts[k];
        }
        let mut fill = grid.starts.clone();
        grid.items = vec![0; grid.starts[nx * ny] as usize];
        for (idx, s) in segs.iter().enumerate() {
            let (i0, j0, i1, j1) = grid.cell_range(s);
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let c = j * nx + i;
                    grid.items[fill[c] as usize] = idx as u32;
                    fill[c] += 1;
                }
            }
        }
        grid
    }

    #[inline]
    fn clamp_i(&self, x: f64) -> usize {
        (((x - self.x0) / self.cell).floor().max(0.0) as usize).min(self.nx - 1)
    }

    #[inline]
    fn clamp_j(&self, y: f64) -> usize {
        (((y - self.y0) / self.cell).floor().max(0.0) as usize).min(self.ny - 1)
    }

    fn cell_range(&self, s: &Seg) -> (usize, usize, usize, usize) {
        let i0 = self.clamp_i(s.a[0].min(s.b[0]));
        let i1 = self.clamp_i(s.a[0].max(s.b[0]));
        let j0 = self.clamp_j(s.a[1].min(s.b[1]));
        let j1 = self.clamp_j(s.a[1].max(s.b[1]));
        (i0, j0, i1, j1)
    }

    #[inline]
    fn cell_items(&self, i: usize, j: usize) -> &[u32] {
        let c = j * self.nx + i;
        &self.items[self.starts[c] as usize..self.starts[c + 1] as usize]
    }

    /// Calls `f` with the ids of all segments whose cells overlap the box (may repeat ids).
    pub fn query_box(&self, bbox: [f64; 4], mut f: impl FnMut(u32)) {
        if bbox[2] < self.x0
            || bbox[3] < self.y0
            || bbox[0] > self.x0 + self.nx as f64 * self.cell
            || bbox[1] > self.y0 + self.ny as f64 * self.cell
        {
            return;
        }
        let (i0, i1) = (self.clamp_i(bbox[0]), self.clamp_i(bbox[2]));
        let (j0, j1) = (self.clamp_j(bbox[1]), self.clamp_j(bbox[3]));
        for j in j0..=j1 {
            for i in i0..=i1 {
                for &id in self.cell_items(i, j) {
                    f(id);
                }
            }
        }
    }

    /// Visits the cells crossed by the segment `p -> q` in order; `visit` returns `true` to stop.
    pub(crate) fn walk_segment(&self, p: Pt, q: Pt, mut visit: impl FnMut(&[u32]) -> bool) {
        // clip to the grid rectangle first, so the walk starts where the segment enters it
        let d0 = sub(q, p);
        let (x1, y1) = (
            self.x0 + self.nx as f64 * self.cell,
            self.y0 + self.ny as f64 * self.cell,
        );
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        for (den, num) in [
            (-d0[0], p[0] - self.x0),
            (d0[0], x1 - p[0]),
            (-d0[1], p[1] - self.y0),
            (d0[1], y1 - p[1]),
        ] {
            if den == 0.0 {
                if num < 0.0 {
                    return;
                }
            } else {
                let t = num / den;
                if den < 0.0 {
                    t0 = t0.max(t);
                } else {
                    t1 = t1.min(t);
                }
            }
        }
        if t0 > t1 {
            return;
        }
        let (p, q) = (
            [p[0] + t0 * d0[0], p[1] + t0 * d0[1]],
            [p[0] + t1 * d0[0], p[1] + t1 * d0[1]],
        );
        let d = sub(q, p);
        let mut i = self.clamp_i(p[0]) as isize;
        let mut j = self.clamp_j(p[1]) as isize;
        let (ti, tj) = (self.clamp_i(q[0]) as isize, self.clamp_j(q[1]) as isize);
        let step_i: isize = if d[0] > 0.0 { 1 } else { -1 };
        let step_j: isize = if d[1] > 0.0 { 1 } else { -1 };
        let next_x = self.x0 + (i + if d[0] > 0.0 { 1 } else { 0 }) as f64 * self.cell;
        let next_y = self.y0 + (j + if d[1] > 0.0 { 1 } else { 0 }) as f64 * self.cell;
        let mut t_max_x = if d[0] != 0.0 {
            (next_x - p[0]) / d[0]
        } else {
            f64::INFINITY
        };
        let mut t_max_y = if d[1] != 0.0 {
            (next_y - p[1]) / d[1]
        } else {
            f64::INFINITY
        };
        let t_delta_x = if d[0] != 0.0 {
            self.cell / d[0].abs()
        } else {
            f64::INFINITY
        };
        let t_delta_y = if d[1] != 0.0 {
            self.cell / d[1].abs()
        } else {
            f64::INFINITY
        };
        let max_steps = (self.nx + self.ny) * 2 + 4;
        for _ in 0..max_steps {
            if visit(self.cell_items(i as usize, j as usize)) {
                return;
            }
            if (i == ti && j == tj) || t_max_x.min(t_max_y) > 1.0 {
                return;
            }
            if t_max_x < t_max_y {
                i += step_i;
                t_max_x += t_delta_x;
            } else {
                j += step_j;
                t_max_y += t_delta_y;
            }
            if i < 0 || j < 0 || i >= self.nx as isize || j >= self.ny as isize {
                return;
            }
        }
    }

    /// Nearest hit of the ray `o + t*d` (t > 0) with the segments, via Amanatides–Woo traversal.
    /// Returns `(segment id, t)`.
    fn ray_cast(&self, segs: &[Seg], o: Pt, d: Pt) -> Option<(usize, f64)> {
        let mut i = self.clamp_i(o[0]) as isize;
        let mut j = self.clamp_j(o[1]) as isize;
        let step_i: isize = if d[0] > 0.0 { 1 } else { -1 };
        let step_j: isize = if d[1] > 0.0 { 1 } else { -1 };
        let next_x = self.x0 + (i + if d[0] > 0.0 { 1 } else { 0 }) as f64 * self.cell;
        let next_y = self.y0 + (j + if d[1] > 0.0 { 1 } else { 0 }) as f64 * self.cell;
        let mut t_max_x = if d[0] != 0.0 {
            (next_x - o[0]) / d[0]
        } else {
            f64::INFINITY
        };
        let mut t_max_y = if d[1] != 0.0 {
            (next_y - o[1]) / d[1]
        } else {
            f64::INFINITY
        };
        let t_delta_x = if d[0] != 0.0 {
            self.cell / d[0].abs()
        } else {
            f64::INFINITY
        };
        let t_delta_y = if d[1] != 0.0 {
            self.cell / d[1].abs()
        } else {
            f64::INFINITY
        };

        let mut best: Option<(usize, f64)> = None;
        loop {
            for &id in self.cell_items(i as usize, j as usize) {
                let s = &segs[id as usize];
                if let Some(t) = ray_segment(o, d, s) {
                    if best.is_none_or(|(_, bt)| t < bt) {
                        best = Some((id as usize, t));
                    }
                }
            }
            let t_exit = t_max_x.min(t_max_y);
            if let Some((_, bt)) = best {
                if bt <= t_exit {
                    return best;
                }
            }
            if t_max_x < t_max_y {
                i += step_i;
                t_max_x += t_delta_x;
            } else {
                j += step_j;
                t_max_y += t_delta_y;
            }
            if i < 0 || j < 0 || i >= self.nx as isize || j >= self.ny as isize {
                return best;
            }
        }
    }
}

/// Parameters `(t, u)` where segment `p + t (q - p)` meets segment `s` at `s.a + u (s.b - s.a)`.
#[inline]
pub(crate) fn segment_params(p: Pt, q: Pt, s: &Seg) -> Option<(f64, f64)> {
    let d = sub(q, p);
    let e = sub(s.b, s.a);
    let den = cross(d, e);
    if den == 0.0 {
        return None;
    }
    let w = sub(s.a, p);
    Some((cross(w, e) / den, cross(w, d) / den))
}

/// Ray parameter `t > 0` where `o + t*d` meets segment `s`, if it does.
#[inline]
fn ray_segment(o: Pt, d: Pt, s: &Seg) -> Option<f64> {
    let e = sub(s.b, s.a);
    let den = cross(d, e);
    if den == 0.0 {
        return None;
    }
    let w = sub(s.a, o);
    let t = cross(w, e) / den;
    let u = cross(w, d) / den;
    let eps = 1e-12;
    if t > eps && u >= -eps && u <= 1.0 + eps {
        Some(t)
    } else {
        None
    }
}

/// Parameter `t` where the ray `o + t*d` meets the supporting line of `s`.
#[inline]
fn ray_line(o: Pt, d: Pt, s: &Seg) -> Option<f64> {
    let e = sub(s.b, s.a);
    let den = cross(d, e);
    if den.abs() <= 1e-14 * norm(d) * norm(e) {
        return None;
    }
    let t = cross(sub(s.a, o), e) / den;
    if t.is_finite() && t > 0.0 {
        Some(t)
    } else {
        None
    }
}

/// Proper or touching intersection point of two segments (None for parallel ones).
fn seg_intersection(s1: &Seg, s2: &Seg) -> Option<Pt> {
    let r = sub(s1.b, s1.a);
    let q = sub(s2.b, s2.a);
    let den = cross(r, q);
    if den == 0.0 {
        return None;
    }
    let w = sub(s2.a, s1.a);
    let t = cross(w, q) / den;
    let u = cross(w, r) / den;
    if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
        Some([s1.a[0] + t * r[0], s1.a[1] + t * r[1]])
    } else {
        None
    }
}

/// Vertices of the GEOS point buffer polygon (counter-clockwise, not closed).
pub fn disc_polygon(c: Pt, r: f64, quad_segs: usize) -> Vec<Pt> {
    let n = 4 * quad_segs.max(1);
    let inc = 2.0 * std::f64::consts::PI / n as f64;
    // GEOS walks clockwise from angle 0 with angle = -(i * inc); we emit the same vertices CCW.
    let mut pts: Vec<Pt> = (0..n)
        .map(|i| {
            let a = -(i as f64 * inc);
            [c[0] + r * a.cos(), c[1] + r * a.sin()]
        })
        .collect();
    pts.reverse();
    pts.rotate_right(1);
    pts
}

/// Cyrus–Beck clipping of a segment to a convex CCW polygon.
fn clip_to_convex(s: &Seg, poly: &[Pt]) -> Option<Seg> {
    let d = sub(s.b, s.a);
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    let n = poly.len();
    for k in 0..n {
        let p = poly[k];
        let q = poly[(k + 1) % n];
        let e = sub(q, p);
        // inside: cross(e, x - p) >= 0
        let num = cross(e, sub(s.a, p));
        let den = cross(e, d);
        if den == 0.0 {
            if num < 0.0 {
                return None;
            }
            continue;
        }
        let t = -num / den;
        if den > 0.0 {
            t0 = t0.max(t);
        } else {
            t1 = t1.min(t);
        }
        if t0 > t1 {
            return None;
        }
    }
    Some(Seg {
        a: [s.a[0] + t0 * d[0], s.a[1] + t0 * d[1]],
        b: [s.a[0] + t1 * d[0], s.a[1] + t1 * d[1]],
    })
}

/// Reusable per-thread scratch buffers.
#[derive(Default)]
pub struct Scratch {
    stamp: Vec<u32>,
    epoch: u32,
    cand: Vec<u32>,
    local: Vec<Seg>,
    local_ids: Vec<u32>,
    dirs: Vec<(f64, Pt)>,
}

/// Visible parts of the walls around an observer, as straight runs in angular order.
pub struct Runs {
    /// `(wall, start, end)`; `wall` is a global wall index, or `None` for the edge of the view disc.
    pub runs: Vec<(Option<u32>, Pt, Pt)>,
}

/// Global obstacle index shared by all observers.
pub struct Obstacles {
    pub segs: Vec<Seg>,
    pub(crate) one_sided: Vec<bool>,
    pub(crate) grid: Grid,
}

impl Obstacles {
    /// Index of two-sided walls (no culling).
    pub fn new(segs: Vec<Seg>, typical_radius: f64) -> Obstacles {
        Obstacles::from_walls(
            segs.into_iter()
                .map(|seg| Wall {
                    seg,
                    one_sided: false,
                })
                .collect(),
            typical_radius,
        )
    }

    pub fn from_walls(walls: Vec<Wall>, typical_radius: f64) -> Obstacles {
        let walls: Vec<Wall> = walls.into_iter().filter(|w| w.seg.a != w.seg.b).collect();
        let one_sided: Vec<bool> = walls.iter().map(|w| w.one_sided).collect();
        let segs: Vec<Seg> = walls.into_iter().map(|w| w.seg).collect();
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
        let extent = (bbox[2] - bbox[0]).max(bbox[3] - bbox[1]).max(1e-9);
        // Cells around a quarter of the view radius, but no more than ~4 cells per segment.
        let by_radius = if typical_radius > 0.0 {
            extent / (typical_radius / 4.0)
        } else {
            1.0
        };
        let by_count = (4.0 * segs.len() as f64).sqrt().max(1.0);
        let cells = by_radius.min(by_count).clamp(1.0, 4096.0) as usize;
        let grid = Grid::build(&segs, bbox, cells);
        Obstacles {
            segs,
            one_sided,
            grid,
        }
    }

    /// True when the segment `p -> q` crosses a wall strictly between its ends (walls in `skip` ignored).
    pub fn segment_blocked(&self, p: Pt, q: Pt, skip: Option<u32>) -> bool {
        let len = norm(sub(q, p));
        if len == 0.0 {
            return false;
        }
        let eps = 1e-9 / len.max(1e-9) + 1e-12;
        let mut blocked = false;
        self.grid.walk_segment(p, q, |items| {
            for &id in items {
                if Some(id) == skip {
                    continue;
                }
                if let Some((t, u)) = segment_params(p, q, &self.segs[id as usize]) {
                    if t > eps && t < 1.0 - eps && (-1e-12..=1.0 + 1e-12).contains(&u) {
                        blocked = true;
                        return true;
                    }
                }
            }
            false
        });
        blocked
    }

    /// Parameters along `p -> q` where it crosses any segment (for parity / inside-length tests).
    pub fn segment_crossings(&self, p: Pt, q: Pt, out: &mut Vec<f64>) {
        out.clear();
        self.grid.walk_segment(p, q, |items| {
            for &id in items {
                if let Some((t, u)) = segment_params(p, q, &self.segs[id as usize]) {
                    if (0.0..=1.0).contains(&t) && (0.0..1.0).contains(&u) {
                        out.push(t);
                    }
                }
            }
            false
        });
        out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out.dedup_by(|b, a| (*b - *a).abs() < 1e-12);
    }

    /// Visibility polygon ring (counter-clockwise, not closed) of observer `o` within radius `r`.
    pub fn visibility(&self, o: Pt, r: f64, quad_segs: usize, sc: &mut Scratch) -> Vec<Pt> {
        if r.is_nan() || r <= 0.0 {
            return Vec::new();
        }
        let Some(runs) = self.visible_runs(o, r, quad_segs, sc) else {
            return disc_polygon(o, r, quad_segs);
        };
        let eps = r * 1e-9;
        let mut ring: Vec<Pt> = Vec::with_capacity(2 * runs.runs.len());
        let mut push = |p: Pt| {
            if let Some(last) = ring.last() {
                if (last[0] - p[0]).abs() <= eps && (last[1] - p[1]).abs() <= eps {
                    return;
                }
            }
            ring.push(p);
        };
        for &(_, pa, pb) in &runs.runs {
            push(pa);
            push(pb);
        }
        if ring.len() > 1 {
            let (f, l) = (ring[0], ring[ring.len() - 1]);
            if (f[0] - l[0]).abs() <= eps && (f[1] - l[1]).abs() <= eps {
                ring.pop();
            }
        }
        // Obstacle vertices are kept even when collinear (e.g. OSM nodes on a straight wall).
        ring
    }

    /// Straight runs of the visibility boundary (absolute coordinates, counter-clockwise); `None` when
    /// no wall is within reach (the boundary is the view disc).
    pub fn visible_runs(&self, o: Pt, r: f64, quad_segs: usize, sc: &mut Scratch) -> Option<Runs> {
        if r.is_nan() || r <= 0.0 {
            return None;
        }
        // 1. candidate segments (deduplicated with an epoch stamp)
        if sc.stamp.len() < self.segs.len() {
            sc.stamp.resize(self.segs.len(), 0);
        }
        sc.epoch = sc.epoch.wrapping_add(1);
        if sc.epoch == 0 {
            sc.stamp.iter_mut().for_each(|v| *v = 0);
            sc.epoch = 1;
        }
        sc.cand.clear();
        let epoch = sc.epoch;
        {
            let stamp = &mut sc.stamp;
            let cand = &mut sc.cand;
            self.grid
                .query_box([o[0] - r, o[1] - r, o[0] + r, o[1] + r], |id| {
                    let k = id as usize;
                    if stamp[k] != epoch {
                        stamp[k] = epoch;
                        cand.push(id);
                    }
                });
        }

        // 2. clip to the view polygon, add its edges. From here on everything is relative to the
        //    observer: UTM-sized coordinates would otherwise eat most of the f64 precision.
        let disc_rel = disc_polygon([0.0, 0.0], r, quad_segs);
        sc.local.clear();
        sc.local_ids.clear();
        let r2 = r * r;
        for &id in &sc.cand {
            let s = &self.segs[id as usize];
            if self.one_sided[id as usize] && cross(sub(s.b, s.a), sub(o, s.a)) > 0.0 {
                continue; // back face: the observer is on the obstacle's interior side
            }
            if dist2_point_segment(o, s) > r2 {
                continue;
            }
            let rel = Seg {
                a: sub(s.a, o),
                b: sub(s.b, o),
            };
            if let Some(c) = clip_to_convex(&rel, &disc_rel) {
                if c.a != c.b {
                    sc.local.push(c);
                    sc.local_ids.push(id);
                }
            }
        }
        let n_obstacle = sc.local.len();
        if n_obstacle == 0 {
            return None;
        }
        let nd = disc_rel.len();
        for k in 0..nd {
            sc.local.push(Seg {
                a: disc_rel[k],
                b: disc_rel[(k + 1) % nd],
            });
        }

        let local = &sc.local;
        let pad = r * 1e-6 + 1e-9;
        let bbox = [-r - pad, -r - pad, r + pad, r + pad];
        let cells = ((local.len() as f64 / 2.0).sqrt().ceil() as usize).clamp(1, 512);
        let grid = Grid::build(local, bbox, cells);

        // 3. critical directions: endpoints + crossings between obstacle segments
        let dirs = &mut sc.dirs;
        dirs.clear();
        let tiny = (r * 1e-12).powi(2);
        let mut push_dir = |v: Pt| {
            if v[0] * v[0] + v[1] * v[1] > tiny {
                dirs.push((v[1].atan2(v[0]), v));
            }
        };
        for s in local.iter() {
            push_dir(s.a);
            push_dir(s.b);
        }
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let items = grid.cell_items(i, j);
                for (ka, &ia) in items.iter().enumerate() {
                    if ia as usize >= n_obstacle {
                        continue;
                    }
                    for &ib in &items[ka + 1..] {
                        if ib as usize >= n_obstacle {
                            continue;
                        }
                        let (sa, sb) = (&local[ia as usize], &local[ib as usize]);
                        if let Some(p) = seg_intersection(sa, sb) {
                            // shared endpoints are already listed
                            if p != sa.a && p != sa.b {
                                push_dir(p);
                            }
                        }
                    }
                }
            }
        }
        dirs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        dirs.dedup_by(|b, a| a.0 == b.0);

        // 4. one ray per angular interval: (nearest wall, its points on both bounding directions)
        let origin = [0.0, 0.0];
        let m = dirs.len();
        let mut spans: Vec<(usize, Pt, Pt)> = Vec::with_capacity(m);
        for k in 0..m {
            let (ang_a, da) = dirs[k];
            let (ang_b_raw, db) = dirs[(k + 1) % m];
            let ang_b = if k + 1 == m {
                ang_b_raw + 2.0 * std::f64::consts::PI
            } else {
                ang_b_raw
            };
            if ang_b - ang_a <= 0.0 {
                continue;
            }
            let ua = [da[0] / norm(da), da[1] / norm(da)];
            let ub = [db[0] / norm(db), db[1] / norm(db)];
            let mut dm = [ua[0] + ub[0], ua[1] + ub[1]];
            if norm(dm) < 1e-6 || ang_b - ang_a >= std::f64::consts::PI {
                let mid_angle = 0.5 * (ang_a + ang_b);
                dm = [mid_angle.cos(), mid_angle.sin()];
            }
            let Some((sid, tm)) = grid.ray_cast(local, origin, dm) else {
                continue;
            };
            let s = &local[sid];
            let hit_mid = [tm * dm[0], tm * dm[1]];
            let pa = ray_line(origin, ua, s).map_or(hit_mid, |t| [t * ua[0], t * ua[1]]);
            let pb = ray_line(origin, ub, s).map_or(hit_mid, |t| [t * ub[0], t * ub[1]]);
            spans.push((sid, pa, pb));
        }
        if spans.is_empty() {
            return None;
        }

        // 5. consecutive intervals on the same wall form one straight run: keep only its ends
        let ns = spans.len();
        let start = (0..ns)
            .find(|&k| spans[k].0 != spans[(k + ns - 1) % ns].0)
            .unwrap_or(0);
        let mut runs = Vec::with_capacity(ns);
        let mut k = 0;
        while k < ns {
            let (sid, pa, mut pb) = spans[(start + k) % ns];
            let mut j = k + 1;
            while j < ns && spans[(start + j) % ns].0 == sid {
                pb = spans[(start + j) % ns].2;
                j += 1;
            }
            let wall = (sid < n_obstacle).then(|| sc.local_ids[sid]);
            runs.push((
                wall,
                [pa[0] + o[0], pa[1] + o[1]],
                [pb[0] + o[0], pb[1] + o[1]],
            ));
            k = j;
        }
        Some(Runs { runs })
    }
}

/// Signed area (positive for counter-clockwise rings).
pub fn ring_area(ring: &[Pt]) -> f64 {
    let n = ring.len();
    let mut a = 0.0;
    for k in 0..n {
        a += cross(ring[k], ring[(k + 1) % n]);
    }
    0.5 * a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(cx: f64, cy: f64, h: f64) -> Vec<Seg> {
        let p = [
            [cx - h, cy - h],
            [cx + h, cy - h],
            [cx + h, cy + h],
            [cx - h, cy + h],
        ];
        (0..4)
            .map(|k| Seg {
                a: p[k],
                b: p[(k + 1) % 4],
            })
            .collect()
    }

    #[test]
    fn empty_scene_is_the_disc() {
        let obs = Obstacles::new(Vec::new(), 10.0);
        let mut sc = Scratch::default();
        let ring = obs.visibility([0.0, 0.0], 10.0, 16, &mut sc);
        assert_eq!(ring.len(), 64);
        let expected = 0.5 * 64.0 * 100.0 * (2.0 * std::f64::consts::PI / 64.0).sin();
        assert!((ring_area(&ring) - expected).abs() < 1e-9);
    }

    #[test]
    fn far_obstacle_is_ignored() {
        let obs = Obstacles::new(square(100.0, 0.0, 1.0), 10.0);
        let mut sc = Scratch::default();
        let ring = obs.visibility([0.0, 0.0], 10.0, 16, &mut sc);
        assert_eq!(ring.len(), 64);
    }

    #[test]
    fn wall_casts_exact_shadow() {
        // a long wall x = 5 from y = -100..100 inside a radius-10 disc: visible part = disc ∩ {x <= 5}
        let wall = vec![Seg {
            a: [5.0, -100.0],
            b: [5.0, 100.0],
        }];
        let obs = Obstacles::new(wall, 10.0);
        let mut sc = Scratch::default();
        let ring = obs.visibility([0.0, 0.0], 10.0, 256, &mut sc);
        let r: f64 = 10.0;
        let d: f64 = 5.0;
        let segment_area = r * r * (d / r).acos() - d * (r * r - d * d).sqrt();
        let expected = std::f64::consts::PI * r * r - segment_area;
        let got = ring_area(&ring);
        assert!(
            (got - expected).abs() / expected < 1e-3,
            "got {got}, expected {expected}"
        );
        assert!(ring.iter().all(|p| p[0] <= 5.0 + 1e-9));
    }

    #[test]
    fn square_shadow_is_a_wedge() {
        let obs = Obstacles::new(square(5.0, 0.0, 1.0), 20.0);
        let mut sc = Scratch::default();
        let ring = obs.visibility([0.0, 0.0], 20.0, 16, &mut sc);
        assert!(ring_area(&ring) > 0.0);
        // the far side of the square and the area right behind it are hidden
        let behind = [10.0, 0.0];
        assert!(!point_in_ring(behind, &ring));
        let beside = [0.0, 10.0];
        assert!(point_in_ring(beside, &ring));
    }

    #[test]
    fn crossing_walls_are_handled() {
        let walls = vec![
            Seg {
                a: [3.0, -3.0],
                b: [6.0, 3.0],
            },
            Seg {
                a: [3.0, 3.0],
                b: [6.0, -3.0],
            },
        ];
        let obs = Obstacles::new(walls, 10.0);
        let mut sc = Scratch::default();
        let ring = obs.visibility([0.0, 0.0], 10.0, 64, &mut sc);
        // the crossing point (4.5, 0) lies on the boundary, nothing beyond it is visible
        assert!(!point_in_ring([4.6, 0.0], &ring));
        assert!(point_in_ring([4.0, 0.0], &ring));
    }

    #[test]
    fn back_face_culling_keeps_the_result() {
        // CCW square: interior on the left of every edge
        let walls: Vec<Wall> = square(5.0, 1.0, 1.0)
            .into_iter()
            .map(|seg| Wall {
                seg,
                one_sided: true,
            })
            .collect();
        let two_sided = Obstacles::new(square(5.0, 1.0, 1.0), 20.0);
        let culled = Obstacles::from_walls(walls, 20.0);
        let mut sc = Scratch::default();
        let a = two_sided.visibility([0.0, 0.0], 20.0, 16, &mut sc);
        let b = culled.visibility([0.0, 0.0], 20.0, 16, &mut sc);
        assert!((ring_area(&a) - ring_area(&b)).abs() < 1e-9);
        assert_eq!(a.len(), b.len());
    }

    fn point_in_ring(p: Pt, ring: &[Pt]) -> bool {
        let mut inside = false;
        let n = ring.len();
        for k in 0..n {
            let (a, b) = (ring[k], ring[(k + 1) % n]);
            if (a[1] > p[1]) != (b[1] > p[1]) {
                let x = a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]);
                if p[0] < x {
                    inside = !inside;
                }
            }
        }
        inside
    }
}
