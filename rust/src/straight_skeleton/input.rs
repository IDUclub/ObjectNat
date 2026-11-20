//! Input validation and preparation: closing points, duplicates, orientation, normalisation,
//! self-intersection and hole containment checks.

use super::geom::*;
use super::SkeletonError;

/// Points closer than this (in normalised units, the larger side of the bounding box being 1)
/// are treated as duplicates.
const DUP_EPS: f64 = 1e-12;
/// Rings with an area below this (normalised units) are rejected as degenerate.
const AREA_EPS: f64 = 1e-12;
/// Distance below which two segments are considered touching (normalised units).
const TOUCH_EPS: f64 = 1e-12;

/// One computed (non-degenerate) edge, mapped back to the input.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CompEdge {
    /// Global index of the input edge (= index of its first vertex in the flattened input).
    pub orig: usize,
    /// Output vertex index at the start of the edge (in the computed orientation).
    pub start: usize,
    /// Output vertex index at the end of the edge.
    pub end: usize,
}

pub(crate) struct Prepared {
    /// Cleaned, oriented rings in normalised coordinates. Ring 0 is counter-clockwise, holes clockwise.
    pub rings: Vec<Vec<V2>>,
    /// `edges[r][j]` is the edge from `rings[r][j]` to `rings[r][j + 1]`.
    pub edges: Vec<Vec<CompEdge>>,
    /// Input edges of zero length: (global edge index, start vertex, end vertex).
    pub zero_edges: Vec<(usize, usize, usize)>,
    /// All input points (closing points removed), in input order.
    pub points: Vec<[f64; 2]>,
    /// Groups of computed vertices (indices into the concatenated `rings`) at the same position:
    /// rings that touch at a vertex.
    pub touch_groups: Vec<Vec<usize>>,
    pub center: V2,
    pub scale: f64,
}

impl Prepared {
    pub(crate) fn to_world(&self, p: V2) -> [f64; 2] {
        [
            p[0] * self.scale + self.center[0],
            p[1] * self.scale + self.center[1],
        ]
    }
}

struct RawRing {
    /// offset of the ring's first point in the flattened input
    offset: usize,
    len: usize,
}

pub(crate) fn prepare(rings: &[Vec<[f64; 2]>]) -> Result<Prepared, SkeletonError> {
    if rings.is_empty() {
        return Err(SkeletonError::EmptyInput);
    }
    // 1. closing points, finiteness, flattened point list
    let mut points: Vec<[f64; 2]> = Vec::new();
    let mut raw: Vec<RawRing> = Vec::with_capacity(rings.len());
    for (ri, ring) in rings.iter().enumerate() {
        let mut len = ring.len();
        if len > 1 && ring[0] == ring[len - 1] {
            len -= 1;
        }
        for (i, p) in ring[..len].iter().enumerate() {
            if !p[0].is_finite() || !p[1].is_finite() {
                return Err(SkeletonError::NonFinite { ring: ri, index: i });
            }
        }
        if len < 3 {
            return Err(SkeletonError::TooFewVertices { ring: ri });
        }
        raw.push(RawRing {
            offset: points.len(),
            len,
        });
        points.extend_from_slice(&ring[..len]);
    }
    // 2. normalisation frame from the bounding box of everything
    let (mut x0, mut y0, mut x1, mut y1) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for p in &points {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    let center = [0.5 * (x0 + x1), 0.5 * (y0 + y1)];
    let scale = (x1 - x0).max(y1 - y0);
    if !scale.is_finite() || scale <= 0.0 {
        return Err(SkeletonError::ZeroArea { ring: 0 });
    }
    let inv = 1.0 / scale;
    let norm_pt = |p: [f64; 2]| -> V2 { [(p[0] - center[0]) * inv, (p[1] - center[1]) * inv] };

    // 3. duplicates (runs of consecutive equal points, cyclic), orientation
    let mut out_rings: Vec<Vec<V2>> = Vec::with_capacity(raw.len());
    let mut out_edges: Vec<Vec<CompEdge>> = Vec::with_capacity(raw.len());
    let mut zero_edges = Vec::new();
    let mut zero_area_rings: Vec<usize> = Vec::new();
    for (ri, rr) in raw.iter().enumerate() {
        let pts: Vec<V2> = (0..rr.len)
            .map(|i| norm_pt(points[rr.offset + i]))
            .collect();
        let m = rr.len;
        // find a start index that is not a duplicate of its predecessor
        let same = |a: usize, b: usize| dist2(pts[a], pts[b]) <= DUP_EPS * DUP_EPS;
        let start = (0..m).find(|&i| !same(i, (i + m - 1) % m));
        let Some(start) = start else {
            return Err(SkeletonError::TooFewVertices { ring: ri });
        };
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for k in 0..m {
            let i = (start + k) % m;
            if k > 0 && same(i, (i + m - 1) % m) {
                runs.last_mut().expect("run exists").1 = i;
            } else {
                runs.push((i, i));
            }
        }
        if runs.len() < 3 {
            return Err(SkeletonError::TooFewVertices { ring: ri });
        }
        let cleaned: Vec<V2> = runs.iter().map(|&(a, _)| pts[a]).collect();
        let area2 = ring_area2(&cleaned);
        if area2.abs() * 0.5 <= AREA_EPS {
            // a figure-eight has zero signed area too: report it as a self-intersection
            if !all_collinear(&cleaned) {
                zero_area_rings.push(ri);
            } else {
                return Err(SkeletonError::ZeroArea { ring: ri });
            }
        }
        let want_ccw = ri == 0;
        let reversed = (area2 > 0.0) != want_ccw;
        // zero-length input edges inside runs
        for &(a, b) in &runs {
            let mut i = a;
            while i != b {
                let j = (i + 1) % m;
                let (s, e) = if reversed { (j, i) } else { (i, j) };
                zero_edges.push((rr.offset + i, rr.offset + s, rr.offset + e));
                i = j;
            }
        }
        let n = runs.len();
        let (ring, edges): (Vec<V2>, Vec<CompEdge>) = if !reversed {
            let edges = (0..n)
                .map(|j| {
                    let a = runs[j];
                    let b = runs[(j + 1) % n];
                    CompEdge {
                        orig: rr.offset + a.1,
                        start: rr.offset + a.1,
                        end: rr.offset + b.0,
                    }
                })
                .collect();
            (cleaned, edges)
        } else {
            // computed ring = runs in reverse order
            let rev: Vec<(usize, usize)> = runs.iter().rev().copied().collect();
            let ring: Vec<V2> = rev.iter().map(|&(a, _)| pts[a]).collect();
            let edges = (0..n)
                .map(|j| {
                    // computed edge from rev[j] to rev[j+1]; in input order rev[j+1] precedes rev[j]
                    let a = rev[j];
                    let b = rev[(j + 1) % n];
                    CompEdge {
                        orig: rr.offset + b.1,
                        start: rr.offset + a.0,
                        end: rr.offset + b.1,
                    }
                })
                .collect();
            (ring, edges)
        };
        out_rings.push(ring);
        out_edges.push(edges);
    }

    let (group_of, touch_groups) = find_touches(&out_rings);
    check_simple(&out_rings, &out_edges, &raw, &group_of)?;
    if let Some(&ring) = zero_area_rings.first() {
        return Err(SkeletonError::ZeroArea { ring });
    }
    check_touch_wedges(&out_rings, &out_edges, &raw, &touch_groups)?;
    check_holes(&out_rings)?;

    Ok(Prepared {
        rings: out_rings,
        edges: out_edges,
        zero_edges,
        points,
        touch_groups,
        center,
        scale,
    })
}

/// Vertices of different rings (or non-consecutive vertices of one ring) at the same position.
/// Returns the group of every computed vertex (`usize::MAX` = alone) and the groups.
fn find_touches(rings: &[Vec<V2>]) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut all: Vec<(V2, usize)> = Vec::new();
    for ring in rings {
        for &p in ring {
            let id = all.len();
            all.push((p, id));
        }
    }
    let n = all.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| all[a].0[0].total_cmp(&all[b].0[0]));
    let mut group_of = vec![usize::MAX; n];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let e2 = DUP_EPS * DUP_EPS;
    for i in 0..n {
        let a = order[i];
        for &b in &order[i + 1..] {
            if all[b].0[0] - all[a].0[0] > DUP_EPS {
                break;
            }
            if dist2(all[a].0, all[b].0) <= e2 {
                match (group_of[a], group_of[b]) {
                    (usize::MAX, usize::MAX) => {
                        group_of[a] = groups.len();
                        group_of[b] = groups.len();
                        groups.push(vec![a, b]);
                    }
                    (g, usize::MAX) => {
                        group_of[b] = g;
                        groups[g].push(b);
                    }
                    (usize::MAX, g) => {
                        group_of[a] = g;
                        groups[g].push(a);
                    }
                    (g, h) if g != h => {
                        let moved = std::mem::take(&mut groups[h]);
                        for &m in &moved {
                            group_of[m] = g;
                        }
                        groups[g].extend(moved);
                    }
                    _ => {}
                }
            }
        }
    }
    groups.retain(|g| !g.is_empty());
    for (k, g) in groups.iter_mut().enumerate() {
        g.sort_unstable();
        for &m in g.iter() {
            group_of[m] = k;
        }
    }
    (group_of, groups)
}

/// Where rings touch at a vertex, their wedges must not overlap: around the point, every
/// vertex's exterior gap (from its incoming to its outgoing edge, counter-clockwise) must contain
/// no edge of another vertex there.
fn check_touch_wedges(
    rings: &[Vec<V2>],
    edges: &[Vec<CompEdge>],
    raw: &[RawRing],
    groups: &[Vec<usize>],
) -> Result<(), SkeletonError> {
    let mut loc: Vec<(usize, usize)> = Vec::new();
    for (r, ring) in rings.iter().enumerate() {
        for j in 0..ring.len() {
            loc.push((r, j));
        }
    }
    for g in groups {
        // rays: (angle, is_in, member)
        let mut rays: Vec<(f64, bool, usize)> = Vec::new();
        for (k, &v) in g.iter().enumerate() {
            let (r, j) = loc[v];
            let ring = &rings[r];
            let n = ring.len();
            let p = ring[j];
            let prev = ring[(j + n - 1) % n];
            let next = ring[(j + 1) % n];
            rays.push((angle(sub(prev, p)), true, k));
            rays.push((angle(sub(next, p)), false, k));
        }
        rays.sort_by(|a, b| a.0.total_cmp(&b.0));
        let m = rays.len();
        // each In ray must be followed directly by the Out ray of the same vertex
        let ok = (0..m).all(|i| {
            let (a, is_in, k) = rays[i];
            if !is_in {
                return true;
            }
            let (b, is_in2, k2) = rays[(i + 1) % m];
            !is_in2 && k2 == k && b != a
        });
        if !ok {
            let (ra, ja) = loc[g[0]];
            let (rb, jb) = loc[g[1]];
            return Err(SkeletonError::SelfIntersection {
                ring_a: ra,
                edge_a: edges[ra][ja].orig - raw[ra].offset,
                ring_b: rb,
                edge_b: edges[rb][jb].orig - raw[rb].offset,
            });
        }
    }
    Ok(())
}

fn all_collinear(pts: &[V2]) -> bool {
    let a = pts[0];
    let far = pts
        .iter()
        .copied()
        .fold(a, |m, p| if dist2(p, a) > dist2(m, a) { p } else { m });
    let d = sub(far, a);
    let l = norm(d);
    if l == 0.0 {
        return true;
    }
    pts.iter()
        .all(|&p| (cross(d, sub(p, a)) / l).abs() <= TOUCH_EPS)
}

#[inline]
fn orient(a: V2, b: V2, c: V2) -> f64 {
    cross(sub(b, a), sub(c, a))
}

fn point_seg_dist2(p: V2, a: V2, b: V2) -> f64 {
    let ab = sub(b, a);
    let l2 = dot(ab, ab);
    let t = if l2 > 0.0 {
        (dot(sub(p, a), ab) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    dist2(p, add(a, mul(ab, t)))
}

fn segs_touch(a: V2, b: V2, c: V2, d: V2) -> bool {
    let d1 = orient(c, d, a);
    let d2 = orient(c, d, b);
    let d3 = orient(a, b, c);
    let d4 = orient(a, b, d);
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
    {
        return true;
    }
    let e2 = TOUCH_EPS * TOUCH_EPS;
    point_seg_dist2(a, c, d) <= e2
        || point_seg_dist2(b, c, d) <= e2
        || point_seg_dist2(c, a, b) <= e2
        || point_seg_dist2(d, a, b) <= e2
}

fn check_simple(
    rings: &[Vec<V2>],
    edges: &[Vec<CompEdge>],
    raw: &[RawRing],
    group_of: &[usize],
) -> Result<(), SkeletonError> {
    struct Seg {
        ring: usize,
        j: usize,
        /// global ids of the endpoints
        ia: usize,
        ib: usize,
        a: V2,
        b: V2,
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
    }
    let mut segs: Vec<Seg> = Vec::new();
    let mut base = 0;
    for (r, ring) in rings.iter().enumerate() {
        let n = ring.len();
        for j in 0..n {
            let a = ring[j];
            let b = ring[(j + 1) % n];
            segs.push(Seg {
                ring: r,
                j,
                ia: base + j,
                ib: base + (j + 1) % n,
                a,
                b,
                xmin: a[0].min(b[0]),
                xmax: a[0].max(b[0]),
                ymin: a[1].min(b[1]),
                ymax: a[1].max(b[1]),
            });
        }
        base += n;
    }
    let same = |x: usize, y: usize| group_of[x] != usize::MAX && group_of[x] == group_of[y];
    let err = |s: &Seg, t: &Seg| {
        let ea = edges[s.ring][s.j].orig - raw[s.ring].offset;
        let eb = edges[t.ring][t.j].orig - raw[t.ring].offset;
        SkeletonError::SelfIntersection {
            ring_a: s.ring,
            edge_a: ea,
            ring_b: t.ring,
            edge_b: eb,
        }
    };
    segs.sort_by(|p, q| p.xmin.total_cmp(&q.xmin));
    for i in 0..segs.len() {
        let s = &segs[i];
        for t in &segs[i + 1..] {
            if t.xmin > s.xmax + TOUCH_EPS {
                break;
            }
            if t.ymin > s.ymax + TOUCH_EPS || s.ymin > t.ymax + TOUCH_EPS {
                continue;
            }
            if s.ring == t.ring {
                let n = rings[s.ring].len();
                let next = |j: usize| (j + 1) % n;
                if next(s.j) == t.j || next(t.j) == s.j {
                    // adjacent: they share one endpoint; they must not fold back onto each other
                    let (first, second) = if next(s.j) == t.j { (s, t) } else { (t, s) };
                    let e2 = TOUCH_EPS * TOUCH_EPS;
                    // shared vertex is first.b == second.a
                    let folds = point_seg_dist2(second.b, first.a, first.b) <= e2
                        || point_seg_dist2(first.a, second.a, second.b) <= e2;
                    if folds {
                        return Err(err(s, t));
                    }
                    continue;
                }
            }
            if segs_touch(s.a, s.b, t.a, t.b) {
                // allowed: the two edges only share an endpoint where rings touch at a vertex
                let shared = [(s.ia, t.ia), (s.ia, t.ib), (s.ib, t.ia), (s.ib, t.ib)]
                    .iter()
                    .filter(|&&(x, y)| same(x, y))
                    .count();
                if shared == 1 {
                    let e2 = TOUCH_EPS * TOUCH_EPS;
                    let (u, w) = if same(s.ia, t.ia) {
                        (s.b, t.b)
                    } else if same(s.ia, t.ib) {
                        (s.b, t.a)
                    } else if same(s.ib, t.ia) {
                        (s.a, t.b)
                    } else {
                        (s.a, t.a)
                    };
                    // the far endpoints must stay off the other edge (no overlap)
                    if point_seg_dist2(u, t.a, t.b) > e2 && point_seg_dist2(w, s.a, s.b) > e2 {
                        continue;
                    }
                }
                return Err(err(s, t));
            }
        }
    }
    Ok(())
}

fn point_in_ring(p: V2, ring: &[V2]) -> bool {
    let n = ring.len();
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let a = ring[i];
        let b = ring[j];
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
            if p[0] < x {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

fn check_holes(rings: &[Vec<V2>]) -> Result<(), SkeletonError> {
    for h in 1..rings.len() {
        // edges only touch at shared vertices, so the middle of an edge decides containment
        let p = mul(add(rings[h][0], rings[h][1]), 0.5);
        if !point_in_ring(p, &rings[0]) {
            return Err(SkeletonError::HoleOutside { ring: h });
        }
        for (g, other) in rings.iter().enumerate().skip(1) {
            if g != h && point_in_ring(p, other) {
                return Err(SkeletonError::NestedHole { ring: h, outer: g });
            }
        }
    }
    Ok(())
}
