//! Inward offset polygons from the skeleton faces.

use super::{Skeleton, SkeletonError};
use std::collections::HashMap;

/// A crossing of the cut level on the face edge `a -> b` (vertex representatives).
#[derive(Clone, Copy)]
struct Crossing {
    a: usize,
    b: usize,
    point: [f64; 2],
    /// position along the face's input edge
    along: f64,
    up: bool,
}

impl Skeleton {
    /// The polygon shrunk inward by `distance`: the cross-section of the skeleton's roof at height
    /// `distance`. Returns closed rings without a closing point, counter-clockwise for outer
    /// boundaries and clockwise for holes; several outer rings appear once the polygon splits.
    ///
    /// `distance <= 0` gives the input rings (oriented as above). At a distance equal to the time of
    /// a ridge or skeleton vertex, the parts of zero width there are left out (the cut is taken a
    /// relative 1e-12 above `distance`); at or beyond the highest vertex the result is empty.
    ///
    /// Returns `SkeletonError::Numerical` if a contour fails to close (a bug: please report it).
    ///
    /// ```ignore
    /// use objectnat_rust::straight_skeleton::Skeleton;
    ///
    /// let sk = Skeleton::build(&[vec![[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]]]).unwrap();
    /// let rings = sk.offset(0.5).unwrap();
    /// assert_eq!(rings.len(), 1);
    /// assert_eq!(rings[0].len(), 4); // the rectangle (0.5, 0.5) - (3.5, 1.5)
    /// ```
    pub fn offset(&self, distance: f64) -> Result<Vec<Vec<[f64; 2]>>, SkeletonError> {
        if !distance.is_finite() {
            return Ok(Vec::new());
        }
        // Cut a relative 1e-12 above the level: where the level is exactly the time of a ridge or
        // a node (common with axis-parallel input), that ridge or node falls below the cut, so no
        // zero-width band and no rings touching at a point are produced. Only parts of zero area are
        // lost.
        let d = (distance * (1.0 + 1e-12)).max(f64::MIN_POSITIVE);
        let same = &self.same;
        let above = |i: usize| self.vertices[i][2] >= d;
        // segments of the level set inside each face: from a crossing going down to one going up
        let mut segments: Vec<(Crossing, Crossing)> = Vec::new();
        for face in &self.faces {
            let vs = &face.vertices;
            let m = vs.len();
            if m < 3 {
                continue;
            }
            let (s, e) = (self.vertices[face.edge.0], self.vertices[face.edge.1]);
            let dir = [e[0] - s[0], e[1] - s[1]];
            let mut cs: Vec<Crossing> = Vec::new();
            for i in 0..m {
                let (u, w) = (vs[i], vs[(i + 1) % m]);
                let (au, aw) = (above(u), above(w));
                if au == aw {
                    continue;
                }
                // interpolate from the same end in both faces sharing this edge, so that both get
                // the identical point
                let (ku, kw) = (same[u], same[w]);
                let (lo, hi) = if (ku, u) < (kw, w) { (u, w) } else { (w, u) };
                let (pl, ph) = (self.vertices[lo], self.vertices[hi]);
                let f = ((d - pl[2]) / (ph[2] - pl[2])).clamp(0.0, 1.0);
                let point = [pl[0] + (ph[0] - pl[0]) * f, pl[1] + (ph[1] - pl[1]) * f];
                let along = (point[0] - s[0]) * dir[0] + (point[1] - s[1]) * dir[1];
                cs.push(Crossing {
                    a: ku,
                    b: kw,
                    point,
                    along,
                    up: aw,
                });
            }
            // along the edge direction the level line leaves the superlevel set at "down"
            // crossings and enters it at "up" crossings, alternately
            cs.sort_by(|x, y| x.along.total_cmp(&y.along).then(x.up.cmp(&y.up)));
            let mut i = 0;
            while i + 1 < cs.len() {
                if !cs[i].up && cs[i + 1].up {
                    segments.push((cs[i], cs[i + 1]));
                    i += 2;
                } else {
                    i += 1;
                }
            }
        }
        // chain: a segment ending on face edge (a, b) continues in the face that has (b, a)
        let mut start_of: HashMap<(usize, usize), usize> = HashMap::with_capacity(segments.len());
        for (k, (from, _)) in segments.iter().enumerate() {
            start_of.insert((from.a, from.b), k);
        }
        let mut used = vec![false; segments.len()];
        let mut rings = Vec::new();
        for k0 in 0..segments.len() {
            if used[k0] {
                continue;
            }
            let mut ring: Vec<[f64; 2]> = Vec::new();
            let mut k = k0;
            loop {
                used[k] = true;
                let (from, to) = segments[k];
                if ring.last() != Some(&from.point) {
                    ring.push(from.point);
                }
                match start_of.get(&(to.b, to.a)) {
                    Some(&nk) if nk == k0 => break,
                    Some(&nk) if !used[nk] => k = nk,
                    _ => return Err(SkeletonError::Numerical),
                }
            }
            while ring.len() > 1 && ring.first() == ring.last() {
                ring.pop();
            }
            if ring.len() < 3 {
                return Err(SkeletonError::Numerical);
            }
            rings.push(ring);
        }
        Ok(rings)
    }
}
