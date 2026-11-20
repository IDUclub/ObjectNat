//! Straight skeleton of polygons with holes, vendored from the geo-straight-skeleton crate 0.1.1
//! (https://github.com/Arinko-inc/geo-straight-skeleton), MIT licence, see LICENSE-MIT in this directory.
//! Local changes: path and module adjustments, geo-types integration removed.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod engine;
mod faces;
mod geom;
mod input;
#[allow(dead_code)] // Retain the upstream offset API for future comparisons.
mod offset;
mod queue;

use std::fmt;

/// The straight skeleton of a polygon with holes.
#[derive(Clone, Debug, PartialEq)]
pub struct Skeleton {
    /// Skeleton vertices as `[x, y, time]`. The first entries are the input points, in input order
    /// (closing points removed, rings one after another, `time` = 0); the skeleton nodes follow.
    /// `time` is the offset distance at which the wavefront reaches the vertex, in input units.
    pub vertices: Vec<[f64; 3]>,
    /// One face per input edge: `faces[k]` belongs to the edge that starts at input point `k`
    /// (and ends at the next point of the same ring).
    pub faces: Vec<Face>,
    /// For every vertex, the representative of the input points treated as one point (duplicates,
    /// rings touching at a vertex); skeleton nodes represent themselves.
    same: Vec<usize>,
}

/// The face swept by one input edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Face {
    /// Indices of the edge's two input points, oriented so that the polygon interior lies on the
    /// left (counter-clockwise outer ring, clockwise holes). For a ring given in the opposite
    /// orientation this is `(k + 1, k)` instead of `(k, k + 1)`.
    pub edge: (usize, usize),
    /// Vertex indices of the face, counter-clockwise, starting with `edge.1` and ending with
    /// `edge.0` (the same order as CGAL's straight skeleton faces). A zero-length input edge
    /// (consecutive duplicate points) gets the two-vertex face `[edge.1, edge.0]`.
    pub vertices: Vec<usize>,
}

/// Why a polygon was rejected. Rings are numbered as given (0 = outer ring); point and edge
/// indices count within the ring, closing point excluded (edge `k` starts at point `k`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SkeletonError {
    /// No rings were given.
    EmptyInput,
    /// A ring has fewer than three distinct points.
    TooFewVertices {
        /// The ring.
        ring: usize,
    },
    /// A coordinate is NaN or infinite.
    NonFinite {
        /// The ring.
        ring: usize,
        /// The point within the ring.
        index: usize,
    },
    /// A ring has (almost) zero area: all its points are collinear.
    ZeroArea {
        /// The ring.
        ring: usize,
    },
    /// Two edges cross, overlap, or touch away from a vertex shared by both rings; or two rings
    /// that share a vertex overlap there.
    SelfIntersection {
        /// Ring of the first edge.
        ring_a: usize,
        /// First edge, by the index of its first point.
        edge_a: usize,
        /// Ring of the second edge.
        ring_b: usize,
        /// Second edge, by the index of its first point.
        edge_b: usize,
    },
    /// A hole is not inside the outer ring.
    HoleOutside {
        /// The hole.
        ring: usize,
    },
    /// A hole lies inside another hole.
    NestedHole {
        /// The inner hole.
        ring: usize,
        /// The hole containing it.
        outer: usize,
    },
    /// The computation failed numerically (a bug: please report the input).
    Numerical,
}

impl fmt::Display for SkeletonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SkeletonError::EmptyInput => write!(f, "no rings were given"),
            SkeletonError::TooFewVertices { ring } => {
                write!(f, "ring {ring} has fewer than three distinct points")
            }
            SkeletonError::NonFinite { ring, index } => {
                write!(f, "point {index} of ring {ring} is not finite")
            }
            SkeletonError::ZeroArea { ring } => write!(f, "ring {ring} has zero area"),
            SkeletonError::SelfIntersection { ring_a, edge_a, ring_b, edge_b } => write!(
                f,
                "edge {edge_a} of ring {ring_a} intersects or touches edge {edge_b} of ring {ring_b}"
            ),
            SkeletonError::HoleOutside { ring } => {
                write!(f, "hole {ring} is not inside the outer ring")
            }
            SkeletonError::NestedHole { ring, outer } => {
                write!(f, "hole {ring} lies inside hole {outer}")
            }
            SkeletonError::Numerical => write!(f, "the straight skeleton computation failed"),
        }
    }
}

impl std::error::Error for SkeletonError {}

impl Skeleton {
    /// Compute the straight skeleton of a polygon.
    ///
    /// `rings[0]` is the outer boundary, the other rings are holes. Rings may be given in either
    /// orientation (they are fixed internally), with or without a closing point equal to the first
    /// point. Consecutive duplicate points are ignored. Collinear points are kept: each one gets its
    /// own face, separated from its neighbour by a skeleton arc perpendicular to the edge (CGAL does
    /// the same).
    pub fn build(rings: &[Vec<[f64; 2]>]) -> Result<Skeleton, SkeletonError> {
        let prep = input::prepare(rings)?;
        let eng = engine::Engine::new(&prep.rings);
        let out = eng.run(&prep.touch_groups).map_err(|e| {
            if cfg!(debug_assertions) && std::env::var_os("SK_TRACE").is_some() {
                eprintln!("engine failed: {e:?}");
            }
            SkeletonError::Numerical
        })?;
        // computed edge -> (start node, end node)
        let mut edge_nodes: Vec<(u32, u32)> = Vec::new();
        let mut comp_edges: Vec<input::CompEdge> = Vec::new();
        let mut base = 0u32;
        for (r, ring) in prep.rings.iter().enumerate() {
            let n = ring.len() as u32;
            for j in 0..n {
                edge_nodes.push((base + j, base + (j + 1) % n));
                comp_edges.push(prep.edges[r][j as usize]);
            }
            base += n;
        }
        let n_input_nodes = base as usize;
        // the face areas must add up to the polygon: a cheap guard against a wrong Ok
        let polygon_area: f64 = prep.rings.iter().map(|r| geom::ring_area2(r)).sum::<f64>() * 0.5;
        let mut canon: Vec<u32> = (0..n_input_nodes as u32).collect();
        for g in &prep.touch_groups {
            for &v in g {
                canon[v] = g[0] as u32;
            }
        }
        let node_faces =
            faces::assemble(&out.nodes, &out.arcs, &edge_nodes, &canon).map_err(|f| {
                if cfg!(debug_assertions) && std::env::var_os("SK_TRACE").is_some() {
                    eprintln!("face {f} could not be assembled");
                }
                SkeletonError::Numerical
            })?;
        let faces_area: f64 = node_faces
            .iter()
            .map(|f| {
                let pts: Vec<geom::V2> = f.iter().map(|&n| out.nodes[n as usize].pos).collect();
                geom::ring_area2(&pts) * 0.5
            })
            .sum();
        if !faces_area.is_finite()
            || (faces_area - polygon_area).abs() > 1e-7 * polygon_area.abs() + 1e-14
        {
            if cfg!(debug_assertions) && std::env::var_os("SK_TRACE").is_some() {
                eprintln!("faces cover {faces_area}, polygon {polygon_area}");
            }
            return Err(SkeletonError::Numerical);
        }

        let n_pts = prep.points.len();
        let mut vertices: Vec<[f64; 3]> = prep.points.iter().map(|p| [p[0], p[1], 0.0]).collect();
        // input nodes: computed vertex k is the start of computed edge k
        let mut node_out: Vec<usize> = vec![usize::MAX; out.nodes.len()];
        for (k, ce) in comp_edges.iter().enumerate() {
            node_out[k] = ce.start;
        }
        let mut faces_out: Vec<Option<Face>> = vec![None; n_pts];
        for (k, nf) in node_faces.iter().enumerate() {
            let ce = comp_edges[k];
            let mut vs = Vec::with_capacity(nf.len());
            let last = nf.len() - 1;
            for (i, &nd) in nf.iter().enumerate() {
                let idx = if i == 0 {
                    ce.end
                } else if i == last {
                    ce.start
                } else {
                    let nd = nd as usize;
                    if node_out[nd] == usize::MAX {
                        let node = out.nodes[nd];
                        let w = prep.to_world(node.pos);
                        node_out[nd] = vertices.len();
                        vertices.push([w[0], w[1], node.t * prep.scale]);
                    }
                    debug_assert!(nd >= n_input_nodes || node_out[nd] != usize::MAX);
                    node_out[nd]
                };
                vs.push(idx);
            }
            faces_out[ce.orig] = Some(Face {
                edge: (ce.start, ce.end),
                vertices: vs,
            });
        }
        for &(orig, s, e) in &prep.zero_edges {
            faces_out[orig] = Some(Face {
                edge: (s, e),
                vertices: vec![e, s],
            });
        }
        let faces = faces_out
            .into_iter()
            .map(|f| f.ok_or(SkeletonError::Numerical))
            .collect::<Result<_, _>>()?;
        // identity of input points: duplicate runs and vertices where rings touch are one point
        let mut parent: Vec<usize> = (0..vertices.len()).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        let union = |p: &mut Vec<usize>, a: usize, b: usize| {
            let (ra, rb) = (find(p, a), find(p, b));
            if ra != rb {
                p[ra.max(rb)] = ra.min(rb);
            }
        };
        for &(_, s, e) in &prep.zero_edges {
            union(&mut parent, s, e);
        }
        let mut base = 0usize;
        for ring in &prep.rings {
            let n = ring.len();
            for j in 0..n {
                // computed vertex base + j ends edge j - 1 and starts edge j
                let prev = comp_edges[base + (j + n - 1) % n].end;
                union(&mut parent, prev, comp_edges[base + j].start);
            }
            base += n;
        }
        for g in &prep.touch_groups {
            for &m in &g[1..] {
                union(&mut parent, comp_edges[g[0]].start, comp_edges[m].start);
            }
        }
        let same: Vec<usize> = (0..vertices.len()).map(|i| find(&mut parent, i)).collect();
        Ok(Skeleton {
            vertices,
            faces,
            same,
        })
    }

    /// The interior arcs of the skeleton: every edge of a face that has at least one endpoint with
    /// `time > 0`, once, as `(a, b)` with `a < b`. Input edges are not included.
    #[allow(dead_code)]
    pub fn arcs(&self) -> Vec<(usize, usize)> {
        let mut out: Vec<(usize, usize)> = Vec::new();
        for f in &self.faces {
            let m = f.vertices.len();
            for i in 0..m {
                let a = f.vertices[i];
                let b = f.vertices[(i + 1) % m];
                if a == b || (self.vertices[a][2] <= 0.0 && self.vertices[b][2] <= 0.0) {
                    continue;
                }
                out.push(if a < b { (a, b) } else { (b, a) });
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}
