//! Straight skeletons of many polygons in parallel, flattened for numpy.

use rayon::prelude::*;

use crate::straight_skeleton::{Skeleton, SkeletonError};

/// Skeletons of a polygon batch, every index global over the batch.
#[derive(Default)]
pub struct Skeletons {
    /// `[x, y, distance to the boundary]` of the vertices of all skeletons.
    pub vertices: Vec<[f64; 3]>,
    /// Whether a vertex is an input point of the polygon (else a skeleton node).
    pub is_input: Vec<bool>,
    /// Polygon of each vertex.
    pub vertex_polygon: Vec<u32>,
    /// Interior arcs as pairs of vertex indices.
    pub arcs: Vec<[u32; 2]>,
    /// Polygon of each arc.
    pub arc_polygon: Vec<u32>,
    /// Vertex indices of all faces, one face after another.
    pub face_vertices: Vec<u32>,
    /// Start of every face in `face_vertices`, plus the total length.
    pub face_offsets: Vec<u64>,
    /// Polygon, ring (0 = exterior) and edge within the ring of each face.
    pub face_polygon: Vec<u32>,
    pub face_ring: Vec<u32>,
    pub face_edge: Vec<u32>,
    /// 0 for a skeleton, else the reason a polygon was rejected (see [`error_code`]).
    pub errors: Vec<u8>,
}

/// Rejection reasons, in the order of [`SkeletonError`].
pub fn error_code(error: &SkeletonError) -> u8 {
    match error {
        SkeletonError::EmptyInput => 1,
        SkeletonError::TooFewVertices { .. } => 2,
        SkeletonError::NonFinite { .. } => 3,
        SkeletonError::ZeroArea { .. } => 4,
        SkeletonError::SelfIntersection { .. } => 5,
        SkeletonError::HoleOutside { .. } => 6,
        SkeletonError::NestedHole { .. } => 7,
        SkeletonError::Numerical => 8,
    }
}

/// Polygon rings from the flat contract of `polygon_arrays`: `rings[r]..rings[r + 1]` are the points of ring `r`
/// (without the closing point), `polygons[p]..polygons[p + 1]` the rings of polygon `p`, exterior first.
fn polygon_rings(
    coords: &[[f64; 2]],
    rings: &[usize],
    polygons: &[usize],
    p: usize,
) -> Vec<Vec<[f64; 2]>> {
    (polygons[p]..polygons[p + 1])
        .map(|r| coords[rings[r]..rings[r + 1]].to_vec())
        .collect()
}

/// Straight skeletons of all polygons, computed in parallel.
pub fn skeletons(coords: &[[f64; 2]], rings: &[usize], polygons: &[usize]) -> Skeletons {
    let n = polygons.len().saturating_sub(1);
    let built: Vec<_> = (0..n)
        .into_par_iter()
        .map(|p| {
            let polygon = polygon_rings(coords, rings, polygons, p);
            let sizes: Vec<usize> = polygon.iter().map(Vec::len).collect();
            (Skeleton::build(&polygon), sizes)
        })
        .collect();

    let mut out = Skeletons {
        face_offsets: vec![0],
        ..Skeletons::default()
    };
    for (p, (result, sizes)) in built.into_iter().enumerate() {
        let skeleton = match result {
            Ok(skeleton) => skeleton,
            Err(error) => {
                out.errors.push(error_code(&error));
                continue;
            }
        };
        out.errors.push(0);
        let base = out.vertices.len() as u32;
        let inputs: usize = sizes.iter().sum();
        for (k, vertex) in skeleton.vertices.iter().enumerate() {
            out.vertices.push(*vertex);
            out.is_input.push(k < inputs);
            out.vertex_polygon.push(p as u32);
        }
        for (a, b) in skeleton.arcs() {
            out.arcs.push([base + a as u32, base + b as u32]);
            out.arc_polygon.push(p as u32);
        }
        // faces[k] belongs to the edge starting at input point k; rings follow one another
        let starts: Vec<usize> = sizes
            .iter()
            .scan(0, |acc, &size| {
                let start = *acc;
                *acc += size;
                Some(start)
            })
            .collect();
        for (k, face) in skeleton.faces.iter().enumerate() {
            let ring = starts.partition_point(|&start| start <= k) - 1;
            out.face_vertices
                .extend(face.vertices.iter().map(|&v| base + v as u32));
            out.face_offsets.push(out.face_vertices.len() as u64);
            out.face_polygon.push(p as u32);
            out.face_ring.push(ring as u32);
            out.face_edge.push((k - starts[ring]) as u32);
        }
    }
    out
}
