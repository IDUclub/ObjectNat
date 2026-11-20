//! Assemble one face per input edge from the arcs traced by the wavefront vertices.

use super::engine::{Arc, Node};
use super::geom::*;

/// Faces of the computed edges, as node sequences `[end, ...skeleton nodes..., start]`.
/// `canon` maps every node to a representative: input vertices at the same position (rings that
/// touch) share one.
pub(crate) fn assemble(
    nodes: &[Node],
    arcs: &[Arc],
    edge_nodes: &[(u32, u32)],
    canon: &[u32],
) -> Result<Vec<Vec<u32>>, usize> {
    let n_faces = edge_nodes.len();
    let c = |n: u32| canon.get(n as usize).copied().unwrap_or(n);
    // directed half-arcs per face: (face, from, to)
    let mut half: Vec<(u32, u32, u32)> = Vec::with_capacity(arcs.len() * 2);
    for a in arcs {
        let (na, nb) = (c(a.a), c(a.b));
        if na == nb {
            continue;
        }
        half.push((a.left, na, nb));
        half.push((a.right, nb, na));
    }
    half.sort_unstable();
    half.dedup();
    let mut first = vec![0usize; n_faces + 1];
    for h in &half {
        first[h.0 as usize + 1] += 1;
    }
    for f in 0..n_faces {
        first[f + 1] += first[f];
    }
    let mut used = vec![false; half.len()];
    let mut faces = Vec::with_capacity(n_faces);
    for (f, &(start, end)) in edge_nodes.iter().enumerate() {
        let (start, end) = (c(start), c(end));
        let lo = first[f];
        let hi = first[f + 1];
        let hs = &half[lo..hi];
        let mut out = vec![end];
        let mut cur = end;
        let mut prev_dir = sub(nodes[end as usize].pos, nodes[start as usize].pos);
        let mut steps = 0usize;
        while cur != start {
            steps += 1;
            if steps > hs.len() + 1 {
                return Err(f);
            }
            // candidates leaving `cur`
            let k0 = hs.partition_point(|h| h.1 < cur);
            let mut best: Option<usize> = None;
            let mut best_ang = f64::INFINITY;
            let back = neg(prev_dir);
            let mut k = k0;
            while k < hs.len() && hs[k].1 == cur {
                if !used[lo + k] {
                    let dir = sub(nodes[hs[k].2 as usize].pos, nodes[cur as usize].pos);
                    // first candidate clockwise from the reversed incoming direction
                    let a = ccw_angle(dir, back);
                    let a = if a <= 0.0 { TAU } else { a };
                    if best.is_none() || a < best_ang {
                        best = Some(k);
                        best_ang = a;
                    }
                }
                k += 1;
            }
            let Some(k) = best else {
                return Err(f);
            };
            used[lo + k] = true;
            let nx = hs[k].2;
            prev_dir = sub(nodes[nx as usize].pos, nodes[cur as usize].pos);
            cur = nx;
            out.push(cur);
        }
        faces.push(out);
    }
    Ok(faces)
}
