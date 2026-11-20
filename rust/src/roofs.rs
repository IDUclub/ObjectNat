//! Piecewise-planar roofs, in a local unit-scale frame to avoid cancellation in projected CRS.
//!
//! Hips use the event-driven, equal-speed wavefront in geo-straight-skeleton 0.1.1. It is a
//! dependency-free f64 Rust implementation (MIT OR Apache-2.0), supports holes and simultaneous
//! events, and has an upstream CGAL oracle corpus. We still validate every result independently.
//! Floating-point failures fall back to a gable; this is not an exact-predicate implementation.

use std::collections::BTreeMap;

use geo_straight_skeleton::Skeleton;
use rayon::prelude::*;

type Pt = [f64; 2];
type Face = Vec<[f64; 3]>;
type Triangle = [Pt; 3];

#[derive(Clone, Copy)]
pub struct RoofSpec {
    pub shape: u8,
    pub eave: f64,
    pub ridge: f64,
    pub direction: f64,
    pub across: bool,
}

pub struct RoofMesh {
    pub faces: Vec<Face>,
    pub edges: Vec<[f64; 6]>,
    pub status: u8,
    pub used_shape: u8,
}

fn sub(a: Pt, b: Pt) -> Pt {
    [a[0] - b[0], a[1] - b[1]]
}
fn dot(a: Pt, b: Pt) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn cross(a: Pt, b: Pt) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}
fn area(r: &[Pt]) -> f64 {
    r.iter()
        .zip(r.iter().cycle().skip(1))
        .take(r.len())
        .map(|(&a, &b)| cross(a, b))
        .sum::<f64>()
        * 0.5
}
fn xy(p: &[f64; 3]) -> Pt {
    [p[0], p[1]]
}

fn clean(r: &[Pt]) -> Vec<Pt> {
    let mut out = Vec::new();
    for &p in r {
        if out.last() != Some(&p) {
            out.push(p);
        }
    }
    if out.len() > 1 && out.first() == out.last() {
        out.pop();
    }
    out
}

fn triangulate(rings: &[Vec<Pt>]) -> Vec<Triangle> {
    let mut coords = Vec::new();
    let mut holes = Vec::new();
    for (k, r) in rings.iter().enumerate() {
        if k > 0 {
            holes.push(coords.len() / 2);
        }
        coords.extend(r.iter().flatten().copied());
    }
    if coords.len() < 6 {
        return Vec::new();
    }
    let Ok(indices) = earcutr::earcut(&coords, &holes, 2) else {
        return Vec::new();
    };
    indices
        .chunks_exact(3)
        .filter_map(|ids| {
            let mut t: Triangle =
                std::array::from_fn(|k| [coords[2 * ids[k]], coords[2 * ids[k] + 1]]);
            if cross(sub(t[1], t[0]), sub(t[2], t[0])) < 0.0 {
                t.swap(1, 2);
            }
            (area(&t) > 1e-16).then_some(t)
        })
        .collect()
}

/// Clip a convex polygon to dot(p, normal) >= offset.
fn clip(poly: &[Pt], normal: Pt, offset: f64) -> Vec<Pt> {
    let mut out = Vec::new();
    for (&a, &b) in poly
        .iter()
        .zip(poly.iter().cycle().skip(1))
        .take(poly.len())
    {
        let da = dot(a, normal) - offset;
        let db = dot(b, normal) - offset;
        if da >= 0.0 {
            out.push(a);
        }
        if (da < 0.0 && db > 0.0) || (da > 0.0 && db < 0.0) {
            let t = da / (da - db);
            out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
    clean(&out)
}

fn overlap(a: &[Pt], b: &Triangle) -> f64 {
    let mut p = a.to_vec();
    for k in 0..3 {
        let edge = sub(b[(k + 1) % 3], b[k]);
        let n = [-edge[1], edge[0]];
        p = clip(&p, n, dot(b[k], n));
    }
    area(&p).abs()
}

fn convex(r: &[Pt]) -> bool {
    r.len() >= 3
        && (0..r.len()).all(|k| {
            cross(
                sub(r[(k + 1) % r.len()], r[k]),
                sub(r[(k + 2) % r.len()], r[k]),
            ) >= -1e-12
        })
}

fn lift(r: &[Pt], height: impl Fn(Pt) -> f64) -> Face {
    r.iter().map(|&p| [p[0], p[1], height(p)]).collect()
}

/// Minimum-area rectangle of the convex hull: centre plus its longer side's unit direction.
fn rectangle(r: &[Pt]) -> (Pt, Pt) {
    let mut points = r.to_vec();
    points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    points.dedup();
    let mut hull: Vec<Pt> = Vec::new();
    for &p in &points {
        while hull.len() >= 2
            && cross(
                sub(hull[hull.len() - 1], hull[hull.len() - 2]),
                sub(p, hull[hull.len() - 1]),
            ) <= 0.0
        {
            hull.pop();
        }
        hull.push(p);
    }
    let lower = hull.len();
    for &p in points.iter().rev().skip(1) {
        while hull.len() > lower
            && cross(
                sub(hull[hull.len() - 1], hull[hull.len() - 2]),
                sub(p, hull[hull.len() - 1]),
            ) <= 0.0
        {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    let mut best = f64::INFINITY;
    let (mut centre, mut axis) = ([0.0; 2], [1.0, 0.0]);
    for (&a, &b) in hull
        .iter()
        .zip(hull.iter().cycle().skip(1))
        .take(hull.len())
    {
        let d = sub(b, a);
        let len = d[0].hypot(d[1]);
        if len == 0.0 {
            continue;
        }
        let u = [d[0] / len, d[1] / len];
        let v = [-u[1], u[0]];
        let bounds = |n| {
            r.iter()
                .map(|&p| dot(p, n))
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
                    (lo.min(x), hi.max(x))
                })
        };
        let (u0, u1) = bounds(u);
        let (v0, v1) = bounds(v);
        let size = (u1 - u0) * (v1 - v0);
        if size < best - 1e-14 {
            best = size;
            centre = [
                u[0] * (u0 + u1) / 2.0 + v[0] * (v0 + v1) / 2.0,
                u[1] * (u0 + u1) / 2.0 + v[1] * (v0 + v1) / 2.0,
            ];
            axis = if u1 - u0 >= v1 - v0 { u } else { v };
        }
    }
    // A stable sign makes the default downslope deterministic.
    if axis[0] < -1e-12 || (axis[0].abs() <= 1e-12 && axis[1] < 0.0) {
        axis = [-axis[0], -axis[1]];
    }
    (centre, axis)
}

fn roof_axis(r: &[Pt], spec: RoofSpec) -> (Pt, Pt) {
    let (c, u) = rectangle(r);
    let n = if spec.direction.is_finite() {
        [spec.direction.sin(), spec.direction.cos()]
    } else if spec.across {
        u
    } else {
        [-u[1], u[0]]
    };
    (c, n)
}

fn flat(rings: &[Vec<Pt>], z: f64) -> Vec<Face> {
    if rings.len() == 1 && area(&rings[0]).abs() > 1e-16 {
        vec![lift(&rings[0], |_| z)]
    } else {
        triangulate(rings).iter().map(|r| lift(r, |_| z)).collect()
    }
}

fn striped(rings: &[Vec<Pt>], spec: RoofSpec, barrel: bool) -> Vec<Face> {
    let (c, n) = roof_axis(&rings[0], spec);
    let centre = dot(c, n);
    let hw = rings
        .iter()
        .flatten()
        .map(|&p| (dot(p, n) - centre).abs())
        .fold(0.0, f64::max);
    if hw <= 1e-14 {
        return flat(rings, spec.ridge);
    }
    let parts = if rings.len() == 1 && convex(&rings[0]) {
        vec![rings[0].clone()]
    } else {
        triangulate(rings).iter().map(|t| t.to_vec()).collect()
    };
    let steps: i32 = if barrel { 4 } else { 1 };
    let mut faces = Vec::new();
    for k in -steps..steps {
        let profile = |j: i32| {
            let angle = std::f64::consts::FRAC_PI_2 * j as f64 / steps as f64;
            if barrel {
                (angle.sin(), angle.cos())
            } else {
                (j as f64, 1.0 - (j as f64).abs())
            }
        };
        let (x0, h0) = profile(k);
        let (x1, h1) = profile(k + 1);
        let lo = centre + hw * x0;
        let hi = centre + hw * x1;
        for part in &parts {
            let p = clip(&clip(part, n, lo), [-n[0], -n[1]], -hi);
            if area(&p) > 1e-16 {
                faces.push(lift(&p, |q| {
                    spec.eave
                        + (spec.ridge - spec.eave)
                            * (h0 + (h1 - h0) * ((dot(q, n) - lo) / (hi - lo)).clamp(0.0, 1.0))
                }));
            }
        }
    }
    faces
}

fn skillion(rings: &[Vec<Pt>], spec: RoofSpec) -> Vec<Face> {
    let (_, n) = roof_axis(&rings[0], spec);
    let (lo, hi) = rings
        .iter()
        .flatten()
        .map(|&p| dot(p, n))
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), d| {
            (lo.min(d), hi.max(d))
        });
    let mut faces = flat(rings, 0.0);
    for p in faces.iter_mut().flatten() {
        p[2] = spec.eave
            + (spec.ridge - spec.eave) * ((hi - dot(xy(p), n)) / (hi - lo)).clamp(0.0, 1.0);
    }
    faces
}

fn apex(rings: &[Vec<Pt>]) -> Option<Pt> {
    if rings.len() != 1 {
        return None;
    }
    let r = &rings[0];
    let mut c = [0.0; 2];
    for (&a, &b) in r.iter().zip(r.iter().cycle().skip(1)).take(r.len()) {
        let w = cross(a, b);
        for j in 0..2 {
            c[j] += (a[j] + b[j]) * w;
        }
    }
    for v in &mut c {
        *v /= 6.0 * area(r);
    }
    // If the area centroid is outside, use the centroid of a representative triangle.
    let inside = crate::insolation::Underside {
        rings: rings.to_vec(),
        z: 0.0,
    };
    if !inside.contains(c) {
        let tris = triangulate(rings);
        let t = tris.iter().max_by(|a, b| area(*a).total_cmp(&area(*b)))?;
        c = [
            (t[0][0] + t[1][0] + t[2][0]) / 3.0,
            (t[0][1] + t[1][1] + t[2][1]) / 3.0,
        ];
    }
    // Visibility kernel of a simple CCW polygon = intersection of all inward halfplanes.
    r.iter()
        .zip(r.iter().cycle().skip(1))
        .take(r.len())
        .all(|(&a, &b)| cross(sub(b, a), sub(c, a)) > 1e-12)
        .then_some(c)
}

fn radial(rings: &[Vec<Pt>], spec: RoofSpec, curved: bool) -> Option<Vec<Face>> {
    let c = apex(rings)?;
    let r = &rings[0];
    let tiers = if curved { 4 } else { 1 };
    let at = |p: Pt, k: usize| {
        let angle = std::f64::consts::FRAC_PI_2 * k as f64 / tiers as f64;
        let (s, h) = if curved {
            (angle.cos(), angle.sin())
        } else {
            (1.0 - k as f64, k as f64)
        };
        [
            c[0] + s * (p[0] - c[0]),
            c[1] + s * (p[1] - c[1]),
            spec.eave + (spec.ridge - spec.eave) * h,
        ]
    };
    let mut faces = Vec::new();
    for k in 0..tiers {
        for (&a, &b) in r.iter().zip(r.iter().cycle().skip(1)).take(r.len()) {
            let mut face = vec![at(a, k), at(b, k)];
            if k + 1 == tiers {
                face.push([c[0], c[1], spec.ridge]);
            } else {
                face.extend([at(b, k + 1), at(a, k + 1)]);
            }
            faces.push(face);
        }
    }
    Some(faces)
}

/// Independent checks: planar facets, height range, no overlap, and all area within F.
fn valid(faces: &[Face], rings: &[Vec<Pt>], spec: RoofSpec) -> bool {
    if faces.is_empty() {
        return false;
    }
    let base = triangulate(rings);
    let wanted: f64 = base.iter().map(|t| area(t)).sum();
    let ztol = (spec.ridge - spec.eave).abs().max(1.0) * 1e-7;
    let mut triangles = Vec::new();
    for face in faces {
        if face.iter().flatten().any(|v| !v.is_finite())
            || face
                .iter()
                .any(|v| v[2] < spec.eave - ztol || v[2] > spec.ridge + ztol)
        {
            return false;
        }
        let r: Vec<_> = face.iter().map(xy).collect();
        if area(&r) <= 1e-16 {
            return false;
        }
        let Some(k) = (1..face.len() - 1)
            .find(|&k| cross(sub(r[k], r[0]), sub(r[k + 1], r[0])).abs() > 1e-16)
        else {
            return false;
        };
        let (u, v) = (sub(r[k], r[0]), sub(r[k + 1], r[0]));
        let (zu, zv) = (face[k][2] - face[0][2], face[k + 1][2] - face[0][2]);
        let d = cross(u, v);
        let slope = [(zu * v[1] - zv * u[1]) / d, (u[0] * zv - v[0] * zu) / d];
        if face
            .iter()
            .any(|p| (p[2] - face[0][2] - dot(sub(xy(p), r[0]), slope)).abs() > ztol)
        {
            return false;
        }
        triangles.extend(triangulate(&[r]));
    }
    let got: f64 = triangles.iter().map(|t| area(t)).sum();
    let tol = wanted.max(1e-8) * 1e-7;
    if wanted <= 1e-16 || (got - wanted).abs() > tol {
        return false;
    }
    for (k, t) in triangles.iter().enumerate() {
        let covered: f64 = base.iter().map(|b| overlap(t, b)).sum();
        if (covered - area(t)).abs() > tol {
            return false;
        }
        for b in &triangles[..k] {
            if overlap(t, b) > tol {
                return false;
            }
        }
    }
    true
}

fn hipped(rings: &[Vec<Pt>], spec: RoofSpec) -> Option<Vec<Face>> {
    let sk = Skeleton::build(rings).ok()?;
    let tmax = sk.vertices.iter().map(|v| v[2]).fold(0.0, f64::max);
    if !tmax.is_finite() || tmax <= 0.0 {
        return None;
    }
    let faces: Vec<Face> = sk
        .faces
        .iter()
        .filter(|f| f.vertices.len() >= 3)
        .map(|f| {
            f.vertices
                .iter()
                .map(|&k| {
                    let v = sk.vertices[k];
                    [
                        v[0],
                        v[1],
                        spec.eave + (spec.ridge - spec.eave) * v[2] / tmax,
                    ]
                })
                .collect()
        })
        .collect();
    valid(&faces, rings, spec).then_some(faces)
}

/// Unique 2D vertex keys (unit-scale XY); roof surfaces are continuous single-valued functions.
fn key(p: Pt) -> [i64; 2] {
    [(p[0] * 1e10).round() as i64, (p[1] * 1e10).round() as i64]
}

fn face_slope(face: &Face) -> Option<Pt> {
    if face.len() < 3 {
        return None;
    }
    let a = face[0];
    for k in 1..face.len() - 1 {
        let (u, v) = (sub(xy(&face[k]), xy(&a)), sub(xy(&face[k + 1]), xy(&a)));
        let d = cross(u, v);
        if d.abs() > 1e-16 {
            let (zu, zv) = (face[k][2] - a[2], face[k + 1][2] - a[2]);
            return Some([(zu * v[1] - zv * u[1]) / d, (u[0] * zv - v[0] * zu) / d]);
        }
    }
    None
}

fn coplanar(a: &Face, b: &Face, sa: Option<Pt>, sb: Option<Pt>, ztol: f64) -> bool {
    let (Some(sa), Some(sb)) = (sa, sb) else {
        return false;
    };
    // Test both complete facets, rather than just their shared edge (always collinear in 3D).
    a.iter()
        .all(|p| (p[2] - b[0][2] - dot(sub(xy(p), xy(&b[0])), sb)).abs() <= ztol)
        && b.iter()
            .all(|p| (p[2] - a[0][2] - dot(sub(xy(p), xy(&a[0])), sa)).abs() <= ztol)
}

fn boundary_edge(a: Pt, b: Pt, boundary: &[Vec<Pt>]) -> bool {
    boundary.iter().any(|ring| {
        ring.iter()
            .zip(ring.iter().cycle().skip(1))
            .take(ring.len())
            .any(|(&p, &q)| {
                let d = sub(q, p);
                let len2 = dot(d, d);
                len2 > 1e-20
                    && [a, b].iter().all(|&v| {
                        let u = sub(v, p);
                        (-1e-10..=1.0 + 1e-10).contains(&(dot(u, d) / len2))
                            && cross(d, u).abs() <= 1e-10 * len2.sqrt()
                    })
            })
    })
}

fn edges(faces: &[Face], boundary: &[Vec<Pt>], flat_z: f64, ztol: f64) -> Vec<[f64; 6]> {
    let mut vertices = BTreeMap::<[i64; 2], [f64; 3]>::new();
    let mut raw = Vec::new();
    for (owner, face) in faces.iter().enumerate() {
        for (&a, &b) in face
            .iter()
            .zip(face.iter().cycle().skip(1))
            .take(face.len())
        {
            raw.push((a, b, owner));
        }
    }
    if faces.is_empty() {
        for r in boundary {
            for (&a, &b) in r.iter().zip(r.iter().cycle().skip(1)).take(r.len()) {
                raw.push(([a[0], a[1], flat_z], [b[0], b[1], flat_z], usize::MAX));
            }
        }
    }
    for &(a, b, _) in &raw {
        for p in [a, b] {
            vertices
                .entry(key(xy(&p)))
                .and_modify(|v| v[2] = v[2].max(p[2]))
                .or_insert(p);
        }
    }
    let mut unique = BTreeMap::<_, Vec<usize>>::new();
    for (a, b, owner) in raw {
        let d = sub(xy(&b), xy(&a));
        let len2 = dot(d, d);
        if len2 <= 1e-20 {
            continue;
        }
        let mut cuts: Vec<_> = vertices
            .iter()
            .filter_map(|(&id, p)| {
                let q = sub(xy(p), xy(&a));
                let t = dot(q, d) / len2;
                ((-1e-10..=1.0 + 1e-10).contains(&t) && cross(d, q).abs() <= 1e-10 * len2.sqrt())
                    .then_some((t, id))
            })
            .collect();
        cuts.sort_by(|a, b| a.0.total_cmp(&b.0));
        for pair in cuts.windows(2) {
            let (u, v) = (pair[0].1, pair[1].1);
            if u != v {
                let adjacent = unique
                    .entry(if u < v { (u, v) } else { (v, u) })
                    .or_default();
                if !adjacent.contains(&owner) {
                    adjacent.push(owner);
                }
            }
        }
    }
    // Only two coplanar neighbours can be merged. Keep boundary edges and ambiguous adjacency;
    // real roof creases remain breakpoints of the piecewise-linear height along every ray.
    let slopes: Vec<_> = faces.iter().map(face_slope).collect();
    unique
        .into_iter()
        .filter_map(|((a, b), adjacent)| {
            let (a, b) = (vertices[&a], vertices[&b]);
            if adjacent.len() == 2
                && !boundary_edge(xy(&a), xy(&b), boundary)
                && coplanar(
                    &faces[adjacent[0]],
                    &faces[adjacent[1]],
                    slopes[adjacent[0]],
                    slopes[adjacent[1]],
                    ztol,
                )
            {
                return None;
            }
            Some([a[0], a[1], a[2], b[0], b[1], b[2]])
        })
        .collect()
}

pub fn roof_mesh(rings: &[Vec<Pt>], spec: RoofSpec) -> RoofMesh {
    let mut rings: Vec<_> = rings
        .iter()
        .map(|r| clean(r))
        .filter(|r| !r.is_empty())
        .collect();
    let origin = rings
        .first()
        .and_then(|r| r.first())
        .copied()
        .unwrap_or([0.0; 2]);
    let scale = rings
        .iter()
        .flatten()
        .map(|&p| {
            let d = sub(p, origin);
            d[0].abs().max(d[1].abs())
        })
        .fold(0.0, f64::max)
        .max(1e-12);
    for (k, r) in rings.iter_mut().enumerate() {
        for p in r.iter_mut() {
            *p = [(p[0] - origin[0]) / scale, (p[1] - origin[1]) / scale];
        }
        if (area(r) > 0.0) != (k == 0) {
            r.reverse();
        }
    }
    let mut used = match spec.shape {
        0..=9 => spec.shape,
        10..=13 => 3,
        14..=15 => 2,
        _ => 0,
    };
    let mut status = u8::from(used != spec.shape);
    if rings.is_empty() || area(&rings[0]).abs() <= 1e-16 || spec.ridge <= spec.eave {
        used = 0;
        status = u8::from(spec.shape != 0 || spec.ridge < spec.eave);
    }
    let flat_z = spec.eave.max(spec.ridge);
    let mut faces = match used {
        0 => flat(&rings, flat_z),
        1 => skillion(&rings, spec),
        2 | 9 => striped(&rings, spec, used == 9),
        3 => hipped(&rings, spec).unwrap_or_else(|| {
            used = 2;
            status = 1;
            striped(&rings, spec, false)
        }),
        _ => radial(&rings, spec, (6..=8).contains(&used)).unwrap_or_else(|| {
            status = 1;
            used = 3;
            hipped(&rings, spec).unwrap_or_else(|| {
                used = 2;
                striped(&rings, spec, false)
            })
        }),
    };
    let ztol = (spec.ridge - spec.eave).abs().max(1.0) * 1e-7;
    let mut edges = edges(&faces, &rings, flat_z, ztol);
    for p in faces.iter_mut().flatten() {
        p[0] = origin[0] + scale * p[0];
        p[1] = origin[1] + scale * p[1];
    }
    for e in &mut edges {
        for k in [0, 3] {
            e[k] = origin[0] + scale * e[k];
            e[k + 1] = origin[1] + scale * e[k + 1];
        }
    }
    RoofMesh {
        faces,
        edges,
        status,
        used_shape: used,
    }
}

pub fn roof_meshes(polygons: &[Vec<Vec<Pt>>], specs: &[RoofSpec]) -> Vec<RoofMesh> {
    polygons
        .par_iter()
        .zip(specs.par_iter())
        .map(|(p, &s)| roof_mesh(p, s))
        .collect()
}

#[cfg(test)]
#[path = "roof_tests.rs"]
mod tests;
