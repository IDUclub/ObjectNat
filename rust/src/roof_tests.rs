use super::*;
use crate::insolation::{Scene, Sun, Underside};
use crate::visibility::Seg;

fn spec(shape: u8) -> RoofSpec {
    RoofSpec {
        shape,
        eave: 10.0,
        ridge: 14.0,
        direction: f64::NAN,
        across: false,
    }
}

fn rect() -> Vec<Vec<Pt>> {
    vec![vec![[0.0, 0.0], [12.0, 0.0], [12.0, 8.0], [0.0, 8.0]]]
}

fn courtyard() -> Vec<Vec<Pt>> {
    vec![
        vec![[0.0, 0.0], [16.0, 0.0], [16.0, 16.0], [0.0, 16.0]],
        vec![[6.0, 6.0], [6.0, 10.0], [10.0, 10.0], [10.0, 6.0]],
    ]
}

fn l_shape() -> Vec<Vec<Pt>> {
    vec![vec![
        [0.0, 0.0],
        [12.0, 0.0],
        [12.0, 4.0],
        [4.0, 4.0],
        [4.0, 12.0],
        [0.0, 12.0],
    ]]
}

fn u_shape() -> Vec<Vec<Pt>> {
    vec![vec![
        [0.0, 0.0],
        [12.0, 0.0],
        [12.0, 12.0],
        [8.0, 12.0],
        [8.0, 4.0],
        [4.0, 4.0],
        [4.0, 12.0],
        [0.0, 12.0],
    ]]
}

fn has_edge(mesh: &RoofMesh, a: [f64; 3], b: [f64; 3]) -> bool {
    let near = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-7);
    mesh.edges.iter().any(|e| {
        (near(&e[..3], &a) && near(&e[3..], &b)) || (near(&e[..3], &b) && near(&e[3..], &a))
    })
}

#[test]
fn rectangle_exact_edges_and_heights() {
    let gable = roof_mesh(&rect(), spec(2));
    assert_eq!(
        (
            gable.status,
            gable.used_shape,
            gable.faces.len(),
            gable.edges.len()
        ),
        (0, 2, 2, 7)
    );
    assert!(has_edge(&gable, [0.0, 4.0, 14.0], [12.0, 4.0, 14.0]));
    assert!(has_edge(&gable, [0.0, 0.0, 10.0], [0.0, 4.0, 14.0]));
    assert!(has_edge(&gable, [0.0, 4.0, 14.0], [0.0, 8.0, 10.0]));
    let hip = roof_mesh(&rect(), spec(3));
    assert_eq!(
        (hip.status, hip.used_shape, hip.faces.len(), hip.edges.len()),
        (0, 3, 4, 9)
    );
    assert!(has_edge(&hip, [4.0, 4.0, 14.0], [8.0, 4.0, 14.0]));
    assert!(has_edge(&hip, [0.0, 0.0, 10.0], [4.0, 4.0, 14.0]));
    let pyramid = roof_mesh(&rect(), spec(4));
    assert_eq!(
        (pyramid.status, pyramid.faces.len(), pyramid.edges.len()),
        (0, 4, 8)
    );
    assert!(has_edge(&pyramid, [0.0, 0.0, 10.0], [6.0, 4.0, 14.0]));
    let shed = roof_mesh(
        &rect(),
        RoofSpec {
            direction: 0.0,
            ..spec(1)
        },
    );
    assert_eq!((shed.status, shed.faces.len(), shed.edges.len()), (0, 1, 4));
    assert!(has_edge(&shed, [0.0, 0.0, 14.0], [12.0, 0.0, 14.0]));
    assert!(has_edge(&shed, [0.0, 8.0, 10.0], [12.0, 8.0, 10.0]));
}

#[test]
fn concave_gabled_roofs_have_no_coplanar_interior_edges() {
    for rings in [l_shape(), u_shape(), courtyard()] {
        let mesh = roof_mesh(&rings, spec(2));
        // Negative tolerance disables pruning for an independent before/after edge inventory.
        let unpruned = edges(&mesh.faces, &rings, 14.0, -1.0);
        assert!(unpruned.len() > mesh.edges.len());
        let mut ridges = 0;
        for e in &mesh.edges {
            let (a, b) = ([e[0], e[1]], [e[3], e[4]]);
            if boundary_edge(a, b, &rings) {
                continue;
            }
            let adjacent: Vec<_> = mesh
                .faces
                .iter()
                .filter(|f| {
                    let ring = f.iter().map(xy).collect();
                    boundary_edge(a, b, &[ring])
                })
                .collect();
            assert_eq!(adjacent.len(), 2);
            assert!(!coplanar(
                adjacent[0],
                adjacent[1],
                face_slope(adjacent[0]),
                face_slope(adjacent[1]),
                4e-7
            ));
            ridges += 1;
        }
        assert!(ridges > 0, "real ridge segments must be retained");
        for e in &unpruned {
            if boundary_edge([e[0], e[1]], [e[3], e[4]], &rings) {
                assert!(has_edge(&mesh, [e[0], e[1], e[2]], [e[3], e[4], e[5]]));
            }
        }
    }
}

#[test]
fn holed_flat_and_skillion_keep_only_footprint_edges() {
    for shape in [0, 1] {
        let rings = courtyard();
        let mesh = roof_mesh(&rings, spec(shape));
        assert_eq!((mesh.status, mesh.used_shape), (0, shape));
        assert!(mesh
            .edges
            .iter()
            .all(|e| boundary_edge([e[0], e[1]], [e[3], e[4]], &rings)));
        let total: f64 = mesh
            .edges
            .iter()
            .map(|e| (e[3] - e[0]).hypot(e[4] - e[1]))
            .sum();
        assert!((total - 80.0).abs() < 1e-7); // outer perimeter 64 + courtyard perimeter 16
    }
}

#[test]
fn explicit_direction_across_and_rotated_rectangle() {
    let across = roof_mesh(
        &rect(),
        RoofSpec {
            across: true,
            ..spec(2)
        },
    );
    assert!(has_edge(&across, [6.0, 0.0, 14.0], [6.0, 8.0, 14.0]));
    let bearing = roof_mesh(
        &rect(),
        RoofSpec {
            across: true,
            direction: 0.0,
            ..spec(2)
        },
    );
    assert!(has_edge(&bearing, [0.0, 4.0, 14.0], [12.0, 4.0, 14.0]));
    let a = 0.37_f64;
    let rotate = |p: Pt| {
        [
            350_000.0 + p[0] * a.cos() - p[1] * a.sin(),
            6_650_000.0 + p[0] * a.sin() + p[1] * a.cos(),
        ]
    };
    let rings = vec![rect()[0].iter().map(|&p| rotate(p)).collect()];
    let mesh = roof_mesh(&rings, spec(2));
    let (p, q) = (rotate([0.0, 4.0]), rotate([12.0, 4.0]));
    assert!(has_edge(&mesh, [p[0], p[1], 14.0], [q[0], q[1], 14.0]));
}

#[test]
fn skeleton_handles_concavities_and_courtyard() {
    for rings in [l_shape(), u_shape(), courtyard()] {
        let mesh = roof_mesh(&rings, spec(3));
        assert_eq!((mesh.status, mesh.used_shape), (0, 3));
        assert!(valid(&mesh.faces, &rings, spec(3)));
    }
}

#[test]
fn curved_roofs_are_planar_and_boundary_is_split() {
    let ring: Vec<_> = (0..16)
        .map(|k| {
            let a = k as f64 * std::f64::consts::TAU / 16.0;
            [10.0 * a.cos(), 10.0 * a.sin()]
        })
        .collect();
    for shape in 6..=8 {
        let rings = vec![ring.clone()];
        let mesh = roof_mesh(&rings, spec(shape));
        assert_eq!(
            (mesh.status, mesh.used_shape, mesh.faces.len()),
            (0, shape, 64)
        );
        assert!(valid(&mesh.faces, &rings, spec(shape)));
    }
    let barrel = roof_mesh(&rect(), spec(9));
    assert_eq!(barrel.faces.len(), 8);
    assert_eq!(barrel.edges.len(), 25);
    assert!(valid(&barrel.faces, &rect(), spec(9)));
    // No output edge may skip a vertex where another facet meets it.
    for e in &barrel.edges {
        let (a, b) = ([e[0], e[1]], [e[3], e[4]]);
        let d = sub(b, a);
        for p in barrel.faces.iter().flatten() {
            let q = sub(xy(p), a);
            let t = dot(q, d) / dot(d, d);
            assert!(!((1e-7..1.0 - 1e-7).contains(&t) && cross(d, q).abs() < 1e-8));
        }
    }
}

#[test]
fn approximations_fallbacks_degeneracies_and_winding() {
    for (shape, used) in [
        (10, 3),
        (11, 3),
        (12, 3),
        (13, 3),
        (14, 2),
        (15, 2),
        (16, 0),
        (255, 0),
    ] {
        let mesh = roof_mesh(&rect(), spec(shape));
        assert_eq!((mesh.status, mesh.used_shape), (1, used));
    }
    for shape in 4..=8 {
        let mesh = roof_mesh(&courtyard(), spec(shape));
        assert_eq!((mesh.status, mesh.used_shape), (1, 3));
    }
    let mesh = roof_mesh(&u_shape(), spec(4));
    assert_eq!((mesh.status, mesh.used_shape), (1, 3));
    let mesh = roof_mesh(
        &rect(),
        RoofSpec {
            ridge: 8.0,
            ..spec(3)
        },
    );
    assert_eq!((mesh.status, mesh.used_shape), (1, 0));
    assert!(mesh.edges.iter().all(|e| e[2] == 10.0 && e[5] == 10.0));
    let line = vec![vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]];
    let mesh = roof_mesh(&line, spec(3));
    assert_eq!((mesh.status, mesh.used_shape), (1, 0));
    assert!(mesh.faces.is_empty());
    assert!(mesh.edges.iter().all(|e| e[2] == 14.0 && e[5] == 14.0));
    assert!(roof_mesh(&[], spec(3)).edges.is_empty());
    let mut rings = courtyard();
    for ring in &mut rings {
        ring.reverse();
        ring.push(ring[0]);
    }
    let mesh = roof_mesh(&rings, spec(3));
    assert_eq!((mesh.status, mesh.used_shape), (0, 3));
    assert!(valid(&mesh.faces, &courtyard(), spec(3)));
    // Exercise the independently checked failure path with a bad skeleton result.
    let mut bad = roof_mesh(&rect(), spec(3)).faces;
    bad[0][0][2] += 2.0;
    assert!(!valid(&bad, &rect(), spec(3)));
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

#[test]
fn random_convex_and_concave_skeletons_validate() {
    let mut rng = Rng(17);
    for k in 0..80 {
        let n = 5 + k % 12;
        let mut ring = Vec::new();
        for j in 0..n {
            let a = std::f64::consts::TAU * j as f64 / n as f64;
            let r = if k % 2 == 0 {
                10.0
            } else {
                4.0 + 6.0 * rng.next()
            };
            ring.push([r * a.cos(), r * a.sin()]);
        }
        let rings = vec![ring];
        let mesh = roof_mesh(&rings, spec(3));
        assert!(
            valid(&mesh.faces, &rings, spec(3)),
            "case {k}, status {}",
            mesh.status
        );
        assert_eq!(
            mesh.used_shape, 3,
            "ordinary polygon should not fall back: case {k}"
        );
    }
}

fn vsub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|k| a[k] - b[k])
}
fn vcross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn vdot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| a[k] * b[k]).sum()
}

// Moller-Trumbore: independent 3D intersection, not the kernel's projected wall test.
// The second result marks intersections within 1e-6 m of a triangle edge as grazing.
fn hit(o: [f64; 3], d: [f64; 3], tri: &[[f64; 3]; 3], limit: f64) -> (bool, bool) {
    let (u, v) = (vsub(tri[1], tri[0]), vsub(tri[2], tri[0]));
    let h = vcross(d, v);
    let det = vdot(u, h);
    if det.abs() < 1e-12 {
        return (false, false);
    }
    let s = vsub(o, tri[0]);
    let a = vdot(s, h) / det;
    let q = vcross(s, u);
    let b = vdot(d, q) / det;
    let t = vdot(v, q) / det;
    if a < -1e-8 || b < -1e-8 || a + b > 1.0 + 1e-8 || t <= 1e-6 || t >= limit - 1e-6 {
        return (false, false);
    }
    let area2 = vdot(vcross(u, v), vcross(u, v)).sqrt();
    let edge_len = |a, b| vdot(vsub(a, b), vsub(a, b)).sqrt();
    let grazing = a * area2 / edge_len(tri[0], tri[2]) < 1e-6
        || b * area2 / edge_len(tri[0], tri[1]) < 1e-6
        || (1.0 - a - b) * area2 / edge_len(tri[1], tri[2]) < 1e-6;
    (true, grazing)
}

fn volume_triangles(mesh: &RoofMesh, rings: &[Vec<Pt>], bottom: f64) -> Vec<[[f64; 3]; 3]> {
    let mut tris = Vec::new();
    for face in &mesh.faces {
        let coords: Vec<_> = face.iter().flat_map(|p| [p[0], p[1]]).collect();
        for ids in earcutr::earcut(&coords, &[], 2).unwrap().chunks_exact(3) {
            tris.push([face[ids[0]], face[ids[1]], face[ids[2]]]);
        }
    }
    // Footprint walls from the face boundary: match each split edge to an original footprint edge.
    for e in &mesh.edges {
        let (a, b) = ([e[0], e[1]], [e[3], e[4]]);
        let on_boundary = rings.iter().any(|r| {
            r.iter()
                .zip(r.iter().cycle().skip(1))
                .take(r.len())
                .any(|(&p, &q)| {
                    let d = sub(q, p);
                    [a, b].iter().all(|&v| {
                        cross(d, sub(v, p)).abs() < 1e-7 && dot(sub(v, p), sub(v, q)) <= 1e-7
                    })
                })
        });
        if on_boundary {
            let (a, b) = ([e[0], e[1], e[2]], [e[3], e[4], e[5]]);
            let (al, bl) = ([a[0], a[1], bottom], [b[0], b[1], bottom]);
            tris.extend([[al, bl, b], [al, b, a]]);
        }
    }
    for t in triangulate(rings) {
        tris.push(t.map(|p| [p[0], p[1], bottom]));
    }
    tris
}

#[test]
fn generated_edge_kernel_agrees_with_brute_force_volume_oracle() {
    let mut rng = Rng(9863);
    let mut compared = 0;
    for rings in [rect(), l_shape(), u_shape(), courtyard()] {
        for shape in 0..=9 {
            let mesh = roof_mesh(&rings, spec(shape));
            let bottom = 5.0;
            let scene = Scene::with_details(
                mesh.edges
                    .iter()
                    .map(|e| Seg {
                        a: [e[0], e[1]],
                        b: [e[3], e[4]],
                    })
                    .collect(),
                mesh.edges.iter().map(|e| [e[2], e[5]]).collect(),
                Some(vec![bottom; mesh.edges.len()]),
                vec![Underside {
                    rings: rings.clone(),
                    z: bottom,
                }],
                None,
            );
            let tris = volume_triangles(&mesh, &rings, bottom);
            for _ in 0..200 {
                // All receivers below B, thus outside the volume even if within its footprint.
                let o = [
                    32.0 * rng.next() - 8.0,
                    32.0 * rng.next() - 8.0,
                    4.0 * rng.next(),
                ];
                let a = std::f64::consts::TAU * rng.next();
                let sun = Sun {
                    dir: [a.cos(), a.sin()],
                    tan_el: 0.1 + 3.0 * rng.next(),
                };
                let d = [sun.dir[0], sun.dir[1], sun.tan_el];
                let hits: Vec<_> = tris.iter().map(|t| hit(o, d, t, f64::INFINITY)).collect();
                if hits.iter().any(|h| h.1) {
                    continue;
                }
                let blocked = hits.iter().any(|h| h.0);
                assert_eq!(
                    scene.sunlit(o, &sun),
                    !blocked,
                    "shape {shape}, sun at {o:?}"
                );
                // Same independent triangle oracle for finite lines, including descending rays.
                let target = [
                    32.0 * rng.next() - 8.0,
                    32.0 * rng.next() - 8.0,
                    16.0 + 4.0 * rng.next(),
                ];
                let d = vsub(target, o);
                let hits: Vec<_> = tris.iter().map(|t| hit(o, d, t, 1.0)).collect();
                if hits.iter().any(|h| h.1) {
                    continue;
                }
                let clear = !hits.iter().any(|h| h.0);
                assert_eq!(
                    scene.line_clear(o, target),
                    clear,
                    "shape {shape}, line {o:?} -> {target:?}"
                );
                assert_eq!(
                    scene.line_clear(target, o),
                    clear,
                    "shape {shape}, descending line"
                );
                compared += 1;
            }
        }
    }
    assert!(
        compared > 7000,
        "too many grazing rays discarded: {compared}"
    );
}
