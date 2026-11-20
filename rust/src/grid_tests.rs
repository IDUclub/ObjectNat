use super::*;
use std::collections::HashSet;

fn rectangle(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<[f64; 2]> {
    vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
}

fn oracle(rings: &[Vec<[f64; 2]>], p: [f64; 2]) -> bool {
    let mut inside = false;
    for ring in rings {
        for i in 0..ring.len() {
            let a = ring[i];
            let b = ring[(i + 1) % ring.len()];
            if (a[1] > p[1]) != (b[1] > p[1])
                && p[0] < a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0])
            {
                inside = !inside;
            }
        }
    }
    inside
}

#[test]
fn grid_boundary_rows_and_holes() {
    let polygons = vec![vec![
        rectangle(0.0, 0.0, 8.0, 8.0),
        rectangle(2.0, 2.0, 6.0, 6.0),
    ]];
    let axis: Vec<_> = (-1..=9).map(f64::from).collect();
    let hits = grid_pairs(&polygons, &axis, &axis);
    let states = area_states(&hits, axis.len() * axis.len());
    for (iy, &y) in axis.iter().enumerate() {
        for (ix, &x) in axis.iter().enumerate() {
            let state = states[iy * axis.len() + ix];
            if state != UNCERTAIN {
                assert_eq!(state == INSIDE, oracle(&polygons[0], [x, y]));
            }
            if (x == 0.0 || x == 8.0 || y == 0.0 || y == 8.0)
                && (0.0..=8.0).contains(&x)
                && (0.0..=8.0).contains(&y)
            {
                assert_eq!(state, UNCERTAIN);
            }
        }
    }
    assert_eq!(states[4 * axis.len() + 4], OUTSIDE); // hole at (3, 3)
    assert_eq!(states[2 * axis.len() + 2], INSIDE); // (1, 1)
}

#[test]
fn translation_and_nearly_horizontal_edges_are_conservative() {
    let x = 1_000_000.0;
    let y = 6_000_000.0;
    let rings = vec![vec![
        [x, y],
        [x + 100.0, y + 1e-6],
        [x + 100.0, y + 10.0],
        [x, y + 10.0],
    ]];
    let polygon = Polygon::new(&rings);
    let mut scratch = Vec::new();
    assert_eq!(polygon.point([x + 50.0, y + 5e-7], &mut scratch), UNCERTAIN);
    assert_eq!(polygon.point([x + 50.0, y + 1.0], &mut scratch), INSIDE);
    assert_eq!(polygon.point([x - 1.0, y + 1.0], &mut scratch), OUTSIDE);
    assert_eq!(
        polygon.point([up(x + 100.0), y + 1.0], &mut scratch),
        UNCERTAIN
    );
}

#[test]
fn tiny_polygons_empty_grids_and_union() {
    let tiny = vec![vec![rectangle(0.25, 0.25, 0.5, 0.5)]];
    let axis = [0.0, 1.0];
    assert!(grid_pairs(&tiny, &axis, &axis).inside.is_empty());
    assert!(grid_pairs(&tiny, &[], &axis).uncertain.is_empty());
    assert!(point_pairs(&[], &[[0.0, 0.0]]).inside.is_empty());
    assert!(point_pairs(&tiny, &[]).uncertain.is_empty());
    let hits = Hits {
        inside: vec![(0, 1)],
        uncertain: vec![(0, 0), (1, 0)],
    };
    assert_eq!(area_states(&hits, 3), [INSIDE, UNCERTAIN, OUTSIDE]);
}

#[test]
fn indexed_points_and_scanlines_match_independent_parity() {
    let mut seed = 42_u64;
    let mut random = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 11) as f64 / (1_u64 << 53) as f64
    };
    let mut polygons = Vec::new();
    for k in 0..12 {
        let outer: Vec<_> = (0..17)
            .map(|i| {
                let angle = i as f64 * std::f64::consts::TAU / 17.0;
                let r = 8.0 + 4.0 * random();
                [k as f64 * 9.0 + r * angle.cos(), r * angle.sin()]
            })
            .collect();
        polygons.push(vec![
            outer,
            rectangle(k as f64 * 9.0 - 1.0, -1.0, k as f64 * 9.0 + 1.0, 1.0),
        ]);
    }
    let points: Vec<_> = (0..2000)
        .map(|_| [random() * 140.0 - 20.0, random() * 40.0 - 20.0])
        .collect();
    let hits = point_pairs(&polygons, &points);
    let inside: HashSet<_> = hits.inside.into_iter().collect();
    let uncertain: HashSet<_> = hits.uncertain.into_iter().collect();
    for (point, &p) in points.iter().enumerate() {
        for (polygon, rings) in polygons.iter().enumerate() {
            if !uncertain.contains(&(point, polygon)) {
                assert_eq!(inside.contains(&(point, polygon)), oracle(rings, p));
            }
        }
    }
    let gx: Vec<_> = (0..220).map(|i| i as f64 * 0.7 - 20.0).collect();
    let gy: Vec<_> = (0..55).map(|i| i as f64 * 0.7 - 20.0).collect();
    let hits = grid_pairs(&polygons, &gx, &gy);
    let inside: HashSet<_> = hits.inside.into_iter().collect();
    let uncertain: HashSet<_> = hits.uncertain.into_iter().collect();
    for (iy, &y) in gy.iter().enumerate() {
        for (ix, &x) in gx.iter().enumerate() {
            let point = iy * gx.len() + ix;
            for (polygon, rings) in polygons.iter().enumerate() {
                if !uncertain.contains(&(point, polygon)) {
                    assert_eq!(inside.contains(&(point, polygon)), oracle(rings, [x, y]));
                }
            }
        }
    }
}

#[test]
fn outward_rounding_encloses_zero_and_negative_values() {
    for x in [-1e100, -1.0, -0.0, 0.0, 1.0, 1e100] {
        assert!(down(x) < x);
        assert!(up(x) > x);
    }
    let a = Interval::shifted(1_000_001.0, 1_000_000.0);
    let b = Interval::shifted(1_000_003.0, 1_000_000.0);
    let c = a.div(b);
    assert!(c.lo < 1.0 / 3.0 && c.hi > 1.0 / 3.0);
}
