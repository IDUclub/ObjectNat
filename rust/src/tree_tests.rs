//! Crown optics and indexed walks against exhaustive opaque/crown intersections.

use super::*;

fn sphere(x: f64, y: f64, z: f64, r: f64) -> Crown {
    Crown {
        x,
        y,
        z,
        r,
        h: r,
        mu_on: -0.2f64.ln() / (2.0 * r),
        mu_off: -0.6f64.ln() / (2.0 * r),
        shape: 0,
    }
}

fn scene(trees: Vec<Crown>) -> Scene {
    Scene::with_details_and_trees(Vec::new(), Vec::new(), None, Vec::new(), None, trees)
}

fn chord_length(tree: &Crown, p: [f64; 3], v: [f64; 3], lo: f64, hi: f64) -> f64 {
    tree.chord(p, v, lo, hi)
        .map_or(0.0, |t| (t[1] - t[0]) * v[0].hypot(v[1]).hypot(v[2]))
}

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 2e-9, "{actual} != {expected}");
}

#[test]
fn analytic_spheroid_chords_and_transmission() {
    let tree = sphere(0.0, 0.0, 5.0, 3.0);
    let v = [1.0, 0.0, 0.0];
    let length = chord_length(&tree, [-10.0, 0.0, 5.0], v, 1e-6, 20.0);
    close(length, 6.0);
    close((-tree.mu_on * length).exp(), 0.2);
    close(chord_length(&tree, [-10.0, 3.0, 5.0], v, 1e-6, 20.0), 0.0); // tangent
    close(chord_length(&tree, [-10.0, 3.1, 5.0], v, 1e-6, 20.0), 0.0); // miss
    let inside = chord_length(&tree, [0.0, 0.0, 5.0], v, 1e-6, 20.0);
    close(inside, 3.0 - 1e-6);
    close(
        (-tree.mu_on * inside).exp(),
        0.2f64.powf((3.0 - 1e-6) / 6.0),
    );
    close(chord_length(&tree, [-10.0, 0.0, 5.0], v, 0.0, 9.0), 2.0); // finite reach
    let oblate = Crown { h: 2.0, ..tree };
    close(
        chord_length(&oblate, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, 20.0),
        4.0,
    );
    for y in [3.0 - 1e-12, 3.0, 3.0 + 1e-12] {
        assert!(chord_length(&tree, [-10.0, y, 5.0], v, 0.0, 20.0).is_finite());
    }
}

#[test]
fn analytic_cone_chords_caps_and_generator() {
    let cone = Crown {
        shape: 1,
        z: 0.0,
        h: 6.0,
        ..sphere(0.0, 0.0, 0.0, 3.0)
    };
    close(
        chord_length(&cone, [-10.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0, 20.0),
        6.0,
    );
    close(
        chord_length(&cone, [-10.0, 0.0, 3.0], [1.0, 0.0, 0.0], 0.0, 20.0),
        3.0,
    );
    close(
        chord_length(&cone, [-10.0, 1.5, 3.0], [1.0, 0.0, 0.0], 0.0, 20.0),
        0.0,
    );
    close(
        chord_length(&cone, [0.0, 0.0, -1.0], [0.0, 0.0, 1.0], 0.0, 20.0),
        6.0,
    );
    close(
        chord_length(&cone, [0.0, 0.0, 3.0], [0.0, 0.0, -1.0], 0.0, 20.0),
        3.0,
    );
    close(
        chord_length(&cone, [-10.0, 0.0, 6.1], [1.0, 0.0, 0.0], 0.0, 20.0),
        0.0,
    );
    // Ray parallel to a generator: the quadratic term vanishes, but the chord is finite.
    close(
        chord_length(&cone, [-2.0, 0.0, 0.0], [0.5, 0.0, 1.0], 0.0, 20.0),
        5.0 * 1.25f64.sqrt(),
    );
}

#[test]
fn trees_multiply_switch_leaves_and_do_not_affect_visibility() {
    let trees = vec![sphere(4.0, 0.0, 4.0, 1.0), sphere(8.0, 0.0, 8.0, 1.0)];
    let scene = scene(trees);
    let sun = Sun {
        dir: [1.0, 0.0],
        tan_el: 1.0,
    };
    close(scene.transmission([0.0; 3], &sun, true), 0.2 * 0.2);
    close(scene.transmission([0.0; 3], &sun, false), 0.6 * 0.6);
    assert!(scene.sunlit([0.0; 3], &sun));
    assert!(scene.line_clear([0.0; 3], [10.0, 0.0, 10.0]));
}

#[test]
fn stats_leaf_state_threshold_and_facing() {
    let tree = sphere(4.0, 0.0, 4.0, 1.0);
    let scene = scene(vec![tree]);
    let hit = Sun {
        dir: [1.0, 0.0],
        tan_el: 1.0,
    };
    let miss = Sun {
        dir: [0.0, 1.0],
        tan_el: 1.0,
    };
    let suns = [hit, hit, miss, hit, hit, hit, miss, hit];
    let (weight, opaque, runs) = day_stats_trees(
        &scene,
        &[[0.0; 3]],
        &suns,
        &[0, 4, 8, 8],
        &[1, 0, 1],
        0.5,
        None,
    );
    close(weight[0], 1.6);
    close(weight[1], 2.8);
    assert_eq!(opaque, vec![4, 4, 0]);
    assert_eq!(runs, vec![1, 4, 0]);
    let (_, opaque, runs) = day_stats_trees(
        &scene,
        &[[0.0; 3]],
        &suns[..4],
        &[0, 4],
        &[0],
        0.5,
        Some((&[[-1.0, 0.0]], 0.0)),
    );
    assert_eq!(opaque, vec![0]);
    assert_eq!(runs, vec![0]);
}

#[test]
fn crown_midpoint_ownership_many_cells_and_mixed_direction_corners() {
    // Distant crowns only establish grid bounds and resolution. The main sphere spans many cells,
    // its chord midpoint is exactly on the centre grid corner, in all four ray directions.
    let mut trees = vec![sphere(0.0, 0.0, 10.0, 4.0)];
    for _ in 0..100 {
        trees.push(sphere(-9.0, -9.0, 1.0, 1.0));
        trees.push(sphere(9.0, 9.0, 1.0, 1.0));
    }
    let scene = scene(trees);
    assert!(scene.grid.cell_count() > 100);
    for dx in [-1.0f64, 1.0] {
        for dy in [-1.0f64, 1.0] {
            let o = [-8.0 * dx, -8.0 * dy];
            let q = [8.0 * dx, 8.0 * dy];
            let (blocked, depth) =
                scene.walk_ray::<false, true, true>(o, q, 10.0, [1.0, 0.0], 0.0, true);
            assert!(!blocked);
            close((-depth).exp(), 0.2);
        }
    }
}

#[test]
fn o7_never_prunes_trees_in_empty_or_low_wall_cells() {
    let tree = sphere(4.0, 0.0, 5.0, 1.0);
    let segs = vec![Seg {
        a: [3.0, -2.0],
        b: [3.0, 2.0],
    }];
    for details in [false, true] {
        let tops = if details {
            vec![[0.5, 1.0]]
        } else {
            vec![[1.0, 1.0]]
        };
        let scene =
            Scene::with_details_and_trees(segs.clone(), tops, None, Vec::new(), None, vec![tree]);
        let sun = Sun {
            dir: [1.0, 0.0],
            tan_el: 1.0,
        };
        let p = [0.0, 0.0, 1.0];
        close(scene.transmission(p, &sun, true), 0.2);
        assert!(scene.sunlit(p, &sun));
    }
}

#[test]
fn opaque_walls_undersides_terrain_and_dense_crown_cutoff() {
    let tree = sphere(4.0, 0.0, 4.0, 1.0);
    let wall = Seg {
        a: [8.0, -2.0],
        b: [8.0, 2.0],
    };
    let wall_scene = Scene::with_details_and_trees(
        vec![wall],
        vec![[12.0, 10.0]],
        Some(vec![1.0]),
        Vec::new(),
        None,
        vec![tree],
    );
    let sun = Sun {
        dir: [1.0, 0.0],
        tan_el: 1.0,
    };
    close(wall_scene.transmission([0.0; 3], &sun, true), 0.0);
    let under = Underside {
        rings: vec![vec![
            [-20.0, -20.0],
            [20.0, -20.0],
            [20.0, 20.0],
            [-20.0, 20.0],
        ]],
        z: 6.0,
    };
    let canopy =
        Scene::with_details_and_trees(Vec::new(), Vec::new(), None, vec![under], None, vec![tree]);
    close(canopy.transmission([0.0; 3], &sun, true), 0.0);
    let terrain = Terrain {
        x0: 0.0,
        y0: -1.0,
        sx: 1.0,
        sy: 1.0,
        nx: 10,
        ny: 3,
        z: vec![10.0; 30],
    };
    let ridge = Scene::with_details_and_trees(
        Vec::new(),
        Vec::new(),
        None,
        Vec::new(),
        Some(terrain),
        vec![tree],
    );
    close(ridge.transmission([0.0; 3], &sun, true), 0.0);
    let dense = scene(vec![tree; 20]);
    close(dense.transmission([0.0; 3], &sun, true), 0.0);
    let (_, opaque, _) = day_stats_trees(&dense, &[[0.0; 3]], &[sun], &[0, 1], &[1], 0.5, None);
    assert_eq!(opaque, vec![1]);
}

fn random(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    (*state >> 11) as f64 / ((1u64 << 53) as f64)
}

#[allow(clippy::too_many_arguments)]
fn brute(
    p: [f64; 3],
    sun: &Sun,
    trees: &[Crown],
    walls: &[Seg],
    tops: &[[f64; 2]],
    bottom: &[f64],
    unders: &[Underside],
    leaf: bool,
) -> f64 {
    let top = trees
        .iter()
        .map(|t| t.z + t.h)
        .chain(tops.iter().flatten().copied())
        .fold(f64::NEG_INFINITY, f64::max);
    let reach = (top - p[2]) / sun.tan_el;
    if reach <= 0.0 {
        return 1.0;
    }
    let o = [p[0], p[1]];
    let q = [p[0] + sun.dir[0] * reach, p[1] + sun.dir[1] * reach];
    let eps = 1e-6 / reach;
    for (k, wall) in walls.iter().enumerate() {
        if let Some((s, u)) = segment_params(o, q, wall) {
            let z = p[2] + s * reach * sun.tan_el;
            if s > eps
                && s <= 1.0
                && (0.0..=1.0).contains(&u)
                && z > bottom[k]
                && z < tops[k][0] + u * (tops[k][1] - tops[k][0])
            {
                return 0.0;
            }
        }
    }
    for poly in unders {
        let s = (poly.z - p[2]) / (reach * sun.tan_el);
        if s > 0.0
            && s <= 1.0
            && poly.contains([o[0] + s * (q[0] - o[0]), o[1] + s * (q[1] - o[1])])
        {
            return 0.0;
        }
    }
    let v = [q[0] - o[0], q[1] - o[1], reach * sun.tan_el];
    let depth: f64 = trees
        .iter()
        .map(|tree| {
            let mu = if leaf { tree.mu_on } else { tree.mu_off };
            mu * chord_length(tree, p, v, eps, 1.0)
        })
        .sum();
    if depth > 13.815_510_557_964_274 {
        0.0
    } else {
        (-depth).exp()
    }
}

#[test]
fn randomized_transmission_and_day_stats_match_exhaustive_primitives() {
    let mut rng = 91u64;
    let mut trees = Vec::new();
    for k in 0..60 {
        let mut tree = sphere(
            40.0 * random(&mut rng) - 20.0,
            40.0 * random(&mut rng) - 20.0,
            3.0 + 15.0 * random(&mut rng),
            1.0 + 5.0 * random(&mut rng),
        );
        tree.shape = (k % 2) as u8;
        tree.h = 2.0 + 6.0 * random(&mut rng);
        trees.push(tree);
    }
    let mut walls = Vec::new();
    let mut tops = Vec::new();
    let mut bottoms = Vec::new();
    for _ in 0..20 {
        let x = 40.0 * random(&mut rng) - 20.0;
        let y = 40.0 * random(&mut rng) - 20.0;
        walls.push(Seg {
            a: [x, y],
            b: [x + 4.0, y + 3.0],
        });
        tops.push([8.0 + 12.0 * random(&mut rng), 8.0 + 12.0 * random(&mut rng)]);
        bottoms.push(3.0 * random(&mut rng));
    }
    let make_unders = || {
        vec![Underside {
            rings: vec![vec![[-8.0, -8.0], [8.0, -8.0], [8.0, 8.0], [-8.0, 8.0]]],
            z: 6.0,
        }]
    };
    let scene = Scene::with_details_and_trees(
        walls.clone(),
        tops.clone(),
        Some(bottoms.clone()),
        make_unders(),
        None,
        trees.clone(),
    );
    let opaque = Scene::with_details(
        walls.clone(),
        tops.clone(),
        Some(bottoms.clone()),
        make_unders(),
        None,
    );
    let mut suns = Vec::new();
    for _ in 0..50 {
        let angle = random(&mut rng) * std::f64::consts::TAU;
        suns.push(Sun {
            dir: [angle.cos(), angle.sin()],
            tan_el: 0.05 + 2.0 * random(&mut rng),
        });
    }
    let mut points = Vec::new();
    for _ in 0..100 {
        points.push([
            60.0 * random(&mut rng) - 30.0,
            60.0 * random(&mut rng) - 30.0,
            5.0 * random(&mut rng),
        ]);
    }
    let offsets = [0, 25, 50];
    let (weighted, baseline, longest) =
        day_stats_trees(&scene, &points, &suns, &offsets, &[1, 0], 0.5, None);
    assert_eq!(
        baseline,
        day_stats(&opaque, &points, &suns, &offsets, None).0
    );
    for (k, &p) in points.iter().enumerate() {
        for day in 0..2 {
            let (mut sum, mut run, mut best) = (0.0, 0u32, 0u32);
            for sun in &suns[offsets[day]..offsets[day + 1]] {
                let expected = brute(
                    p,
                    sun,
                    &trees,
                    &walls,
                    &tops,
                    &bottoms,
                    &make_unders(),
                    day == 0,
                );
                close(scene.transmission(p, sun, day == 0), expected);
                sum += expected;
                run = if expected >= 0.5 { run + 1 } else { 0 };
                best = best.max(run);
            }
            close(weighted[k * 2 + day], sum);
            assert_eq!(longest[k * 2 + day], best);
        }
    }
}
