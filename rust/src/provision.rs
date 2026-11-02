//! Gravity-probabilistic service provision on a sparse OD matrix.
//!
//! Same model as ObjectNat 2.1 (`provision_calculating.py`), rewritten over sparse entries:
//!
//! * candidate links are building-service pairs with a finite cost `d <= 3 * threshold`; the model
//!   works with `a = d + 1`;
//! * each iteration every active service spreads its remaining capacity over the active buildings with
//!   `a <= selection_range`, keeping those whose gravity weight `1 / a^2` is at or above the
//!   `best_houses` quantile, by `capacity_left` weighted draws;
//! * each building then accepts at most its remaining demand: `demand_left` draws over the offers,
//!   weighted by the offers, capped by each offer;
//! * exhausted buildings/services and those left without candidate links drop out, `selection_range`
//!   grows by 1.5 and `best_houses` shrinks with the number of active objects.
//!
//! Randomness is a deterministic stream per (seed, iteration, object), so results do not depend on
//! the number of threads. Services and buildings are processed in parallel within an iteration.

use rayon::prelude::*;

const MAX_ITERATIONS: usize = 10_000;

/// splitmix64: seeds the generator and mixes stream ids.
#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// xoshiro256** generator.
struct Rng([u64; 4]);

impl Rng {
    fn new(seed: u64, iteration: u64, kind: u64, id: u64) -> Rng {
        let mut s =
            splitmix64(seed) ^ splitmix64(iteration.wrapping_mul(2).wrapping_add(kind)) ^ id;
        let mut state = [0u64; 4];
        for v in state.iter_mut() {
            s = splitmix64(s);
            *v = s;
        }
        Rng(state)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform float in [0, 1).
    #[inline]
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// `draws` weighted draws with replacement over `weights` (like `numpy.random.choice(p=...)`):
/// returns how many times each item was drawn.
fn weighted_counts(weights: &[f64], draws: i64, rng: &mut Rng) -> Vec<i64> {
    let mut counts = vec![0i64; weights.len()];
    if draws <= 0 || weights.is_empty() {
        return counts;
    }
    let mut cdf = Vec::with_capacity(weights.len());
    let mut acc = 0.0;
    for &w in weights {
        acc += w;
        cdf.push(acc);
    }
    if acc.is_nan() || acc <= 0.0 || !acc.is_finite() {
        return counts;
    }
    for _ in 0..draws {
        let u = rng.uniform() * acc;
        let k = cdf.partition_point(|&c| c <= u).min(weights.len() - 1);
        counts[k] += 1;
    }
    counts
}

/// Linear-interpolation quantile of an ascending slice (pandas' default).
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// Compressed adjacency: for each owner, its (other, cost) pairs.
struct Csr {
    start: Vec<usize>,
    other: Vec<u32>,
    cost: Vec<f64>,
}

impl Csr {
    fn build(n: usize, owner: &[u32], other: &[u32], cost: &[f64]) -> Csr {
        let mut start = vec![0usize; n + 1];
        for &o in owner {
            start[o as usize + 1] += 1;
        }
        for k in 0..n {
            start[k + 1] += start[k];
        }
        let mut fill = start.clone();
        let mut out_other = vec![0u32; owner.len()];
        let mut out_cost = vec![0f64; owner.len()];
        for e in 0..owner.len() {
            let o = owner[e] as usize;
            out_other[fill[o]] = other[e];
            out_cost[fill[o]] = cost[e];
            fill[o] += 1;
        }
        Csr {
            start,
            other: out_other,
            cost: out_cost,
        }
    }

    #[inline]
    fn row(&self, k: usize) -> (&[u32], &[f64]) {
        let (a, b) = (self.start[k], self.start[k + 1]);
        (&self.other[a..b], &self.cost[a..b])
    }
}

/// Inputs of the allocation; `rows`/`cols` index buildings/services, `cost` is the OD cost.
pub struct ProvisionInput<'a> {
    pub rows: &'a [u32],
    pub cols: &'a [u32],
    pub cost: &'a [f64],
    pub demand: &'a [i64],
    pub capacity: &'a [i64],
    pub threshold: f64,
    pub seed: u64,
}

/// Allocated flows as (building, service, units) triples; a pair may appear more than once
/// (once per iteration), callers sum duplicates.
pub fn allocate(input: &ProvisionInput) -> (Vec<u32>, Vec<u32>, Vec<i64>) {
    let nb = input.demand.len();
    let ns = input.capacity.len();
    let limit = input.threshold * 3.0;

    // candidate links (finite cost within 3 * threshold), as allocation cost a = d + 1
    let mut eb = Vec::new();
    let mut es = Vec::new();
    let mut ea = Vec::new();
    for k in 0..input.cost.len() {
        let d = input.cost[k];
        let (b, s) = (input.rows[k] as usize, input.cols[k] as usize);
        if d.is_finite() && d <= limit && b < nb && s < ns {
            eb.push(input.rows[k]);
            es.push(input.cols[k]);
            ea.push(d + 1.0);
        }
    }
    let by_service = Csr::build(ns, &es, &eb, &ea);
    let by_building = Csr::build(nb, &eb, &es, &ea);

    let mut demand_left: Vec<i64> = input.demand.to_vec();
    let mut capacity_left: Vec<i64> = input.capacity.to_vec();
    let mut active_b: Vec<bool> = demand_left.iter().map(|&d| d > 0).collect();
    let mut active_s: Vec<bool> = capacity_left.iter().map(|&c| c > 0).collect();
    drop_inactive(
        &by_building,
        &by_service,
        &demand_left,
        &capacity_left,
        &mut active_b,
        &mut active_s,
    );

    let mut selection_range = (input.threshold + 1.0) / 2.0;
    let mut best_houses = 0.9f64;
    let (mut out_b, mut out_s, mut out_v) = (Vec::new(), Vec::new(), Vec::new());

    for iteration in 0..MAX_ITERATIONS {
        let objects_n =
            active_b.iter().filter(|&&a| a).count() + active_s.iter().filter(|&&a| a).count();
        if !active_b.iter().any(|&a| a) || !active_s.iter().any(|&a| a) {
            break;
        }

        // 1. services spread capacity over nearby buildings
        let offers: Vec<(u32, u32, i64)> = (0..ns)
            .into_par_iter()
            .filter(|&s| active_s[s] && capacity_left[s] > 0)
            .flat_map_iter(|s| {
                let (blds, costs) = by_service.row(s);
                let mut cand_b = Vec::new();
                let mut cand_p = Vec::new();
                for (&b, &a) in blds.iter().zip(costs) {
                    if active_b[b as usize] && a <= selection_range {
                        cand_b.push(b);
                        cand_p.push(1.0 / (a * a));
                    }
                }
                let mut picked = Vec::new();
                if !cand_b.is_empty() {
                    let total: f64 = cand_p.iter().sum();
                    let probs: Vec<f64> = cand_p.iter().map(|p| p / total).collect();
                    let mut sorted = probs.clone();
                    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
                    let cut = quantile_sorted(&sorted, best_houses);
                    let (keep_b, keep_p): (Vec<u32>, Vec<f64>) = cand_b
                        .iter()
                        .zip(&probs)
                        .filter(|(_, &p)| p >= cut)
                        .map(|(&b, &p)| (b, p))
                        .unzip();
                    let mut rng = Rng::new(input.seed, iteration as u64, 0, s as u64);
                    let counts = weighted_counts(&keep_p, capacity_left[s], &mut rng);
                    for (b, c) in keep_b.into_iter().zip(counts) {
                        if c > 0 {
                            picked.push((b, s as u32, c));
                        }
                    }
                }
                picked.into_iter()
            })
            .collect();

        // 2. buildings accept at most their remaining demand
        let mut offers = offers;
        offers.par_sort_unstable_by_key(|&(b, s, _)| (b, s));
        let mut groups: Vec<(usize, usize)> = Vec::new();
        let mut k = 0;
        while k < offers.len() {
            let mut j = k + 1;
            while j < offers.len() && offers[j].0 == offers[k].0 {
                j += 1;
            }
            groups.push((k, j));
            k = j;
        }
        let accepted: Vec<(u32, u32, i64)> = groups
            .par_iter()
            .flat_map_iter(|&(a, z)| {
                let group = &offers[a..z];
                let b = group[0].0 as usize;
                let weights: Vec<f64> = group.iter().map(|o| o.2 as f64).collect();
                let mut rng = Rng::new(input.seed, iteration as u64, 1, b as u64);
                let counts = weighted_counts(&weights, demand_left[b], &mut rng);
                group
                    .iter()
                    .zip(counts)
                    .map(|(o, c)| (o.0, o.1, o.2.min(c)))
                    .filter(|o| o.2 > 0)
                    .collect::<Vec<_>>()
                    .into_iter()
            })
            .collect();

        for &(b, s, v) in &accepted {
            demand_left[b as usize] -= v;
            capacity_left[s as usize] -= v;
            out_b.push(b);
            out_s.push(s);
            out_v.push(v);
        }

        drop_inactive(
            &by_building,
            &by_service,
            &demand_left,
            &capacity_left,
            &mut active_b,
            &mut active_s,
        );
        selection_range *= 1.5;
        if best_houses <= 0.1 {
            best_houses = 0.0;
        } else {
            let objects_n_new =
                active_b.iter().filter(|&&a| a).count() + active_s.iter().filter(|&&a| a).count();
            best_houses = objects_n_new as f64 / (objects_n as f64 / best_houses);
        }
    }
    (out_b, out_s, out_v)
}

/// Drops exhausted objects, then buildings without a reachable active service, then services
/// without a reachable active building (the order of the reference implementation).
fn drop_inactive(
    by_building: &Csr,
    by_service: &Csr,
    demand_left: &[i64],
    capacity_left: &[i64],
    active_b: &mut [bool],
    active_s: &mut [bool],
) {
    for (a, &d) in active_b.iter_mut().zip(demand_left) {
        *a = *a && d > 0;
    }
    for (a, &c) in active_s.iter_mut().zip(capacity_left) {
        *a = *a && c > 0;
    }
    for (b, active) in active_b.iter_mut().enumerate() {
        if *active {
            let (svcs, _) = by_building.row(b);
            *active = svcs.iter().any(|&s| active_s[s as usize]);
        }
    }
    for (s, active) in active_s.iter_mut().enumerate() {
        if *active {
            let (blds, _) = by_service.row(s);
            *active = blds.iter().any(|&b| active_b[b as usize]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(
        rows: &[u32],
        cols: &[u32],
        cost: &[f64],
        demand: &[i64],
        capacity: &[i64],
    ) -> Vec<(u32, u32, i64)> {
        let (b, s, v) = allocate(&ProvisionInput {
            rows,
            cols,
            cost,
            demand,
            capacity,
            threshold: 10.0,
            seed: 0,
        });
        let mut out: Vec<_> = b
            .into_iter()
            .zip(s)
            .zip(v)
            .map(|((b, s), v)| (b, s, v))
            .collect();
        out.sort();
        out
    }

    fn totals(flows: &[(u32, u32, i64)], nb: usize, ns: usize) -> (Vec<i64>, Vec<i64>) {
        let mut by_b = vec![0; nb];
        let mut by_s = vec![0; ns];
        for &(b, s, v) in flows {
            by_b[b as usize] += v;
            by_s[s as usize] += v;
        }
        (by_b, by_s)
    }

    #[test]
    fn never_exceeds_demand_or_capacity() {
        let rows = [0, 0, 1, 1, 2, 2];
        let cols = [0, 1, 0, 1, 0, 1];
        let cost = [1.0, 5.0, 2.0, 2.0, 8.0, 3.0];
        let demand = [30, 20, 25];
        let capacity = [40, 10];
        let flows = run(&rows, &cols, &cost, &demand, &capacity);
        let (by_b, by_s) = totals(&flows, 3, 2);
        assert!(by_b.iter().zip(&demand).all(|(f, d)| f <= d));
        assert!(by_s.iter().zip(&capacity).all(|(f, c)| f <= c));
        // capacity is scarce: all of it is used
        assert_eq!(by_s.iter().sum::<i64>(), 50);
    }

    #[test]
    fn demand_is_met_when_capacity_is_plenty() {
        let flows = run(&[0, 1], &[0, 0], &[2.0, 4.0], &[7, 3], &[100]);
        let (by_b, _) = totals(&flows, 2, 1);
        assert_eq!(by_b, vec![7, 3]);
    }

    #[test]
    fn links_beyond_three_thresholds_are_ignored() {
        let flows = run(&[0, 1], &[0, 0], &[2.0, 31.0], &[5, 5], &[100]);
        let (by_b, _) = totals(&flows, 2, 1);
        assert_eq!(by_b, vec![5, 0]);
    }

    #[test]
    fn same_seed_same_result() {
        let rows = [0, 0, 1, 1, 2, 2, 3];
        let cols = [0, 1, 0, 1, 0, 1, 1];
        let cost = [1.0, 5.0, 2.0, 2.0, 8.0, 3.0, 4.0];
        let a = run(&rows, &cols, &cost, &[30, 20, 25, 9], &[40, 10]);
        let b = run(&rows, &cols, &cost, &[30, 20, 25, 9], &[40, 10]);
        assert_eq!(a, b);
    }

    #[test]
    fn quantile_matches_pandas_linear_interpolation() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert!((quantile_sorted(&v, 0.9) - 3.7).abs() < 1e-12);
        assert_eq!(quantile_sorted(&v, 0.0), 1.0);
    }
}
