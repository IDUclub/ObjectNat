//! Conservative containment pruning for regular point buffers.

struct Disc {
    position: [f64; 2],
    radius: f64,
    inscribed: f64,
}

struct Node {
    lower: [f64; 2],
    upper: [f64; 2],
    largest: usize,
    children: Option<Box<[Node; 2]>>,
}

impl Node {
    fn build(indices: &mut [usize], discs: &[Disc]) -> Self {
        let largest = *indices
            .iter()
            .max_by(|&&a, &&b| discs[a].inscribed.total_cmp(&discs[b].inscribed))
            .unwrap();
        let mut lower = discs[indices[0]].position;
        let mut upper = lower;
        for &index in indices.iter() {
            for axis in 0..2 {
                let coordinate = discs[index].position[axis];
                lower[axis] = lower[axis].min(coordinate);
                upper[axis] = upper[axis].max(coordinate);
            }
        }
        let children = if indices.len() <= 16 {
            None
        } else {
            let axis = usize::from(upper[1] - lower[1] > upper[0] - lower[0]);
            let mid = indices.len() / 2;
            indices.select_nth_unstable_by(mid, |&a, &b| {
                discs[a].position[axis].total_cmp(&discs[b].position[axis])
            });
            let (left, right) = indices.split_at_mut(mid);
            Some(Box::new([
                Self::build(left, discs),
                Self::build(right, discs),
            ]))
        };
        Self {
            lower,
            upper,
            largest,
            children,
        }
    }

    fn contains(&self, disc: &Disc, indices: &[usize], discs: &[Disc]) -> bool {
        let largest = &discs[self.largest];
        let reach = largest.inscribed - disc.radius;
        if reach <= 0.0 {
            return false;
        }
        let [x, y] = disc.position;
        let dx = (self.lower[0] - x).max(x - self.upper[0]).max(0.0);
        let dy = (self.lower[1] - y).max(y - self.upper[1]).max(0.0);
        if dx * dx + dy * dy > reach * reach {
            return false;
        }
        if contains(largest, disc) {
            return true;
        }
        if let Some(children) = &self.children {
            let (left, right) = indices.split_at(indices.len() / 2);
            let [a, b] = children.as_ref();
            let mut branches = [(a, left), (b, right)];
            if discs[b.largest].inscribed > discs[a.largest].inscribed {
                branches.swap(0, 1);
            }
            branches
                .into_iter()
                .any(|(node, indices)| node.contains(disc, indices, discs))
        } else {
            indices.iter().any(|&index| contains(&discs[index], disc))
        }
    }
}

fn contains(outer: &Disc, inner: &Disc) -> bool {
    let reach = outer.inscribed - inner.radius;
    let dx = outer.position[0] - inner.position[0];
    let dy = outer.position[1] - inner.position[1];
    reach > 0.0 && reach.is_finite() && dx * dx + dy * dy < reach * reach
}

pub fn keep_buffers(points: Vec<[f64; 2]>, radii: Vec<f64>, quad_segs: usize) -> Vec<bool> {
    let cosine = (std::f64::consts::PI / (4.0 * quad_segs as f64)).cos();
    let discs: Vec<_> = points
        .into_iter()
        .zip(radii)
        .map(|(position, radius)| {
            // Allow for rounding of GEOS vertices at the scale of their coordinates.
            let margin = 32.0 * f64::EPSILON * position[0].abs().max(position[1].abs()).max(radius);
            Disc {
                position,
                radius,
                inscribed: radius * cosine - margin,
            }
        })
        .collect();
    let mut keep: Vec<_> = discs.iter().map(|disc| disc.radius > 0.0).collect();
    let mut indices: Vec<_> = discs
        .iter()
        .enumerate()
        .filter(|(_, disc)| {
            disc.radius > 0.0
                && disc.radius.is_finite()
                && disc.position.iter().all(|x| x.is_finite())
        })
        .map(|(index, _)| index)
        .collect();
    if indices.is_empty() {
        return keep;
    }
    let min_radius = indices
        .iter()
        .map(|&index| discs[index].radius)
        .fold(f64::INFINITY, f64::min);
    let max_inscribed = indices
        .iter()
        .map(|&index| discs[index].inscribed)
        .fold(0.0, f64::max);
    if min_radius >= max_inscribed {
        return keep;
    }
    let root = Node::build(&mut indices, &discs);
    for &index in &indices {
        keep[index] = !root.contains(&discs[index], &indices, &discs);
    }
    keep
}
