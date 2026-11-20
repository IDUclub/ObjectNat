//! Direct sunlight and lines of sight among buildings and terrain (2.5D scene).
//!
//! Walls carry linearly varying roof tops and optional flat bottoms. Raised volumes also have
//! polygonal undersides. Terrain is an optional height grid. A receiver at height `z` sees the sun when the
//! ray towards it — horizontal direction `dir`, rising `tan_el` metres per metre — passes above every
//! wall it crosses and above the terrain. The ray only needs to be followed while it is still below the
//! highest point of the scene, so high sun means short rays.

use rayon::prelude::*;

use crate::visibility::{segment_params, Grid, Seg};

/// Height grid: `z[j * nx + i]` is the ground height at `(x0 + i * sx, y0 + j * sy)`; NaN is no data.
pub struct Terrain {
    pub x0: f64,
    pub y0: f64,
    pub sx: f64,
    pub sy: f64,
    pub nx: usize,
    pub ny: usize,
    pub z: Vec<f64>,
}

impl Terrain {
    /// Bilinear ground height at `(x, y)`: `None` outside the grid, NaN next to missing data.
    pub fn at(&self, x: f64, y: f64) -> Option<f64> {
        let fx = (x - self.x0) / self.sx;
        let fy = (y - self.y0) / self.sy;
        let (mx, my) = ((self.nx - 1) as f64, (self.ny - 1) as f64);
        if !(0.0..=mx).contains(&fx) || !(0.0..=my).contains(&fy) {
            return None;
        }
        let i = (fx.floor() as usize).min(self.nx.saturating_sub(2));
        let j = (fy.floor() as usize).min(self.ny.saturating_sub(2));
        let (i1, j1) = ((i + 1).min(self.nx - 1), (j + 1).min(self.ny - 1));
        let (tx, ty) = (fx - i as f64, fy - j as f64);
        let z = |a: usize, b: usize| self.z[b * self.nx + a];
        let v = (1.0 - tx) * (1.0 - ty) * z(i, j)
            + tx * (1.0 - ty) * z(i1, j)
            + (1.0 - tx) * ty * z(i, j1)
            + tx * ty * z(i1, j1);
        Some(v)
    }

    fn max(&self) -> f64 {
        self.z
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// One sun position: unit horizontal direction towards the sun in the scene's coordinates and the
/// tangent of its elevation.
#[derive(Clone, Copy, Debug)]
pub struct Sun {
    pub dir: [f64; 2],
    pub tan_el: f64,
}

/// Walls with roof heights and optional terrain, indexed for ray walks.
pub struct Scene {
    segs: Vec<Seg>,
    tops: Vec<f64>,
    // Construction arrays are released by pack(); details remains a dispatch marker.
    details: Option<(Vec<f64>, Vec<f64>)>,
    undersides: Option<Undersides>,
    grid: Grid,
    terrain: Option<Terrain>,
    z_max: f64,
    wall_starts: Vec<usize>,
    cell_top: Vec<f64>,
    flat: Vec<FlatWall>,
    detailed: Vec<DetailedWall>,
    trees: Option<Trees>,
}

/// A spheroid (shape 0, centre z, vertical semi-axis h) or an upright cone (shape 1,
/// base z, full height h). Extinction is per metre of the three-dimensional chord.
#[derive(Clone, Copy, Debug)]
pub struct Crown {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub r: f64,
    pub h: f64,
    pub mu_on: f64,
    pub mu_off: f64,
    pub shape: u8,
}

impl Crown {
    fn bbox(&self) -> Seg {
        Seg {
            a: [self.x - self.r, self.y - self.r],
            b: [self.x + self.r, self.y + self.r],
        }
    }

    /// Intersection parameters on p + t*v, clipped to (lo, hi]. Tangencies have zero length.
    fn chord(&self, p: [f64; 3], v: [f64; 3], mut lo: f64, mut hi: f64) -> Option<[f64; 2]> {
        let x = (p[0] - self.x) / self.r;
        let y = (p[1] - self.y) / self.r;
        let z = (p[2] - self.z) / self.h;
        let dx = v[0] / self.r;
        let dy = v[1] / self.r;
        let dz = v[2] / self.h;
        let (a, b, c) = if self.shape == 0 {
            (
                dx * dx + dy * dy + dz * dz,
                x * dx + y * dy + z * dz,
                x * x + y * y + z * z - 1.0,
            )
        } else {
            if dz == 0.0 {
                if !(0.0..=1.0).contains(&z) {
                    return None;
                }
            } else {
                let t0 = -z / dz;
                let t1 = (1.0 - z) / dz;
                lo = lo.max(t0.min(t1));
                hi = hi.min(t0.max(t1));
            }
            (
                dx * dx + dy * dy - dz * dz,
                x * dx + y * dy + (1.0 - z) * dz,
                x * x + y * y - (1.0 - z).powi(2),
            )
        };
        if hi <= lo {
            return None;
        }
        // The cone can have a negative or zero quadratic coefficient along a generator.
        // Clipping to its vertical slab first leaves a single convex intersection interval.
        let scale = dx * dx + dy * dy + dz * dz;
        if a.abs() <= 8.0 * f64::EPSILON * scale {
            if b == 0.0 {
                return (c < 0.0).then_some([lo, hi]);
            }
            let root = -c / (2.0 * b);
            if b > 0.0 {
                hi = hi.min(root);
            } else {
                lo = lo.max(root);
            }
        } else {
            let discriminant = b * b - a * c;
            // Roundoff in b²-a*c must not turn an exact tangent into a short absorbing chord.
            let tangent_tolerance = 8.0 * f64::EPSILON * (b * b + (a * c).abs());
            if discriminant <= tangent_tolerance {
                return (a < 0.0).then_some([lo, hi]);
            }
            // Stable roots avoid cancellation for receivers near a crown surface.
            let q = -b - discriminant.sqrt().copysign(b);
            let t0 = q / a;
            let t1 = c / q;
            let (first, last) = (t0.min(t1), t0.max(t1));
            if a > 0.0 {
                lo = lo.max(first);
                hi = hi.min(last);
            } else if lo < first {
                hi = hi.min(first);
            } else {
                lo = lo.max(last);
            }
        }
        (hi > lo).then_some([lo, hi])
    }
}

struct Trees {
    crowns: Vec<Crown>,
    starts: Vec<usize>,
    z_max: f64,
}

impl Trees {
    fn new(crowns: Vec<Crown>) -> Self {
        let z_max = crowns
            .iter()
            .map(|t| t.z + t.h)
            .fold(f64::NEG_INFINITY, f64::max);
        Self {
            crowns,
            starts: Vec::new(),
            z_max,
        }
    }

    fn pack(&mut self, grid: &Grid) {
        self.starts = vec![0; grid.cell_count() + 1];
        for tree in &self.crowns {
            grid.box_cells(&tree.bbox(), |c| self.starts[c + 1] += 1);
        }
        for c in 0..grid.cell_count() {
            self.starts[c + 1] += self.starts[c];
        }
        let mut packed = vec![self.crowns[0]; *self.starts.last().unwrap()];
        let mut fill = self.starts.clone();
        for tree in &self.crowns {
            grid.box_cells(&tree.bbox(), |c| {
                packed[fill[c]] = *tree;
                fill[c] += 1;
            });
        }
        self.crowns = packed;
    }
}

// Records are duplicated in bbox cell order; the ray loop never gathers wall data by id.
struct FlatWall {
    seg: Seg,
    top: f64,
}

struct DetailedWall {
    seg: Seg,
    top: f64,
    top_b: f64,
    bottom: f64,
}

/// Rings use even-odd filling; orientation and an optional repeated closing vertex do not matter.
pub struct Underside {
    pub rings: Vec<Vec<[f64; 2]>>,
    pub z: f64,
}

impl Underside {
    pub fn contains(&self, p: [f64; 2]) -> bool {
        let mut inside = false;
        for ring in &self.rings {
            for k in 0..ring.len() {
                let (a, b) = (ring[k], ring[(k + 1) % ring.len()]);
                if (a[1] > p[1]) != (b[1] > p[1])
                    && p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
                {
                    inside = !inside;
                }
            }
        }
        inside
    }
}

struct Undersides {
    polygons: Vec<Underside>,
    boxes: Vec<Seg>,
    starts: Vec<usize>,
    ids: Vec<u32>,
}

fn wall_grid(segs: &[Seg], boxes: &[Seg]) -> Grid {
    let mut bbox = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for s in segs.iter().chain(boxes) {
        bbox[0] = bbox[0].min(s.a[0].min(s.b[0]));
        bbox[1] = bbox[1].min(s.a[1].min(s.b[1]));
        bbox[2] = bbox[2].max(s.a[0].max(s.b[0]));
        bbox[3] = bbox[3].max(s.a[1].max(s.b[1]));
    }
    if segs.is_empty() && boxes.is_empty() {
        bbox = [0.0, 0.0, 1.0, 1.0];
    }
    let cells = (4.0 * segs.len() as f64).sqrt().clamp(1.0, 4096.0) as usize;
    Grid::build(segs, bbox, cells)
}

impl Undersides {
    fn new(polygons: Vec<Underside>) -> Self {
        let boxes: Vec<Seg> = polygons
            .iter()
            .map(|p| {
                let mut a = [f64::INFINITY; 2];
                let mut b = [f64::NEG_INFINITY; 2];
                for v in p.rings.iter().flatten() {
                    for k in 0..2 {
                        a[k] = a[k].min(v[k]);
                        b[k] = b[k].max(v[k]);
                    }
                }
                Seg { a, b }
            })
            .collect();

        Self {
            polygons,
            boxes,
            starts: Vec::new(),
            ids: Vec::new(),
        }
    }

    fn blocked(&self, o: [f64; 2], q: [f64; 2], z: f64, rise: f64, lo: f64, hi: f64) -> bool {
        if rise == 0.0 {
            return false;
        }
        self.polygons
            .iter()
            .zip(&self.boxes)
            .any(|(poly, b)| Self::crossing(poly, b, o, q, z, rise, [lo, hi], None))
    }

    #[allow(clippy::too_many_arguments)]
    fn crossing(
        poly: &Underside,
        b: &Seg,
        o: [f64; 2],
        q: [f64; 2],
        z: f64,
        rise: f64,
        limits: [f64; 2],
        cell: Option<[f64; 4]>,
    ) -> bool {
        let s = (poly.z - z) / rise;
        if s <= limits[0] || s >= limits[1] {
            return false;
        }
        let p = [o[0] + s * (q[0] - o[0]), o[1] + s * (q[1] - o[1])];
        if cell.is_some_and(|c| p[0] < c[0] || p[1] < c[1] || p[0] > c[2] || p[1] > c[3]) {
            return false;
        }
        p[0] >= b.a[0] && p[0] <= b.b[0] && p[1] >= b.a[1] && p[1] <= b.b[1] && poly.contains(p)
    }
}

impl Scene {
    pub fn new(segs: Vec<Seg>, tops: Vec<f64>, terrain: Option<Terrain>) -> Scene {
        let mut scene = Self::unpacked(segs, tops, terrain);
        scene.pack();
        scene
    }

    fn unpacked(segs: Vec<Seg>, tops: Vec<f64>, terrain: Option<Terrain>) -> Scene {
        let (segs, tops): (Vec<Seg>, Vec<f64>) = segs
            .into_iter()
            .zip(tops)
            .filter(|(s, top)| {
                s.a != s.b && top.is_finite() && s.a.iter().chain(s.b.iter()).all(|v| v.is_finite())
            })
            .unzip();
        let grid = Grid::build(&[], [0.0, 0.0, 1.0, 1.0], 1);
        let z_max = tops
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
            .max(terrain.as_ref().map_or(f64::NEG_INFINITY, Terrain::max));
        Scene {
            segs,
            tops,
            details: None,
            undersides: None,
            grid,
            terrain,
            z_max,
            wall_starts: Vec::new(),
            cell_top: Vec::new(),
            flat: Vec::new(),
            detailed: Vec::new(),
            trees: None,
        }
    }

    pub fn with_details(
        segs: Vec<Seg>,
        tops: Vec<[f64; 2]>,
        bottoms: Option<Vec<f64>>,
        undersides: Vec<Underside>,
        terrain: Option<Terrain>,
    ) -> Scene {
        Self::with_details_and_trees(segs, tops, bottoms, undersides, terrain, Vec::new())
    }

    pub fn with_details_and_trees(
        segs: Vec<Seg>,
        tops: Vec<[f64; 2]>,
        bottoms: Option<Vec<f64>>,
        undersides: Vec<Underside>,
        terrain: Option<Terrain>,
        trees: Vec<Crown>,
    ) -> Scene {
        let mut walls = Vec::new();
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut low = Vec::new();
        for (k, (s, top)) in segs.into_iter().zip(tops).enumerate() {
            let bottom = bottoms.as_ref().map_or(f64::NEG_INFINITY, |v| v[k]);
            if s.a != s.b
                && top.iter().all(|v| v.is_finite())
                && s.a.iter().chain(s.b.iter()).all(|v| v.is_finite())
                && !bottom.is_nan()
            {
                walls.push(s);
                a.push(top[0]);
                b.push(top[1]);
                low.push(bottom);
            }
        }
        let mut scene = Self::unpacked(walls, a, terrain);
        scene.z_max = b.iter().copied().fold(scene.z_max, f64::max);
        if b != scene.tops || low.iter().any(|&z| z != f64::NEG_INFINITY) {
            scene.details = Some((b, low));
        }
        if !undersides.is_empty() {
            scene.z_max = undersides.iter().map(|p| p.z).fold(scene.z_max, f64::max);
            scene.undersides = Some(Undersides::new(undersides));
        }
        if !trees.is_empty() {
            scene.trees = Some(Trees::new(trees));
        }
        scene.pack();
        scene
    }

    fn pack(&mut self) {
        let boxes = self
            .undersides
            .as_ref()
            .map_or(&[][..], |u| u.boxes.as_slice());
        if let Some(trees) = &self.trees {
            let mut all_boxes = boxes.to_vec();
            all_boxes.extend(trees.crowns.iter().map(Crown::bbox));
            // Tree-only scenes also need a useful grid resolution, rather than one giant cell.
            let mut bbox = [
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ];
            for b in self.segs.iter().chain(&all_boxes) {
                bbox[0] = bbox[0].min(b.a[0].min(b.b[0]));
                bbox[1] = bbox[1].min(b.a[1].min(b.b[1]));
                bbox[2] = bbox[2].max(b.a[0].max(b.b[0]));
                bbox[3] = bbox[3].max(b.a[1].max(b.b[1]));
            }
            let cells = (4.0 * (self.segs.len() + trees.crowns.len()) as f64)
                .sqrt()
                .clamp(1.0, 4096.0) as usize;
            self.grid = Grid::build(&self.segs, bbox, cells);
        } else {
            self.grid = wall_grid(&self.segs, boxes);
        }
        self.wall_starts = Vec::with_capacity(self.grid.cell_count() + 1);
        self.wall_starts.push(0);
        self.cell_top = vec![f64::NEG_INFINITY; self.grid.cell_count()];
        for c in 0..self.grid.cell_count() {
            for &id in self.grid.ids(c) {
                let k = id as usize;
                let top_b = self.details.as_ref().map_or(self.tops[k], |(b, _)| b[k]);
                self.cell_top[c] = self.cell_top[c].max(self.tops[k].max(top_b));
                if let Some((b, bottom)) = &self.details {
                    self.detailed.push(DetailedWall {
                        seg: self.segs[k],
                        top: self.tops[k],
                        top_b: b[k],
                        bottom: bottom[k],
                    });
                } else {
                    self.flat.push(FlatWall {
                        seg: self.segs[k],
                        top: self.tops[k],
                    });
                }
            }
            self.wall_starts.push(if self.details.is_some() {
                self.detailed.len()
            } else {
                self.flat.len()
            });
        }
        if let Some(u) = &mut self.undersides {
            u.starts = vec![0; self.grid.cell_count() + 1];
            for b in &u.boxes {
                self.grid.box_cells(b, |c| u.starts[c + 1] += 1);
            }
            for c in 0..self.grid.cell_count() {
                u.starts[c + 1] += u.starts[c];
            }
            u.ids = vec![0; *u.starts.last().unwrap()];
            let mut fill = u.starts.clone();
            for (id, b) in u.boxes.iter().enumerate() {
                self.grid.box_cells(b, |c| {
                    self.cell_top[c] = self.cell_top[c].max(u.polygons[id].z);
                    u.ids[fill[c]] = id as u32;
                    fill[c] += 1;
                });
            }
        }
        if let Some(trees) = &mut self.trees {
            trees.pack(&self.grid);
        }
        self.grid.clear_ids();
        self.segs = Vec::new();
        self.tops = Vec::new();
        if self.details.is_some() {
            self.details = Some((Vec::new(), Vec::new()));
        }
    }

    // Compile a separate original flat-top loop, with no interpolation or bottom loads.
    #[inline(always)]
    fn wall_blocked<const DETAILS: bool, const SUN: bool>(
        &self,
        o: [f64; 2],
        q: [f64; 2],
        z: f64,
        rise: [f64; 2],
        eps: f64,
    ) -> bool {
        self.walk_ray::<DETAILS, SUN, false>(o, q, z, rise, eps, false)
            .0
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_ray<const DETAILS: bool, const SUN: bool, const TREES: bool>(
        &self,
        o: [f64; 2],
        q: [f64; 2],
        z: f64,
        rise: [f64; 2],
        eps: f64,
        leaf_on: bool,
    ) -> (bool, f64) {
        let mut blocked = false;
        let mut depth = 0.0;
        let mut visit = |c: usize, s_enter: f64, s_exit: f64| {
            if TREES {
                // This MUST precede O7 wall-height pruning. A low wall's cell may contain a tall crown.
                if let Some(trees) = &self.trees {
                    let v = [q[0] - o[0], q[1] - o[1], rise[0] * rise[1]];
                    for tree in &trees.crowns[trees.starts[c]..trees.starts[c + 1]] {
                        if let Some([lo, hi]) = tree.chord([o[0], o[1], z], v, eps, 1.0) {
                            let mid = (lo + hi) * 0.5;
                            let owner = self.grid.cell_at([o[0] + mid * v[0], o[1] + mid * v[1]]);
                            if owner == c {
                                let mu = if leaf_on { tree.mu_on } else { tree.mu_off };
                                depth += mu * (hi - lo) * v[0].hypot(v[1]).hypot(v[2]);
                                if depth > 13.815_510_557_964_274 {
                                    return true; // exp(-depth) < 1e-6
                                }
                            }
                        }
                    }
                }
            }
            // Preserve the arithmetic order of the wall intersection test.
            let height_at = |s| {
                if SUN {
                    z + s * rise[0] * rise[1]
                } else {
                    z + s * rise[0]
                }
            };
            let entry_height = height_at(s_enter);
            let can_skip = if SUN {
                // Undersides can cross exactly at cell entry, including grid entry.
                let has_undersides = self
                    .undersides
                    .as_ref()
                    .is_some_and(|u| u.starts[c] != u.starts[c + 1]);
                self.cell_top[c] <= entry_height
                    && (!has_undersides || self.cell_top[c] < entry_height)
            } else {
                self.cell_top[c] < entry_height.min(height_at(s_exit)) - 1e-9
            };
            if s_enter.is_finite() && s_exit.is_finite() && can_skip {
                return false;
            }
            let range = self.wall_starts[c]..self.wall_starts[c + 1];
            for k in range {
                let seg = if DETAILS {
                    &self.detailed[k].seg
                } else {
                    &self.flat[k].seg
                };
                if let Some((s, u)) = segment_params(o, q, seg) {
                    if s > eps
                        && (if SUN { s <= 1.0 } else { s < 1.0 - eps })
                        && (0.0..=1.0).contains(&u)
                    {
                        let height = if SUN {
                            z + s * rise[0] * rise[1]
                        } else {
                            z + s * rise[0]
                        };
                        let mut top = if DETAILS {
                            self.detailed[k].top
                        } else {
                            self.flat[k].top
                        };
                        if DETAILS {
                            let wall = &self.detailed[k];
                            if height <= wall.bottom {
                                continue;
                            }
                            top += u * (wall.top_b - top);
                        }
                        if top > height + if SUN { 0.0 } else { 1e-9 } {
                            blocked = true;
                            return true;
                        }
                    }
                }
            }
            if let Some(undersides) = &self.undersides {
                let rise = if SUN { rise[0] * rise[1] } else { rise[0] };
                if rise != 0.0 {
                    let limits = if SUN {
                        [0.0, 1.0 + f64::EPSILON]
                    } else {
                        [eps, 1.0 - eps]
                    };
                    let cell = self.grid.bounds(c);
                    for &id in &undersides.ids[undersides.starts[c]..undersides.starts[c + 1]] {
                        let k = id as usize;
                        if Undersides::crossing(
                            &undersides.polygons[k],
                            &undersides.boxes[k],
                            o,
                            q,
                            z,
                            rise,
                            limits,
                            Some(cell),
                        ) {
                            blocked = true;
                            return true;
                        }
                    }
                }
            }
            false
        };
        if TREES {
            self.grid.walk_cells_owned(o, q, &mut visit);
        } else {
            self.grid.walk_cells(o, q, &mut visit);
        }
        (blocked, depth)
    }

    /// Crown-weighted sunlight; old sunlit/line_clear never consult the tree index or tree height.
    fn transmission(&self, p: [f64; 3], sun: &Sun, leaf_on: bool) -> f64 {
        if sun.tan_el <= 0.0 {
            return 0.0;
        }
        let top = self
            .trees
            .as_ref()
            .map_or(self.z_max, |trees| self.z_max.max(trees.z_max));
        let reach = (top - p[2]) / sun.tan_el;
        if reach.is_nan() || reach <= 0.0 {
            return 1.0;
        }
        let o = [p[0], p[1]];
        let q = [o[0] + sun.dir[0] * reach, o[1] + sun.dir[1] * reach];
        let eps = 1e-6 / reach;
        let (blocked, depth) = if self.details.is_some() {
            self.walk_ray::<true, true, true>(o, q, p[2], [reach, sun.tan_el], eps, leaf_on)
        } else {
            self.walk_ray::<false, true, true>(o, q, p[2], [reach, sun.tan_el], eps, leaf_on)
        };
        if blocked || depth > 13.815_510_557_964_274 {
            return 0.0;
        }
        // Terrain is opaque and retains the original sampling/reach convention. Trees cannot change
        // the building-only result; there is no need to sample above the old highest opaque blocker.
        if let Some(terrain) = &self.terrain {
            let step = terrain.sx.min(terrain.sy).abs();
            let opaque_reach = (self.z_max - p[2]) / sun.tan_el;
            let mut d = step;
            while d <= opaque_reach {
                match terrain.at(o[0] + sun.dir[0] * d, o[1] + sun.dir[1] * d) {
                    Some(h) if h > p[2] + d * sun.tan_el + 1e-6 => return 0.0,
                    Some(_) => {}
                    None => break,
                }
                d += step;
            }
        }
        (-depth).exp()
    }

    /// True when the receiver at `(x, y)` and height `z` gets direct sunlight from `sun`.
    pub fn sunlit(&self, p: [f64; 3], sun: &Sun) -> bool {
        if sun.tan_el <= 0.0 {
            return false;
        }
        let reach = (self.z_max - p[2]) / sun.tan_el;
        if reach.is_nan() || reach <= 0.0 {
            return true;
        }
        let o = [p[0], p[1]];
        let q = [o[0] + sun.dir[0] * reach, o[1] + sun.dir[1] * reach];
        let min_t = 1e-6 / reach; // a receiver on a wall is not shaded by that wall
        let blocked = if self.details.is_some() {
            self.wall_blocked::<true, true>(o, q, p[2], [reach, sun.tan_el], min_t)
        } else {
            self.wall_blocked::<false, true>(o, q, p[2], [reach, sun.tan_el], min_t)
        };
        if blocked {
            return false;
        }
        if let Some(terrain) = &self.terrain {
            let step = terrain.sx.min(terrain.sy).abs();
            let mut d = step;
            while d <= reach {
                match terrain.at(o[0] + sun.dir[0] * d, o[1] + sun.dir[1] * d) {
                    Some(h) if h > p[2] + d * sun.tan_el + 1e-6 => return false,
                    Some(_) => {}
                    None => break,
                }
                d += step;
            }
        }
        true
    }
}

impl Scene {
    /// True when nothing blocks the straight line between the 3D points `e` (eye) and `t` (target): every
    /// wall it crosses has its roof below the line there, and so does the terrain. Walls touching either
    /// end are ignored, so targets on a roof or right at a wall can be seen.
    pub fn line_clear(&self, e: [f64; 3], t: [f64; 3]) -> bool {
        let (o, q) = ([e[0], e[1]], [t[0], t[1]]);
        let len = (q[0] - o[0]).hypot(q[1] - o[1]);
        if len == 0.0 {
            let rise = t[2] - e[2];
            let eps = 1e-6 / rise.abs();
            return !self
                .undersides
                .as_ref()
                .is_some_and(|u| u.blocked(o, q, e[2], rise, eps, 1.0 - eps));
        }
        let eps = 1e-6 / len;
        let rise = t[2] - e[2];
        let blocked = if self.details.is_some() {
            self.wall_blocked::<true, false>(o, q, e[2], [rise, 1.0], eps)
        } else {
            self.wall_blocked::<false, false>(o, q, e[2], [rise, 1.0], eps)
        };
        if blocked {
            return false;
        }
        if let Some(terrain) = &self.terrain {
            let step = terrain.sx.min(terrain.sy).abs();
            let (dx, dy) = ((q[0] - o[0]) / len, (q[1] - o[1]) / len);
            let mut d = step;
            while d < len - step * 0.5 {
                if let Some(h) = terrain.at(o[0] + dx * d, o[1] + dy * d) {
                    if h > e[2] + d / len * rise + 1e-6 {
                        return false;
                    }
                }
                d += step;
            }
        }
        true
    }
}

/// For every target `targets[k]`, 1 when it is visible from `eyes[owner[k]]`.
pub fn line_of_sight(
    scene: &Scene,
    eyes: &[[f64; 3]],
    targets: &[[f64; 3]],
    owner: &[usize],
) -> Vec<u8> {
    targets
        .par_iter()
        .zip(owner.par_iter())
        .map(|(t, &k)| u8::from(scene.line_clear(eyes[k], *t)))
        .collect()
}

/// Per receiver and day: the number of sunlit sun positions and the longest run of consecutive sunlit
/// positions. `suns[day_offsets[d]..day_offsets[d + 1]]` are the positions of day `d` in time order.
/// Both results are `receivers.len() x days` matrices (row-major).
///
/// Receivers on a facade pass `facing`: the facade's outward unit normal per receiver and the least dot
/// product between it and the horizontal direction to the sun; the sun counts only in front of the facade.
pub fn day_stats(
    scene: &Scene,
    receivers: &[[f64; 3]],
    suns: &[Sun],
    day_offsets: &[usize],
    facing: Option<(&[[f64; 2]], f64)>,
) -> (Vec<u32>, Vec<u32>) {
    let days = day_offsets.len().saturating_sub(1);
    let mut counts = vec![0u32; receivers.len() * days];
    let mut longest = vec![0u32; receivers.len() * days];
    if days == 0 {
        return (counts, longest);
    }
    counts
        .par_chunks_mut(days)
        .zip(longest.par_chunks_mut(days))
        .zip(receivers.par_iter().enumerate())
        .for_each(|((count_row, run_row), (k, p))| {
            let in_front = |sun: &Sun| {
                facing.is_none_or(|(normals, min_dot)| {
                    let n = normals[k];
                    n[0] * sun.dir[0] + n[1] * sun.dir[1] > min_dot
                })
            };
            for d in 0..days {
                let (mut count, mut run, mut best) = (0u32, 0u32, 0u32);
                for sun in &suns[day_offsets[d]..day_offsets[d + 1]] {
                    if in_front(sun) && scene.sunlit(*p, sun) {
                        count += 1;
                        run += 1;
                        best = best.max(run);
                    } else {
                        run = 0;
                    }
                }
                count_row[d] = count;
                run_row[d] = best;
            }
        });
    (counts, longest)
}

/// Weighted counts, opaque-only counts and thresholded longest runs, row-major receiver x day.
/// The opaque-only trace deliberately uses the original reach (without crown tops), including its
/// original floating-point operation order and terrain sampling. Dense crowns can terminate the
/// weighted walk early without losing the independent opaque-only count.
pub fn day_stats_trees(
    scene: &Scene,
    receivers: &[[f64; 3]],
    suns: &[Sun],
    day_offsets: &[usize],
    leaf_on: &[u8],
    threshold: f64,
    facing: Option<(&[[f64; 2]], f64)>,
) -> (Vec<f64>, Vec<u32>, Vec<u32>) {
    let days = day_offsets.len().saturating_sub(1);
    let mut weighted = vec![0.0; receivers.len() * days];
    let mut opaque = vec![0u32; receivers.len() * days];
    let mut longest = vec![0u32; receivers.len() * days];
    if days == 0 {
        return (weighted, opaque, longest);
    }
    weighted
        .par_chunks_mut(days)
        .zip(opaque.par_chunks_mut(days))
        .zip(longest.par_chunks_mut(days))
        .zip(receivers.par_iter().enumerate())
        .for_each(|(((weight_row, opaque_row), run_row), (k, p))| {
            for d in 0..days {
                let (mut run, mut best) = (0u32, 0u32);
                for sun in &suns[day_offsets[d]..day_offsets[d + 1]] {
                    let in_front = facing.is_none_or(|(normals, min_dot)| {
                        normals[k][0] * sun.dir[0] + normals[k][1] * sun.dir[1] > min_dot
                    });
                    let clear = in_front && scene.sunlit(*p, sun);
                    opaque_row[d] += u32::from(clear);
                    let transmitted = if clear {
                        scene.transmission(*p, sun, leaf_on[d] != 0)
                    } else {
                        0.0
                    };
                    weight_row[d] += transmitted;
                    if in_front && sun.tan_el > 0.0 && transmitted >= threshold {
                        run += 1;
                        best = best.max(run);
                    } else {
                        run = 0;
                    }
                }
                run_row[d] = best;
            }
        });
    (weighted, opaque, longest)
}

#[cfg(test)]
#[path = "tree_tests.rs"]
mod tree_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<Seg> {
        let c = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        (0..4)
            .map(|k| Seg {
                a: c[k],
                b: c[(k + 1) % 4],
            })
            .collect()
    }

    fn sun(azimuth_deg: f64, elevation_deg: f64) -> Sun {
        let a = azimuth_deg.to_radians();
        Sun {
            dir: [a.sin(), a.cos()],
            tan_el: elevation_deg.to_radians().tan(),
        }
    }

    #[test]
    fn building_shadow_length_follows_the_elevation() {
        // 10 m high building, x 0..10, y 0..10; sun from the south at 45 degrees -> 10 m shadow north
        let scene = Scene::new(square(0.0, 0.0, 10.0, 10.0), vec![10.0; 4], None);
        let s = sun(180.0, 45.0);
        assert!(!scene.sunlit([5.0, 15.0, 0.0], &s));
        assert!(!scene.sunlit([5.0, 19.9, 0.0], &s));
        assert!(scene.sunlit([5.0, 20.1, 0.0], &s));
        assert!(scene.sunlit([5.0, -5.0, 0.0], &s)); // south side faces the sun
        assert!(scene.sunlit([5.0, 15.0, 6.0], &s)); // a receiver higher up sees over the roof
        assert!(!scene.sunlit([5.0, 15.0, 0.0], &sun(180.0, -1.0))); // below the horizon
    }

    #[test]
    fn terrain_ridge_casts_a_shadow() {
        // flat ground at 0 with a 20 m ridge along y = 50 (rows 4..6 of a 10 m grid)
        let (nx, ny) = (11, 11);
        let z: Vec<f64> = (0..nx * ny)
            .map(|k| {
                if (4..=6).contains(&(k / nx)) {
                    20.0
                } else {
                    0.0
                }
            })
            .collect();
        let terrain = Terrain {
            x0: 0.0,
            y0: 0.0,
            sx: 10.0,
            sy: 10.0,
            nx,
            ny,
            z,
        };
        let scene = Scene::new(Vec::new(), Vec::new(), Some(terrain));
        let s = sun(0.0, 30.0); // sun in the north: the ridge shades the ground south of it
        assert!(!scene.sunlit([50.0, 30.0, 0.0], &s));
        assert!(scene.sunlit([50.0, 80.0, 0.0], &s));
        assert_eq!(scene.terrain.as_ref().unwrap().at(50.0, 35.0), Some(10.0));
    }

    #[test]
    fn line_of_sight_over_and_behind_a_building() {
        // 10 m high building x 0..10; eye at 1.6 m west of it
        let scene = Scene::new(square(0.0, -5.0, 10.0, 5.0), vec![10.0; 4], None);
        let eye = [-10.0, 0.0, 1.6];
        assert!(!scene.line_clear(eye, [20.0, 0.0, 0.0])); // ground behind the building
        assert!(scene.line_clear(eye, [20.0, 0.0, 40.0])); // high enough to show over it
        assert!(scene.line_clear(eye, [-20.0, 0.0, 0.0])); // nothing in between
        assert!(scene.line_clear([-10.0, 0.0, 20.0], [5.0, 0.0, 10.0])); // its roof seen from above
        assert!(!scene.line_clear(eye, [5.0, 0.0, 10.0])); // but not from below
        let eyes = [eye];
        let seen = line_of_sight(
            &scene,
            &eyes,
            &[[20.0, 0.0, 0.0], [-20.0, 0.0, 0.0]],
            &[0, 0],
        );
        assert_eq!(seen, vec![0, 1]);
    }

    #[test]
    fn day_stats_count_and_longest_run() {
        // a wall east of the receiver blocks the morning sun only
        let wall = vec![Seg {
            a: [10.0, -50.0],
            b: [10.0, 50.0],
        }];
        let scene = Scene::new(wall, vec![30.0], None);
        let suns = [
            sun(90.0, 20.0),  // east, low: blocked
            sun(135.0, 30.0), // south-east: blocked
            sun(180.0, 40.0), // south: free
            sun(225.0, 30.0), // south-west: free
            sun(270.0, 20.0), // west: free
        ];
        let (counts, longest) = day_stats(&scene, &[[0.0, 0.0, 0.0]], &suns, &[0, 5, 5], None);
        assert_eq!(counts, vec![3, 0]);
        assert_eq!(longest, vec![3, 0]);
        let (counts, _) = day_stats(&scene, &[[0.0, 0.0, 0.0]], &suns[..], &[0, 2, 5], None);
        assert_eq!(counts, vec![0, 3]);
        // a facade facing west sees only the south-west and west sun
        let west: &[[f64; 2]] = &[[-1.0, 0.0]];
        let (counts, _) = day_stats(
            &scene,
            &[[0.0, 0.0, 0.0]],
            &suns,
            &[0, 5],
            Some((west, 0.0)),
        );
        assert_eq!(counts, vec![2]);
    }

    #[test]
    fn sloped_wall_shadow_and_line_end_exclusions() {
        // Top varies from 4 to 12 along x; a 45-degree south sun casts a 6 m shadow at x=2.
        let scene = Scene::with_details(
            vec![Seg {
                a: [0.0, 0.0],
                b: [8.0, 0.0],
            }],
            vec![[4.0, 12.0]],
            None,
            Vec::new(),
            None,
        );
        let s = Sun {
            dir: [0.0, -1.0],
            tan_el: 1.0,
        };
        assert!(!scene.sunlit([2.0, 5.9, 0.0], &s));
        assert!(scene.sunlit([2.0, 6.1, 0.0], &s));
        assert!(!scene.sunlit([6.0, 9.9, 0.0], &s));
        assert!(scene.sunlit([6.0, 10.1, 0.0], &s));
        assert!(!scene.line_clear([2.0, 1.0, 5.0], [2.0, -1.0, 5.0]));
        assert!(scene.line_clear([2.0, 1.0, 6.0], [2.0, -1.0, 6.0]));
        assert!(scene.line_clear([2.0, 1.0, 1.0], [2.0, 0.0, 1.0]));
        assert!(scene.sunlit([2.0, 0.0, 1.0], &s));
    }

    fn canopy(rings: Vec<Vec<[f64; 2]>>, walls: bool) -> Scene {
        let segs = if walls {
            square(-100.0, -100.0, 100.0, 100.0)
        } else {
            Vec::new()
        };
        let n = segs.len();
        Scene::with_details(
            segs,
            vec![[10.0, 10.0]; n],
            Some(vec![5.0; n]),
            vec![Underside { rings, z: 5.0 }],
            None,
        )
    }

    #[test]
    fn raised_walls_allow_rays_below_the_bottom() {
        let scene = Scene::with_details(
            square(0.0, -5.0, 10.0, 5.0),
            vec![[10.0, 10.0]; 4],
            Some(vec![5.0; 4]),
            Vec::new(),
            None,
        );
        assert!(scene.sunlit(
            [-10.0, 0.0, 0.0],
            &Sun {
                dir: [1.0, 0.0],
                tan_el: 0.1
            }
        ));
        assert!(!scene.sunlit(
            [-10.0, 0.0, 5.0],
            &Sun {
                dir: [1.0, 0.0],
                tan_el: 0.1
            }
        ));
        assert!(scene.line_clear([-10.0, 0.0, 4.0], [20.0, 0.0, 4.0]));
        assert!(scene.line_clear([-10.0, 0.0, 5.0], [20.0, 0.0, 5.0])); // bottom is strict
        assert!(!scene.line_clear([-10.0, 0.0, 6.0], [20.0, 0.0, 6.0]));
    }

    #[test]
    fn underside_blocks_without_a_wall_crossing_in_the_ray_reach() {
        let rings = vec![vec![
            [-100.0, -100.0],
            [100.0, -100.0],
            [100.0, 100.0],
            [-100.0, 100.0],
        ]];
        let scene = canopy(rings.clone(), true);
        let s = Sun {
            dir: [1.0, 0.0],
            tan_el: 1.0,
        };
        assert!(!scene.sunlit([0.0, 0.0, 0.0], &s)); // reach=10, walls are 100 m away
        assert!(scene.sunlit([0.0, 0.0, 5.0], &s)); // B is not strictly above the receiver
        assert!(!scene.line_clear([0.0, 0.0, 0.0], [10.0, 0.0, 8.0]));
        assert!(!scene.line_clear([10.0, 0.0, 8.0], [0.0, 0.0, 0.0]));
        assert!(!scene.line_clear([0.0, 0.0, 0.0], [0.0, 0.0, 8.0])); // vertical
        assert!(scene.line_clear([0.0, 0.0, 0.0], [10.0, 0.0, 5.0])); // endpoint exclusion
                                                                      // The shared bbox index must also work when there are no walls at all.
        assert!(!canopy(rings, false).sunlit([0.0, 0.0, 0.0], &s));
    }

    #[test]
    fn underside_holes_and_bbox_false_positives() {
        let scene = canopy(
            vec![
                vec![
                    [-100.0, -100.0],
                    [100.0, -100.0],
                    [100.0, 100.0],
                    [-100.0, 100.0],
                ],
                vec![[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]],
            ],
            false,
        );
        let s = Sun {
            dir: [1.0, 0.0],
            tan_el: 1.0,
        };
        assert!(scene.sunlit([0.0, 0.0, 0.0], &s));
        assert!(!scene.sunlit([20.0, 0.0, 0.0], &s));
        assert!(scene.sunlit([200.0, 0.0, 0.0], &s));
        assert!(scene.line_clear([0.0, 0.0, 0.0], [1.0, 0.0, 8.0]));
        assert!(!scene.line_clear([20.0, 0.0, 0.0], [21.0, 0.0, 8.0]));
    }

    #[test]
    fn many_undersides_match_brute_force() {
        let mut polygons = Vec::new();
        // Many large polygons repeated across cells. Most crossings are in a hole.
        for k in 0..48 {
            let z = 1.0 + k as f64;
            polygons.push(Underside {
                rings: vec![
                    vec![
                        [-100.0, -100.0],
                        [100.0, -100.0],
                        [100.0, 100.0],
                        [-100.0, 100.0],
                    ],
                    vec![[-60.0, -60.0], [60.0, -60.0], [60.0, 60.0], [-60.0, 60.0]],
                ],
                z,
            });
        }
        let scene = Scene::with_details(
            square(-100.0, -100.0, 100.0, 100.0),
            vec![[0.0, 0.0]; 4],
            None,
            polygons,
            None,
        );
        let undersides = scene.undersides.as_ref().unwrap();
        for k in 0..1000 {
            let o = [-120.0 + (k % 151) as f64, -80.0 + (k % 127) as f64];
            let q = [120.0 - (k % 139) as f64, 80.0 - (k % 113) as f64];
            for (z, rise) in [(0.0, 60.0), (60.0, -60.0), (20.0, 0.0)] {
                let (lo, hi) = (1e-6, 1.0 - 1e-6);
                let expected = rise != 0.0
                    && undersides.polygons.iter().any(|p| {
                        let s = (p.z - z) / rise;
                        s > lo
                            && s < hi
                            && p.contains([o[0] + s * (q[0] - o[0]), o[1] + s * (q[1] - o[1])])
                    });
                assert_eq!(
                    scene.wall_blocked::<false, false>(o, q, z, [rise, 1.0], lo),
                    expected
                );
            }
        }
        assert!(!undersides.blocked([0.0, 0.0], [0.0, 0.0], 0.0, 60.0, 0.0, 1.0));
        assert!(undersides.blocked([80.0, 0.0], [80.0, 0.0], 0.0, 60.0, 0.0, 1.0));
        assert!(!undersides.blocked([80.0, 0.0], [80.0, 0.0], 0.0, 1.0, 0.0, 1.0));
    }

    #[test]
    fn underside_crossings_on_cell_edges_and_segment_ends() {
        let scene = Scene::with_details(
            square(-100.0, -100.0, 100.0, 100.0),
            vec![[0.0, 0.0]; 4],
            None,
            vec![Underside {
                rings: vec![vec![
                    [-100.0, -100.0],
                    [100.0, -100.0],
                    [100.0, 100.0],
                    [-100.0, 100.0],
                ]],
                z: 5.0,
            }],
            None,
        );
        let u = scene.undersides.as_ref().unwrap();
        for c in 0..scene.grid.cell_count() {
            let b = scene.grid.bounds(c);
            for p in [[b[0], b[1]], [b[2], b[3]], [b[0], (b[1] + b[3]) * 0.5]] {
                for d in [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0]] {
                    let e = [p[0] - d[0], p[1] - d[1], 4.0];
                    let t = [p[0] + d[0], p[1] + d[1], 6.0];
                    let expected =
                        !u.blocked([e[0], e[1]], [t[0], t[1]], 4.0, 2.0, 1e-6, 1.0 - 1e-6);
                    assert_eq!(scene.line_clear(e, t), expected);
                }
                assert!(scene.line_clear([p[0], p[1], 4.0], [p[0], p[1], 5.0]));
                assert_eq!(
                    scene.line_clear([p[0], p[1], 4.0], [p[0], p[1], 6.0]),
                    !u.polygons[0].contains(p)
                );
                let sun = Sun {
                    dir: [1.0, 0.0],
                    tan_el: 1.0,
                };
                // Crossing exactly at q, with z_max provided by this underside alone.
                let only = canopy(u.polygons[0].rings.clone(), false);
                assert_eq!(
                    only.sunlit([p[0] - 1.0, p[1], 4.0], &sun),
                    !u.polygons[0].contains(p)
                );
            }
        }
    }

    #[test]
    fn randomized_packed_rays_match_all_obstacles() {
        // Fixed seed, no external random dependency; include sparse and dense scenes.
        let mut state = 0x942e_173b_u64;
        let mut random = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 11) as f64 / ((1_u64 << 53) as f64)
        };
        for count in [0, 1, 40, 120] {
            let mut walls = Vec::new();
            let mut tops = Vec::new();
            let mut bottoms = Vec::new();
            for wall_id in 0..count {
                walls.push(Seg {
                    a: [random() * 100.0 - 50.0, random() * 100.0 - 50.0],
                    b: [random() * 100.0 - 50.0, random() * 100.0 - 50.0],
                });
                let scale = if wall_id % 7 == 0 { 400.0 } else { 4.0 };
                tops.push([10.0 + random() * scale, 10.0 + random() * scale]);
                bottoms.push(random() * 10.0);
            }
            let mut polygons = Vec::new();
            for k in 0..32 {
                let x = random() * 100.0 - 50.0;
                let y = random() * 100.0 - 50.0;
                let r = if k % 4 == 0 {
                    150.0
                } else {
                    1.0 + random() * 20.0
                };
                let ring = |r| {
                    vec![
                        [x - r, y - r],
                        [x + r, y - r],
                        [x + r, y + r],
                        [x - r, y + r],
                    ]
                };
                let mut rings = vec![ring(r)];
                if k % 2 == 0 {
                    rings.push(ring(r * 0.7));
                }
                polygons.push(Underside {
                    rings,
                    z: 1.0 + random() * 40.0,
                });
            }
            // Oracle retains the original wall arrays and loops over every polygon.
            let copies = polygons
                .iter()
                .map(|p| Underside {
                    rings: p.rings.clone(),
                    z: p.z,
                })
                .collect();
            let detailed = Scene::with_details(
                walls.clone(),
                tops.clone(),
                Some(bottoms.clone()),
                copies,
                None,
            );
            let flat_tops: Vec<f64> = tops.iter().map(|t| t[0]).collect();
            let flat = Scene::new(walls.clone(), flat_tops, None);
            for scene in [&detailed, &flat] {
                for k in 0..2000 {
                    let e = [
                        random() * 240.0 - 120.0,
                        random() * 240.0 - 120.0,
                        random() * 60.0 - 5.0,
                    ];
                    let mut t = [
                        random() * 240.0 - 120.0,
                        random() * 240.0 - 120.0,
                        random() * 60.0 - 5.0,
                    ];
                    if k % 10 == 0 {
                        t[0] = e[0];
                        t[1] = e[1];
                    }
                    if k % 3 == 0 {
                        t[2] = e[2] - 100.0 * random();
                    }
                    let elevation = match k % 3 {
                        0 => 0.01 + random(),
                        1 => 80.0 + random() * 9.99,
                        _ => random() * 90.0 - 1.0,
                    };
                    let sun = sun(random() * 360.0, elevation);
                    for is_sun in [false, true] {
                        let reach = (scene.z_max - e[2]) / sun.tan_el;
                        let len = (t[0] - e[0]).hypot(t[1] - e[1]);
                        let o = [e[0], e[1]];
                        let q = if is_sun {
                            [o[0] + sun.dir[0] * reach, o[1] + sun.dir[1] * reach]
                        } else {
                            [t[0], t[1]]
                        };
                        let rise = if is_sun {
                            reach * sun.tan_el
                        } else {
                            t[2] - e[2]
                        };
                        let eps = if is_sun {
                            1e-6 / reach
                        } else if len == 0.0 {
                            1e-6 / rise.abs()
                        } else {
                            1e-6 / len
                        };
                        let mut blocked = false;
                        if is_sun || len != 0.0 {
                            for (i, wall) in walls.iter().enumerate() {
                                if let Some((s, u)) = segment_params(o, q, wall) {
                                    if s > eps
                                        && (if is_sun { s <= 1.0 } else { s < 1.0 - eps })
                                        && (0.0..=1.0).contains(&u)
                                    {
                                        let height = if is_sun {
                                            e[2] + s * reach * sun.tan_el
                                        } else {
                                            e[2] + s * rise
                                        };
                                        let mut top = tops[i][0];
                                        if scene.details.is_some() {
                                            if height <= bottoms[i] {
                                                continue;
                                            }
                                            top += u * (tops[i][1] - top);
                                        }
                                        blocked |= top > height + if is_sun { 0.0 } else { 1e-9 };
                                    }
                                }
                            }
                        }
                        if scene.undersides.is_some() && rise != 0.0 {
                            for p in &polygons {
                                let s = (p.z - e[2]) / rise;
                                let (lo, hi) = if is_sun {
                                    (0.0, 1.0 + f64::EPSILON)
                                } else {
                                    (eps, 1.0 - eps)
                                };
                                blocked |= s > lo
                                    && s < hi
                                    && p.contains([
                                        o[0] + s * (q[0] - o[0]),
                                        o[1] + s * (q[1] - o[1]),
                                    ]);
                            }
                        }
                        let expected = if is_sun && sun.tan_el <= 0.0 {
                            false
                        } else if is_sun && (reach.is_nan() || reach <= 0.0) {
                            true
                        } else {
                            !blocked
                        };
                        let got = if is_sun {
                            scene.sunlit(e, &sun)
                        } else {
                            scene.line_clear(e, t)
                        };
                        assert_eq!(got, expected, "walls={count}, case={k}, sun={is_sun}");
                    }
                }
            }
        }
    }

    #[test]
    fn descending_line_uses_cell_exit_height() {
        let scene = Scene::new(
            vec![
                Seg {
                    a: [0.0, -1.0],
                    b: [0.0, 1.0],
                },
                Seg {
                    a: [10.0, -1.0],
                    b: [10.0, 1.0],
                },
            ],
            vec![100.0, 5.0],
            None,
        );
        assert!(!scene.line_clear([1.0, 0.0, 20.0], [11.0, 0.0, 0.0]));
        assert!(scene.line_clear([1.0, 0.0, 30.0], [11.0, 0.0, 10.0]));
    }

    #[test]
    fn flat_and_equal_endpoint_tops_agree_bit_for_bit() {
        let a = Scene::new(square(0.0, 0.0, 10.0, 10.0), vec![10.0; 4], None);
        let b = Scene::with_details(
            square(0.0, 0.0, 10.0, 10.0),
            vec![[10.0, 10.0]; 4],
            None,
            Vec::new(),
            None,
        );
        assert!(b.details.is_none());
        for k in 0..1000 {
            let p = [
                (k % 31) as f64 - 10.0,
                (k % 37) as f64 - 10.0,
                (k % 11) as f64,
            ];
            let s = sun((k % 360) as f64, 1.0 + (k % 88) as f64);
            assert_eq!(a.sunlit(p, &s), b.sunlit(p, &s));
            assert_eq!(
                a.line_clear(p, [20.0, 20.0, 15.0]),
                b.line_clear(p, [20.0, 20.0, 15.0])
            );
        }
    }
}
