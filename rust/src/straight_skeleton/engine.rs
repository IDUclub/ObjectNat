//! Wavefront propagation.
//!
//! The wavefront is a set of cycles of moving vertices. Each wavefront edge ("piece") lies on the
//! offset line of one input edge; a vertex sits where two pieces meet and moves along the bisector
//! of the interior sector between them. Events are found with an indexed priority queue:
//!
//! * edge event: the two endpoints of a piece meet;
//! * split event: a reflex vertex reaches a piece (or the endpoint of a piece).
//!
//! Every event is resolved by the same local operation, [`Engine::surgery`]: at the event point it
//! collects every live vertex and every piece that passes through that point at that time, groups
//! the vertices into chains ("passes" of the wavefront through the point), orders the passes by the
//! angular gap they leave outside the wavefront, and reconnects consecutive passes with new
//! vertices. Simultaneous events at one point (four edges of a square, the centre of a regular
//! polygon, two reflex vertices meeting head-on, a split that coincides with an edge collapse) are
//! therefore resolved in one step, whatever order the queue produced them in.
//!
//! A new vertex whose interior sector has zero width (two anti-parallel pieces lying on top of each
//! other, as in a rectangle once its short edges are gone) is "zipped": it slides along the overlap
//! at the same instant until it reaches the end of the shorter piece, and the surgery runs again
//! there. This is how ridges between parallel edges are produced.

use super::geom::*;
use super::queue::EventQueue;

/// Position tolerance (normalised units: the larger side of the bounding box is 1).
pub(crate) const EPS_P: f64 = 1e-12;
/// An inconsistent new vertex whose rays are closer than this (radians) is closed as a sliver.
const SLIVER: f64 = 1e-3;
/// Largest collection radius an inconsistent surgery may widen to.
const EPS_P_MAX: f64 = 1e-7;
/// Time tolerance used when merging skeleton nodes.
pub(crate) const EPS_T: f64 = 1e-12;
/// Interior sectors narrower than this are treated as zero-width (zipped).
pub(crate) const EPS_ANG: f64 = 1e-10;
/// Minimum |det| of the three-line system for using it instead of vertex trajectories.
const EPS_DET: f64 = 1e-6;
/// A vertex is reflex when its interior sector exceeds π by more than this.
const EPS_REFLEX: f64 = 1e-12;

const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
struct Line {
    /// unit direction of the input edge
    d: V2,
    /// inward unit normal (left of `d`)
    n: V2,
    /// offset line at time t: n·p = c + t
    c: f64,
}

#[derive(Clone, Debug)]
struct Vert {
    pos: V2,
    t0: f64,
    vel: V2,
    in_p: u32,
    out_p: u32,
    node: u32,
    /// interior sector angle, in [0, 2π]
    sigma: f64,
    alive: bool,
    reflex: bool,
    zip: bool,
    alive_idx: u32,
    reflex_idx: u32,
    split_line: u32,
    split_time: f64,
    mark: u32,
    visit: u32,
}

#[derive(Clone, Debug)]
struct Piece {
    line: u32,
    start: u32,
    end: u32,
    alive: bool,
    alive_idx: u32,
    mark: u32,
    modified: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Node {
    pub pos: V2,
    pub t: f64,
}

/// A skeleton arc traced by one wavefront vertex: from node `a` to node `b`, with the face of
/// line `left` on one side (the vertex's incoming piece) and `right` on the other.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Arc {
    pub a: u32,
    pub b: u32,
    pub left: u32,
    pub right: u32,
}

pub(crate) struct Output {
    pub nodes: Vec<Node>,
    pub arcs: Vec<Arc>,
}

#[derive(Debug)]
pub(crate) enum EngineError {
    /// The propagation did not terminate within the iteration budget.
    Runaway,
    /// Wavefront vertices were still alive when no event was left.
    Unfinished,
}

struct Pass {
    in_p: u32,
    out_p: u32,
    /// interior sector of the pass before the event
    sigma: f64,
    /// a single vertex, no collapsed piece
    single: bool,
    /// a piece passing through the point (to be split)
    split: bool,
}

struct Plan {
    st: u32,
    vs: Vec<u32>,
    es: Vec<u32>,
    passes: Vec<Pass>,
    pairs: Vec<(usize, usize, bool)>,
    dead_verts: Vec<u32>,
    dead_pieces: Vec<u32>,
    whole_cycles: bool,
    consistent: bool,
}

impl Plan {
    /// Every pass is a lone vertex joined back to itself: nothing to do.
    fn is_noop(&self) -> bool {
        !self.whole_cycles
            && self
                .pairs
                .iter()
                .all(|&(a, b, _)| a == b && self.passes[a].single)
    }
}

struct Ray {
    ang: f64,
    is_in: bool,
    pass: usize,
    cluster: usize,
}

enum Hit {
    Interior(u32),
    Vertex(u32),
    Nothing,
}

pub(crate) struct Engine {
    lines: Vec<Line>,
    verts: Vec<Vert>,
    pieces: Vec<Piece>,
    line_pieces: Vec<Vec<u32>>,
    alive_verts: Vec<u32>,
    alive_pieces: Vec<u32>,
    /// kinematics of each live vertex (pos.x, pos.y, vel.x, vel.y, t0), parallel to `alive_verts`
    alive_kin: Vec<[f64; 5]>,
    /// line of each live piece (n.x, n.y, c), parallel to `alive_pieces`
    alive_piece_nc: Vec<[f64; 3]>,
    reflex: Vec<u32>,
    /// kinematics of each reflex vertex (pos.x, pos.y, vel.x, vel.y, t0), parallel to `reflex`
    reflex_kin: Vec<[f64; 5]>,
    /// current split time of each reflex vertex, parallel to `reflex`
    reflex_best: Vec<f64>,
    queue: EventQueue,
    nodes: Vec<Node>,
    n_input: usize,
    arcs: Vec<Arc>,
    now: f64,
    stamp: u32,
    zips: Vec<u32>,
    modified: Vec<u32>,
    new_verts: Vec<u32>,
    /// node to use for the next surgery instead of looking one up (rings touching at time 0)
    node_override: u32,
}

/// Debug tracing (debug builds only): set `SK_TRACE=1`.
#[cfg(debug_assertions)]
fn trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("SK_TRACE").is_some())
}

#[cfg(not(debug_assertions))]
#[inline(always)]
fn trace() -> bool {
    false
}

// Queue slots: even = edge event of a piece, odd = split event of a vertex. At equal times the
// queue pops even slots first, so an edge collapse is resolved before a split at the same instant.
#[inline]
fn eslot(p: u32) -> u32 {
    2 * p
}
#[inline]
fn sslot(v: u32) -> u32 {
    2 * v + 1
}

impl Engine {
    pub(crate) fn new(rings: &[Vec<V2>]) -> Engine {
        let total: usize = rings.iter().map(|r| r.len()).sum();
        let mut e = Engine {
            lines: Vec::with_capacity(total),
            verts: Vec::with_capacity(total * 3),
            pieces: Vec::with_capacity(total * 2),
            line_pieces: Vec::with_capacity(total),
            alive_verts: Vec::with_capacity(total),
            alive_pieces: Vec::with_capacity(total),
            alive_piece_nc: Vec::with_capacity(total),
            alive_kin: Vec::with_capacity(total),
            reflex: Vec::new(),
            reflex_kin: Vec::new(),
            reflex_best: Vec::new(),
            queue: EventQueue::new(),
            nodes: Vec::with_capacity(total * 2),
            n_input: total,
            arcs: Vec::with_capacity(total * 3),
            now: 0.0,
            stamp: 0,
            zips: Vec::new(),
            modified: Vec::new(),
            new_verts: Vec::new(),
            node_override: NONE,
        };
        for ring in rings {
            let n = ring.len();
            let vb = e.verts.len() as u32;
            let eb = e.lines.len() as u32;
            for j in 0..n {
                let a = ring[j];
                let b = ring[(j + 1) % n];
                let v = sub(b, a);
                let l = norm(v);
                let d = [v[0] / l, v[1] / l];
                let nn = [-d[1], d[0]];
                e.lines.push(Line {
                    d,
                    n: nn,
                    c: dot(nn, a),
                });
                e.line_pieces.push(Vec::new());
            }
            for j in 0..n as u32 {
                let pid = e.new_piece(eb + j, vb + j, vb + (j + 1) % n as u32);
                debug_assert_eq!(pid, eb + j);
            }
            for (j, &p) in ring.iter().enumerate() {
                let in_p = eb + ((j + n - 1) % n) as u32;
                let out_p = eb + j as u32;
                let d_in = e.lines[in_p as usize].d;
                let d_out = e.lines[out_p as usize].d;
                let sigma = ccw_angle(d_out, neg(d_in));
                let id = e.verts.len() as u32;
                e.nodes.push(Node { pos: p, t: 0.0 });
                e.push_vert(p, 0.0, in_p, out_p, id, sigma, false);
            }
        }
        e
    }

    fn new_piece(&mut self, line: u32, start: u32, end: u32) -> u32 {
        let id = self.pieces.len() as u32;
        self.pieces.push(Piece {
            line,
            start,
            end,
            alive: true,
            alive_idx: self.alive_pieces.len() as u32,
            mark: 0,
            modified: 0,
        });
        self.alive_pieces.push(id);
        let ln = self.lines[line as usize];
        self.alive_piece_nc.push([ln.n[0], ln.n[1], ln.c]);
        self.line_pieces[line as usize].push(id);
        id
    }

    /// `overlap`: the vertex joins two pieces that may lie on top of each other (it comes out of an
    /// event); a sector below `EPS_ANG` then means zero width. An input vertex is never zero-width,
    /// however sharp: it moves (fast) like any other.
    #[allow(clippy::too_many_arguments)]
    fn push_vert(
        &mut self,
        pos: V2,
        t: f64,
        in_p: u32,
        out_p: u32,
        node: u32,
        sigma: f64,
        overlap: bool,
    ) -> u32 {
        let id = self.verts.len() as u32;
        let sigma = sigma.clamp(0.0, TAU);
        let zip = overlap && sigma < EPS_ANG;
        let reflex = !zip && sigma > PI + EPS_REFLEX;
        let vel = if zip {
            [0.0, 0.0]
        } else {
            let s = sigma.clamp(f64::MIN_POSITIVE.sqrt(), TAU - EPS_ANG);
            let d_out = self.lines[self.pieces[out_p as usize].line as usize].d;
            let bis = rot(d_out, 0.5 * s);
            mul(bis, 1.0 / (0.5 * s).sin())
        };
        self.verts.push(Vert {
            pos,
            t0: t,
            vel,
            in_p,
            out_p,
            node,
            sigma,
            alive: true,
            reflex,
            zip,
            alive_idx: self.alive_verts.len() as u32,
            reflex_idx: NONE,
            split_line: NONE,
            split_time: f64::INFINITY,
            mark: 0,
            visit: 0,
        });
        self.alive_verts.push(id);
        self.alive_kin.push([pos[0], pos[1], vel[0], vel[1], t]);
        self.pieces[in_p as usize].end = id;
        self.pieces[out_p as usize].start = id;
        if reflex {
            self.verts[id as usize].reflex_idx = self.reflex.len() as u32;
            self.reflex.push(id);
            self.reflex_kin.push([pos[0], pos[1], vel[0], vel[1], t]);
            self.reflex_best.push(f64::INFINITY);
        }
        id
    }

    #[inline]
    fn pos(&self, v: u32, t: f64) -> V2 {
        let vv = &self.verts[v as usize];
        [
            vv.pos[0] + vv.vel[0] * (t - vv.t0),
            vv.pos[1] + vv.vel[1] * (t - vv.t0),
        ]
    }

    #[inline]
    fn line_of(&self, p: u32) -> Line {
        self.lines[self.pieces[p as usize].line as usize]
    }

    fn kill_vert(&mut self, v: u32, node: u32) {
        let (in_p, out_p, start) = {
            let vv = &self.verts[v as usize];
            (vv.in_p, vv.out_p, vv.node)
        };
        let left = self.pieces[in_p as usize].line;
        let right = self.pieces[out_p as usize].line;
        self.arcs.push(Arc {
            a: start,
            b: node,
            left,
            right,
        });
        let vv = &mut self.verts[v as usize];
        vv.alive = false;
        let ai = vv.alive_idx as usize;
        let ri = vv.reflex_idx;
        vv.reflex_idx = NONE;
        self.alive_verts.swap_remove(ai);
        self.alive_kin.swap_remove(ai);
        if ai < self.alive_verts.len() {
            let moved = self.alive_verts[ai];
            self.verts[moved as usize].alive_idx = ai as u32;
        }
        if ri != NONE {
            let ri = ri as usize;
            self.reflex.swap_remove(ri);
            self.reflex_kin.swap_remove(ri);
            self.reflex_best.swap_remove(ri);
            if ri < self.reflex.len() {
                let moved = self.reflex[ri];
                self.verts[moved as usize].reflex_idx = ri as u32;
            }
        }
        self.queue.remove(sslot(v));
    }

    fn kill_piece(&mut self, p: u32) {
        let pc = &mut self.pieces[p as usize];
        if !pc.alive {
            return;
        }
        pc.alive = false;
        let ai = pc.alive_idx as usize;
        let line = pc.line as usize;
        self.alive_pieces.swap_remove(ai);
        self.alive_piece_nc.swap_remove(ai);
        if ai < self.alive_pieces.len() {
            let moved = self.alive_pieces[ai];
            self.pieces[moved as usize].alive_idx = ai as u32;
        }
        let lp = &mut self.line_pieces[line];
        if let Some(k) = lp.iter().position(|&x| x == p) {
            lp.swap_remove(k);
        }
        self.queue.remove(eslot(p));
    }

    fn node_at(&mut self, p: V2, t: f64) -> u32 {
        let e2 = EPS_P * EPS_P;
        for i in (self.n_input..self.nodes.len()).rev() {
            let nd = self.nodes[i];
            if nd.t < t - 4.0 * EPS_T {
                break;
            }
            if (nd.t - t).abs() <= EPS_T && dist2(nd.pos, p) <= e2 {
                return i as u32;
            }
        }
        self.nodes.push(Node { pos: p, t });
        (self.nodes.len() - 1) as u32
    }

    // ---------------------------------------------------------------- events

    /// The point and time at which the offset lines of `a`, `b` and `c` meet, when the system is
    /// well conditioned (no two of the lines parallel with the same orientation).
    fn three_lines(&self, a: u32, b: u32, c: u32) -> Option<(V2, f64)> {
        let (la, lb, lc) = (
            self.lines[a as usize],
            self.lines[b as usize],
            self.lines[c as usize],
        );
        // rows: n_i . p - t = c_i ; subtract row a from rows b and c to eliminate t
        let r1 = sub(lb.n, la.n);
        let r2 = sub(lc.n, la.n);
        let det = cross(r1, r2);
        if det.abs() < EPS_DET {
            return None;
        }
        let k1 = lb.c - la.c;
        let k2 = lc.c - la.c;
        let x = (k1 * r2[1] - k2 * r1[1]) / det;
        let y = (r1[0] * k2 - r2[0] * k1) / det;
        let t = dot(la.n, [x, y]) - la.c;
        if x.is_finite() && y.is_finite() && t.is_finite() {
            Some(([x, y], t))
        } else {
            None
        }
    }

    /// Collapse point and time of piece `p`, from the three lines when possible.
    fn edge_collapse(&self, p: u32, t_traj: f64) -> (V2, f64) {
        let pc = &self.pieces[p as usize];
        let (u, w) = (pc.start, pc.end);
        let la = self.pieces[self.verts[u as usize].in_p as usize].line;
        let lc = self.pieces[self.verts[w as usize].out_p as usize].line;
        if let Some((q, t3)) = self.three_lines(la, pc.line, lc) {
            if t3 >= self.now - EPS_T
                && (t3 - t_traj).abs() <= 1e-7 + 1e-3 * (t_traj - self.now).abs()
            {
                return (q, t3.max(self.now));
            }
        }
        let q = mul(add(self.pos(u, t_traj), self.pos(w, t_traj)), 0.5);
        (q, t_traj)
    }

    fn update_edge_event(&mut self, p: u32) {
        let pc = &self.pieces[p as usize];
        if !pc.alive {
            return;
        }
        let (u, w) = (pc.start, pc.end);
        let (vu, vw) = (&self.verts[u as usize], &self.verts[w as usize]);
        if vu.zip || vw.zip {
            self.queue.remove(eslot(p));
            return;
        }
        let d = self.lines[pc.line as usize].d;
        let t = self.now;
        let l0 = dot(d, sub(self.pos(w, t), self.pos(u, t)));
        let rate = dot(d, sub(vw.vel, vu.vel));
        if rate < 0.0 {
            let tc = t + l0.max(0.0) / -rate;
            if tc.is_finite() {
                let (_, tc) = self.edge_collapse(p, tc);
                self.queue.set(eslot(p), tc);
                return;
            }
        }
        self.queue.remove(eslot(p));
    }

    /// Time at which reflex vertex `r` reaches piece `p`, if it does so before `limit`.
    fn split_time(&self, r: u32, p: u32, limit: f64) -> Option<f64> {
        let pc = &self.pieces[p as usize];
        if pc.start == r || pc.end == r {
            return None;
        }
        let rv = &self.verts[r as usize];
        let ln = self.lines[pc.line as usize];
        let t = self.now;
        let pr = self.pos(r, t);
        let dist0 = dot(ln.n, pr) - ln.c - t;
        if dist0 < -EPS_P {
            return None;
        }
        let closing = 1.0 - dot(ln.n, rv.vel);
        if closing <= 1e-12 {
            return None;
        }
        let th = t + dist0.max(0.0) / closing;
        // the three-line time below may differ from the trajectory by this much
        let slack = 1e-7 + 1e-3 * (th - t);
        if th.is_nan() || th >= limit + slack {
            return None;
        }
        let (vs, ve) = (&self.verts[pc.start as usize], &self.verts[pc.end as usize]);
        if vs.zip || ve.zip {
            return None;
        }
        // cheap test on the trajectories first
        {
            let q = add(pr, mul(rv.vel, th - t));
            let ss = dot(ln.d, self.pos(pc.start, th));
            let se = dot(ln.d, self.pos(pc.end, th));
            let sq = dot(ln.d, q);
            let margin = 1e-6 + slack * (1.0 + norm(rv.vel) + norm(vs.vel) + norm(ve.vel));
            if se < ss - margin || sq < ss - margin || sq > se + margin {
                return None;
            }
        }
        let (q, th) = self.split_point(r, pc.line, th);
        if th >= limit {
            return None;
        }
        // a vertex created at this instant has already been reconciled with everything at its
        // position (the surgery that created it saw all of it)
        if th - t <= EPS_T && rv.t0 >= t - EPS_T && dist2(q, pr) <= EPS_P * EPS_P {
            return None;
        }
        let ss = dot(ln.d, self.pos(pc.start, th));
        let se = dot(ln.d, self.pos(pc.end, th));
        let sq = dot(ln.d, q);
        let tol = 0.5 * EPS_P;
        if se < ss - tol || sq < ss - tol || sq > se + tol {
            return None;
        }
        Some(th)
    }

    /// Point and time at which reflex vertex `r` reaches the offset line `line`.
    fn split_point(&self, r: u32, line: u32, t_traj: f64) -> (V2, f64) {
        let rv = &self.verts[r as usize];
        let la = self.pieces[rv.in_p as usize].line;
        let lb = self.pieces[rv.out_p as usize].line;
        if let Some((q, t3)) = self.three_lines(la, lb, line) {
            if t3 >= self.now - EPS_T
                && (t3 - t_traj).abs() <= 1e-7 + 1e-3 * (t_traj - self.now).abs()
            {
                return (q, t3.max(self.now));
            }
        }
        (self.pos(r, t_traj), t_traj)
    }

    fn set_split(&mut self, r: u32, time: f64, line: u32) {
        let rv = &mut self.verts[r as usize];
        rv.split_time = time;
        rv.split_line = line;
        if rv.reflex_idx != NONE {
            self.reflex_best[rv.reflex_idx as usize] = time;
        }
        if line != NONE {
            self.queue.set(sslot(r), time);
        } else {
            self.queue.remove(sslot(r));
        }
    }

    /// Cheap necessary condition for `r` (at `pr`, moving with `vel`) to reach the line `nc`
    /// before `limit`: the same arithmetic as the start of [`Engine::split_time`].
    #[inline]
    fn may_reach(nc: [f64; 3], pr: V2, vel: V2, t: f64, limit: f64) -> bool {
        let dist0 = nc[0] * pr[0] + nc[1] * pr[1] - nc[2] - t;
        if dist0 < -EPS_P {
            return false;
        }
        let closing = 1.0 - (nc[0] * vel[0] + nc[1] * vel[1]);
        if closing <= 1e-12 {
            return false;
        }
        let th = t + dist0.max(0.0) / closing;
        th < limit + 1e-7 + 1e-3 * (th - t)
    }

    /// Full search for the first split event of reflex vertex `r`.
    fn rescan(&mut self, r: u32) {
        let mut best = f64::INFINITY;
        let mut best_line = NONE;
        let t = self.now;
        let pr = self.pos(r, t);
        let vel = self.verts[r as usize].vel;
        for k in 0..self.alive_pieces.len() {
            if !Self::may_reach(self.alive_piece_nc[k], pr, vel, t, best) {
                continue;
            }
            let p = self.alive_pieces[k];
            if let Some(th) = self.split_time(r, p, best) {
                best = th;
                best_line = self.pieces[p as usize].line;
            }
        }
        if trace() {
            eprintln!("  rescan v{r}: best line {best_line} at {best}");
        }
        self.set_split(r, best, best_line);
    }

    /// A piece changed: does any reflex vertex now reach it earlier than its current event?
    fn recheck(&mut self, p: u32) {
        let line = self.pieces[p as usize].line;
        let ln = self.lines[line as usize];
        let nc = [ln.n[0], ln.n[1], ln.c];
        let t = self.now;
        for k in 0..self.reflex.len() {
            let [px, py, vx, vy, t0] = self.reflex_kin[k];
            let limit = self.reflex_best[k];
            let pr = [px + vx * (t - t0), py + vy * (t - t0)];
            if !Self::may_reach(nc, pr, [vx, vy], t, limit) {
                continue;
            }
            let r = self.reflex[k];
            if let Some(th) = self.split_time(r, p, limit) {
                self.set_split(r, th, line);
            }
        }
    }

    /// Find the live piece of `line` that contains point `q` at time `t`.
    fn find_on_line(&self, line: u32, r: u32, q: V2, t: f64) -> Hit {
        let ln = self.lines[line as usize];
        let sq = dot(ln.d, q);
        // A piece endpoint that shares a supporting line with `r` is exactly where `r` meets the
        // line (both lie on the shared line and on `line`). Fast vertices make the numeric test
        // below unreliable for this case.
        let (ra, rb) = {
            let rv = &self.verts[r as usize];
            (
                self.pieces[rv.in_p as usize].line,
                self.pieces[rv.out_p as usize].line,
            )
        };
        for &p in &self.line_pieces[line as usize] {
            let pc = &self.pieces[p as usize];
            if pc.start == r || pc.end == r {
                continue;
            }
            for (w, other) in [
                (
                    pc.start,
                    self.pieces[self.verts[pc.start as usize].in_p as usize].line,
                ),
                (
                    pc.end,
                    self.pieces[self.verts[pc.end as usize].out_p as usize].line,
                ),
            ] {
                if (other == ra || other == rb) && other != line && !self.verts[w as usize].zip {
                    let lo = self.lines[other as usize];
                    if cross(lo.d, ln.d).abs() > 1e-9 && dist2(self.pos(w, t), q) <= 1e-6 {
                        return Hit::Vertex(w);
                    }
                }
            }
        }
        let mut best = Hit::Nothing;
        let mut best_perp = f64::INFINITY;
        for &p in &self.line_pieces[line as usize] {
            let pc = &self.pieces[p as usize];
            if pc.start == r || pc.end == r {
                continue;
            }
            if self.verts[pc.start as usize].zip || self.verts[pc.end as usize].zip {
                continue;
            }
            let ss = dot(ln.d, self.pos(pc.start, t));
            let se = dot(ln.d, self.pos(pc.end, t));
            if se < ss - EPS_P || sq < ss - EPS_P || sq > se + EPS_P {
                continue;
            }
            let perp = (dot(ln.n, q) - ln.c - t).abs();
            if perp > 4.0 * EPS_P || perp >= best_perp {
                continue;
            }
            best_perp = perp;
            best = if sq - ss <= EPS_P {
                Hit::Vertex(pc.start)
            } else if se - sq <= EPS_P {
                Hit::Vertex(pc.end)
            } else {
                Hit::Interior(p)
            };
        }
        best
    }

    // ---------------------------------------------------------------- surgery

    fn piece_through(&self, p: u32, q: V2, t: f64, radius: f64) -> bool {
        let pc = &self.pieces[p as usize];
        let ln = self.lines[pc.line as usize];
        let perp = (dot(ln.n, q) - ln.c - t).abs();
        if perp > radius {
            return false;
        }
        let ss = dot(ln.d, self.pos(pc.start, t));
        let se = dot(ln.d, self.pos(pc.end, t));
        let sq = dot(ln.d, q);
        ss < sq && sq < se
    }

    /// Sector of the vertex that would join pass `a`'s incoming piece to pass `b`'s outgoing one.
    fn pair_sector(&self, a: &Pass, b: &Pass) -> f64 {
        let d_in = self.line_of(a.in_p).d;
        let d_out = if b.split {
            self.line_of(b.in_p).d
        } else {
            self.line_of(b.out_p).d
        };
        ccw_angle(d_out, neg(d_in))
    }

    /// Collect everything at `p` at time `t` within `radius`, and work out how to reconnect it.
    /// Nothing is modified (apart from marks).
    fn plan(
        &mut self,
        t: f64,
        p: V2,
        forced: &[u32],
        forced_piece: u32,
        radius: f64,
    ) -> Option<Plan> {
        self.stamp += 1;
        let st = self.stamp;
        let e2 = radius * radius;
        // 1. vertices at p
        let mut vs: Vec<u32> = Vec::new();
        for &v in forced {
            let vv = &mut self.verts[v as usize];
            if vv.alive && vv.mark != st {
                vv.mark = st;
                vs.push(v);
            }
        }
        for k in 0..self.alive_verts.len() {
            let [px, py, vx, vy, t0] = self.alive_kin[k];
            let q = [px + vx * (t - t0), py + vy * (t - t0)];
            if dist2(q, p) > e2 {
                continue;
            }
            let v = self.alive_verts[k];
            if self.verts[v as usize].mark != st {
                self.verts[v as usize].mark = st;
                vs.push(v);
            }
        }
        // 2. pieces passing through p
        let mut es: Vec<u32> = Vec::new();
        if forced_piece != NONE {
            let pc = &self.pieces[forced_piece as usize];
            if pc.alive
                && self.verts[pc.start as usize].mark != st
                && self.verts[pc.end as usize].mark != st
            {
                self.pieces[forced_piece as usize].mark = st;
                es.push(forced_piece);
            }
        }
        for k in 0..self.alive_pieces.len() {
            let nc = self.alive_piece_nc[k];
            if (nc[0] * p[0] + nc[1] * p[1] - nc[2] - t).abs() > radius {
                continue;
            }
            let pid = self.alive_pieces[k];
            let pc = &self.pieces[pid as usize];
            if pc.mark == st
                || self.verts[pc.start as usize].mark == st
                || self.verts[pc.end as usize].mark == st
            {
                continue;
            }
            if self.verts[pc.start as usize].zip || self.verts[pc.end as usize].zip {
                continue;
            }
            if self.piece_through(pid, p, t, radius) {
                self.pieces[pid as usize].mark = st;
                es.push(pid);
            }
        }
        if vs.is_empty() {
            return None;
        }

        // 3. passes through p
        let mut passes: Vec<Pass> = Vec::new();
        let mut dead_verts: Vec<u32> = Vec::new();
        let mut dead_pieces: Vec<u32> = Vec::new();
        let mut whole_cycles = false;
        let mut consistent = true;
        for &v in &vs {
            if self.verts[v as usize].visit == st {
                continue;
            }
            // walk back to the start of the chain
            let mut s = v;
            let mut whole = false;
            loop {
                let pv = self.pieces[self.verts[s as usize].in_p as usize].start;
                if self.verts[pv as usize].mark != st {
                    break;
                }
                if pv == v {
                    whole = true;
                    break;
                }
                s = pv;
            }
            if whole {
                whole_cycles = true;
                let mut c = v;
                loop {
                    self.verts[c as usize].visit = st;
                    dead_verts.push(c);
                    let op = self.verts[c as usize].out_p;
                    dead_pieces.push(op);
                    c = self.pieces[op as usize].end;
                    if c == v {
                        break;
                    }
                }
                continue;
            }
            let in_p = self.verts[s as usize].in_p;
            let mut c = s;
            let mut tsum = 0.0;
            let mut single = true;
            loop {
                let cv = &mut self.verts[c as usize];
                cv.visit = st;
                tsum += PI - cv.sigma;
                dead_verts.push(c);
                let op = cv.out_p;
                let nx = self.pieces[op as usize].end;
                if self.verts[nx as usize].mark != st {
                    break;
                }
                dead_pieces.push(op);
                single = false;
                c = nx;
            }
            let out_p = self.verts[c as usize].out_p;
            let sigma = PI - tsum;
            // The chain plus the one vertex at the far end of both its pieces is the whole cycle:
            // a polygon whose lines all meet here, so it vanishes here, wherever rounding put that
            // vertex (a sliver tip can be off by a lot). Exception: two exactly anti-parallel
            // pieces lying on top of each other, a zero-width strip that is zipped instead.
            let a0 = self.pieces[in_p as usize].start;
            if a0 == self.pieces[out_p as usize].end && sigma.abs() > EPS_ANG {
                let av = &mut self.verts[a0 as usize];
                av.mark = st;
                av.visit = st;
                dead_verts.push(a0);
                dead_pieces.push(in_p);
                dead_pieces.push(out_p);
                whole_cycles = true;
                continue;
            }
            // a chain that turns by more than half a turn cannot collapse on its own: something
            // nearby is missing from the event
            if !(-EPS_ANG..=TAU + EPS_ANG).contains(&sigma) {
                consistent = false;
            }
            passes.push(Pass {
                in_p,
                out_p,
                sigma: sigma.clamp(0.0, TAU),
                single,
                split: false,
            });
        }
        for &e in &es {
            passes.push(Pass {
                in_p: e,
                out_p: NONE,
                sigma: PI,
                single: false,
                split: true,
            });
        }

        // 4. pair the rays: every In ray is joined to the Out ray just before it (clockwise)
        let mut pairs: Vec<(usize, usize, bool)> = Vec::new();
        if !passes.is_empty() {
            let mut rays: Vec<Ray> = Vec::with_capacity(2 * passes.len());
            for (k, ps) in passes.iter().enumerate() {
                let din = self.line_of(ps.in_p).d;
                let dout = if ps.split {
                    din
                } else {
                    self.line_of(ps.out_p).d
                };
                rays.push(Ray {
                    ang: angle(neg(din)),
                    is_in: true,
                    pass: k,
                    cluster: 0,
                });
                rays.push(Ray {
                    ang: angle(dout),
                    is_in: false,
                    pass: k,
                    cluster: 0,
                });
            }
            rays.sort_by(|a, b| {
                a.ang
                    .total_cmp(&b.ang)
                    .then(a.is_in.cmp(&b.is_in))
                    .then(a.pass.cmp(&b.pass))
            });
            // start right after the widest angular gap, so that no group of tied rays wraps around
            let nr = rays.len();
            let mut best = 0;
            let mut best_gap = -1.0;
            for i in 0..nr {
                let a = rays[i].ang;
                let b = if i + 1 < nr {
                    rays[i + 1].ang
                } else {
                    rays[0].ang + TAU
                };
                if b - a > best_gap {
                    best_gap = b - a;
                    best = (i + 1) % nr;
                }
            }
            rays.rotate_left(best);
            // groups of (numerically) tied rays: Out rays first, so a zero-width sector is interior
            let mut cl = 0;
            for i in 0..nr {
                if i > 0 {
                    let mut d = rays[i].ang - rays[i - 1].ang;
                    if d < 0.0 {
                        d += TAU;
                    }
                    if d >= EPS_ANG {
                        cl += 1;
                    }
                }
                rays[i].cluster = cl;
            }
            rays.sort_by(|a, b| a.cluster.cmp(&b.cluster).then(a.is_in.cmp(&b.is_in)));
            if let Some(first_out) = rays.iter().position(|r| !r.is_in) {
                rays.rotate_left(first_out);
            }
            let mut stack: Vec<usize> = Vec::new();
            let mut lone_in: Vec<usize> = Vec::new();
            for (i, r) in rays.iter().enumerate() {
                if r.is_in {
                    match stack.pop() {
                        Some(o) => pairs.push((r.pass, rays[o].pass, rays[o].cluster == r.cluster)),
                        None => lone_in.push(i),
                    }
                } else {
                    stack.push(i);
                }
            }
            if !lone_in.is_empty() {
                consistent = false;
            }
            // Events are collisions. If the rays say every pass just separates (nothing would
            // change), the passes overlap by a rounding error: pair them as colliding instead,
            // ordered by the middle of their exterior gaps.
            if !whole_cycles
                && passes.len() >= 2
                && pairs.iter().all(|&(a, b, _)| a == b && passes[a].single)
            {
                let mut order: Vec<(f64, usize)> = passes
                    .iter()
                    .enumerate()
                    .map(|(k, ps)| {
                        let alpha = angle(neg(self.line_of(ps.in_p).d));
                        let mut mid = alpha + 0.5 * (TAU - ps.sigma);
                        if mid >= TAU {
                            mid -= TAU;
                        }
                        (mid, k)
                    })
                    .collect();
                order.sort_by(|a, b| a.0.total_cmp(&b.0));
                let m = order.len();
                pairs = (0..m)
                    .map(|k| (order[(k + 1) % m].1, order[k].1, false))
                    .collect();
            }
            // In a collision every ray lies inside the other passes' sectors, so the sector of a
            // new vertex joining two different passes fits inside both of them.
            for &(ia, ob, tied) in &pairs {
                if ia != ob
                    && !tied
                    && self.pair_sector(&passes[ia], &passes[ob])
                        > passes[ia].sigma.min(passes[ob].sigma) + 1e-9
                {
                    consistent = false;
                }
            }
            // inconsistent configurations: pair whatever is left in order
            for (i, o) in lone_in.into_iter().zip(stack) {
                pairs.push((rays[i].pass, rays[o].pass, false));
            }
        }
        Some(Plan {
            st,
            vs,
            es,
            passes,
            pairs,
            dead_verts,
            dead_pieces,
            whole_cycles,
            consistent,
        })
    }

    /// Resolve everything that happens at point `p` at time `t`. Returns false if nothing changed.
    ///
    /// When the reconnection comes out inconsistent (a chain turning by more than half a turn, or
    /// unmatched rays), a participant was missed because rounding put it slightly too far away;
    /// the collection radius is then widened tenfold, up to `EPS_P_MAX`.
    fn surgery(&mut self, t: f64, p: V2, forced: &[u32], forced_piece: u32) -> bool {
        let mut radius = EPS_P;
        let mut prev: Option<Plan> = None;
        let plan = loop {
            let Some(plan) = self.plan(t, p, forced, forced_piece, radius) else {
                return false;
            };
            // widening until nothing changes does not resolve an inconsistency: keep the last plan
            // that does something (its inverted slivers are closed below)
            if plan.is_noop() {
                if let Some(pp) = prev {
                    break pp;
                }
            }
            if trace() {
                eprintln!(
                    "  plan r={radius:e} consistent={} verts={:?} pieces={:?} pairs={:?} sigmas={:?}",
                    plan.consistent,
                    plan.vs,
                    plan.es,
                    plan.pairs,
                    plan.passes.iter().map(|p| p.sigma).collect::<Vec<_>>()
                );
            }
            if plan.consistent || radius >= EPS_P_MAX {
                break plan;
            }
            radius *= 10.0;
            prev = Some(plan);
        };
        if plan.is_noop() {
            return false;
        }
        let Plan {
            st,
            vs,
            es,
            mut passes,
            pairs,
            dead_verts,
            dead_pieces,
            ..
        } = plan;
        let node = if self.node_override != NONE {
            self.node_override
        } else {
            self.node_at(p, t)
        };
        if trace() {
            eprintln!(
                "surgery t={t:.12} p=({:.12},{:.12}) node={node} verts={:?} pieces={:?}",
                p[0],
                p[1],
                vs.iter()
                    .map(|&v| (
                        v,
                        self.pieces[self.verts[v as usize].in_p as usize].line,
                        self.pieces[self.verts[v as usize].out_p as usize].line,
                        self.verts[v as usize].sigma
                    ))
                    .collect::<Vec<_>>(),
                es.iter()
                    .map(|&e| (e, self.pieces[e as usize].line))
                    .collect::<Vec<_>>()
            );
        }

        // 5. apply: split pieces, remove what collapsed, create the new vertices
        for ps in passes.iter_mut() {
            if ps.split {
                let e = ps.in_p;
                let w = self.pieces[e as usize].end;
                let line = self.pieces[e as usize].line;
                let e2p = self.new_piece(line, NONE, w);
                self.verts[w as usize].in_p = e2p;
                ps.out_p = e2p;
            }
        }
        for &v in &dead_verts {
            self.kill_vert(v, node);
        }
        for &pc in &dead_pieces {
            self.kill_piece(pc);
        }
        self.new_verts.clear();
        self.modified.clear();
        for &(ia, ob, tied) in &pairs {
            let a = &passes[ia];
            let b = &passes[ob];
            let (in_p, out_p) = (a.in_p, b.out_p);
            let d_in = self.line_of(in_p).d;
            let d_out = self.line_of(out_p).d;
            let geo = ccw_angle(d_out, neg(d_in));
            let sigma = if tied {
                if geo < PI {
                    geo
                } else {
                    0.0
                }
            } else if ia != ob && geo > a.sigma.min(b.sigma) + 1e-9 && TAU - geo < SLIVER {
                // an inverted sliver: the two pieces already overlap by a hair (an event next to
                // this one was resolved a rounding error late); close it like a zero-width sector
                0.0
            } else if ia == ob {
                let mut sg = geo;
                for cand in [geo - TAU, geo + TAU] {
                    if (cand - a.sigma).abs() < (sg - a.sigma).abs() {
                        sg = cand;
                    }
                }
                sg
            } else {
                geo
            };
            let nv = self.push_vert(p, t, in_p, out_p, node, sigma, true);
            if trace() {
                eprintln!(
                    "   new v{nv}: in line {} out line {} sigma {sigma:.6}",
                    self.pieces[in_p as usize].line, self.pieces[out_p as usize].line
                );
            }
            self.new_verts.push(nv);
            for pc in [in_p, out_p] {
                if self.pieces[pc as usize].modified != st {
                    self.pieces[pc as usize].modified = st;
                    self.modified.push(pc);
                }
            }
        }
        // 6. new events
        let new_verts = std::mem::take(&mut self.new_verts);
        let modified = std::mem::take(&mut self.modified);
        for &nv in &new_verts {
            let vv = &self.verts[nv as usize];
            if vv.zip {
                self.zips.push(nv);
            }
        }
        for &pc in &modified {
            self.update_edge_event(pc);
        }
        for &nv in &new_verts {
            if self.verts[nv as usize].reflex {
                self.rescan(nv);
            }
        }
        for &pc in &modified {
            self.recheck(pc);
        }
        self.new_verts = new_verts;
        self.modified = modified;
        true
    }

    /// Slide a zero-width vertex along its overlapping pieces to the end of the shorter one.
    fn zip(&mut self, z: u32, t: f64) {
        let (zin, zout, p) = {
            let zv = &self.verts[z as usize];
            (zv.in_p, zv.out_p, zv.pos)
        };
        let a0 = self.pieces[zin as usize].start;
        let b1 = self.pieces[zout as usize].end;
        let u = self.line_of(zout).d;
        let pa = self.pos(a0, t);
        let pb = self.pos(b1, t);
        let la = dot(u, sub(pa, p));
        let lb = dot(u, sub(pb, p));
        let (q, forced): (V2, [u32; 3]) = if a0 == b1 {
            // A two-vertex cycle. If the other vertex is zero-width too, the two pieces lie on top
            // of each other and the cycle is a segment: close it at the far end. Otherwise both
            // vertices sit where the same two lines cross, which is here; the other one is a
            // sliver tip whose extrapolated position is not trustworthy.
            if self.verts[a0 as usize].zip {
                (pa, [z, a0, NONE])
            } else {
                (p, [z, a0, NONE])
            }
        } else if (la - lb).abs() <= EPS_P {
            (mul(add(pa, pb), 0.5), [z, a0, b1])
        } else if la > lb {
            (pb, [z, b1, NONE])
        } else {
            (pa, [z, a0, NONE])
        };
        {
            let zv = &mut self.verts[z as usize];
            zv.pos = q;
            zv.t0 = t;
            zv.vel = [0.0, 0.0];
            let ai = zv.alive_idx as usize;
            self.alive_kin[ai] = [q[0], q[1], 0.0, 0.0, t];
        }
        let n = if forced[2] == NONE { 2 } else { 3 };
        let f = forced;
        let _ = self.surgery(t, q, &f[..n], NONE);
    }

    fn drain_zips(&mut self) {
        while let Some(z) = self.zips.pop() {
            if self.verts[z as usize].alive {
                let t = self.now;
                self.zip(z, t);
            }
        }
    }

    /// Run the propagation. `touch_groups` lists input vertices at the same position (rings
    /// touching at a vertex); they are reconnected at time 0, at the node of the group's first vertex.
    pub(crate) fn run(mut self, touch_groups: &[Vec<usize>]) -> Result<Output, EngineError> {
        for g in touch_groups {
            let ids: Vec<u32> = g.iter().map(|&v| v as u32).collect();
            let canon = ids[0];
            let p = self.verts[canon as usize].pos;
            self.node_override = canon;
            let _ = self.surgery(0.0, p, &ids, NONE);
            self.node_override = NONE;
        }
        // initial events
        for k in 0..self.alive_pieces.len() {
            let p = self.alive_pieces[k];
            self.update_edge_event(p);
        }
        for k in 0..self.reflex.len() {
            let r = self.reflex[k];
            self.rescan(r);
        }
        for k in 0..self.alive_verts.len() {
            let v = self.alive_verts[k];
            if self.verts[v as usize].zip {
                self.zips.push(v);
            }
        }
        self.drain_zips();
        let budget = 200 * self.n_input as u64 + 10_000;
        let mut iters: u64 = 0;
        while let Some((t, slot)) = self.queue.pop() {
            iters += 1;
            if iters > budget {
                return Err(EngineError::Runaway);
            }
            let t = t.max(self.now);
            self.now = t;
            if trace() {
                eprintln!(
                    "pop t={t:.12} {} {}",
                    if slot & 1 == 0 { "edge" } else { "split" },
                    slot >> 1
                );
            }
            if slot & 1 == 0 {
                let p = slot >> 1;
                let (u, w) = (self.pieces[p as usize].start, self.pieces[p as usize].end);
                let (q, _) = self.edge_collapse(p, t);
                let _ = self.surgery(t, q, &[u, w], NONE);
            } else {
                let r = slot >> 1;
                let line = self.verts[r as usize].split_line;
                self.verts[r as usize].split_time = f64::INFINITY;
                let ri = self.verts[r as usize].reflex_idx;
                if ri != NONE {
                    self.reflex_best[ri as usize] = f64::INFINITY;
                }
                let (q, _) = self.split_point(r, line, t);
                let hit = self.find_on_line(line, r, q, t);
                if trace() {
                    eprintln!(
                        "  split v{r} on line {line} at ({:.12},{:.12}): {}",
                        q[0],
                        q[1],
                        match hit {
                            Hit::Interior(p) => format!("interior of piece {p}"),
                            Hit::Vertex(v) => format!("vertex v{v}"),
                            Hit::Nothing => "nothing".to_string(),
                        }
                    );
                }
                let changed = match hit {
                    Hit::Interior(pid) => self.surgery(t, q, &[r], pid),
                    Hit::Vertex(v) => self.surgery(t, q, &[r, v], NONE),
                    Hit::Nothing => false,
                };
                if !changed && self.verts[r as usize].alive {
                    self.rescan(r);
                }
            }
            self.drain_zips();
        }
        if !self.alive_verts.is_empty() {
            if trace() {
                for &v in &self.alive_verts {
                    let vv = &self.verts[v as usize];
                    eprintln!(
                        "alive v{v}: lines {}->{} sigma {} pos {:?} t0 {} vel {:?} prev v{} next v{}",
                        self.pieces[vv.in_p as usize].line,
                        self.pieces[vv.out_p as usize].line,
                        vv.sigma,
                        vv.pos,
                        vv.t0,
                        vv.vel,
                        self.pieces[vv.in_p as usize].start,
                        self.pieces[vv.out_p as usize].end
                    );
                }
            }
            return Err(EngineError::Unfinished);
        }
        let _ = self.queue.len();
        Ok(Output {
            nodes: self.nodes,
            arcs: self.arcs,
        })
    }
}
