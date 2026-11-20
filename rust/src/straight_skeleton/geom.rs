//! Small 2D vector helpers.

pub(crate) type V2 = [f64; 2];

pub(crate) const PI: f64 = std::f64::consts::PI;
pub(crate) const TAU: f64 = std::f64::consts::TAU;

#[inline]
pub(crate) fn sub(a: V2, b: V2) -> V2 {
    [a[0] - b[0], a[1] - b[1]]
}

#[inline]
pub(crate) fn add(a: V2, b: V2) -> V2 {
    [a[0] + b[0], a[1] + b[1]]
}

#[inline]
pub(crate) fn mul(a: V2, s: f64) -> V2 {
    [a[0] * s, a[1] * s]
}

#[inline]
pub(crate) fn dot(a: V2, b: V2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

#[inline]
pub(crate) fn cross(a: V2, b: V2) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

#[inline]
pub(crate) fn norm(a: V2) -> f64 {
    a[0].hypot(a[1])
}

#[inline]
pub(crate) fn dist2(a: V2, b: V2) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

#[inline]
pub(crate) fn neg(a: V2) -> V2 {
    [-a[0], -a[1]]
}

/// Rotate by `ang` radians counter-clockwise.
#[inline]
pub(crate) fn rot(a: V2, ang: f64) -> V2 {
    let (s, c) = ang.sin_cos();
    [a[0] * c - a[1] * s, a[0] * s + a[1] * c]
}

/// Angle of `a` in `[0, 2π)`.
#[inline]
pub(crate) fn angle(a: V2) -> f64 {
    let t = a[1].atan2(a[0]);
    if t < 0.0 {
        t + TAU
    } else {
        t
    }
}

/// Counter-clockwise angle from `a` to `b`, in `[0, 2π)`.
#[inline]
pub(crate) fn ccw_angle(a: V2, b: V2) -> f64 {
    let t = cross(a, b).atan2(dot(a, b));
    if t < 0.0 {
        t + TAU
    } else {
        t
    }
}

/// Twice the signed area of a ring (positive for counter-clockwise).
pub(crate) fn ring_area2(pts: &[V2]) -> f64 {
    let n = pts.len();
    let mut a = 0.0;
    for i in 0..n {
        let p = pts[i];
        let q = pts[(i + 1) % n];
        a += cross(p, q);
    }
    a
}
