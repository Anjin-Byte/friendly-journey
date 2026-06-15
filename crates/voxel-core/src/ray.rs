//! Rays and an independent ray–AABB slab test.
//!
//! The canonical [`Ray`] is `f64` — it is "the truth" against which both the
//! `f32` GPU-mirror traversal and the GPU kernel are measured (adversarial
//! review R1, the tiered oracle). [`ray_aabb`] is an independent
//! intersection primitive used to grid-clip the DDA and, in tests, as the
//! sharp oracle that the incremental traversal is validated against.

use glam::DVec3;

/// A ray with an `f64` origin and direction.
///
/// The direction need not be normalized; traversal parameters `t` are measured
/// in units of `‖dir‖`. The hit *voxel* is invariant to the direction's
/// magnitude, which is what the differential tests compare.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    /// Ray origin.
    pub origin: DVec3,
    /// Ray direction (not required to be unit length).
    pub dir: DVec3,
}

impl Ray {
    /// Constructs a ray from an origin and direction.
    #[must_use]
    pub const fn new(origin: DVec3, dir: DVec3) -> Self {
        Self { origin, dir }
    }

    /// The point `origin + t · dir`.
    #[must_use]
    pub fn at(&self, t: f64) -> DVec3 {
        self.origin + self.dir * t
    }
}

/// Intersects the ray `origin + t·dir` with the axis-aligned box `[lo, hi]`
/// using the slab method, returning the entry/exit parameters `(t_near, t_far)`
/// with `t_near ≤ t_far`, or `None` if the ray misses the box.
///
/// Handles a zero direction component (a ray parallel to a slab): the ray hits
/// only if its origin already lies within that slab. The returned interval can
/// have a negative `t_near` (box surrounds or is behind the origin); callers
/// that want forward hits check `t_far >= 0`.
///
/// ```
/// use glam::DVec3;
/// use voxel_core::ray_aabb;
/// let hit = ray_aabb(DVec3::new(-1.0, 0.5, 0.5), DVec3::X, DVec3::ZERO, DVec3::ONE);
/// let (near, far) = hit.unwrap();
/// assert!((near - 1.0).abs() < 1e-12 && (far - 2.0).abs() < 1e-12);
/// ```
#[must_use]
pub fn ray_aabb(origin: DVec3, dir: DVec3, lo: DVec3, hi: DVec3) -> Option<(f64, f64)> {
    let o = origin.to_array();
    let d = dir.to_array();
    let lo = lo.to_array();
    let hi = hi.to_array();

    let mut t_near = f64::NEG_INFINITY;
    let mut t_far = f64::INFINITY;

    for axis in 0..3 {
        if d[axis] == 0.0 {
            // Parallel to this slab: miss unless the origin is inside it.
            if o[axis] < lo[axis] || o[axis] > hi[axis] {
                return None;
            }
        } else {
            let inv = 1.0 / d[axis];
            let mut t1 = (lo[axis] - o[axis]) * inv;
            let mut t2 = (hi[axis] - o[axis]) * inv;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            t_near = t_near.max(t1);
            t_far = t_far.min(t2);
            if t_near > t_far {
                return None;
            }
        }
    }
    Some((t_near, t_far))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_through_unit_box() {
        let (near, far) = ray_aabb(
            DVec3::new(-1.0, 0.5, 0.5),
            DVec3::X,
            DVec3::ZERO,
            DVec3::ONE,
        )
        .unwrap();
        assert!((near - 1.0).abs() < 1e-12);
        assert!((far - 2.0).abs() < 1e-12);
    }

    #[test]
    fn parallel_and_outside_misses() {
        // Parallel to the x slab, but y is above the box.
        assert!(ray_aabb(DVec3::new(0.5, 2.0, 0.5), DVec3::X, DVec3::ZERO, DVec3::ONE).is_none());
    }

    #[test]
    fn origin_inside_gives_negative_near() {
        let (near, far) = ray_aabb(DVec3::splat(0.5), DVec3::X, DVec3::ZERO, DVec3::ONE).unwrap();
        assert!(near < 0.0 && far > 0.0);
    }

    #[test]
    fn miss_to_the_side() {
        assert!(
            ray_aabb(
                DVec3::new(-1.0, 5.0, 5.0),
                DVec3::X,
                DVec3::ZERO,
                DVec3::ONE
            )
            .is_none()
        );
    }
}
