//! Tier-A reference traversal: a dense `f64` Amanatides–Woo grid march.
//!
//! This is "the truth" of the tiered oracle (adversarial review R1): an
//! obviously-correct single-level DDA in `f64`, used to validate every other
//! traversal in the workspace. It is itself validated, in tests, against an
//! independent brute-force ray–AABB-per-voxel oracle ([`ray_aabb`]) — a
//! *sharp oracle* with no shared code path (Engineering Codex: *Sharp Oracles*).
//!
//! It implements `idea.md` §7 at a single (voxel) level: no hierarchy, no
//! skipping — that is what P2+ add and diff against this.

use glam::DVec3;

use crate::ray::{Ray, ray_aabb};
use crate::{OccupancyField, VoxelCoord};

/// The result of a traversal: the first occupied voxel the ray enters, with the
/// ray parameter at which it was entered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// The first occupied voxel along the ray.
    pub voxel: VoxelCoord,
    /// Ray parameter `t` at which the ray enters `voxel` (units of `‖dir‖`).
    pub t_enter: f64,
}

/// Truncates a world coordinate to a voxel index, clamped into `[0, n)`.
///
/// The single audited `f64 → u32` conversion in the traversal: the argument is
/// clamped non-negative and below `n ≤ 2³¹` before the cast, so neither
/// truncation nor sign loss can discard information.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn floor_to_index(v: f64, n: u32) -> u32 {
    let max = f64::from(n - 1);
    v.floor().clamp(0.0, max) as u32
}

/// Marches `ray` through `field` and returns the first occupied voxel, or
/// `None` for a miss.
///
/// The grid occupies world box `[0, n]³` (one world unit per voxel). Ties in
/// the DDA step (two `t_max` equal) break toward the lowest axis index — a
/// fixed convention the `f32` mirror and GPU kernel must copy so the
/// differential is exact off the grazing set.
#[must_use]
pub fn first_hit<F: OccupancyField>(field: &F, ray: &Ray) -> Option<Hit> {
    let n = field.resolution().voxels_per_axis();
    let n_world = f64::from(n);

    // Grid-clip: find where the ray meets the grid AABB.
    let (t0, t1) = ray_aabb(ray.origin, ray.dir, DVec3::ZERO, DVec3::splat(n_world))?;
    if t1 < 0.0 {
        return None; // grid entirely behind the origin
    }
    let t_entry = t0.max(0.0);

    let o = ray.origin.to_array();
    let d = ray.dir.to_array();
    let entry = [
        o[0] + t_entry * d[0],
        o[1] + t_entry * d[1],
        o[2] + t_entry * d[2],
    ];

    let mut voxel = [
        floor_to_index(entry[0], n),
        floor_to_index(entry[1], n),
        floor_to_index(entry[2], n),
    ];

    // Per-axis DDA setup (A&W, voxel size 1).
    let mut step = [0i64; 3];
    let mut t_max = [f64::INFINITY; 3];
    let mut t_delta = [f64::INFINITY; 3];
    for a in 0..3 {
        if d[a] > 0.0 {
            step[a] = 1;
            let next = f64::from(voxel[a] + 1);
            t_max[a] = t_entry + (next - entry[a]) / d[a];
            t_delta[a] = 1.0 / d[a];
        } else if d[a] < 0.0 {
            step[a] = -1;
            let next = f64::from(voxel[a]);
            t_max[a] = t_entry + (next - entry[a]) / d[a];
            t_delta[a] = -1.0 / d[a];
        } // else: parallel, t_max/t_delta stay +inf, step 0.
    }

    let mut t_current = t_entry;
    loop {
        let c = VoxelCoord::new(voxel[0], voxel[1], voxel[2]);
        if field.is_occupied(c) {
            return Some(Hit {
                voxel: c,
                t_enter: t_current,
            });
        }

        // Step along the axis with the smallest t_max (lowest index on ties).
        let mut axis = 0;
        if t_max[1] < t_max[axis] {
            axis = 1;
        }
        if t_max[2] < t_max[axis] {
            axis = 2;
        }
        if !t_max[axis].is_finite() {
            return None; // all remaining axes parallel; ray leaves through none
        }

        // A finite t_max means a nonzero direction on this axis, hence a ±1
        // step. Advance in u32 and bounds-check without any cast.
        match step[axis] {
            1 => {
                if voxel[axis] + 1 >= n {
                    return None; // stepped out the far face
                }
                voxel[axis] += 1;
            }
            -1 => {
                if voxel[axis] == 0 {
                    return None; // stepped out the near face
                }
                voxel[axis] -= 1;
            }
            _ => unreachable!("a finite-t_max axis always has a ±1 step"),
        }
        t_current = t_max[axis];
        t_max[axis] += t_delta[axis];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Resolution;
    use crate::fixtures::{Checkerboard, Empty, OctantFractal, SingleVoxel, Solid};
    use proptest::prelude::*;

    fn res(n: u32) -> Resolution {
        Resolution::new(n).unwrap()
    }

    /// Independent sharp oracle: the first occupied voxel by ray parameter,
    /// found by testing the ray against every occupied voxel's AABB. `O(n³)`,
    /// no shared code with the DDA — small grids only.
    fn brute_force_first_hit<F: OccupancyField>(field: &F, ray: &Ray) -> Option<Hit> {
        let n = field.resolution().voxels_per_axis();
        let mut best: Option<Hit> = None;
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let c = VoxelCoord::new(x, y, z);
                    if !field.is_occupied(c) {
                        continue;
                    }
                    let lo = DVec3::new(f64::from(x), f64::from(y), f64::from(z));
                    let hi = lo + DVec3::ONE;
                    if let Some((tn, tf)) = ray_aabb(ray.origin, ray.dir, lo, hi) {
                        if tf >= 0.0 {
                            let t = tn.max(0.0);
                            if best.is_none_or(|b| t < b.t_enter) {
                                best = Some(Hit {
                                    voxel: c,
                                    t_enter: t,
                                });
                            }
                        }
                    }
                }
            }
        }
        best
    }

    #[test]
    fn axis_aligned_hit_is_exact() {
        // Ray down +x at the center of row (·, 0, 0) hits the single voxel (5,0,0).
        let f = SingleVoxel {
            resolution: res(8),
            voxel: VoxelCoord::new(5, 0, 0),
        };
        let ray = Ray::new(DVec3::new(-1.0, 0.5, 0.5), DVec3::X);
        let hit = first_hit(&f, &ray).unwrap();
        assert_eq!(hit.voxel, VoxelCoord::new(5, 0, 0));
        assert!((hit.t_enter - 6.0).abs() < 1e-12); // enters x=5 at t=6 from x=-1
    }

    #[test]
    fn empty_field_always_misses() {
        let f = Empty {
            resolution: res(32),
        };
        let ray = Ray::new(DVec3::new(-1.0, 8.0, 8.0), DVec3::X);
        assert!(first_hit(&f, &ray).is_none());
    }

    #[test]
    fn solid_field_hits_entry_voxel() {
        let f = Solid {
            resolution: res(32),
        };
        let ray = Ray::new(DVec3::new(-5.0, 4.5, 4.5), DVec3::X);
        let hit = first_hit(&f, &ray).unwrap();
        assert_eq!(hit.voxel, VoxelCoord::new(0, 4, 4));
    }

    #[test]
    fn ray_pointing_away_misses() {
        let f = Solid { resolution: res(8) };
        let ray = Ray::new(DVec3::new(-1.0, 4.0, 4.0), DVec3::NEG_X);
        assert!(first_hit(&f, &ray).is_none());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(2000))]

        // Tier-A agrees with the independent brute-force slab oracle on
        // hit/miss and entry distance, over random rays and fixtures. The hit
        // *voxel* can differ by one at exact corner grazes (a measure-zero set
        // where both answers are defensible), so we assert on the robust
        // invariants: same hit/miss, and entry `t` within tolerance.
        #[test]
        fn dda_matches_brute_force_slab(
            ox in -4.0f64..12.0, oy in -4.0f64..12.0, oz in -4.0f64..12.0,
            dx in -1.0f64..1.0, dy in -1.0f64..1.0, dz in -1.0f64..1.0,
            fixture in 0u8..3,
        ) {
            // Skip near-zero directions (degenerate ray).
            let dir = DVec3::new(dx, dy, dz);
            prop_assume!(dir.length() > 1e-3);
            let ray = Ray::new(DVec3::new(ox, oy, oz), dir);
            let r = res(8);

            macro_rules! check {
                ($field:expr) => {{
                    let dda = first_hit(&$field, &ray);
                    let bf = brute_force_first_hit(&$field, &ray);
                    prop_assert_eq!(dda.is_some(), bf.is_some());
                    if let (Some(a), Some(b)) = (dda, bf) {
                        prop_assert!((a.t_enter - b.t_enter).abs() < 1e-6,
                            "t mismatch: dda={} bf={} dir={:?}", a.t_enter, b.t_enter, dir);
                    }
                }};
            }
            match fixture {
                0 => check!(Checkerboard { resolution: r }),
                1 => check!(OctantFractal::sierpinski_tetrahedron(r)),
                _ => check!(OctantFractal::cantor_dust(r)),
            }
        }
    }
}
