//! P2: a two-level MIP traversal — a coarse brick skip level over `8³` bitmask
//! leaves (`idea.md` §11.2).
//!
//! This is the first hierarchical step: a *dense* coarse grid stores, per `8³`
//! brick, a non-empty flag (the skip signal) and the brick's 512-bit leaf. The
//! [`first_hit`] march runs Amanatides–Woo at the brick level, skipping empty
//! bricks in one step, and on an occupied brick descends to a fresh voxel-level
//! walk inside it. It exists to validate the **skip** mechanic and the
//! **descend recompute** (§7.3) in isolation, diffed against the Tier-A oracle,
//! before sparsity (P3) and the full School-B layout (P4). It is dense, so it is
//! for small/medium resolutions only.

use glam::DVec3;

use crate::dda::DdaWalker;
use crate::leaf::LeafBrick;
use crate::oracle::Hit;
use crate::ray::{Ray, ray_aabb};
use crate::{OccupancyField, Resolution, VoxelCoord};

/// A dense two-level structure: a non-empty bit and an `8³` leaf per brick.
#[derive(Debug, Clone)]
pub struct BrickGrid {
    resolution: Resolution,
    /// Bricks per axis (`n / 8`).
    bricks_per_axis: u32,
    /// One bit per brick: set iff the brick has any occupied voxel.
    coarse: Vec<u64>,
    /// One leaf per brick, indexed `bx + by·bpa + bz·bpa²`.
    leaves: Vec<LeafBrick>,
}

impl BrickGrid {
    /// Builds the dense two-level grid from an occupancy field.
    ///
    /// # Panics
    /// Panics if `bricks³` exceeds addressable memory. This is dense; for large
    /// resolutions use the sparse builder (P3+).
    #[must_use]
    pub fn from_field<F: OccupancyField>(field: &F) -> Self {
        let resolution = field.resolution();
        let bpa = resolution.voxels_per_axis() / 8;
        let brick_count = usize::try_from(u64::from(bpa).pow(3))
            .expect("dense BrickGrid too large for this platform; use the sparse builder");

        let mut leaves = vec![LeafBrick::EMPTY; brick_count];
        let mut coarse = vec![0u64; brick_count.div_ceil(64)];

        for bz in 0..bpa {
            for by in 0..bpa {
                for bx in 0..bpa {
                    let mut leaf = LeafBrick::EMPTY;
                    for lz in 0..8 {
                        for ly in 0..8 {
                            for lx in 0..8 {
                                let c = VoxelCoord::new(bx * 8 + lx, by * 8 + ly, bz * 8 + lz);
                                if field.is_occupied(c) {
                                    leaf.set_local(lx, ly, lz);
                                }
                            }
                        }
                    }
                    let idx = Self::brick_index(bpa, [bx, by, bz]);
                    if !leaf.is_empty() {
                        coarse[idx / 64] |= 1u64 << (idx % 64);
                    }
                    leaves[idx] = leaf;
                }
            }
        }

        Self {
            resolution,
            bricks_per_axis: bpa,
            coarse,
            leaves,
        }
    }

    /// The grid resolution.
    #[must_use]
    pub fn resolution(&self) -> Resolution {
        self.resolution
    }

    /// Bricks per axis (`n / 8`).
    #[must_use]
    pub fn bricks_per_axis(&self) -> u32 {
        self.bricks_per_axis
    }

    fn brick_index(bpa: u32, b: [u32; 3]) -> usize {
        let bpa = bpa as usize;
        b[0] as usize + b[1] as usize * bpa + b[2] as usize * bpa * bpa
    }

    /// Whether brick `b` has any occupied voxel (the coarse skip test). `false`
    /// for out-of-range brick coordinates.
    #[must_use]
    pub fn is_brick_occupied(&self, b: [u32; 3]) -> bool {
        if b[0] >= self.bricks_per_axis
            || b[1] >= self.bricks_per_axis
            || b[2] >= self.bricks_per_axis
        {
            return false;
        }
        let idx = Self::brick_index(self.bricks_per_axis, b);
        (self.coarse[idx / 64] >> (idx % 64)) & 1 == 1
    }

    /// The leaf brick at `b`.
    fn leaf(&self, b: [u32; 3]) -> &LeafBrick {
        &self.leaves[Self::brick_index(self.bricks_per_axis, b)]
    }
}

/// Marches `ray` through the two-level grid, returning the first occupied voxel
/// or `None`. The result is identical to the Tier-A oracle on the same field;
/// that equivalence is what the differential test checks.
#[must_use]
pub fn first_hit(grid: &BrickGrid, ray: &Ray) -> Option<Hit> {
    let n_world = f64::from(grid.resolution.voxels_per_axis());
    let (t0, t1) = ray_aabb(ray.origin, ray.dir, DVec3::ZERO, DVec3::splat(n_world))?;
    if t1 < 0.0 {
        return None;
    }
    let t_entry = t0.max(0.0);

    // Coarse level: walk bricks of edge 8.
    let mut brick = DdaWalker::enter(ray, [0.0; 3], grid.bricks_per_axis, 8.0, t_entry);
    loop {
        let b = brick.cell();
        if grid.is_brick_occupied(b) {
            // Descend: a fresh voxel walk whose t_max is recomputed from the
            // brick entry point (the §7.3 recompute), not inherited.
            let leaf = grid.leaf(b);
            let origin = [
                f64::from(b[0]) * 8.0,
                f64::from(b[1]) * 8.0,
                f64::from(b[2]) * 8.0,
            ];
            let mut voxel = DdaWalker::enter(ray, origin, 8, 1.0, brick.t_entry());
            loop {
                let lv = voxel.cell();
                if leaf.get_local(lv[0], lv[1], lv[2]) {
                    return Some(Hit {
                        voxel: VoxelCoord::new(
                            b[0] * 8 + lv[0],
                            b[1] * 8 + lv[1],
                            b[2] * 8 + lv[2],
                        ),
                        t_enter: voxel.t_entry(),
                    });
                }
                if !voxel.step() {
                    break; // exited the brick with no hit — ascend
                }
            }
        }
        if !brick.step() {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{Checkerboard, OctantFractal, SingleVoxel};
    use crate::oracle;

    fn res(n: u32) -> Resolution {
        Resolution::new(n).unwrap()
    }

    /// Deterministic splitmix64, so the differential rays are reproducible.
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[allow(clippy::cast_precision_loss)]
    fn unit(state: &mut u64) -> f64 {
        // 53-bit mantissa worth of randomness in [0, 1).
        (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn materialization_matches_field() {
        let field = OctantFractal::sierpinski_tetrahedron(res(128));
        let grid = BrickGrid::from_field(&field);
        // Every occupied voxel implies its brick is flagged occupied.
        let n = field.resolution().voxels_per_axis();
        for z in (0..n).step_by(3) {
            for y in (0..n).step_by(3) {
                for x in (0..n).step_by(3) {
                    let c = VoxelCoord::new(x, y, z);
                    if field.is_occupied(c) {
                        assert!(grid.is_brick_occupied([x / 8, y / 8, z / 8]));
                    }
                }
            }
        }
    }

    #[test]
    fn exact_hit_through_single_voxel() {
        let field = SingleVoxel {
            resolution: res(32),
            voxel: VoxelCoord::new(20, 3, 3),
        };
        let grid = BrickGrid::from_field(&field);
        let ray = Ray::new(DVec3::new(-1.0, 3.5, 3.5), DVec3::X);
        let hit = first_hit(&grid, &ray).unwrap();
        assert_eq!(hit.voxel, VoxelCoord::new(20, 3, 3));
        assert!((hit.t_enter - 21.0).abs() < 1e-9);
    }

    #[test]
    fn two_level_matches_oracle_on_random_rays() {
        // Build each grid once; cast many deterministic rays; require the
        // two-level march to agree with the single-level oracle on hit/miss and
        // entry distance (the hit voxel may differ by one only at exact corner
        // grazes, where t still agrees within tolerance).
        let r = res(128);
        let nf = f64::from(r.voxels_per_axis());
        let checker = Checkerboard { resolution: r };
        let frac = OctantFractal::sierpinski_tetrahedron(r);
        let checker_grid = BrickGrid::from_field(&checker);
        let frac_grid = BrickGrid::from_field(&frac);

        let mut state = 0xC0FF_EE12_3456_789Au64;
        let mut compared = 0u32;
        for _ in 0..4000 {
            let origin = DVec3::new(
                unit(&mut state) * (nf + 8.0) - 4.0,
                unit(&mut state) * (nf + 8.0) - 4.0,
                unit(&mut state) * (nf + 8.0) - 4.0,
            );
            let dir = DVec3::new(
                unit(&mut state) * 2.0 - 1.0,
                unit(&mut state) * 2.0 - 1.0,
                unit(&mut state) * 2.0 - 1.0,
            );
            if dir.length() < 1e-3 {
                continue;
            }
            let ray = Ray::new(origin, dir);

            for (field_hit, grid) in [
                (oracle::first_hit(&checker, &ray), &checker_grid),
                (oracle::first_hit(&frac, &ray), &frac_grid),
            ] {
                let mip_hit = first_hit(grid, &ray);
                assert_eq!(
                    field_hit.is_some(),
                    mip_hit.is_some(),
                    "hit/miss disagreement, dir={dir:?}"
                );
                if let (Some(a), Some(b)) = (field_hit, mip_hit) {
                    assert!(
                        (a.t_enter - b.t_enter).abs() < 1e-6,
                        "t mismatch: oracle={} mip={} dir={dir:?}",
                        a.t_enter,
                        b.t_enter
                    );
                }
                compared += 1;
            }
        }
        assert!(compared > 1000, "too few comparisons: {compared}");
    }
}
