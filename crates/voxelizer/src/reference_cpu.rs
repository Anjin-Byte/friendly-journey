//! CPU reference voxelizer — the conservative-superset oracle for the GPU path
//! (bit-exact on tangent-free meshes; at f32 tangent voxels the GPU may over-mark
//! by one, never under-mark).
//!
//! [`voxelize_surface_cpu`] rasterizes a triangle mesh into a dense occupancy
//! grid using the Akenine-Möller triangle/box separating-axis-theorem (SAT)
//! overlap test (`triangle_box_overlap`). The differential tests in
//! `tests/differential.rs` compare this output against [`crate::gpu`] to guard
//! the WGSL implementation.

use glam::Vec3;

use crate::core::{
    DispatchStats, MeshInput, TileSpec, VoxelGrid, VoxelOccupancy, VoxelizationOutput, VoxelizeOpts,
};

/// Separating-axis-theorem test for triangle/axis-aligned-box overlap.
///
/// Returns `true` if the triangle `(v0, v1, v2)` overlaps the box centered at
/// `box_center` with half-extents `box_half`. Tests the 13 SAT axes: the 9
/// edge-cross-axis products, the 3 box face normals, and the triangle normal.
///
/// Crate-internal: the SAT overlap test the reference voxelizer and its
/// differential tests share. (Not part of the public API — the bake uses
/// nearest-surface owners per design D2, not this overlap.)
#[must_use]
pub(crate) fn triangle_box_overlap(
    box_center: Vec3,
    box_half: Vec3,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
) -> bool {
    let v0 = v0 - box_center;
    let v1 = v1 - box_center;
    let v2 = v2 - box_center;

    let e0 = v1 - v0;
    let e1 = v2 - v1;
    let e2 = v0 - v2;

    let axes = [
        Vec3::new(0.0, -e0.z, e0.y),
        Vec3::new(0.0, -e1.z, e1.y),
        Vec3::new(0.0, -e2.z, e2.y),
        Vec3::new(e0.z, 0.0, -e0.x),
        Vec3::new(e1.z, 0.0, -e1.x),
        Vec3::new(e2.z, 0.0, -e2.x),
        Vec3::new(-e0.y, e0.x, 0.0),
        Vec3::new(-e1.y, e1.x, 0.0),
        Vec3::new(-e2.y, e2.x, 0.0),
    ];

    for axis in &axes {
        let p0 = v0.dot(*axis);
        let p1 = v1.dot(*axis);
        let p2 = v2.dot(*axis);
        let min_p = p0.min(p1.min(p2));
        let max_p = p0.max(p1.max(p2));
        let r = box_half.x * axis.x.abs() + box_half.y * axis.y.abs() + box_half.z * axis.z.abs();
        if min_p > r || max_p < -r {
            return false;
        }
    }

    if v0.x.min(v1.x.min(v2.x)) > box_half.x
        || v0.x.max(v1.x.max(v2.x)) < -box_half.x
        || v0.y.min(v1.y.min(v2.y)) > box_half.y
        || v0.y.max(v1.y.max(v2.y)) < -box_half.y
        || v0.z.min(v1.z.min(v2.z)) > box_half.z
        || v0.z.max(v1.z.max(v2.z)) < -box_half.z
    {
        return false;
    }

    let normal = e0.cross(e1);
    let d = -normal.dot(v0);
    let r = box_half.x * normal.x.abs() + box_half.y * normal.y.abs() + box_half.z * normal.z.abs();
    let s = normal.dot(Vec3::ZERO) + d;
    if s.abs() > r {
        return false;
    }

    true
}

/// Deterministically hashes an owner id into an opaque RGBA color.
///
/// Uses the Numerical-Recipes LCG constants to spread successive ids across the
/// color space; the alpha channel is fixed at `255`.
fn hash_color(id: u32) -> u32 {
    let mut x = id.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let r = (x & 0xff) as u8;
    x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let g = (x & 0xff) as u8;
    x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let b = (x & 0xff) as u8;
    u32::from_le_bytes([r, g, b, 255])
}

/// Voxelizes a triangle mesh into a dense occupancy grid on the CPU.
///
/// This is the reference oracle: for each triangle it scans the voxels in its
/// (epsilon-padded) grid-space AABB and marks every voxel whose unit cube
/// overlaps the triangle per `triangle_box_overlap`. When `opts.store_owner`
/// is set, each voxel records the lowest triangle index that covered it; when
/// `opts.store_color` is set, those owners are hashed into colors. `tiles` is
/// accepted for signature parity with the GPU path and only feeds the reported
/// tile count in [`DispatchStats`].
pub fn voxelize_surface_cpu(
    mesh: &MeshInput,
    grid: &VoxelGrid,
    tiles: &TileSpec,
    opts: &VoxelizeOpts,
) -> VoxelizationOutput {
    let dims = grid.dims();
    let num_voxels = (dims[0] as usize) * (dims[1] as usize) * (dims[2] as usize);
    let word_count = num_voxels.div_ceil(32);
    let mut occupancy = vec![0u32; word_count];
    let mut owner = if opts.store_owner {
        vec![u32::MAX; num_voxels]
    } else {
        Vec::new()
    };
    let mut color = if opts.store_color {
        vec![0u32; num_voxels]
    } else {
        Vec::new()
    };

    let to_grid = grid.world_to_grid_matrix();
    let half = Vec3::splat(0.5);

    for (tri_index, tri) in mesh.triangles.iter().enumerate() {
        let v0 = to_grid.transform_point3(tri[0]);
        let v1 = to_grid.transform_point3(tri[1]);
        let v2 = to_grid.transform_point3(tri[2]);

        let min_v = v0.min(v1).min(v2) - Vec3::splat(opts.epsilon);
        let max_v = v0.max(v1).max(v2) + Vec3::splat(opts.epsilon);
        let min = [
            min_v.x.floor().max(0.0) as i32,
            min_v.y.floor().max(0.0) as i32,
            min_v.z.floor().max(0.0) as i32,
        ];
        let max = [
            max_v.x.floor().min((dims[0] - 1) as f32) as i32,
            max_v.y.floor().min((dims[1] - 1) as f32) as i32,
            max_v.z.floor().min((dims[2] - 1) as f32) as i32,
        ];

        for z in min[2]..=max[2] {
            for y in min[1]..=max[1] {
                for x in min[0]..=max[0] {
                    let center = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                    if triangle_box_overlap(center, half, v0, v1, v2) {
                        let linear = (x as usize)
                            + (dims[0] as usize)
                                * ((y as usize) + (dims[1] as usize) * (z as usize));
                        let word = linear >> 5;
                        let bit = linear & 31;
                        occupancy[word] |= 1u32 << bit;
                        if opts.store_owner {
                            let owner_ref = &mut owner[linear];
                            let tri_u = tri_index as u32;
                            if tri_u < *owner_ref {
                                *owner_ref = tri_u;
                            }
                        }
                    }
                }
            }
        }
    }

    if opts.store_color {
        for (index, color_out) in color.iter_mut().enumerate() {
            let owner_id = if opts.store_owner {
                owner[index]
            } else {
                u32::MAX
            };
            if owner_id != u32::MAX {
                *color_out = hash_color(owner_id);
            }
        }
    }

    VoxelizationOutput {
        occupancy: VoxelOccupancy::from_words(grid.resolution, occupancy),
        owner_id: if opts.store_owner { Some(owner) } else { None },
        color_rgba: if opts.store_color { Some(color) } else { None },
        stats: DispatchStats {
            triangles: mesh.triangles.len() as u32,
            tiles: u32::try_from(tiles.num_tiles_total()).unwrap_or(u32::MAX),
            voxels: num_voxels as u64,
            gpu_time_ms: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::TileSpec;
    use proptest::prelude::*;

    #[test]
    fn cpu_voxelizes_single_triangle() {
        let grid = VoxelGrid::new(voxel_core::Resolution::new(8).unwrap(), Vec3::ZERO, 1.0);
        let tiles = TileSpec::new([2, 2, 2], grid.dims()).expect("tiles");
        let mesh = MeshInput {
            triangles: vec![[
                Vec3::new(0.1, 0.1, 0.1),
                Vec3::new(1.2, 0.1, 0.1),
                Vec3::new(0.1, 1.2, 0.1),
            ]],
            material_ids: None,
            uvs: None,
            appearance: None,
        };
        let output = voxelize_surface_cpu(&mesh, &grid, &tiles, &VoxelizeOpts::default());
        let occupied = output.occupancy.count_occupied() > 0;
        assert!(occupied, "expected at least one occupied voxel");
    }

    // --- Oracle self-validation: pin the SAT overlap test against analytic cases,
    // so the GPU differential cannot pass by CPU and WGSL being co-buggy. ---

    const HALF: Vec3 = Vec3::splat(0.5);

    #[test]
    fn sat_triangle_containing_box_center_overlaps() {
        // A triangle whose plane passes through the unit box at the origin.
        let c = Vec3::ZERO;
        assert!(triangle_box_overlap(
            c,
            HALF,
            Vec3::new(-2.0, -2.0, 0.0),
            Vec3::new(2.0, -2.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
        ));
    }

    #[test]
    fn sat_far_triangle_is_separated() {
        // All vertices well outside on +x → AABB separation, no overlap.
        let c = Vec3::ZERO;
        assert!(!triangle_box_overlap(
            c,
            HALF,
            Vec3::new(10.0, 10.0, 10.0),
            Vec3::new(11.0, 10.0, 10.0),
            Vec3::new(10.0, 11.0, 10.0),
        ));
    }

    #[test]
    fn sat_offset_plane_misses_box() {
        // Triangle lying entirely in the z = 2 plane: its AABB is separated from the
        // box in z, so it must not overlap the box centered at the origin.
        let c = Vec3::ZERO;
        assert!(!triangle_box_overlap(
            c,
            HALF,
            Vec3::new(-2.0, -2.0, 2.0),
            Vec3::new(2.0, -2.0, 2.0),
            Vec3::new(0.0, 2.0, 2.0),
        ));
    }

    #[test]
    fn sat_vertex_inside_box_overlaps() {
        // A tiny triangle with a vertex inside the box.
        let c = Vec3::ZERO;
        assert!(triangle_box_overlap(
            c,
            HALF,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(3.0, 0.1, 0.0),
            Vec3::new(0.1, 3.0, 0.0),
        ));
    }

    /// Independent transcription of the GPU `voxelize.wgsl` SAT — the
    /// `axis_test` / `plane_box_intersects` / `triangle_box_overlap` formulas and
    /// the 9 edge-cross axes, verbatim — fed the inputs the shader's host
    /// precompute supplies (world-space plane `normal`/`d` and the triangle AABB).
    /// Diffing it against the reference [`triangle_box_overlap`] pins the two SAT
    /// implementations in lockstep: an un-mirrored edit to the shader (a flipped
    /// axis, a `>` vs `>=`, a sign error) fails the parity test below, so the GPU
    /// differential cannot pass by the CPU and WGSL being co-buggy. Mirrors the
    /// `wgsl_rank` transcription guard in `leaf.rs`.
    // Single-char `a`/`b`/`c`/`d`/`r` mirror the WGSL shader names verbatim.
    #[allow(clippy::many_single_char_names)]
    fn wgsl_voxelize_sat(center: Vec3, half: Vec3, a: Vec3, b: Vec3, c: Vec3) -> bool {
        // Host precompute (per voxelize.wgsl's caller): plane + triangle AABB.
        let normal = (b - a).cross(c - b);
        let d = -normal.dot(a);
        let tri_min = a.min(b).min(c);
        let tri_max = a.max(b).max(c);

        // --- triangle_box_overlap body (verbatim from voxelize.wgsl) ---
        let (v0, v1, v2) = (a - center, b - center, c - center);
        let (e0, e1, e2) = (v1 - v0, v2 - v1, v0 - v2);
        let box_min = center - half;
        let box_max = center + half;
        if tri_min.x > box_max.x || tri_max.x < box_min.x {
            return false;
        }
        if tri_min.y > box_max.y || tri_max.y < box_min.y {
            return false;
        }
        if tri_min.z > box_max.z || tri_max.z < box_min.z {
            return false;
        }
        // plane_box_intersects(normal, d, center, half)
        let pr = half.x * normal.x.abs() + half.y * normal.y.abs() + half.z * normal.z.abs();
        if (normal.dot(center) + d).abs() > pr {
            return false;
        }
        // The 9 edge × box-axis cross products.
        let axes = [
            Vec3::new(0.0, -e0.z, e0.y),
            Vec3::new(0.0, -e1.z, e1.y),
            Vec3::new(0.0, -e2.z, e2.y),
            Vec3::new(e0.z, 0.0, -e0.x),
            Vec3::new(e1.z, 0.0, -e1.x),
            Vec3::new(e2.z, 0.0, -e2.x),
            Vec3::new(-e0.y, e0.x, 0.0),
            Vec3::new(-e1.y, e1.x, 0.0),
            Vec3::new(-e2.y, e2.x, 0.0),
        ];
        for axis in axes {
            let p0 = v0.dot(axis);
            let p1 = v1.dot(axis);
            let p2 = v2.dot(axis);
            let min_p = p0.min(p1).min(p2);
            let max_p = p0.max(p1).max(p2);
            let r = half.x * axis.x.abs() + half.y * axis.y.abs() + half.z * axis.z.abs();
            // WGSL axis_test returns `!(min_p > r || max_p < -r)`.
            if min_p > r || max_p < -r {
                return false;
            }
        }
        true
    }

    /// The transcribed GPU SAT must agree with the reference oracle over a fixed
    /// random corpus. A deterministic LCG (not proptest) avoids shrinking onto a
    /// measure-zero floating-point tangent where the two algebraically-equal but
    /// differently-rounded plane tests could disagree by an ULP.
    #[test]
    fn wgsl_voxelize_sat_matches_reference() {
        const N: u32 = 8192;
        let mut state: u32 = 0x9E37_79B9;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from(u16::try_from(state >> 16).unwrap()) / f32::from(u16::MAX) * 4.0 - 2.0 // [-2, 2]
        };
        let mut hits = 0u32;
        for _ in 0..N {
            let a = Vec3::new(next(), next(), next());
            let b = Vec3::new(next(), next(), next());
            let c = Vec3::new(next(), next(), next());
            let center = Vec3::new(next(), next(), next()) * 0.5; // [-1, 1]
            let half = Vec3::splat(0.3 + (next() + 2.0) / 4.0 * 0.5); // [0.3, 0.8]
            assert_eq!(
                wgsl_voxelize_sat(center, half, a, b, c),
                triangle_box_overlap(center, half, a, b, c),
                "WGSL SAT diverged from the CPU reference: tri {a:?} {b:?} {c:?}, box c={center:?} h={half:?}"
            );
            if triangle_box_overlap(center, half, a, b, c) {
                hits += 1;
            }
        }
        // The corpus must exercise BOTH outcomes or the parity check is vacuous.
        assert!(
            hits > 0 && hits < N,
            "corpus must contain both overlaps and misses (got {hits}/{N})"
        );
    }

    /// Brute-force oracle: marks every voxel of an `n³` grid whose unit cube the
    /// SAT says overlaps any triangle (grid space == world). The voxelizer's
    /// AABB-scan output must equal this exhaustive set.
    fn bruteforce_occupied_set(
        mesh: &MeshInput,
        n: u32,
    ) -> std::collections::BTreeSet<(u32, u32, u32)> {
        let mut set = std::collections::BTreeSet::new();
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let center = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                    if mesh
                        .triangles
                        .iter()
                        .any(|t| triangle_box_overlap(center, HALF, t[0], t[1], t[2]))
                    {
                        set.insert((x, y, z));
                    }
                }
            }
        }
        set
    }

    /// `epsilon` only pads the candidate-voxel AABB scan; it must never change the
    /// SAT-decided occupancy. A finite mesh voxelized at several `epsilon >= 0`
    /// values must yield bit-identical occupancy.
    #[test]
    fn epsilon_ge_zero_does_not_change_occupancy() {
        use voxel_core::Resolution;
        let grid = VoxelGrid::new(Resolution::new(8).unwrap(), Vec3::ZERO, 1.0);
        let tiles = TileSpec::new([2, 2, 2], grid.dims()).unwrap();
        let mesh = MeshInput {
            triangles: vec![
                [
                    Vec3::new(1.3, 1.1, 2.4),
                    Vec3::new(6.2, 2.7, 3.1),
                    Vec3::new(2.9, 6.4, 4.8),
                ],
                [
                    Vec3::new(0.6, 5.2, 1.7),
                    Vec3::new(5.1, 0.8, 6.3),
                    Vec3::new(3.3, 3.3, 3.3),
                ],
            ],
            material_ids: None,
            uvs: None,
            appearance: None,
        };
        let mut reference: Option<Vec<u32>> = None;
        for epsilon in [0.0_f32, 1e-4, 0.5, 2.0] {
            let opts = VoxelizeOpts {
                epsilon,
                store_owner: false,
                store_color: false,
            };
            let out = voxelize_surface_cpu(&mesh, &grid, &tiles, &opts);
            let words = out.occupancy.words().to_vec();
            match &reference {
                None => reference = Some(words),
                Some(r) => assert_eq!(
                    r, &words,
                    "epsilon {epsilon} changed occupancy (epsilon must only pad the AABB scan)"
                ),
            }
        }
    }

    /// The voxelizer's occupancy must equal the exhaustive brute-force SAT set
    /// over *every* voxel — catching MISSED voxels (the over-marking proptest
    /// above only catches spurious ones).
    #[test]
    fn occupancy_equals_bruteforce_sat() {
        use voxel_core::{OccupancyField, Resolution, VoxelCoord};
        let grid = VoxelGrid::new(Resolution::new(8).unwrap(), Vec3::ZERO, 1.0);
        let tiles = TileSpec::new([2, 2, 2], grid.dims()).unwrap();
        let meshes = [
            MeshInput {
                triangles: vec![[
                    Vec3::new(0.5, 0.5, 0.5),
                    Vec3::new(7.0, 1.0, 1.0),
                    Vec3::new(1.0, 7.0, 5.0),
                ]],
                material_ids: None,
                uvs: None,
                appearance: None,
            },
            MeshInput {
                triangles: vec![
                    [
                        Vec3::new(1.2, 1.2, 1.2),
                        Vec3::new(6.6, 2.1, 5.4),
                        Vec3::new(2.5, 6.3, 3.0),
                    ],
                    [
                        Vec3::new(4.0, 0.7, 6.8),
                        Vec3::new(0.9, 4.4, 2.2),
                        Vec3::new(6.1, 5.5, 1.3),
                    ],
                ],
                material_ids: None,
                uvs: None,
                appearance: None,
            },
        ];
        for mesh in &meshes {
            let out = voxelize_surface_cpu(mesh, &grid, &tiles, &VoxelizeOpts::default());
            let mut got = std::collections::BTreeSet::new();
            for z in 0..8u32 {
                for y in 0..8u32 {
                    for x in 0..8u32 {
                        if out.occupancy.is_occupied(VoxelCoord::new(x, y, z)) {
                            got.insert((x, y, z));
                        }
                    }
                }
            }
            let expected = bruteforce_occupied_set(mesh, 8);
            assert_eq!(
                got, expected,
                "voxelizer occupancy must equal the exhaustive SAT set (no missed/spurious voxels)"
            );
        }
    }

    /// With duplicate/overlapping triangles, each occupied voxel's owner must be
    /// the *lowest* covering triangle index; empty voxels keep `u32::MAX`.
    #[test]
    fn owner_is_lowest_triangle_index() {
        use voxel_core::Resolution;
        let grid = VoxelGrid::new(Resolution::new(8).unwrap(), Vec3::ZERO, 1.0);
        let tiles = TileSpec::new([2, 2, 2], grid.dims()).unwrap();
        // Three identical triangles: every voxel one covers, all three cover.
        let tri = [
            Vec3::new(0.5, 0.5, 1.5),
            Vec3::new(6.5, 1.0, 1.5),
            Vec3::new(1.0, 6.5, 1.5),
        ];
        let mesh = MeshInput {
            triangles: vec![tri, tri, tri],
            material_ids: None,
            uvs: None,
            appearance: None,
        };
        let opts = VoxelizeOpts {
            epsilon: 1e-4,
            store_owner: true,
            store_color: false,
        };
        let out = voxelize_surface_cpu(&mesh, &grid, &tiles, &opts);
        let owner = out.owner_id.expect("store_owner produces owner_id");
        let dims = grid.dims();
        let mut saw_owned = false;
        for (linear, &o) in owner.iter().enumerate() {
            let x = linear % dims[0] as usize;
            let y = (linear / dims[0] as usize) % dims[1] as usize;
            let z = linear / (dims[0] as usize * dims[1] as usize);
            let center = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
            let covered = triangle_box_overlap(center, HALF, tri[0], tri[1], tri[2]);
            if covered {
                assert_eq!(o, 0, "covered voxel must own the lowest (index 0) triangle");
                saw_owned = true;
            } else {
                assert_eq!(o, u32::MAX, "empty voxel must keep u32::MAX owner");
            }
        }
        assert!(saw_owned, "the test triangle must cover at least one voxel");
    }

    /// A triangle entirely in negative space, and one entirely beyond the grid,
    /// must each mark zero voxels (the AABB scan clamps to the grid).
    #[test]
    fn fully_outside_triangle_marks_nothing() {
        use voxel_core::Resolution;
        let grid = VoxelGrid::new(Resolution::new(8).unwrap(), Vec3::ZERO, 1.0);
        let tiles = TileSpec::new([2, 2, 2], grid.dims()).unwrap();
        let cases = [
            // Wholly negative.
            [
                Vec3::new(-10.0, -10.0, -10.0),
                Vec3::new(-8.0, -10.0, -10.0),
                Vec3::new(-10.0, -8.0, -10.0),
            ],
            // Wholly beyond the 8-voxel grid.
            [
                Vec3::new(20.0, 20.0, 20.0),
                Vec3::new(22.0, 20.0, 20.0),
                Vec3::new(20.0, 22.0, 20.0),
            ],
        ];
        for tri in cases {
            let mesh = MeshInput {
                triangles: vec![tri],
                material_ids: None,
                uvs: None,
                appearance: None,
            };
            let out = voxelize_surface_cpu(&mesh, &grid, &tiles, &VoxelizeOpts::default());
            assert_eq!(
                out.occupancy.count_occupied(),
                0,
                "a fully-outside triangle must mark nothing (tri {tri:?})"
            );
        }
    }

    proptest! {
        /// Every voxel the rasterizer marks occupied must genuinely overlap the
        /// triangle (re-verified by the SAT directly) — guards the AABB-scan bounds
        /// and bit packing against marking spurious voxels. Grid space == world here.
        #[test]
        fn occupied_voxels_actually_overlap(
            ax in 0.5f32..7.5, ay in 0.5f32..7.5, az in 0.5f32..7.5,
            bx in 0.5f32..7.5, by in 0.5f32..7.5, bz in 0.5f32..7.5,
            cx in 0.5f32..7.5, cy in 0.5f32..7.5, cz in 0.5f32..7.5,
        ) {
            use voxel_core::{OccupancyField, Resolution, VoxelCoord};
            let grid = VoxelGrid::new(Resolution::new(8).unwrap(), Vec3::ZERO, 1.0);
            let tiles = TileSpec::new([2, 2, 2], grid.dims()).unwrap();
            let v0 = Vec3::new(ax, ay, az);
            let v1 = Vec3::new(bx, by, bz);
            let v2 = Vec3::new(cx, cy, cz);
            let mesh = MeshInput {
                triangles: vec![[v0, v1, v2]],
                material_ids: None,
                uvs: None,
                appearance: None,
            };
            let out = voxelize_surface_cpu(&mesh, &grid, &tiles, &VoxelizeOpts::default());
            for z in 0..8u32 {
                for y in 0..8u32 {
                    for x in 0..8u32 {
                        if out.occupancy.is_occupied(VoxelCoord::new(x, y, z)) {
                            let center = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
                            prop_assert!(
                                triangle_box_overlap(center, HALF, v0, v1, v2),
                                "voxel ({x},{y},{z}) marked occupied but does not overlap the triangle"
                            );
                        }
                    }
                }
            }
        }
    }
}
