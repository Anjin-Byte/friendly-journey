//! Compressed sparse row (CSR) builders for triangle-to-cell assignment.
//! (Conservative Uniform-Grid Binning with CSR (CUGB-CSR))
//! This module precomputes spatial lookup tables that map grid partitions
//! (tiles or bricks) to candidate triangle indices. The resulting CSR
//! structures are used by voxelization paths to limit intersection work to
//! relevant triangles per partition.
//!
//! Two partitioning schemes are supported:
//! - [`TileTriangleCsr`]: regular tile grid from [`TileSpec`]
//! - [`BrickTriangleCsr`]: sparse brick set derived from triangle coverage
//!
//! Both outputs follow the standard CSR shape:
//! - `*_offsets.len() == cell_count + 1`
//! - for cell `i`, triangle range is `tri_indices[offsets[i]..offsets[i + 1]]`
//! - `offsets[0] == 0`
//! - `offsets` is monotonic non-decreasing
//! - `offsets.last() == tri_indices.len()`

use glam::Vec3;

use crate::core::{MeshInput, TileSpec, VoxelGrid};

/// CSR mapping from regular tiles to candidate triangles.
///
/// Tile indexing uses X-major layout:
/// `tile = tx + nx * ty + nx * ny * tz`.
#[derive(Debug, Clone)]
pub struct TileTriangleCsr {
    /// Prefix-sum offsets into [`Self::tri_indices`], length = `tile_count` + 1.
    pub tile_offsets: Vec<u32>,
    /// Flattened triangle index list for all tiles.
    pub tri_indices: Vec<u32>,
    /// Number of triangle references per tile (same cardinality as `tile_count`).
    ///
    /// This is equivalent to `tile_offsets[i + 1] - tile_offsets[i]`.
    pub tri_counts: Vec<u32>,
}

/// CSR mapping from sparse bricks to candidate triangles.
///
/// Unlike [`TileTriangleCsr`], only bricks touched by at least one triangle
/// are emitted.
#[derive(Debug, Clone)]
pub struct BrickTriangleCsr {
    /// World-grid origins for each emitted brick, sorted by `(z, y, x)`.
    pub brick_origins: Vec<[u32; 3]>,
    /// Prefix-sum offsets into [`Self::tri_indices`], length = `brick_count` + 1.
    pub brick_offsets: Vec<u32>,
    /// Flattened triangle index list for all emitted bricks.
    pub tri_indices: Vec<u32>,
}

/// Build a dense tile CSR over the full tile lattice.
///
/// Each triangle is transformed into grid space, expanded by `epsilon`, and
/// assigned to every overlapping tile in the regular tile grid.
///
/// This function performs a standard two-pass CSR build:
/// 1. Count references per tile.
/// 2. Prefix-sum counts to offsets, then scatter triangle indices.
///
/// # Parameters
/// - `mesh`: source triangles in world space.
/// - `grid`: voxel grid transform and bounds.
/// - `tiles`: tile dimensions and lattice shape.
/// - `epsilon`: conservative expansion margin in grid units.
///
/// # Returns
/// A [`TileTriangleCsr`] that includes all tiles from `tiles`.
// Two-pass CSR build (count, then scatter) shares the same per-triangle tile
// expansion inline; splitting it would obscure the count/scatter symmetry.
#[allow(clippy::too_many_lines)]
pub fn build_tile_csr(
    mesh: &MeshInput,
    grid: &VoxelGrid,
    tiles: &TileSpec,
    epsilon: f32,
) -> TileTriangleCsr {
    let num_tiles_total = tiles.num_tiles_total() as usize;
    let mut counts = vec![0u32; num_tiles_total];
    let to_grid = grid.world_to_grid_matrix();
    let grid_dims = grid.dims();

    for tri in &mesh.triangles {
        let v0 = to_grid.transform_point3(tri[0]);
        let v1 = to_grid.transform_point3(tri[1]);
        let v2 = to_grid.transform_point3(tri[2]);
        let min_v = v0.min(v1).min(v2) - Vec3::splat(epsilon);
        let max_v = v0.max(v1).max(v2) + Vec3::splat(epsilon);

        let max_x = grid_dims[0].saturating_sub(1) as i32;
        let max_y = grid_dims[1].saturating_sub(1) as i32;
        let max_z = grid_dims[2].saturating_sub(1) as i32;
        let min_voxel = [
            min_v.x.floor() as i32,
            min_v.y.floor() as i32,
            min_v.z.floor() as i32,
        ];
        let max_voxel = [
            max_v.x.floor() as i32,
            max_v.y.floor() as i32,
            max_v.z.floor() as i32,
        ];
        let min_voxel = [
            min_voxel[0].clamp(0, max_x),
            min_voxel[1].clamp(0, max_y),
            min_voxel[2].clamp(0, max_z),
        ];
        let max_voxel = [
            max_voxel[0].clamp(0, max_x),
            max_voxel[1].clamp(0, max_y),
            max_voxel[2].clamp(0, max_z),
        ];

        let min_tile = [
            (min_voxel[0].div_euclid(tiles.tile_dims[0] as i32))
                .clamp(0, tiles.num_tiles[0] as i32 - 1),
            (min_voxel[1].div_euclid(tiles.tile_dims[1] as i32))
                .clamp(0, tiles.num_tiles[1] as i32 - 1),
            (min_voxel[2].div_euclid(tiles.tile_dims[2] as i32))
                .clamp(0, tiles.num_tiles[2] as i32 - 1),
        ];
        let max_tile = [
            (max_voxel[0].div_euclid(tiles.tile_dims[0] as i32))
                .clamp(0, tiles.num_tiles[0] as i32 - 1),
            (max_voxel[1].div_euclid(tiles.tile_dims[1] as i32))
                .clamp(0, tiles.num_tiles[1] as i32 - 1),
            (max_voxel[2].div_euclid(tiles.tile_dims[2] as i32))
                .clamp(0, tiles.num_tiles[2] as i32 - 1),
        ];

        if min_tile[0] > max_tile[0] || min_tile[1] > max_tile[1] || min_tile[2] > max_tile[2] {
            continue;
        }

        for tz in min_tile[2]..=max_tile[2] {
            for ty in min_tile[1]..=max_tile[1] {
                for tx in min_tile[0]..=max_tile[0] {
                    let tile_index = (tx as u32)
                        + tiles.num_tiles[0] * (ty as u32)
                        + tiles.num_tiles[0] * tiles.num_tiles[1] * (tz as u32);
                    counts[tile_index as usize] += 1;
                }
            }
        }
    }

    let mut offsets = vec![0u32; num_tiles_total + 1];
    for i in 0..num_tiles_total {
        offsets[i + 1] = offsets[i] + counts[i];
    }

    let mut cursor = offsets.clone();
    let mut tri_indices = vec![0u32; offsets[num_tiles_total] as usize];
    for (tri_index, tri) in mesh.triangles.iter().enumerate() {
        let v0 = to_grid.transform_point3(tri[0]);
        let v1 = to_grid.transform_point3(tri[1]);
        let v2 = to_grid.transform_point3(tri[2]);
        let min_v = v0.min(v1).min(v2) - Vec3::splat(epsilon);
        let max_v = v0.max(v1).max(v2) + Vec3::splat(epsilon);

        let min_voxel = [
            min_v.x.floor() as i32,
            min_v.y.floor() as i32,
            min_v.z.floor() as i32,
        ];
        let max_voxel = [
            max_v.x.floor() as i32,
            max_v.y.floor() as i32,
            max_v.z.floor() as i32,
        ];

        let min_tile = [
            (min_voxel[0].div_euclid(tiles.tile_dims[0] as i32))
                .clamp(0, tiles.num_tiles[0] as i32 - 1),
            (min_voxel[1].div_euclid(tiles.tile_dims[1] as i32))
                .clamp(0, tiles.num_tiles[1] as i32 - 1),
            (min_voxel[2].div_euclid(tiles.tile_dims[2] as i32))
                .clamp(0, tiles.num_tiles[2] as i32 - 1),
        ];
        let max_tile = [
            (max_voxel[0].div_euclid(tiles.tile_dims[0] as i32))
                .clamp(0, tiles.num_tiles[0] as i32 - 1),
            (max_voxel[1].div_euclid(tiles.tile_dims[1] as i32))
                .clamp(0, tiles.num_tiles[1] as i32 - 1),
            (max_voxel[2].div_euclid(tiles.tile_dims[2] as i32))
                .clamp(0, tiles.num_tiles[2] as i32 - 1),
        ];

        if min_tile[0] > max_tile[0] || min_tile[1] > max_tile[1] || min_tile[2] > max_tile[2] {
            continue;
        }

        for tz in min_tile[2]..=max_tile[2] {
            for ty in min_tile[1]..=max_tile[1] {
                for tx in min_tile[0]..=max_tile[0] {
                    let tile_index = (tx as u32)
                        + tiles.num_tiles[0] * (ty as u32)
                        + tiles.num_tiles[0] * tiles.num_tiles[1] * (tz as u32);
                    let write = cursor[tile_index as usize];
                    tri_indices[write as usize] = tri_index as u32;
                    cursor[tile_index as usize] += 1;
                }
            }
        }
    }

    TileTriangleCsr {
        tile_offsets: offsets,
        tri_indices,
        tri_counts: counts,
    }
}

/// Build a sparse brick CSR from triangle coverage.
///
/// Triangles are transformed into grid space, expanded by `epsilon`, and
/// associated with every overlapping brick of size `brick_dim^3`.
/// Only bricks with at least one triangle are emitted.
///
/// # Parameters
/// - `mesh`: source triangles in world space.
/// - `grid`: voxel grid transform and bounds.
/// - `brick_dim`: edge length of each brick in voxels.
/// - `epsilon`: conservative expansion margin in grid units.
///
/// # Returns
/// A [`BrickTriangleCsr`] where:
/// - `brick_origins` are sorted by `(z, y, x)` for stable iteration order.
/// - `brick_offsets` and `tri_indices` follow CSR invariants.
// Single-pass HashMap build with inline per-triangle brick clamping/rejection;
// splitting it would obscure the bounds logic.
#[allow(clippy::too_many_lines)]
pub fn build_brick_csr(
    mesh: &MeshInput,
    grid: &VoxelGrid,
    brick_dim: u32,
    epsilon: f32,
) -> BrickTriangleCsr {
    use std::collections::HashMap;

    let to_grid = grid.world_to_grid_matrix();
    let grid_dims = grid.dims();
    // Per-axis brick extent: number of bricks covering the grid (rounded up).
    let num_bricks = [
        grid_dims[0].div_ceil(brick_dim) as i32,
        grid_dims[1].div_ceil(brick_dim) as i32,
        grid_dims[2].div_ceil(brick_dim) as i32,
    ];
    let mut brick_map: HashMap<(u32, u32, u32), Vec<u32>> = HashMap::new();

    for (tri_index, tri) in mesh.triangles.iter().enumerate() {
        let v0 = to_grid.transform_point3(tri[0]);
        let v1 = to_grid.transform_point3(tri[1]);
        let v2 = to_grid.transform_point3(tri[2]);
        let min_v = v0.min(v1).min(v2) - Vec3::splat(epsilon);
        let max_v = v0.max(v1).max(v2) + Vec3::splat(epsilon);

        let min_voxel = [
            min_v.x.floor() as i32,
            min_v.y.floor() as i32,
            min_v.z.floor() as i32,
        ];
        let max_voxel = [
            max_v.x.floor() as i32,
            max_v.y.floor() as i32,
            max_v.z.floor() as i32,
        ];

        // Raw (pre-clamp) brick range on each axis. Used both to clamp into the
        // grid's brick extent and to reject triangles whose range lies wholly
        // outside `[0, num_bricks)` on any axis (which would otherwise emit a
        // phantom (0,0,0) or out-of-grid brick that no in-grid voxel touches).
        let raw_min_brick = [
            min_voxel[0].div_euclid(brick_dim as i32),
            min_voxel[1].div_euclid(brick_dim as i32),
            min_voxel[2].div_euclid(brick_dim as i32),
        ];
        let raw_max_brick = [
            max_voxel[0].div_euclid(brick_dim as i32),
            max_voxel[1].div_euclid(brick_dim as i32),
            max_voxel[2].div_euclid(brick_dim as i32),
        ];

        // Reject when the true brick range is entirely below 0 or at/above the
        // brick extent on any axis (no in-grid bricks possible).
        if (0..3).any(|ax| raw_max_brick[ax] < 0 || raw_min_brick[ax] >= num_bricks[ax]) {
            continue;
        }

        let min_brick = [
            raw_min_brick[0].clamp(0, num_bricks[0] - 1),
            raw_min_brick[1].clamp(0, num_bricks[1] - 1),
            raw_min_brick[2].clamp(0, num_bricks[2] - 1),
        ];
        let max_brick = [
            raw_max_brick[0].clamp(0, num_bricks[0] - 1),
            raw_max_brick[1].clamp(0, num_bricks[1] - 1),
            raw_max_brick[2].clamp(0, num_bricks[2] - 1),
        ];

        if min_brick[0] > max_brick[0] || min_brick[1] > max_brick[1] || min_brick[2] > max_brick[2]
        {
            continue;
        }

        for bz in min_brick[2]..=max_brick[2] {
            for by in min_brick[1]..=max_brick[1] {
                for bx in min_brick[0]..=max_brick[0] {
                    let key = (bx as u32, by as u32, bz as u32);
                    brick_map.entry(key).or_default().push(tri_index as u32);
                }
            }
        }
    }

    // Diagnostic: report CSR memory usage
    let total_refs: usize = brick_map.values().map(Vec::len).sum();
    let brick_count = brick_map.len();
    // HashMap overhead: ~64 bytes per entry + Vec overhead per bucket
    let estimated_bytes = brick_count * 80 + total_refs * 4;
    #[cfg(target_arch = "wasm32")]
    {
        let msg = format!(
            "[build_brick_csr] bricks={}, tri_refs={}, est_memory={}MB, triangles={}, grid_dims={:?}, brick_dim={}",
            brick_count,
            total_refs,
            estimated_bytes / (1024 * 1024),
            mesh.triangles.len(),
            grid.dims(),
            brick_dim
        );
        web_sys::console::log_1(&msg.into());
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (brick_count, total_refs, estimated_bytes);
    }

    let mut brick_origins: Vec<[u32; 3]> = brick_map
        .keys()
        .map(|(x, y, z)| [x * brick_dim, y * brick_dim, z * brick_dim])
        .collect();
    brick_origins.sort_by_key(|origin| (origin[2], origin[1], origin[0]));

    let mut brick_offsets = Vec::with_capacity(brick_origins.len() + 1);
    let mut tri_indices = Vec::new();
    brick_offsets.push(0);
    for origin in &brick_origins {
        let key = (
            origin[0] / brick_dim,
            origin[1] / brick_dim,
            origin[2] / brick_dim,
        );
        if let Some(list) = brick_map.get(&key) {
            tri_indices.extend(list.iter().copied());
        }
        brick_offsets.push(tri_indices.len() as u32);
    }

    BrickTriangleCsr {
        brick_origins,
        brick_offsets,
        tri_indices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{MeshInput, TileSpec, VoxelGrid};
    use glam::Vec3;

    #[test]
    fn csr_invariants_hold() {
        let grid = VoxelGrid::new(voxel_core::Resolution::new(8).unwrap(), Vec3::ZERO, 1.0);
        let tiles = TileSpec::new([4, 4, 4], grid.dims()).expect("tiles");
        let mesh = MeshInput {
            triangles: vec![
                [
                    Vec3::new(0.1, 0.1, 0.1),
                    Vec3::new(1.2, 0.1, 0.1),
                    Vec3::new(0.1, 1.2, 0.1),
                ],
                [
                    Vec3::new(4.0, 4.0, 4.0),
                    Vec3::new(5.0, 4.0, 4.0),
                    Vec3::new(4.0, 5.0, 4.0),
                ],
            ],
            material_ids: None,
            uvs: None,
            appearance: None,
        };
        let csr = build_tile_csr(&mesh, &grid, &tiles, 1e-4);
        assert_eq!(csr.tile_offsets[0], 0);
        for window in csr.tile_offsets.windows(2) {
            assert!(window[0] <= window[1]);
        }
        assert_eq!(
            csr.tile_offsets.last().copied().unwrap(),
            csr.tri_indices.len() as u32
        );

        // Strengthen: the brick CSR over the same in-grid mesh must also satisfy
        // the CSR invariants, emit at least one brick, and place every emitted
        // brick origin inside the grid (no phantom/out-of-grid bricks).
        let brick_dim = 4u32;
        let brick = build_brick_csr(&mesh, &grid, brick_dim, 1e-4);
        assert_eq!(brick.brick_offsets[0], 0);
        for window in brick.brick_offsets.windows(2) {
            assert!(window[0] <= window[1], "brick_offsets must be monotonic");
        }
        assert_eq!(
            brick.brick_offsets.last().copied().unwrap(),
            brick.tri_indices.len() as u32,
            "last offset == tri_indices length"
        );
        assert_eq!(
            brick.brick_offsets.len(),
            brick.brick_origins.len() + 1,
            "offsets length == brick_count + 1"
        );
        assert!(
            !brick.brick_origins.is_empty(),
            "in-grid geometry must emit at least one brick"
        );
        let dims = grid.dims();
        for o in &brick.brick_origins {
            for ax in 0..3 {
                assert!(
                    o[ax] < dims[ax],
                    "brick origin {o:?} axis {ax} must be inside the grid (dim {})",
                    dims[ax]
                );
            }
        }
    }

    /// Geometry wholly in negative grid space must emit **zero** bricks: before
    /// the fix, the per-axis `clamp(0, …)` collapsed an all-negative range onto a
    /// phantom `(0,0,0)` brick that no in-grid voxel touches.
    #[test]
    fn build_brick_csr_rejects_geometry_in_negative_space() {
        let grid = VoxelGrid::new(voxel_core::Resolution::new(8).unwrap(), Vec3::ZERO, 1.0);
        // All vertices at -10 (grid == world here): every voxel index is negative.
        let mesh = MeshInput {
            triangles: vec![[
                Vec3::new(-10.0, -10.0, -10.0),
                Vec3::new(-8.0, -10.0, -10.0),
                Vec3::new(-10.0, -8.0, -10.0),
            ]],
            material_ids: None,
            uvs: None,
            appearance: None,
        };
        let csr = build_brick_csr(&mesh, &grid, 4, 1e-4);
        assert!(
            csr.brick_origins.is_empty(),
            "negative-space geometry must emit no bricks (got {:?})",
            csr.brick_origins
        );
        assert_eq!(csr.brick_offsets, vec![0]);
        assert!(csr.tri_indices.is_empty());
    }

    /// Geometry wholly beyond the grid's far corner must emit **zero** bricks:
    /// before the fix the brick range was never clamped to the grid's brick
    /// extent, so an out-of-grid brick could be emitted.
    #[test]
    fn build_brick_csr_rejects_geometry_beyond_grid() {
        let grid = VoxelGrid::new(voxel_core::Resolution::new(8).unwrap(), Vec3::ZERO, 1.0);
        // All vertices at +20, well past the 8-voxel grid on every axis.
        let mesh = MeshInput {
            triangles: vec![[
                Vec3::new(20.0, 20.0, 20.0),
                Vec3::new(22.0, 20.0, 20.0),
                Vec3::new(20.0, 22.0, 20.0),
            ]],
            material_ids: None,
            uvs: None,
            appearance: None,
        };
        let csr = build_brick_csr(&mesh, &grid, 4, 1e-4);
        assert!(
            csr.brick_origins.is_empty(),
            "beyond-grid geometry must emit no bricks (got {:?})",
            csr.brick_origins
        );
        assert_eq!(csr.brick_offsets, vec![0]);
        assert!(csr.tri_indices.is_empty());
    }

    /// A triangle that straddles the grid's far boundary (part inside, part
    /// outside) must clamp its brick range to the grid's brick extent: every
    /// emitted brick origin stays inside the grid.
    #[test]
    fn build_brick_csr_clamps_boundary_brick_to_grid_extent() {
        let grid = VoxelGrid::new(voxel_core::Resolution::new(8).unwrap(), Vec3::ZERO, 1.0);
        let dims = grid.dims();
        // Spans from inside the top brick (x≈6) out past the grid (x≈12).
        let mesh = MeshInput {
            triangles: vec![[
                Vec3::new(6.0, 6.0, 6.0),
                Vec3::new(12.0, 6.0, 6.0),
                Vec3::new(6.0, 12.0, 6.0),
            ]],
            material_ids: None,
            uvs: None,
            appearance: None,
        };
        let brick_dim = 4u32;
        let csr = build_brick_csr(&mesh, &grid, brick_dim, 1e-4);
        assert!(
            !csr.brick_origins.is_empty(),
            "the in-grid portion must still emit a brick"
        );
        // num_bricks per axis = ceil(8/4) = 2 → valid origins are {0, 4}.
        for o in &csr.brick_origins {
            for ax in 0..3 {
                assert!(
                    o[ax] < dims[ax],
                    "clamped brick origin {o:?} axis {ax} must stay in [0, {})",
                    dims[ax]
                );
                assert_eq!(o[ax] % brick_dim, 0, "origin must be brick-aligned");
            }
        }
    }
}
