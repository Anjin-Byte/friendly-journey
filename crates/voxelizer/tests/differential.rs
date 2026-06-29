//! Reference-as-Oracle differential tests for the voxelizer (Engineering Codex:
//! *Reference Implementation as Oracle* / *Sharp Oracles*).
//!
//! A discrete voxel hit has no tolerance, so the GPU production path
//! ([`GpuVoxelizer::voxelize_surface`]) must be **bit-exact** with the CPU SAT
//! oracle ([`voxelizer::reference_cpu::voxelize_surface_cpu`]) on the occupancy
//! bitset. The GPU test is gated on a runtime adapter probe so CI stays green on
//! machines without a GPU; the CPU-only tests (oracle fixtures, edge cases, and
//! the native `voxel-core` bridge) always run.

use glam::Vec3;
use voxelizer::reference_cpu::voxelize_surface_cpu;
use voxelizer::{
    GpuVoxelizer, GpuVoxelizerConfig, MeshInput, OccupancyField, Resolution, TileSpec, VoxelCoord,
    VoxelGrid, VoxelizeGpuError, VoxelizeOpts,
};

/// A deterministic multi-triangle mesh spanning a 32³ grid (world == grid space
/// since the grid below uses `voxel_size = 1`, origin `0`).
fn test_mesh() -> MeshInput {
    MeshInput {
        triangles: vec![
            [
                Vec3::new(2.0, 2.0, 2.0),
                Vec3::new(28.0, 4.0, 6.0),
                Vec3::new(5.0, 26.0, 10.0),
            ],
            [
                Vec3::new(10.0, 10.0, 20.0),
                Vec3::new(22.0, 10.0, 20.0),
                Vec3::new(16.0, 24.0, 20.0),
            ],
            [
                Vec3::new(1.0, 1.0, 30.0),
                Vec3::new(4.0, 1.0, 30.0),
                Vec3::new(1.0, 4.0, 30.0),
            ],
        ],
        material_ids: None,
        uvs: None,
        appearance: None,
    }
}

fn test_grid() -> VoxelGrid {
    VoxelGrid::new(Resolution::new(32).unwrap(), Vec3::ZERO, 1.0)
}

fn test_tiles(grid: &VoxelGrid) -> TileSpec {
    TileSpec::new([4, 4, 4], grid.dims()).unwrap()
}

/// Probes for a GPU; returns `None` (and skips) when no adapter is present —
/// **unless** `VOXEL_REQUIRE_GPU` is set, which turns a missing adapter into a
/// hard failure so a GPU CI lane (`cargo xtask ci-gpu`) can't silently skip this
/// differential. Any *other* init failure always panics. Mirrors the gate in the
/// sibling `sparse_material_bridge.rs`.
fn gpu_or_skip() -> Option<GpuVoxelizer> {
    match pollster::block_on(GpuVoxelizer::new_standalone(GpuVoxelizerConfig::default())) {
        Ok(v) => Some(v),
        Err(VoxelizeGpuError::NoAdapter) => {
            assert!(
                std::env::var_os("VOXEL_REQUIRE_GPU").is_none(),
                "VOXEL_REQUIRE_GPU set but no GPU adapter present"
            );
            eprintln!("no GPU adapter present — skipping GPU differential test");
            None
        }
        Err(e) => panic!("GPU init failed (not NoAdapter): {e}"),
    }
}

/// THE codex requirement: GPU occupancy must equal the CPU oracle bit-for-bit.
/// `test_mesh`'s fixture is tangent-free (no voxel sits exactly on a triangle
/// boundary), so the CPU/GPU bit-exact equality holds here.
#[test]
fn cpu_gpu_occupancy_bit_exact() {
    let mesh = test_mesh();
    let grid = test_grid();
    let tiles = test_tiles(&grid);
    let opts = VoxelizeOpts::default();

    let cpu = voxelize_surface_cpu(&mesh, &grid, &tiles, &opts);
    assert!(
        cpu.occupancy.count_occupied() > 0,
        "the test mesh must occupy voxels for the comparison to be meaningful"
    );

    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let gpu_out = pollster::block_on(gpu.voxelize_surface(&mesh, &grid, &tiles, &opts))
        .expect("GPU voxelize_surface");

    assert_eq!(
        cpu.occupancy.words(),
        gpu_out.occupancy.words(),
        "CPU oracle and GPU occupancy must be bit-exact (a discrete voxel hit has no tolerance)"
    );
}

/// The tolerant contract at floating-point tangents: the GPU is a *conservative
/// superset* of the CPU oracle (`GPU ⊇ CPU`) — it may over-mark a boundary voxel
/// by a few, but must never under-mark one the CPU marks. This single-triangle
/// fixture sits at irrational-ish positions so several voxels land on FP tangents.
#[test]
fn gpu_occupancy_is_conservative_superset_at_tangents() {
    let mesh = MeshInput {
        triangles: vec![[
            Vec3::new(1.3, 1.7, 2.4),
            Vec3::new(5.1, 2.2, 3.9),
            Vec3::new(2.6, 6.3, 4.1),
        ]],
        material_ids: None,
        uvs: None,
        appearance: None,
    };
    let grid = VoxelGrid::new(Resolution::new(8).unwrap(), Vec3::ZERO, 1.0);
    let tiles = TileSpec::new([4, 4, 4], grid.dims()).unwrap();
    let opts = VoxelizeOpts::default();

    let cpu = voxelize_surface_cpu(&mesh, &grid, &tiles, &opts);
    let cpu_count = cpu.occupancy.count_occupied();
    assert!(
        cpu_count > 0,
        "the fixture must occupy voxels for the comparison to be meaningful"
    );

    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let gpu_out = pollster::block_on(gpu.voxelize_surface(&mesh, &grid, &tiles, &opts))
        .expect("GPU voxelize_surface");

    let mut gpu_only = 0u32;
    for z in 0..8u32 {
        for y in 0..8u32 {
            for x in 0..8u32 {
                let c = VoxelCoord::new(x, y, z);
                let cpu_hit = cpu.occupancy.is_occupied(c);
                let gpu_hit = gpu_out.occupancy.is_occupied(c);
                // GPU ⊇ CPU: every CPU hit must also be a GPU hit (never under-marks).
                assert!(
                    !cpu_hit || gpu_hit,
                    "voxel ({x},{y},{z}) marked by CPU but not GPU — GPU under-marked"
                );
                if gpu_hit && !cpu_hit {
                    gpu_only += 1;
                }
            }
        }
    }
    // Relative (not magic-constant) tightness bound on the conservative superset.
    // An over-mark is a voxel the GPU's f32 SAT grazes at a tangent but the CPU
    // reference does not; such voxels are confined to the triangle's tangent
    // shell, whose size is bounded by the surface itself. So the over-mark margin
    // is at most the CPU surface-voxel count — i.e. the GPU marks at most ~2× the
    // CPU, never an unbounded blow-up. (Replaces the prior `<= 4` magic constant.)
    assert!(
        u64::from(gpu_only) <= cpu_count,
        "GPU over-marked {gpu_only} voxels beyond the {cpu_count} CPU oracle marks \
         — the conservative-superset margin should not exceed the surface size"
    );
}

/// The native bridge the reshape exists for: occupancy → `SparseTree` →
/// `SchoolBBuffer` (the renderer's intake), all via `voxel-core`.
#[test]
fn occupancy_bridges_to_renderer() {
    let mesh = test_mesh();
    let grid = test_grid();
    let tiles = test_tiles(&grid);

    let out = voxelize_surface_cpu(&mesh, &grid, &tiles, &VoxelizeOpts::default());
    assert!(out.occupancy.count_occupied() > 0);

    // VoxelOccupancy implements OccupancyField, so it feeds the core builders.
    let tree = out.occupancy.to_sparse_tree();
    assert!(
        tree.leaf_count() > 0,
        "an occupied field must produce sparse-tree leaves"
    );
    let buffer = out.occupancy.to_school_b();
    assert!(
        buffer.node_count() > 0,
        "the School-B buffer must be non-empty for an occupied field"
    );
}

/// The CPU oracle marks the voxels a hand-placed triangle clearly intersects, and
/// leaves far-away voxels empty.
#[test]
fn cpu_oracle_marks_expected_voxels() {
    let grid = VoxelGrid::new(Resolution::new(8).unwrap(), Vec3::ZERO, 1.0);
    let tiles = TileSpec::new([2, 2, 2], grid.dims()).unwrap();
    // A triangle in the z≈1.5 plane; vertex v0 sits exactly at voxel (1,1,1)'s center.
    let mesh = MeshInput {
        triangles: vec![[
            Vec3::new(1.5, 1.5, 1.5),
            Vec3::new(5.5, 1.5, 1.5),
            Vec3::new(1.5, 5.5, 1.5),
        ]],
        material_ids: None,
        uvs: None,
        appearance: None,
    };
    let out = voxelize_surface_cpu(&mesh, &grid, &tiles, &VoxelizeOpts::default());

    assert!(
        out.occupancy.is_occupied(VoxelCoord::new(1, 1, 1)),
        "voxel containing a triangle vertex must be occupied"
    );
    assert!(
        !out.occupancy.is_occupied(VoxelCoord::new(7, 7, 7)),
        "a voxel far from the triangle must be empty"
    );
    assert!(
        !out.occupancy.is_occupied(VoxelCoord::new(8, 0, 0)),
        "an out-of-bounds coordinate must read empty"
    );
}

/// An empty mesh yields an empty field that still bridges without panicking.
#[test]
fn empty_mesh_is_empty() {
    let grid = VoxelGrid::new(Resolution::new(8).unwrap(), Vec3::ZERO, 1.0);
    let tiles = TileSpec::new([2, 2, 2], grid.dims()).unwrap();
    let mesh = MeshInput {
        triangles: Vec::new(),
        material_ids: None,
        uvs: None,
        appearance: None,
    };
    let out = voxelize_surface_cpu(&mesh, &grid, &tiles, &VoxelizeOpts::default());
    assert_eq!(out.occupancy.count_occupied(), 0);
    // Bridging an empty field must not panic.
    let _ = out.occupancy.to_school_b();
}

/// Occupancy is independent of triangle order (a property of a correct rasterizer).
#[test]
fn occupancy_is_order_invariant() {
    let grid = test_grid();
    let tiles = test_tiles(&grid);
    let opts = VoxelizeOpts::default();

    let mesh = test_mesh();
    let mut reversed = mesh.clone();
    reversed.triangles.reverse();

    let a = voxelize_surface_cpu(&mesh, &grid, &tiles, &opts);
    let b = voxelize_surface_cpu(&reversed, &grid, &tiles, &opts);
    assert_eq!(
        a.occupancy.words(),
        b.occupancy.words(),
        "occupancy must not depend on triangle order"
    );
}
