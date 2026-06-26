//! GPU-accelerated voxelization using wgpu compute shaders.
//!
//! This module provides `GpuVoxelizer` for high-performance surface voxelization
//! with both dense and sparse output modes.

use bytemuck::{Pod, Zeroable};

use crate::core::VoxelizeOpts;
use crate::error::VoxelizeGpuError;

mod buffers;
mod compact_attrs;
mod compact_positions;
mod compact_voxels;
mod dense;
mod pipelines;
mod shaders;
mod sparse;

pub(crate) use buffers::{map_buffer_f32, map_buffer_u32};
use pipelines::create_pipelines;

// Re-export the Params type for use by dense/sparse modules
pub(crate) use self::params::Params;

mod params {
    use bytemuck::{Pod, Zeroable};

    /// Uniform parameters for the voxelizer compute shader (mirrors the WGSL
    /// `Params` struct; std140 layout with `vec4` padding).
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct Params {
        /// Voxel grid dimensions `[x, y, z, _]`.
        pub(crate) grid_dims: [u32; 4],
        /// Tile (dense) or brick (sparse) edge lengths `[x, y, z, _]`.
        pub(crate) tile_dims: [u32; 4],
        /// Number of tiles per axis `[x, y, z, _]` (unused on the sparse path).
        pub(crate) num_tiles_xyz: [u32; 4],
        /// Total triangle count in the triangle buffer.
        pub(crate) num_triangles: u32,
        /// Total number of tiles (dense) or bricks (sparse) to process.
        pub(crate) num_tiles: u32,
        /// Voxels per tile/brick (`tile_dims.x * y * z`).
        pub(crate) tile_voxels: u32,
        /// Non-zero to write the per-voxel owner triangle index.
        pub(crate) store_owner: u32,
        /// Non-zero to write the per-voxel hashed color.
        pub(crate) store_color: u32,
        /// Non-zero to accumulate debug counters into the debug buffer.
        pub(crate) debug: u32,
        /// The dispatch's x and y workgroup extents, so the shader can linearize a
        /// 3-D `wg_id` back to a flat tile index (a dense grid can need more tiles
        /// than the per-dimension dispatch limit). `[0, 0]` for 1-D dispatches.
        pub(crate) dispatch_xy: [u32; 2],
    }
}

/// Configuration for the GPU voxelizer.
#[derive(Debug, Clone)]
pub struct GpuVoxelizerConfig {
    /// Workgroup size for compute shaders (0 = auto-detect).
    pub workgroup_size: u32,
    /// Number of tiles processed per workgroup.
    pub tiles_per_workgroup: u32,
}

impl Default for GpuVoxelizerConfig {
    fn default() -> Self {
        Self {
            workgroup_size: 0,
            tiles_per_workgroup: 2,
        }
    }
}

const MAX_TILES_PER_WORKGROUP: u32 = 4;

/// Uniform parameters for the position-compaction shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CompactParams {
    /// Brick edge length in voxels.
    pub(crate) brick_dim: u32,
    /// Number of bricks in the brick-origin buffer.
    pub(crate) brick_count: u32,
    /// Capacity of the output positions buffer (in positions).
    pub(crate) max_positions: u32,
    /// Padding to align `origin_world` to 16 bytes.
    pub(crate) _pad0: u32,
    /// World-space grid origin `[x, y, z, voxel_size]`.
    pub(crate) origin_world: [f32; 4],
}

/// Uniform parameters for the attribute-compaction shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CompactAttrsParams {
    /// Brick edge length in voxels.
    pub(crate) brick_dim: u32,
    /// Number of bricks in the brick-origin buffer.
    pub(crate) brick_count: u32,
    /// Capacity of the output attribute buffers (in entries).
    pub(crate) max_entries: u32,
    /// Padding to align `grid_dims` to 16 bytes.
    pub(crate) _pad0: u32,
    /// Voxel grid dimensions `[x, y, z, _]`.
    pub(crate) grid_dims: [u32; 4],
}

/// Uniform parameters for the compact-voxel (material-resolving) shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CompactVoxelsParams {
    /// Brick edge length in voxels.
    pub(crate) brick_dim: u32,
    /// Number of bricks in the brick-origin buffer.
    pub(crate) brick_count: u32,
    /// Capacity of the output voxel buffer (in voxels).
    pub(crate) max_entries: u32,
    /// Length, in `u32` words, of the packed-`u16` material table.
    pub(crate) material_table_len: u32,
    /// Global voxel-space origin offset `[x, y, z, _]`.
    pub(crate) g_origin: [i32; 4],
}

/// GPU-accelerated voxelizer using wgpu compute shaders.
///
/// Supports both dense voxelization (full grid) and sparse voxelization
/// (brick-based, only allocating storage for occupied regions).
pub struct GpuVoxelizer {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) compact_pipeline: wgpu::ComputePipeline,
    pub(crate) compact_bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) compact_attrs_pipeline: wgpu::ComputePipeline,
    pub(crate) compact_attrs_bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) compact_voxels_pipeline: wgpu::ComputePipeline,
    pub(crate) compact_voxels_bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) workgroup_size: u32,
    pub(crate) tiles_per_workgroup: u32,
    pub(crate) max_invocations: u32,
    pub(crate) brick_dim: u32,
    pub(crate) max_storage_buffer_binding_size: u64,
    pub(crate) max_buffer_size: u64,
    pub(crate) max_storage_buffers_per_shader_stage: u32,
    pub(crate) max_compute_workgroups_per_dimension: u32,
}

/// Summary of GPU device limits relevant to voxelization.
#[derive(Debug, Clone, Copy)]
pub struct GpuLimitsSummary {
    /// Maximum compute invocations (threads) per workgroup.
    pub max_invocations_per_workgroup: u32,
    /// Maximum number of storage buffers bindable in a single compute stage.
    pub max_storage_buffers_per_shader_stage: u32,
    /// Maximum size, in bytes, of a single storage-buffer binding.
    pub max_storage_buffer_binding_size: u64,
    /// Maximum workgroups dispatchable along a single dimension.
    pub max_compute_workgroups_per_dimension: u32,
}

impl GpuVoxelizer {
    /// Builds a voxelizer on an existing device + queue (dependency injection),
    /// so it can share a single GPU with the renderer (`voxel-gpu`) instead of
    /// creating a second device. The composition root creates the device once
    /// (e.g. via `voxel-gpu`'s `GpuContext`) and injects its handles into both
    /// adapters. Prefer this when a process voxelizes *and* renders — it avoids a
    /// duplicate device and lets output be handed to the renderer on the same GPU.
    pub async fn from_device(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: GpuVoxelizerConfig,
    ) -> Result<Self, VoxelizeGpuError> {
        let device = device.clone();
        let queue = queue.clone();

        let limits = device.limits();
        let max_invocations = limits.max_compute_invocations_per_workgroup;

        let (workgroup_size, tiles_per_workgroup) =
            compute_workgroup_params(&config, max_invocations);

        let max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size;
        let max_buffer_size = limits.max_buffer_size;
        let max_storage_buffers_per_shader_stage = limits.max_storage_buffers_per_shader_stage;
        let max_compute_workgroups_per_dimension = limits.max_compute_workgroups_per_dimension;

        let brick_dim = (max_invocations as f32).cbrt().floor() as u32;
        let brick_dim = brick_dim.clamp(2, 8);

        let pipelines = create_pipelines(&device, workgroup_size, tiles_per_workgroup).await?;

        Ok(Self {
            device,
            queue,
            pipeline: pipelines.pipeline,
            bind_group_layout: pipelines.bind_group_layout,
            compact_pipeline: pipelines.compact_pipeline,
            compact_bind_group_layout: pipelines.compact_bind_group_layout,
            compact_attrs_pipeline: pipelines.compact_attrs_pipeline,
            compact_attrs_bind_group_layout: pipelines.compact_attrs_bind_group_layout,
            compact_voxels_pipeline: pipelines.compact_voxels_pipeline,
            compact_voxels_bind_group_layout: pipelines.compact_voxels_bind_group_layout,
            workgroup_size,
            tiles_per_workgroup,
            max_invocations,
            brick_dim,
            max_storage_buffer_binding_size,
            max_buffer_size,
            max_storage_buffers_per_shader_stage,
            max_compute_workgroups_per_dimension,
        })
    }

    /// Creates a voxelizer that owns its own wgpu device — convenient for
    /// standalone / headless use (CLI voxelize-only, tests). When voxelizing *and*
    /// rendering in one process, prefer [`from_device`](Self::from_device) to share
    /// a single GPU device with the renderer.
    pub async fn new_standalone(config: GpuVoxelizerConfig) -> Result<Self, VoxelizeGpuError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|_| VoxelizeGpuError::NoAdapter)?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await?;
        Self::from_device(&device, &queue, config).await
    }

    /// Returns the wgpu device.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Returns the wgpu queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Returns the brick dimension used for sparse voxelization.
    pub fn brick_dim(&self) -> u32 {
        self.brick_dim
    }

    /// Returns a summary of GPU device limits.
    pub fn limits_summary(&self) -> GpuLimitsSummary {
        GpuLimitsSummary {
            max_invocations_per_workgroup: self.max_invocations,
            max_storage_buffers_per_shader_stage: self.max_storage_buffers_per_shader_stage,
            max_storage_buffer_binding_size: self.max_storage_buffer_binding_size,
            max_compute_workgroups_per_dimension: self.max_compute_workgroups_per_dimension,
        }
    }

    /// Returns the workgroup size used by compute shaders.
    pub fn workgroup_size(&self) -> u32 {
        self.workgroup_size
    }

    /// Validates that the workgroup count fits within device limits.
    pub(crate) fn ensure_workgroups_fit(
        &self,
        workgroups: u32,
        label: &'static str,
    ) -> Result<(), VoxelizeGpuError> {
        if workgroups > self.max_compute_workgroups_per_dimension {
            return Err(VoxelizeGpuError::WorkgroupsExceeded {
                label,
                workgroups,
                limit: self.max_compute_workgroups_per_dimension,
            });
        }
        Ok(())
    }

    /// Factors a flat workgroup count into a 3-D dispatch whose every dimension is
    /// within `max_compute_workgroups_per_dimension`, so a dense grid with more
    /// tiles than the per-dimension limit dispatches in one pass (the shader
    /// linearizes `wg_id` back to the flat index). `x * y * z >= workgroups`;
    /// surplus workgroups are guarded by the shader's `tile_index < num_tiles`.
    /// `z` exceeds the limit only for an astronomically large grid — the caller
    /// rejects that case.
    pub(crate) fn dense_workgroup_dims(&self, workgroups: u32) -> (u32, u32, u32) {
        let max = self.max_compute_workgroups_per_dimension.max(1);
        let x = workgroups.min(max).max(1);
        let rem = workgroups.div_ceil(x);
        let y = rem.min(max).max(1);
        let z = rem.div_ceil(y).max(1);
        (x, y, z)
    }

    /// Validates that a buffer size fits within device limits.
    ///
    /// Checks both the per-binding storage limit and the device-wide
    /// `max_buffer_size` (which `create_buffer` enforces with a raw panic if
    /// exceeded), reporting whichever limit is smaller.
    pub(crate) fn ensure_storage_fits(
        &self,
        bytes: u64,
        label: &'static str,
    ) -> Result<(), VoxelizeGpuError> {
        if bytes > self.max_storage_buffer_binding_size || bytes > self.max_buffer_size {
            return Err(VoxelizeGpuError::StorageExceeded {
                label,
                bytes,
                limit: self
                    .max_storage_buffer_binding_size
                    .min(self.max_buffer_size),
            });
        }
        Ok(())
    }

    /// Validates `brick_dim` and returns the per-brick voxel count (`brick_dim³`).
    ///
    /// Thin instance-method wrapper over the free [`validate_brick_dim`] so the
    /// compaction validators can call `self.validate_brick_dim(..)`.
    ///
    /// # Errors
    /// Returns [`VoxelizeGpuError::InvalidBrickDim`] for a zero or cube-overflowing
    /// `brick_dim`.
    #[allow(clippy::unused_self)]
    pub(crate) fn validate_brick_dim(&self, brick_dim: u32) -> Result<u32, VoxelizeGpuError> {
        validate_brick_dim(brick_dim)
    }

    /// Computes the maximum number of bricks that can be processed in one dispatch.
    pub(crate) fn max_bricks_per_dispatch(&self, brick_dim: u32, opts: &VoxelizeOpts) -> usize {
        let brick_voxels = u64::from(brick_dim)
            .saturating_mul(u64::from(brick_dim))
            .saturating_mul(u64::from(brick_dim));
        let words_per_brick = brick_voxels.div_ceil(32);
        let max_storage = self.max_storage_buffer_binding_size;
        let max_workgroups = u64::from(self.max_compute_workgroups_per_dimension);

        let occupancy_bytes = words_per_brick.saturating_mul(4);
        let mut max_bricks = if occupancy_bytes > 0 {
            max_storage / occupancy_bytes
        } else {
            max_storage
        };

        if opts.store_owner {
            let owner_bytes = brick_voxels.saturating_mul(4);
            if owner_bytes > 0 {
                max_bricks = max_bricks.min(max_storage / owner_bytes);
            }
        }
        if opts.store_color {
            let color_bytes = brick_voxels.saturating_mul(4);
            if color_bytes > 0 {
                max_bricks = max_bricks.min(max_storage / color_bytes);
            }
        }

        let max_bricks = max_bricks.min(max_workgroups).max(1);
        usize::try_from(max_bricks).unwrap_or(1)
    }

    /// Creates an empty position buffer (used for zero-result compaction).
    pub(crate) fn empty_position_buffer(&self) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer.compact.empty_positions"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }
}

/// Validates `brick_dim` and returns the per-brick voxel count (`brick_dim³`).
///
/// Rejects `brick_dim == 0` or a cube that overflows `u32` (the saturating `u64`
/// widening mirrors [`GpuVoxelizer::max_bricks_per_dispatch`]); on success the
/// returned `brick_dim * brick_dim * brick_dim` is known to fit `u32`, so the
/// compaction validators' downstream multiply can no longer panic. Free fn (no
/// device state) so it is testable without a GPU.
///
/// # Errors
/// Returns [`VoxelizeGpuError::InvalidBrickDim`] for a zero or cube-overflowing
/// `brick_dim`.
pub(crate) fn validate_brick_dim(brick_dim: u32) -> Result<u32, VoxelizeGpuError> {
    if brick_dim == 0 || u64::from(brick_dim).pow(3) > u64::from(u32::MAX) {
        return Err(VoxelizeGpuError::InvalidBrickDim { got: brick_dim });
    }
    Ok(brick_dim * brick_dim * brick_dim)
}

/// Computes workgroup size and tiles per workgroup from config and device limits.
fn compute_workgroup_params(config: &GpuVoxelizerConfig, max_invocations: u32) -> (u32, u32) {
    let mut tiles_per_workgroup = config.tiles_per_workgroup.max(1);
    tiles_per_workgroup = tiles_per_workgroup.min(MAX_TILES_PER_WORKGROUP);

    let mut workgroup_size = if config.workgroup_size == 0 {
        let per_tile = max_invocations / tiles_per_workgroup;
        // Lower bound is `min(32, max_invocations)` so the clamp range stays
        // well-ordered when a device exposes fewer than 32 invocations (clamp
        // panics if its `min` exceeds its `max`).
        per_tile.clamp(32.min(max_invocations), max_invocations)
    } else {
        config.workgroup_size
    };

    if workgroup_size > max_invocations {
        workgroup_size = max_invocations;
    }

    if workgroup_size.saturating_mul(tiles_per_workgroup) > max_invocations {
        let max_tiles = (max_invocations / workgroup_size).max(1);
        tiles_per_workgroup = tiles_per_workgroup.min(max_tiles);
    }

    (workgroup_size, tiles_per_workgroup)
}

#[cfg(test)]
mod tests {
    use super::{GpuVoxelizerConfig, MAX_TILES_PER_WORKGROUP, compute_workgroup_params};

    /// `compute_workgroup_params` must always emit a usable `(workgroup_size,
    /// tiles_per_workgroup)` whose product fits the device's invocation budget,
    /// no matter how degenerate the config or the limit is. Pure, no GPU.
    #[test]
    fn compute_workgroup_params_invariants_hold() {
        for &workgroup_size in &[0u32, 1, u32::MAX] {
            for &tiles_per_workgroup in &[0u32, 1, 4, 9] {
                for &max_invocations in &[16u32, 31, 32, 64, 256, 1024] {
                    let config = GpuVoxelizerConfig {
                        workgroup_size,
                        tiles_per_workgroup,
                    };
                    let (ws, tpw) = compute_workgroup_params(&config, max_invocations);
                    assert!(
                        (1..=max_invocations).contains(&ws),
                        "workgroup_size {ws} out of 1..={max_invocations} \
                         (cfg ws={workgroup_size}, tpw={tiles_per_workgroup})"
                    );
                    assert!(
                        (1..=MAX_TILES_PER_WORKGROUP).contains(&tpw),
                        "tiles_per_workgroup {tpw} out of 1..={MAX_TILES_PER_WORKGROUP} \
                         (cfg ws={workgroup_size}, tpw={tiles_per_workgroup})"
                    );
                    assert!(
                        ws.saturating_mul(tpw) <= max_invocations,
                        "ws {ws} * tpw {tpw} exceeds max_invocations {max_invocations} \
                         (cfg ws={workgroup_size}, tpw={tiles_per_workgroup})"
                    );
                }
            }
        }
    }

    /// `from_device` builds a working voxelizer on an externally-created device,
    /// and two voxelizers can share ONE device — the point of the injection
    /// interface (renderer + voxelizer on a single GPU). GPU-gated.
    #[test]
    fn from_device_shares_one_injected_device() {
        use crate::core::{MeshInput, TileSpec, VoxelGrid, VoxelizeOpts};
        use crate::gpu::GpuVoxelizer;
        use crate::reference_cpu::voxelize_surface_cpu;
        use glam::Vec3;
        use voxel_core::Resolution;

        // Create ONE device the way a host (e.g. voxel-gpu's GpuContext) would.
        let made = pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .ok()?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .ok()?;
            Some((device, queue))
        });
        let Some((device, queue)) = made else {
            return; // no adapter — skip
        };

        // Two voxelizers injected with the SAME device + queue (one shared GPU).
        let vox = pollster::block_on(GpuVoxelizer::from_device(
            &device,
            &queue,
            GpuVoxelizerConfig::default(),
        ))
        .expect("from_device on a shared device");
        let _second = pollster::block_on(GpuVoxelizer::from_device(
            &device,
            &queue,
            GpuVoxelizerConfig::default(),
        ))
        .expect("a second voxelizer can share the same device");

        let grid = VoxelGrid::new(Resolution::new(8).unwrap(), Vec3::ZERO, 1.0);
        let tiles = TileSpec::new([2, 2, 2], grid.dims()).unwrap();
        let opts = VoxelizeOpts::default();
        let mesh = MeshInput {
            triangles: vec![[
                Vec3::new(1.5, 1.5, 1.5),
                Vec3::new(6.0, 2.0, 2.0),
                Vec3::new(2.0, 6.0, 3.0),
            ]],
            material_ids: None,
        };
        let out = pollster::block_on(vox.voxelize_surface(&mesh, &grid, &tiles, &opts))
            .expect("voxelize on an injected device");
        let cpu = voxelize_surface_cpu(&mesh, &grid, &tiles, &opts);

        assert!(
            out.occupancy.count_occupied() > 0,
            "the injected-device voxelizer must produce occupancy"
        );
        // GPU ⊇ CPU (never under-marks) — the same tolerant contract as the differential.
        let under: u32 = cpu
            .occupancy
            .words()
            .iter()
            .zip(out.occupancy.words())
            .map(|(c, g)| (c & !g).count_ones())
            .sum();
        assert_eq!(
            under, 0,
            "injected-device voxelizer under-marked vs the CPU oracle"
        );
    }

    /// Regression for the `u32` overflow of the global occupancy index at 2048³:
    /// `gx + n·gy + n²·gz` reaches 8.6e9 > `u32::MAX`, overflowing for `gz > 1024`,
    /// so the GPU silently dropped the top half of the model. Needs a device with
    /// the adapter's real storage limits (the 2048³ occupancy buffer is ~1 GiB —
    /// `new_standalone`'s default 128 MiB would reject it), so it builds its own.
    /// Heavy (≈1 GiB buffers), so it is `#[ignore]`d — run with `--ignored`.
    #[test]
    #[ignore = "2048³ needs a ~1 GiB occupancy buffer + a high-limit GPU; run explicitly"]
    fn gpu_no_u32_overflow_at_2048() {
        use crate::core::{MeshInput, TileSpec, VoxelGrid, VoxelizeOpts};
        use crate::gpu::GpuVoxelizer;
        use crate::reference_cpu::voxelize_surface_cpu;
        use glam::Vec3;
        use voxel_core::Resolution;

        // Default limits cap storage at 128 MiB → 2048³ would StorageExceeded.
        // Build a device with the adapter's real storage/buffer limits instead.
        let made = pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .ok()?;
            let al = adapter.limits();
            let limits = wgpu::Limits {
                max_storage_buffer_binding_size: al.max_storage_buffer_binding_size,
                max_buffer_size: al.max_buffer_size,
                ..wgpu::Limits::default()
            };
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_limits: limits,
                    ..Default::default()
                })
                .await
                .ok()?;
            Some((device, queue))
        });
        let Some((device, queue)) = made else {
            return; // no adapter — skip
        };
        let vox = pollster::block_on(GpuVoxelizer::from_device(
            &device,
            &queue,
            GpuVoxelizerConfig::default(),
        ))
        .expect("from_device on a high-limit device");

        let grid = VoxelGrid::new(Resolution::new(2048).unwrap(), Vec3::ZERO, 1.0);
        let tiles = TileSpec::new([4, 4, 4], grid.dims()).unwrap();
        let opts = VoxelizeOpts {
            epsilon: 1e-4,
            store_owner: false,
            store_color: false,
        };
        // A vertical quad at y=1024 spanning x,z fully (thin in y → the CPU oracle
        // scans one slab, stays feasible). Its voxels reach gz≈2044, deep in the
        // gz>1024 region whose global index overflowed u32 before the fix.
        let mesh = MeshInput {
            triangles: vec![
                [
                    Vec3::new(4.0, 1024.0, 4.0),
                    Vec3::new(2044.0, 1024.0, 4.0),
                    Vec3::new(2044.0, 1024.0, 2044.0),
                ],
                [
                    Vec3::new(4.0, 1024.0, 4.0),
                    Vec3::new(2044.0, 1024.0, 2044.0),
                    Vec3::new(4.0, 1024.0, 2044.0),
                ],
            ],
            material_ids: None,
        };

        let cpu = voxelize_surface_cpu(&mesh, &grid, &tiles, &opts);
        assert!(
            cpu.occupancy.count_occupied() > 1_000_000,
            "the fixture must occupy many high-z voxels"
        );
        let gpu = pollster::block_on(vox.voxelize_surface(&mesh, &grid, &tiles, &opts))
            .expect("voxelize at 2048³");
        // The fix's invariant: no under-marking (the overflow dropped ~half).
        let under: u64 = cpu
            .occupancy
            .words()
            .iter()
            .zip(gpu.occupancy.words())
            .map(|(c, g)| u64::from((c & !g).count_ones()))
            .sum();
        assert_eq!(
            under, 0,
            "GPU under-marked {under} voxels at 2048³ — the u32 occupancy-index overflow regressed"
        );
    }
}
