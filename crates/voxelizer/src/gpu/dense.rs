//! Dense voxelization - full grid output.

use bytemuck;
use wgpu::util::DeviceExt;

use crate::core::{
    DispatchStats, MeshInput, TileSpec, VoxelGrid, VoxelOccupancy, VoxelizationOutput, VoxelizeOpts,
};
use crate::csr::build_tile_csr;
use crate::error::VoxelizeGpuError;

use super::map_buffer_u32;
use super::{GpuVoxelizer, Params};

impl GpuVoxelizer {
    /// Voxelizes a mesh surface into a dense voxel grid.
    ///
    /// Returns occupancy bitfield, optional owner IDs, and optional colors.
    // Stays `async` to preserve the public API contract (the differential test
    // and callers `pollster::block_on(...)` it); the readback path is now
    // synchronous.
    #[allow(clippy::unused_async)]
    pub async fn voxelize_surface(
        &self,
        mesh: &MeshInput,
        grid: &VoxelGrid,
        tiles: &TileSpec,
        opts: &VoxelizeOpts,
    ) -> Result<VoxelizationOutput, VoxelizeGpuError> {
        grid.validate()
            .map_err(|e| VoxelizeGpuError::PipelineValidation(e.to_string()))?;
        mesh.validate()
            .map_err(|e| VoxelizeGpuError::PipelineValidation(e.to_string()))?;
        opts.validate()
            .map_err(|e| VoxelizeGpuError::PipelineValidation(e.to_string()))?;
        tiles
            .validate(self.max_invocations)
            .map_err(|e| VoxelizeGpuError::PipelineValidation(e.to_string()))?;

        self.validate_dense_storage(grid, opts)?;

        // An empty mesh creates a zero-length triangle buffer, which panics
        // inside wgpu when bound. Short-circuit to the same all-empty output the
        // CPU oracle produces for an empty mesh.
        if mesh.triangles.is_empty() {
            return Ok(empty_dense_output(grid, opts));
        }

        let csr = build_tile_csr(mesh, grid, tiles, opts.epsilon);
        let tri_data = prepare_triangle_data(mesh, grid);
        let buffers = self.create_dense_buffers(mesh, grid, tiles, opts, &csr, &tri_data);
        let bind_group = self.create_dense_bind_group(&buffers);

        self.dispatch_dense(&bind_group, tiles)?;

        let output = self.readback_dense(&buffers, grid, mesh, tiles, opts)?;
        Ok(output)
    }
}

/// The all-empty dense output for a mesh with no triangles.
///
/// Mirrors [`crate::reference_cpu::voxelize_surface_cpu`] on an empty mesh:
/// occupancy is all-zero (`ceil(n³/32)` words), `owner_id` is
/// `Some(vec![u32::MAX; n³])` when `store_owner`, `color_rgba` is
/// `Some(vec![0; n³])` when `store_color`, and `stats.triangles == 0`. Built on
/// the CPU so the empty case touches no GPU buffers (a zero-length triangle
/// buffer panics inside wgpu when bound).
fn empty_dense_output(grid: &VoxelGrid, opts: &VoxelizeOpts) -> VoxelizationOutput {
    let num_voxels = grid.num_voxels() as usize;
    let word_count = num_voxels.div_ceil(32);
    VoxelizationOutput {
        occupancy: VoxelOccupancy::from_words(grid.resolution, vec![0u32; word_count]),
        owner_id: if opts.store_owner {
            Some(vec![u32::MAX; num_voxels])
        } else {
            None
        },
        color_rgba: if opts.store_color {
            Some(vec![0u32; num_voxels])
        } else {
            None
        },
        stats: DispatchStats {
            triangles: 0,
            tiles: 0,
            voxels: grid.num_voxels(),
            gpu_time_ms: None,
        },
    }
}

// === Validation ===

impl GpuVoxelizer {
    fn validate_dense_storage(
        &self,
        grid: &VoxelGrid,
        opts: &VoxelizeOpts,
    ) -> Result<(), VoxelizeGpuError> {
        let num_voxels = grid.num_voxels() as usize;
        let word_count = num_voxels.div_ceil(32);
        let occupancy_bytes = (word_count as u64).saturating_mul(4);
        self.ensure_storage_fits(occupancy_bytes, "dense occupancy")?;

        if opts.store_owner {
            let owner_bytes = (num_voxels as u64).saturating_mul(4);
            self.ensure_storage_fits(owner_bytes, "dense owner")?;
        }
        if opts.store_color {
            let color_bytes = (num_voxels as u64).saturating_mul(4);
            self.ensure_storage_fits(color_bytes, "dense color")?;
        }
        Ok(())
    }
}

// === Triangle Data Preparation ===

fn prepare_triangle_data(mesh: &MeshInput, grid: &VoxelGrid) -> Vec<[f32; 4]> {
    let to_grid = grid.world_to_grid_matrix();
    let mut tri_data = Vec::with_capacity(mesh.triangles.len() * 6);

    for tri in &mesh.triangles {
        let p0 = to_grid.transform_point3(tri[0]);
        let p1 = to_grid.transform_point3(tri[1]);
        let p2 = to_grid.transform_point3(tri[2]);
        let min = p0.min(p1).min(p2);
        let max = p0.max(p1).max(p2);
        let normal = (p1 - p0).cross(p2 - p0);
        let d = -normal.dot(p0);

        tri_data.push([p0.x, p0.y, p0.z, 0.0]);
        tri_data.push([p1.x, p1.y, p1.z, 0.0]);
        tri_data.push([p2.x, p2.y, p2.z, 0.0]);
        tri_data.push([min.x, min.y, min.z, 0.0]);
        tri_data.push([max.x, max.y, max.z, 0.0]);
        tri_data.push([normal.x, normal.y, normal.z, d]);
    }

    tri_data
}

// === Buffer Creation ===

/// A minimal (16-byte) storage buffer for an unused binding.
///
/// Used for the owner/color bindings when their store flag is off: the WGSL
/// never writes them (every write is guarded), so a placeholder is sufficient to
/// satisfy the bind-group layout without allocating `n³·4` bytes.
fn dummy_storage_buffer(device: &wgpu::Device, label: &'static str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: 16,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

struct DenseBuffers {
    triangles: wgpu::Buffer,
    tile_offsets: wgpu::Buffer,
    tri_indices: wgpu::Buffer,
    occupancy: wgpu::Buffer,
    owner: wgpu::Buffer,
    color: wgpu::Buffer,
    params: wgpu::Buffer,
    brick_origins: wgpu::Buffer,
    debug: wgpu::Buffer,
    word_count: usize,
    num_voxels: usize,
}

impl GpuVoxelizer {
    fn create_dense_buffers(
        &self,
        mesh: &MeshInput,
        grid: &VoxelGrid,
        tiles: &TileSpec,
        opts: &VoxelizeOpts,
        csr: &crate::csr::TileTriangleCsr,
        tri_data: &[[f32; 4]],
    ) -> DenseBuffers {
        let num_voxels = grid.num_voxels() as usize;
        let word_count = num_voxels.div_ceil(32);

        let triangles = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.triangles"),
                contents: bytemuck::cast_slice(tri_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let tile_offsets = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.tile_offsets"),
                contents: bytemuck::cast_slice(&csr.tile_offsets),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let tri_indices = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.tri_indices"),
                contents: bytemuck::cast_slice(&csr.tri_indices),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let occupancy_init = vec![0u32; word_count];
        let occupancy = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.occupancy"),
                contents: bytemuck::cast_slice(&occupancy_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        // Owner/color are only written when their store flag is set (the WGSL
        // guards every owner_id/color_rgba write behind params.store_owner /
        // params.store_color). When a flag is OFF, allocating the full n³·4-byte
        // storage is pure waste — and at large n it can exceed the device's
        // buffer limit and panic. Bind a tiny dummy buffer in that case, mirroring
        // the `dummy_brick_origin` pattern below; readback skips it when off.
        let owner = if opts.store_owner {
            let owner_init = vec![u32::MAX; num_voxels];
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxelizer.owner"),
                    contents: bytemuck::cast_slice(&owner_init),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                })
        } else {
            dummy_storage_buffer(&self.device, "voxelizer.owner_dummy")
        };

        let color = if opts.store_color {
            let color_init = vec![0u32; num_voxels];
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxelizer.color"),
                    contents: bytemuck::cast_slice(&color_init),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                })
        } else {
            dummy_storage_buffer(&self.device, "voxelizer.color_dummy")
        };

        let params = self.create_dense_params(mesh, grid, tiles, opts);
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let dummy_brick_origin = [[0u32, 0u32, 0u32, 0u32]];
        let brick_origins = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.brick_origins_dummy"),
                contents: bytemuck::cast_slice(&dummy_brick_origin),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let debug_init = [0u32, 0u32, 0u32];
        let debug = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.debug"),
                contents: bytemuck::cast_slice(&debug_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        DenseBuffers {
            triangles,
            tile_offsets,
            tri_indices,
            occupancy,
            owner,
            color,
            params: params_buf,
            brick_origins,
            debug,
            word_count,
            num_voxels,
        }
    }

    // Takes `&self` for symmetry with `create_sparse_params` and the other
    // buffer-builder methods, though it needs no device state.
    #[allow(clippy::unused_self)]
    fn create_dense_params(
        &self,
        mesh: &MeshInput,
        grid: &VoxelGrid,
        tiles: &TileSpec,
        opts: &VoxelizeOpts,
    ) -> Params {
        let grid_dims = grid.dims();
        let num_tiles = u32::try_from(tiles.num_tiles_total()).unwrap_or(u32::MAX);
        let workgroups = num_tiles.div_ceil(self.tiles_per_workgroup);
        let (dispatch_x, dispatch_y, _dispatch_z) = self.dense_workgroup_dims(workgroups);
        Params {
            grid_dims: [grid_dims[0], grid_dims[1], grid_dims[2], 0],
            tile_dims: [
                tiles.tile_dims[0],
                tiles.tile_dims[1],
                tiles.tile_dims[2],
                0,
            ],
            num_tiles_xyz: [
                tiles.num_tiles[0],
                tiles.num_tiles[1],
                tiles.num_tiles[2],
                0,
            ],
            num_triangles: mesh.triangles.len() as u32,
            num_tiles,
            tile_voxels: tiles.tile_dims[0] * tiles.tile_dims[1] * tiles.tile_dims[2],
            store_owner: u32::from(opts.store_owner),
            store_color: u32::from(opts.store_color),
            debug: 0,
            dispatch_xy: [dispatch_x, dispatch_y],
        }
    }

    fn create_dense_bind_group(&self, buffers: &DenseBuffers) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxelizer.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.triangles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.tile_offsets.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.tri_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.occupancy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.owner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.color.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: buffers.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: buffers.brick_origins.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: buffers.debug.as_entire_binding(),
                },
            ],
        })
    }
}

// === Dispatch ===

impl GpuVoxelizer {
    fn dispatch_dense(
        &self,
        bind_group: &wgpu::BindGroup,
        tiles: &TileSpec,
    ) -> Result<(), VoxelizeGpuError> {
        let num_tiles = u32::try_from(tiles.num_tiles_total()).unwrap_or(u32::MAX);
        let workgroups = num_tiles.div_ceil(self.tiles_per_workgroup);
        // Spread the workgroups over 3 dispatch dimensions so each stays within the
        // device's per-dimension limit (the shader linearizes wg_id). Only a grid
        // needing z beyond the limit is unrepresentable in one dispatch.
        let (dx, dy, dz) = self.dense_workgroup_dims(workgroups);
        if dz > self.max_compute_workgroups_per_dimension {
            return Err(VoxelizeGpuError::WorkgroupsExceeded {
                label: "dense dispatch",
                workgroups,
                limit: self.max_compute_workgroups_per_dimension,
            });
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxelizer.encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voxelizer.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(dx, dy, dz);
        }

        self.queue.submit([encoder.finish()]);
        Ok(())
    }
}

// === Readback ===

impl GpuVoxelizer {
    fn readback_dense(
        &self,
        buffers: &DenseBuffers,
        grid: &VoxelGrid,
        mesh: &MeshInput,
        tiles: &TileSpec,
        opts: &VoxelizeOpts,
    ) -> Result<VoxelizationOutput, VoxelizeGpuError> {
        let read_occupancy = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer.read_occupancy"),
            size: (buffers.word_count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Owner/color readback buffers exist only when their flag is set — when
        // off, `buffers.owner`/`buffers.color` are 16-byte dummies (Fix 3), so a
        // full `num_voxels`-sized copy from them would be out of range.
        let read_owner = opts.store_owner.then(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxelizer.read_owner"),
                size: (buffers.num_voxels * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let read_color = opts.store_color.then(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("voxelizer.read_color"),
                size: (buffers.num_voxels * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxelizer.readback"),
            });

        encoder.copy_buffer_to_buffer(
            &buffers.occupancy,
            0,
            &read_occupancy,
            0,
            read_occupancy.size(),
        );
        if let Some(read_owner) = &read_owner {
            encoder.copy_buffer_to_buffer(&buffers.owner, 0, read_owner, 0, read_owner.size());
        }
        if let Some(read_color) = &read_color {
            encoder.copy_buffer_to_buffer(&buffers.color, 0, read_color, 0, read_color.size());
        }

        self.queue.submit([encoder.finish()]);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let occupancy = map_buffer_u32(&read_occupancy, &self.device)?;
        let owner = match &read_owner {
            Some(buf) => Some(map_buffer_u32(buf, &self.device)?),
            None => None,
        };
        let color = match &read_color {
            Some(buf) => Some(map_buffer_u32(buf, &self.device)?),
            None => None,
        };

        Ok(VoxelizationOutput {
            occupancy: VoxelOccupancy::from_words(grid.resolution, occupancy),
            owner_id: owner,
            color_rgba: color,
            stats: DispatchStats {
                triangles: mesh.triangles.len() as u32,
                tiles: u32::try_from(tiles.num_tiles_total()).unwrap_or(u32::MAX),
                voxels: grid.num_voxels(),
                gpu_time_ms: None,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec3;
    use voxel_core::Resolution;

    use crate::core::{MeshInput, TileSpec, VoxelGrid, VoxelizeOpts};
    use crate::error::VoxelizeGpuError;
    use crate::gpu::{GpuVoxelizer, GpuVoxelizerConfig};
    use crate::reference_cpu::voxelize_surface_cpu;

    /// Builds a GPU voxelizer, or returns `None` (skip) when no adapter exists.
    /// Panics on any non-`NoAdapter` init failure so a real regression is loud.
    fn gpu_or_skip() -> Option<GpuVoxelizer> {
        match pollster::block_on(GpuVoxelizer::new_standalone(GpuVoxelizerConfig::default())) {
            Ok(v) => Some(v),
            Err(VoxelizeGpuError::NoAdapter) => None,
            Err(e) => panic!("GPU init failed (not NoAdapter): {e}"),
        }
    }

    fn grid_n(n: u32) -> VoxelGrid {
        VoxelGrid::new(Resolution::new(n).unwrap(), Vec3::ZERO, 1.0)
    }

    /// An empty mesh must short-circuit to the same all-empty output the CPU
    /// oracle produces — never panic inside wgpu on a zero-length triangle buffer.
    #[test]
    fn empty_mesh_returns_empty_grid_not_panic() {
        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let grid = grid_n(8);
        let tiles = TileSpec::new([2, 2, 2], grid.dims()).unwrap();
        let opts = VoxelizeOpts::default();
        let mesh = MeshInput {
            triangles: Vec::new(),
            material_ids: None,
        };

        let gpu_out = pollster::block_on(gpu.voxelize_surface(&mesh, &grid, &tiles, &opts))
            .expect("empty mesh must voxelize to an empty grid, not error");
        let cpu_out = voxelize_surface_cpu(&mesh, &grid, &tiles, &opts);

        assert_eq!(
            gpu_out.occupancy.words(),
            cpu_out.occupancy.words(),
            "empty-mesh occupancy must equal the CPU oracle (all zero)"
        );
        assert_eq!(gpu_out.occupancy.count_occupied(), 0);
        assert_eq!(gpu_out.owner_id, cpu_out.owner_id, "empty owner channel");
        assert_eq!(
            gpu_out.color_rgba, cpu_out.color_rgba,
            "empty color channel"
        );
        assert_eq!(gpu_out.stats.triangles, 0);
    }

    /// With both store flags OFF at n=512, the owner/color buffers must NOT be
    /// allocated (they'd be 512³·4 ≈ 537 MiB each and could exceed the device
    /// buffer limit — the flag-off path used to be *less* safe than flag-on).
    /// The occupancy-only path (16 MiB) must run, or — if a device limit is hit
    /// (storage-binding size, or the dense dispatch's workgroup-per-dimension
    /// cap at this grid size) — return a *typed* error; never a raw wgpu panic.
    #[test]
    fn store_flags_off_n512_no_panic() {
        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let grid = grid_n(512);
        let tiles = TileSpec::new([4, 4, 4], grid.dims()).unwrap();
        let opts = VoxelizeOpts {
            epsilon: 1e-4,
            store_owner: false,
            store_color: false,
        };
        // A single small in-grid triangle so the path actually dispatches.
        let mesh = MeshInput {
            triangles: vec![[
                Vec3::new(1.5, 1.5, 1.5),
                Vec3::new(8.0, 2.0, 2.0),
                Vec3::new(2.0, 8.0, 4.0),
            ]],
            material_ids: None,
        };
        match pollster::block_on(gpu.voxelize_surface(&mesh, &grid, &tiles, &opts)) {
            Ok(out) => {
                // Occupancy-only ran: 512³/32·4 = 16 MiB occupancy buffer.
                assert_eq!(out.occupancy.words().len(), (512u64.pow(3) / 32) as usize);
                assert!(out.owner_id.is_none(), "store_owner off ⇒ no owner channel");
                assert!(
                    out.color_rgba.is_none(),
                    "store_color off ⇒ no color channel"
                );
                assert!(
                    out.occupancy.count_occupied() > 0,
                    "the triangle occupies voxels"
                );
            }
            // Any typed device-limit error is acceptable — the fix's contract is
            // "no raw wgpu panic", and these are caught and returned as errors.
            Err(
                VoxelizeGpuError::StorageExceeded { .. }
                | VoxelizeGpuError::WorkgroupsExceeded { .. },
            ) => {}
            Err(e) => panic!("expected Ok or a typed device-limit error, got {e}"),
        }
    }

    /// `validate_dense_storage` must reject a buffer that exceeds the per-binding
    /// storage limit with a typed `StorageExceeded` (rather than letting
    /// `create_buffer` panic). Drives a resolution whose owner storage exceeds the
    /// device's storage-binding limit.
    #[test]
    fn validate_dense_storage_rejects_oversized_with_storage_exceeded() {
        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let limits = gpu.limits_summary();
        // Smallest 8·4^k resolution whose owner storage (n³·4 bytes) exceeds the
        // per-binding limit. n³·4 > limit ⇒ n > (limit/4)^(1/3).
        let mut n = 8u32;
        while (u64::from(n)).pow(3).saturating_mul(4) <= limits.max_storage_buffer_binding_size {
            n = n.saturating_mul(4);
            assert!(
                n <= 8192,
                "no in-range resolution exceeded the binding limit"
            );
        }
        let grid = grid_n(n);
        let opts = VoxelizeOpts {
            epsilon: 1e-4,
            store_owner: true,
            store_color: false,
        };
        let err = gpu
            .validate_dense_storage(&grid, &opts)
            .expect_err("oversized owner storage must be rejected");
        assert!(
            matches!(err, VoxelizeGpuError::StorageExceeded { .. }),
            "expected StorageExceeded, got {err}"
        );
    }

    /// Every (`store_owner`, `store_color`) combination at n=128 must voxelize
    /// successfully, with the requested channels present (and color requiring
    /// owner per the API contract).
    #[test]
    fn store_flags_on_n128_each_combo_ok() {
        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let grid = grid_n(128);
        let tiles = TileSpec::new([4, 4, 4], grid.dims()).unwrap();
        let mesh = MeshInput {
            triangles: vec![[
                Vec3::new(2.0, 2.0, 2.0),
                Vec3::new(60.0, 4.0, 8.0),
                Vec3::new(6.0, 60.0, 12.0),
            ]],
            material_ids: None,
        };
        // (store_owner, store_color) — color requires owner, so skip (false,true).
        for (owner, color) in [(false, false), (true, false), (true, true)] {
            let opts = VoxelizeOpts {
                epsilon: 1e-4,
                store_owner: owner,
                store_color: color,
            };
            let out = pollster::block_on(gpu.voxelize_surface(&mesh, &grid, &tiles, &opts))
                .unwrap_or_else(|e| {
                    panic!("n128 combo (owner={owner}, color={color}) failed: {e}")
                });
            assert_eq!(out.owner_id.is_some(), owner, "owner channel presence");
            assert_eq!(out.color_rgba.is_some(), color, "color channel presence");
            assert!(
                out.occupancy.count_occupied() > 0,
                "triangle must occupy voxels"
            );
        }
    }

    /// An over-large tile (`[128,128,128]` = 2.1M voxels, far past any device's
    /// per-workgroup invocation limit) must return a typed `PipelineValidation`
    /// error (wrapping `TileTooLarge`), not panic.
    #[test]
    fn over_large_tile_returns_pipeline_validation_err() {
        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let grid = grid_n(128);
        let tiles = TileSpec::new([128, 128, 128], grid.dims()).unwrap();
        let opts = VoxelizeOpts::default();
        let mesh = MeshInput {
            triangles: vec![[
                Vec3::new(2.0, 2.0, 2.0),
                Vec3::new(60.0, 4.0, 8.0),
                Vec3::new(6.0, 60.0, 12.0),
            ]],
            material_ids: None,
        };
        let err = pollster::block_on(gpu.voxelize_surface(&mesh, &grid, &tiles, &opts))
            .expect_err("an over-large tile must be rejected");
        assert!(
            matches!(err, VoxelizeGpuError::PipelineValidation(_)),
            "expected PipelineValidation, got {err}"
        );
    }

    /// A zero tile dimension is rejected by `TileSpec::new` itself (pure, no GPU).
    #[test]
    fn zero_tile_dim_rejected_by_tilespec() {
        let grid = grid_n(8);
        assert!(
            TileSpec::new([0, 4, 4], grid.dims()).is_err(),
            "TileSpec::new must reject a zero tile dimension"
        );
    }

    /// The dense path must voxelize n=512: with tile `[4,4,4]` that is 128³ ≈ 2.1M
    /// tiles ⇒ far more workgroups than the 65,535 per-dimension dispatch limit.
    /// Before the 3-D dispatch this returned `WorkgroupsExceeded`; now it must run.
    /// The fixture is a planar triangle at z≈250 spanning most of x/y, so its
    /// occupied voxels live in HIGH linear-index tiles (≈1.0M) — dispatched at
    /// `wg_id.y > 0`, directly exercising the shader's `wg_id` linearization. A
    /// thin AABB keeps the CPU oracle cheap. The result must be a conservative
    /// superset of the oracle: no CPU-marked voxel may be missing (which would
    /// mean a high-index tile was skipped/mismapped by the chunked dispatch).
    #[test]
    fn dense_n512_dispatches_via_3d_chunking() {
        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let grid = grid_n(512);
        let tiles = TileSpec::new([4, 4, 4], grid.dims()).unwrap();
        let opts = VoxelizeOpts {
            epsilon: 1e-4,
            store_owner: false,
            store_color: false,
        };
        let mesh = MeshInput {
            triangles: vec![[
                Vec3::new(10.0, 10.0, 250.0),
                Vec3::new(500.0, 30.0, 250.0),
                Vec3::new(30.0, 500.0, 250.0),
            ]],
            material_ids: None,
        };

        let gpu_out = pollster::block_on(gpu.voxelize_surface(&mesh, &grid, &tiles, &opts))
            .expect("n=512 must dispatch via 3-D workgroup chunking, not exceed the per-dim limit");
        let cpu_out = voxelize_surface_cpu(&mesh, &grid, &tiles, &opts);

        let cpu_count = cpu_out.occupancy.count_occupied();
        assert!(
            cpu_count > 1000,
            "planar fixture should mark many high-index-tile voxels (got {cpu_count})"
        );

        let cpu_w = cpu_out.occupancy.words();
        let gpu_w = gpu_out.occupancy.words();
        assert_eq!(cpu_w.len(), gpu_w.len());
        let mut under = 0u64; // CPU-marked but GPU-missed — must be 0
        let mut over = 0u64; // GPU over-marked — a small FP-tangent margin only
        for (c, g) in cpu_w.iter().zip(gpu_w) {
            under += u64::from((c & !g).count_ones());
            over += u64::from((g & !c).count_ones());
        }
        assert_eq!(
            under, 0,
            "chunked dispatch under-marked {under} voxels — a high-index tile was skipped/mismapped"
        );
        assert!(
            over * 50 < cpu_count,
            "GPU over-marked {over} vs {cpu_count} CPU — beyond a small FP-tangent margin"
        );
    }
}
