//! Sparse voxelization - brick-based output for memory efficiency.

use bytemuck;
use wgpu::util::DeviceExt;

use crate::core::{
    CompactVoxel, DispatchStats, MeshInput, SparseVoxelizationOutput, VoxelGrid, VoxelizeOpts,
};
use crate::csr::{BrickTriangleCsr, build_brick_csr};
use crate::error::VoxelizeGpuError;

use super::map_buffer_u32;
use super::{GpuVoxelizer, Params};

impl GpuVoxelizer {
    /// Voxelizes a mesh surface into sparse brick-based output.
    ///
    /// Only allocates storage for bricks that contain geometry.
    // Stays `async` to preserve the public GPU-orchestration API contract
    // (callers `.await` it); the readback path is now synchronous.
    #[allow(clippy::unused_async)]
    pub async fn voxelize_surface_sparse(
        &self,
        mesh: &MeshInput,
        grid: &VoxelGrid,
        opts: &VoxelizeOpts,
    ) -> Result<SparseVoxelizationOutput, VoxelizeGpuError> {
        grid.validate()
            .map_err(|e| VoxelizeGpuError::PipelineValidation(e.to_string()))?;
        mesh.validate()
            .map_err(|e| VoxelizeGpuError::PipelineValidation(e.to_string()))?;
        opts.validate()
            .map_err(|e| VoxelizeGpuError::PipelineValidation(e.to_string()))?;

        let brick_dim = self.brick_dim;
        let csr = build_brick_csr(mesh, grid, brick_dim, opts.epsilon);

        // No bricks (empty mesh or all-outside geometry) → binding a zero-length
        // triangle buffer panics inside wgpu. Mirror the chunked sibling's guard
        // and return an empty output. The chunked path returns an empty Vec; here
        // a single empty `SparseVoxelizationOutput` is the analogue.
        if csr.brick_origins.is_empty() {
            return Ok(empty_sparse_output(mesh, grid, brick_dim, opts));
        }

        self.run_sparse(mesh, grid, opts, brick_dim, csr)
    }

    /// Voxelizes a mesh and compacts the output into `CompactVoxel` tuples.
    ///
    /// This is the high-level orchestrator for the ADR-0009 pipeline:
    /// 1. Runs sparse voxelization to get occupancy + `owner_id`
    /// 2. Runs the compact voxels shader to resolve materials and compute global coords
    ///
    /// # Arguments
    /// * `mesh` — input triangles
    /// * `grid` — voxel grid specification
    /// * `opts` — voxelization options (must have `store_owner = true`)
    /// * `material_table` — packed u16 material IDs (two per u32), indexed by triangle
    /// * `g_origin` — global voxel-space origin offset
    pub async fn compact_surface_sparse(
        &self,
        mesh: &MeshInput,
        grid: &VoxelGrid,
        opts: &VoxelizeOpts,
        material_table: &[u32],
        g_origin: [i32; 3],
    ) -> Result<Vec<CompactVoxel>, VoxelizeGpuError> {
        if !opts.store_owner {
            return Err(VoxelizeGpuError::OwnerRequired);
        }

        // Use chunked voxelization to stay within GPU dispatch limits,
        // then compact each chunk independently and merge results.
        let chunks = self
            .voxelize_surface_sparse_chunked(mesh, grid, opts, 0)
            .await?;

        let mut all_voxels: Vec<CompactVoxel> = Vec::new();

        for sparse in &chunks {
            let owner_id = sparse.owner_id.as_ref().ok_or_else(|| {
                VoxelizeGpuError::PipelineValidation(
                    "voxelize_surface_sparse did not produce owner_id".to_string(),
                )
            })?;

            let max_entries = sparse
                .occupancy
                .iter()
                .map(|w| w.count_ones())
                .sum::<u32>()
                .max(1);

            let voxels = self
                .compact_sparse_voxels(
                    &sparse.occupancy,
                    owner_id,
                    &sparse.brick_origins,
                    sparse.brick_dim,
                    max_entries,
                    material_table,
                    g_origin,
                )
                .await?;

            all_voxels.extend(voxels);
        }

        Ok(all_voxels)
    }

    /// Voxelizes in chunks to handle large meshes within GPU limits.
    // Stays `async` to preserve the public GPU-orchestration API contract
    // (callers `.await` it); the readback path is now synchronous.
    #[allow(clippy::unused_async)]
    pub async fn voxelize_surface_sparse_chunked(
        &self,
        mesh: &MeshInput,
        grid: &VoxelGrid,
        opts: &VoxelizeOpts,
        chunk_size: usize,
    ) -> Result<Vec<SparseVoxelizationOutput>, VoxelizeGpuError> {
        grid.validate()
            .map_err(|e| VoxelizeGpuError::PipelineValidation(e.to_string()))?;
        mesh.validate()
            .map_err(|e| VoxelizeGpuError::PipelineValidation(e.to_string()))?;
        opts.validate()
            .map_err(|e| VoxelizeGpuError::PipelineValidation(e.to_string()))?;

        let brick_dim = self.brick_dim;
        let csr = build_brick_csr(mesh, grid, brick_dim, opts.epsilon);

        if csr.brick_origins.is_empty() {
            return Ok(Vec::new());
        }

        let chunk_size =
            self.compute_chunk_size(brick_dim, opts, chunk_size, csr.brick_origins.len());
        self.process_chunks(mesh, grid, opts, brick_dim, &csr, chunk_size)
    }

    fn run_sparse(
        &self,
        mesh: &MeshInput,
        grid: &VoxelGrid,
        opts: &VoxelizeOpts,
        brick_dim: u32,
        csr: BrickTriangleCsr,
    ) -> Result<SparseVoxelizationOutput, VoxelizeGpuError> {
        let brick_count = csr.brick_origins.len() as u32;

        self.validate_sparse_storage(brick_dim, brick_count, opts)?;

        let tri_data = prepare_sparse_triangle_data(mesh, grid);
        let buffers = self.create_sparse_buffers(brick_dim, opts, &csr, &tri_data);
        let params = self.create_sparse_params(mesh, grid, brick_dim, brick_count, opts);
        let bind_group = self.create_sparse_bind_group(&buffers, &params);

        self.dispatch_sparse(&bind_group, brick_count)?;

        let output =
            self.readback_sparse(&buffers, mesh, grid, brick_dim, csr.brick_origins, opts)?;

        Ok(output)
    }
}

/// The empty sparse output for a mesh that touches no in-grid bricks.
///
/// Used by the non-chunked entry when `build_brick_csr` emits zero bricks (empty
/// mesh or geometry wholly outside the grid). No GPU buffers are bound (a
/// zero-length triangle buffer panics inside wgpu). `owner_id`/`color_rgba` are
/// empty `Vec`s under their store flags so consumers still see the requested
/// channels; `stats.triangles` reflects the input mesh.
fn empty_sparse_output(
    mesh: &MeshInput,
    grid: &VoxelGrid,
    brick_dim: u32,
    opts: &VoxelizeOpts,
) -> SparseVoxelizationOutput {
    SparseVoxelizationOutput {
        brick_dim,
        brick_origins: Vec::new(),
        occupancy: Vec::new(),
        owner_id: if opts.store_owner {
            Some(Vec::new())
        } else {
            None
        },
        color_rgba: if opts.store_color {
            Some(Vec::new())
        } else {
            None
        },
        debug_flags: [0, 0, 0],
        debug_workgroups: 0,
        debug_tested: 0,
        debug_hits: 0,
        stats: DispatchStats {
            triangles: mesh.triangles.len() as u32,
            tiles: 0,
            voxels: grid.num_voxels(),
            gpu_time_ms: None,
        },
    }
}

// === Chunked Processing ===

impl GpuVoxelizer {
    fn compute_chunk_size(
        &self,
        brick_dim: u32,
        opts: &VoxelizeOpts,
        requested: usize,
        total_bricks: usize,
    ) -> usize {
        let max_bricks = self
            .max_bricks_per_dispatch(brick_dim, opts)
            .min(total_bricks);
        let chunk_size = if requested == 0 {
            max_bricks
        } else {
            requested.min(max_bricks)
        };
        chunk_size.max(1)
    }

    fn process_chunks(
        &self,
        mesh: &MeshInput,
        grid: &VoxelGrid,
        opts: &VoxelizeOpts,
        brick_dim: u32,
        csr: &BrickTriangleCsr,
        chunk_size: usize,
    ) -> Result<Vec<SparseVoxelizationOutput>, VoxelizeGpuError> {
        let mut chunks = Vec::new();
        let brick_count = csr.brick_origins.len();
        let mut start = 0usize;

        while start < brick_count {
            let end = (start + chunk_size).min(brick_count);
            let sub_csr = extract_chunk_csr(csr, start, end);
            let output = self.run_sparse(mesh, grid, opts, brick_dim, sub_csr)?;
            chunks.push(output);
            start = end;
        }

        Ok(chunks)
    }
}

fn extract_chunk_csr(csr: &BrickTriangleCsr, start: usize, end: usize) -> BrickTriangleCsr {
    let offset_start = csr.brick_offsets[start] as usize;
    let offset_end = csr.brick_offsets[end] as usize;
    let tri_indices = csr.tri_indices[offset_start..offset_end].to_vec();

    let base = csr.brick_offsets[start];
    let brick_offsets: Vec<u32> = (start..=end)
        .map(|idx| csr.brick_offsets[idx] - base)
        .collect();

    let brick_origins = csr.brick_origins[start..end].to_vec();

    BrickTriangleCsr {
        brick_origins,
        brick_offsets,
        tri_indices,
    }
}

// === Validation ===

impl GpuVoxelizer {
    fn validate_sparse_storage(
        &self,
        brick_dim: u32,
        brick_count: u32,
        opts: &VoxelizeOpts,
    ) -> Result<(), VoxelizeGpuError> {
        let workgroups = brick_count.div_ceil(self.tiles_per_workgroup);
        self.ensure_workgroups_fit(workgroups, "sparse dispatch")?;

        let brick_voxels = (brick_dim * brick_dim * brick_dim) as usize;
        let words_per_brick = brick_voxels.div_ceil(32);

        let occupancy_bytes = (words_per_brick as u64)
            .saturating_mul(4)
            .saturating_mul(u64::from(brick_count));
        self.ensure_storage_fits(occupancy_bytes, "sparse occupancy")?;

        if opts.store_owner {
            let owner_bytes = (brick_voxels as u64)
                .saturating_mul(4)
                .saturating_mul(u64::from(brick_count));
            self.ensure_storage_fits(owner_bytes, "sparse owner")?;
        }
        if opts.store_color {
            let color_bytes = (brick_voxels as u64)
                .saturating_mul(4)
                .saturating_mul(u64::from(brick_count));
            self.ensure_storage_fits(color_bytes, "sparse color")?;
        }

        Ok(())
    }
}

// === Triangle Data ===

fn prepare_sparse_triangle_data(mesh: &MeshInput, grid: &VoxelGrid) -> Vec<[f32; 4]> {
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

struct SparseBuffers {
    triangles: wgpu::Buffer,
    brick_origins: wgpu::Buffer,
    brick_offsets: wgpu::Buffer,
    tri_indices: wgpu::Buffer,
    occupancy: wgpu::Buffer,
    owner: wgpu::Buffer,
    color: wgpu::Buffer,
    debug: wgpu::Buffer,
    occupancy_len: usize,
    owner_len: usize,
    color_len: usize,
}

impl GpuVoxelizer {
    fn create_sparse_buffers(
        &self,
        brick_dim: u32,
        _opts: &VoxelizeOpts,
        csr: &BrickTriangleCsr,
        tri_data: &[[f32; 4]],
    ) -> SparseBuffers {
        let brick_voxels = (brick_dim * brick_dim * brick_dim) as usize;
        let words_per_brick = brick_voxels.div_ceil(32);
        let brick_count = csr.brick_origins.len();

        let triangles = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer_sparse.triangles"),
                contents: bytemuck::cast_slice(tri_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let brick_origin_data: Vec<[u32; 4]> = csr
            .brick_origins
            .iter()
            .map(|o| [o[0], o[1], o[2], 0])
            .collect();

        let brick_origins = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer_sparse.brick_origins"),
                contents: bytemuck::cast_slice(&brick_origin_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let brick_offsets = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer_sparse.brick_offsets"),
                contents: bytemuck::cast_slice(&csr.brick_offsets),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let tri_indices = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer_sparse.tri_indices"),
                contents: bytemuck::cast_slice(&csr.tri_indices),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let occupancy_len = words_per_brick * brick_count;
        let occupancy_init = vec![0u32; occupancy_len];
        let occupancy = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer_sparse.occupancy"),
                contents: bytemuck::cast_slice(&occupancy_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let owner_len = brick_voxels * brick_count;
        let owner_init = vec![u32::MAX; owner_len];
        let owner = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer_sparse.owner"),
                contents: bytemuck::cast_slice(&owner_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let color_len = brick_voxels * brick_count;
        let color_init = vec![0u32; color_len];
        let color = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer_sparse.color"),
                contents: bytemuck::cast_slice(&color_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let debug_init = [0u32, 0u32, 0u32];
        let debug = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer_sparse.debug"),
                contents: bytemuck::cast_slice(&debug_init),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        SparseBuffers {
            triangles,
            brick_origins,
            brick_offsets,
            tri_indices,
            occupancy,
            owner,
            color,
            debug,
            occupancy_len,
            owner_len,
            color_len,
        }
    }

    fn create_sparse_params(
        &self,
        mesh: &MeshInput,
        grid: &VoxelGrid,
        brick_dim: u32,
        brick_count: u32,
        opts: &VoxelizeOpts,
    ) -> wgpu::Buffer {
        let brick_voxels = brick_dim * brick_dim * brick_dim;

        let grid_dims = grid.dims();
        let params = Params {
            grid_dims: [grid_dims[0], grid_dims[1], grid_dims[2], 0],
            tile_dims: [brick_dim, brick_dim, brick_dim, 0],
            num_tiles_xyz: [0, 0, 0, 0],
            num_triangles: mesh.triangles.len() as u32,
            num_tiles: brick_count,
            tile_voxels: brick_voxels,
            store_owner: u32::from(opts.store_owner),
            store_color: u32::from(opts.store_color),
            debug: 1,
            // 1-D brick dispatch → no wg_id linearization needed.
            dispatch_xy: [0, 0],
        };

        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer_sparse.params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    fn create_sparse_bind_group(
        &self,
        buffers: &SparseBuffers,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxelizer_sparse.bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.triangles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.brick_offsets.as_entire_binding(),
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
                    resource: params.as_entire_binding(),
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
    // Returns `Result` for signature parity with `dispatch_dense` (workgroup-fit
    // validation happens earlier in `validate_sparse_storage`).
    #[allow(clippy::unnecessary_wraps)]
    fn dispatch_sparse(
        &self,
        bind_group: &wgpu::BindGroup,
        brick_count: u32,
    ) -> Result<(), VoxelizeGpuError> {
        let workgroups = brick_count.div_ceil(self.tiles_per_workgroup);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxelizer_sparse.encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voxelizer_sparse.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit([encoder.finish()]);
        Ok(())
    }
}

// === Readback ===

impl GpuVoxelizer {
    fn readback_sparse(
        &self,
        buffers: &SparseBuffers,
        mesh: &MeshInput,
        grid: &VoxelGrid,
        brick_dim: u32,
        brick_origins: Vec<[u32; 3]>,
        opts: &VoxelizeOpts,
    ) -> Result<SparseVoxelizationOutput, VoxelizeGpuError> {
        let read_occupancy = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer_sparse.read_occupancy"),
            size: (buffers.occupancy_len * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let read_owner = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer_sparse.read_owner"),
            size: (buffers.owner_len * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let read_color = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer_sparse.read_color"),
            size: (buffers.color_len * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let read_debug = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer_sparse.read_debug"),
            size: 12,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxelizer_sparse.readback"),
            });

        encoder.copy_buffer_to_buffer(
            &buffers.occupancy,
            0,
            &read_occupancy,
            0,
            read_occupancy.size(),
        );
        encoder.copy_buffer_to_buffer(&buffers.owner, 0, &read_owner, 0, read_owner.size());
        encoder.copy_buffer_to_buffer(&buffers.color, 0, &read_color, 0, read_color.size());
        encoder.copy_buffer_to_buffer(&buffers.debug, 0, &read_debug, 0, read_debug.size());

        self.queue.submit([encoder.finish()]);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let occupancy = map_buffer_u32(&read_occupancy, &self.device)?;
        let owner = map_buffer_u32(&read_owner, &self.device)?;
        let color = map_buffer_u32(&read_color, &self.device)?;
        let debug = map_buffer_u32(&read_debug, &self.device)?;

        let brick_count = brick_origins.len() as u32;

        Ok(SparseVoxelizationOutput {
            brick_dim,
            brick_origins,
            occupancy,
            owner_id: if opts.store_owner { Some(owner) } else { None },
            color_rgba: if opts.store_color { Some(color) } else { None },
            debug_flags: [0, 0, 0],
            debug_workgroups: *debug.first().unwrap_or(&0),
            debug_tested: *debug.get(1).unwrap_or(&0),
            debug_hits: *debug.get(2).unwrap_or(&0),
            stats: DispatchStats {
                triangles: mesh.triangles.len() as u32,
                tiles: brick_count,
                voxels: grid.num_voxels(),
                gpu_time_ms: None,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use glam::Vec3;
    use voxel_core::{OccupancyField, Resolution, VoxelCoord};

    use crate::core::{MeshInput, SparseVoxelizationOutput, TileSpec, VoxelGrid, VoxelizeOpts};
    use crate::error::VoxelizeGpuError;
    use crate::gpu::{GpuVoxelizer, GpuVoxelizerConfig};

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

    /// Decodes a sparse output into the set of global occupied voxel coords by
    /// walking each brick's packed occupancy bits.
    fn sparse_voxel_set(out: &SparseVoxelizationOutput) -> BTreeSet<(u32, u32, u32)> {
        let bd = out.brick_dim;
        let brick_voxels = bd * bd * bd;
        let words_per_brick = brick_voxels.div_ceil(32) as usize;
        let mut set = BTreeSet::new();
        for (b, origin) in out.brick_origins.iter().enumerate() {
            let base_word = b * words_per_brick;
            for linear in 0..brick_voxels {
                let word = base_word + (linear >> 5) as usize;
                let bit = linear & 31;
                if (out.occupancy[word] >> bit) & 1 == 1 {
                    let vx = linear % bd;
                    let vy = (linear / bd) % bd;
                    let vz = linear / (bd * bd);
                    set.insert((origin[0] + vx, origin[1] + vy, origin[2] + vz));
                }
            }
        }
        set
    }

    fn dense_voxel_set(out: &crate::core::VoxelizationOutput, n: u32) -> BTreeSet<(u32, u32, u32)> {
        let mut set = BTreeSet::new();
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    if out.occupancy.is_occupied(VoxelCoord::new(x, y, z)) {
                        set.insert((x, y, z));
                    }
                }
            }
        }
        set
    }

    /// An empty mesh on the non-chunked sparse path must early-return an empty
    /// output (no bricks), never panic on a zero-length triangle buffer.
    #[test]
    fn empty_mesh_nonchunked_returns_empty_not_panic() {
        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let grid = grid_n(8);
        let opts = VoxelizeOpts::default();
        let mesh = MeshInput {
            triangles: Vec::new(),
            material_ids: None,
        };
        let out = pollster::block_on(gpu.voxelize_surface_sparse(&mesh, &grid, &opts))
            .expect("empty mesh must produce an empty sparse output, not error");
        assert!(out.brick_origins.is_empty(), "no bricks for an empty mesh");
        assert!(
            out.occupancy.is_empty(),
            "no occupancy words for an empty mesh"
        );
        assert_eq!(out.stats.triangles, 0);
    }

    /// The union of all sparse brick occupancy must equal the dense occupancy set
    /// for the same mesh — including a triangle that straddles a brick boundary.
    #[test]
    fn sparse_union_equals_dense() {
        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let opts = VoxelizeOpts::default();
        for n in [8u32, 32] {
            let grid = grid_n(n);
            let tiles = TileSpec::new([4, 4, 4], grid.dims()).unwrap();
            let f = n as f32;
            // A large triangle straddling brick boundaries plus a second offset one.
            let mesh = MeshInput {
                triangles: vec![
                    [
                        Vec3::new(1.0, 1.0, 1.0),
                        Vec3::new(f - 1.0, 2.0, 3.0),
                        Vec3::new(2.0, f - 1.0, 5.0),
                    ],
                    [
                        Vec3::new(2.0, 2.0, f * 0.5),
                        Vec3::new(f - 2.0, 3.0, f * 0.5),
                        Vec3::new(3.0, f - 2.0, f * 0.5),
                    ],
                ],
                material_ids: None,
            };
            let dense = pollster::block_on(gpu.voxelize_surface(&mesh, &grid, &tiles, &opts))
                .expect("dense voxelize");
            let sparse = pollster::block_on(gpu.voxelize_surface_sparse(&mesh, &grid, &opts))
                .expect("sparse voxelize");
            let dense_set = dense_voxel_set(&dense, n);
            let sparse_set = sparse_voxel_set(&sparse);
            assert!(!dense_set.is_empty(), "n={n}: mesh must occupy voxels");
            assert_eq!(
                sparse_set, dense_set,
                "n={n}: sparse brick union must equal the dense occupancy set"
            );
        }
    }

    /// Geometry wholly in negative space must emit zero bricks (no phantom
    /// (0,0,0) brick) on the GPU sparse path.
    #[test]
    fn sparse_no_phantom_brick_for_outside_geometry() {
        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let grid = grid_n(8);
        let opts = VoxelizeOpts::default();
        let mesh = MeshInput {
            triangles: vec![[
                Vec3::new(-10.0, -10.0, -10.0),
                Vec3::new(-8.0, -10.0, -10.0),
                Vec3::new(-10.0, -8.0, -10.0),
            ]],
            material_ids: None,
        };
        let out = pollster::block_on(gpu.voxelize_surface_sparse(&mesh, &grid, &opts))
            .expect("outside-geometry mesh must not error");
        assert!(
            out.brick_origins.is_empty(),
            "negative-space geometry must emit no bricks (got {:?})",
            out.brick_origins
        );
    }

    /// `compact_surface_sparse` requires `store_owner`; with it false it must
    /// return `OwnerRequired` before any GPU work.
    #[test]
    fn owner_required_error_when_store_owner_false() {
        let Some(gpu) = gpu_or_skip() else {
            return;
        };
        let grid = grid_n(8);
        let opts = VoxelizeOpts {
            epsilon: 1e-4,
            store_owner: false,
            store_color: false,
        };
        let mesh = MeshInput {
            triangles: vec![[
                Vec3::new(1.0, 1.0, 1.0),
                Vec3::new(6.0, 2.0, 2.0),
                Vec3::new(2.0, 6.0, 4.0),
            ]],
            material_ids: None,
        };
        let err =
            pollster::block_on(gpu.compact_surface_sparse(&mesh, &grid, &opts, &[], [0, 0, 0]))
                .expect_err("store_owner=false must be rejected");
        assert!(
            matches!(err, VoxelizeGpuError::OwnerRequired),
            "expected OwnerRequired, got {err}"
        );
    }
}
