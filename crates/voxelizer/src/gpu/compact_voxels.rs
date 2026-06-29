//! Compact voxel extraction — produces (vx, vy, vz, material) tuples from sparse voxelizer output.
//!
//! GPU-side material resolution via packed u16 material table (ADR-0009).

use wgpu::util::DeviceExt;

use super::{CompactVoxelsParams, GpuVoxelizer, map_buffer_u32};
use crate::core::CompactVoxel;
use crate::error::VoxelizeGpuError;

impl GpuVoxelizer {
    /// Compacts sparse voxel data into `CompactVoxel` tuples with resolved materials.
    ///
    /// # Arguments
    /// * `occupancy` — per-brick occupancy bitmasks
    /// * `owner_id` — per-voxel owner triangle index (0xFFFFFFFF = no owner)
    /// * `brick_origins` — origin of each brick in grid space
    /// * `brick_dim` — dimension of each brick (e.g. 8)
    /// * `max_entries` — maximum output voxels (output buffer size)
    /// * `material_table` — packed u16 material IDs (two per u32 word)
    /// * `g_origin` — global voxel-space origin offset
    // Stays `async` to preserve the public GPU-orchestration API contract
    // (callers `.await` it); the readback path is now synchronous.
    // Many arguments: this is a GPU dispatch entry binding several input buffers.
    #[allow(clippy::unused_async, clippy::too_many_arguments)]
    pub async fn compact_sparse_voxels(
        &self,
        occupancy: &[u32],
        owner_id: &[u32],
        brick_origins: &[[u32; 3]],
        brick_dim: u32,
        max_entries: u32,
        material_table: &[u32],
        g_origin: [i32; 3],
    ) -> Result<Vec<CompactVoxel>, VoxelizeGpuError> {
        if max_entries == 0 {
            return Ok(Vec::new());
        }

        let brick_count = brick_origins.len() as u32;
        if brick_count == 0 || occupancy.is_empty() {
            return Ok(Vec::new());
        }

        self.validate_compact_voxels_inputs(
            occupancy,
            owner_id,
            brick_origins,
            brick_dim,
            max_entries,
        )?;

        let buffers = self.create_compact_voxels_buffers(
            occupancy,
            owner_id,
            brick_origins,
            brick_dim,
            max_entries,
            material_table,
            g_origin,
        );

        let count = self.dispatch_compact_voxels(&buffers, brick_count, max_entries)?;

        if count == 0 {
            return Ok(Vec::new());
        }

        let voxels = self.readback_voxels(&buffers, count, max_entries)?;
        Ok(voxels)
    }
}

// === Validation ===

impl GpuVoxelizer {
    fn validate_compact_voxels_inputs(
        &self,
        occupancy: &[u32],
        owner_id: &[u32],
        brick_origins: &[[u32; 3]],
        brick_dim: u32,
        max_entries: u32,
    ) -> Result<(), VoxelizeGpuError> {
        let brick_count = brick_origins.len() as u32;

        if brick_count > self.max_compute_workgroups_per_dimension {
            return Err(VoxelizeGpuError::WorkgroupsExceeded {
                label: "compact voxels",
                workgroups: brick_count,
                limit: self.max_compute_workgroups_per_dimension,
            });
        }

        let brick_voxels = self.validate_brick_dim(brick_dim)?;
        let words_per_brick = brick_voxels.div_ceil(32);
        let expected_words = words_per_brick as usize * brick_origins.len();

        if occupancy.len() < expected_words {
            return Err(VoxelizeGpuError::PipelineValidation(
                "occupancy buffer too small for brick list".to_string(),
            ));
        }

        let expected_attrs = brick_voxels as usize * brick_origins.len();
        if owner_id.len() < expected_attrs {
            return Err(VoxelizeGpuError::PipelineValidation(
                "owner_id buffer too small for brick list".to_string(),
            ));
        }

        let occupancy_bytes = (occupancy.len() as u64).saturating_mul(4);
        self.ensure_storage_fits(occupancy_bytes, "compact voxels occupancy")?;

        let brick_origins_bytes = (brick_origins.len() as u64).saturating_mul(16);
        self.ensure_storage_fits(brick_origins_bytes, "compact voxels brick origins")?;

        let owner_bytes = (owner_id.len() as u64).saturating_mul(4);
        self.ensure_storage_fits(owner_bytes, "compact voxels owner")?;

        // Output buffer: 16 bytes per CompactVoxel
        let out_bytes = u64::from(max_entries).saturating_mul(16);
        if out_bytes > 0 {
            self.ensure_storage_fits(out_bytes, "compact voxels output")?;
        }

        Ok(())
    }
}

// === Buffer Creation ===

struct CompactVoxelsBuffers {
    occupancy: wgpu::Buffer,
    brick_origins: wgpu::Buffer,
    owner: wgpu::Buffer,
    material_table: wgpu::Buffer,
    out_voxels: wgpu::Buffer,
    counter: wgpu::Buffer,
    params: wgpu::Buffer,
}

impl GpuVoxelizer {
    // Many arguments: builds the full set of GPU buffers for the voxel dispatch.
    #[allow(clippy::too_many_arguments)]
    fn create_compact_voxels_buffers(
        &self,
        occupancy: &[u32],
        owner_id: &[u32],
        brick_origins: &[[u32; 3]],
        brick_dim: u32,
        max_entries: u32,
        material_table: &[u32],
        g_origin: [i32; 3],
    ) -> CompactVoxelsBuffers {
        let brick_origin_data: Vec<[u32; 4]> = brick_origins
            .iter()
            .map(|o| [o[0], o[1], o[2], 0])
            .collect();

        let occupancy_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.compact_voxels.occupancy"),
                contents: bytemuck::cast_slice(occupancy),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let brick_origins_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.compact_voxels.brick_origins"),
                contents: bytemuck::cast_slice(&brick_origin_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let owner_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.compact_voxels.owner"),
                contents: bytemuck::cast_slice(owner_id),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Material table — ensure at least 4 bytes (wgpu requires non-zero buffers)
        let mat_table_contents = if material_table.is_empty() {
            bytemuck::cast_slice(&[0u32])
        } else {
            bytemuck::cast_slice(material_table)
        };
        let material_table_buf =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("voxelizer.compact_voxels.material_table"),
                    contents: mat_table_contents,
                    usage: wgpu::BufferUsages::STORAGE,
                });

        // Output: 16 bytes per CompactVoxel (4x u32/i32)
        let out_bytes = u64::from(max_entries).saturating_mul(16).max(16);
        let out_voxels_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer.compact_voxels.out_voxels"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let counter_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.compact_voxels.counter"),
                contents: bytemuck::cast_slice(&[0u32]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let params = CompactVoxelsParams {
            brick_dim,
            brick_count: brick_origins.len() as u32,
            max_entries,
            material_table_len: material_table.len() as u32,
            g_origin: [g_origin[0], g_origin[1], g_origin[2], 0],
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.compact_voxels.params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        CompactVoxelsBuffers {
            occupancy: occupancy_buf,
            brick_origins: brick_origins_buf,
            owner: owner_buf,
            material_table: material_table_buf,
            out_voxels: out_voxels_buf,
            counter: counter_buf,
            params: params_buf,
        }
    }
}

// === Dispatch ===

impl GpuVoxelizer {
    fn dispatch_compact_voxels(
        &self,
        buffers: &CompactVoxelsBuffers,
        brick_count: u32,
        max_entries: u32,
    ) -> Result<u32, VoxelizeGpuError> {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxelizer.compact_voxels.bind_group"),
            layout: &self.compact_voxels_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.occupancy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.brick_origins.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.owner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.material_table.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.out_voxels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.params.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxelizer.compact_voxels.encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voxelizer.compact_voxels.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compact_voxels_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(brick_count, 1, 1);
        }

        let read_counter = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer.compact_voxels.read_counter"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&buffers.counter, 0, &read_counter, 0, 4);
        self.queue.submit([encoder.finish()]);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let counter = map_buffer_u32(&read_counter, &self.device)?;
        let raw = counter.first().copied().unwrap_or(0);
        // Tripwire (docs/materials/09 step 2): the shader emits one voxel per
        // occupied bit, so its count must not exceed `max_entries` (the host
        // occupancy popcount the output buffer is sized for). A raw count ABOVE it
        // means the `.min` below silently dropped voxels — an occupancy/owner
        // desync. Structurally impossible today, but guard loudly so a future
        // regression is not a silent, incomplete tree.
        debug_assert!(
            raw <= max_entries,
            "compact_voxels truncated: shader emitted {raw} voxels > max_entries {max_entries}"
        );
        Ok(raw.min(max_entries))
    }
}

// === Readback ===

impl GpuVoxelizer {
    fn readback_voxels(
        &self,
        buffers: &CompactVoxelsBuffers,
        count: u32,
        max_entries: u32,
    ) -> Result<Vec<CompactVoxel>, VoxelizeGpuError> {
        // Read back the full output buffer (16 bytes per voxel)
        let size = u64::from(max_entries) * 16;

        let read_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer.compact_voxels.readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxelizer.compact_voxels.readback_encoder"),
            });
        encoder.copy_buffer_to_buffer(&buffers.out_voxels, 0, &read_buf, 0, size);
        self.queue.submit([encoder.finish()]);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let raw = map_buffer_u32(&read_buf, &self.device)?;
        // Each CompactVoxel is 4 u32s (16 bytes)
        let all_voxels: &[CompactVoxel] = bytemuck::cast_slice(&raw);
        Ok(all_voxels[..count as usize].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use crate::error::VoxelizeGpuError;
    use crate::gpu::validate_brick_dim;

    /// `brick_dim = 1626` overflows a `u32` cube; the voxel-compaction validator
    /// must return a typed error, not panic. Pure, no GPU. (`VoxelizeGpuError`
    /// isn't `PartialEq`, so we match the variant + field.)
    #[test]
    fn brick_dim_cube_does_not_overflow_u32() {
        assert!(matches!(
            validate_brick_dim(1626),
            Err(VoxelizeGpuError::InvalidBrickDim { got: 1626 })
        ));
    }

    /// `brick_dim = 0` is rejected up front. Pure, no GPU.
    #[test]
    fn zero_brick_dim_rejected() {
        assert!(matches!(
            validate_brick_dim(0),
            Err(VoxelizeGpuError::InvalidBrickDim { got: 0 })
        ));
    }
}
