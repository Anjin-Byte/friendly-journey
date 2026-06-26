//! Sparse voxel compaction - extracts attributes (indices, owners, colors) from sparse data.

use wgpu::util::DeviceExt;

use super::{CompactAttrsParams, GpuVoxelizer, map_buffer_u32};
use crate::error::VoxelizeGpuError;

impl GpuVoxelizer {
    /// Compacts sparse voxel data into indices, owner IDs, and colors.
    // Stays `async` to preserve the public GPU-orchestration API contract
    // (callers `.await` it); the readback path is now synchronous.
    // Many arguments: this is a GPU dispatch entry binding several input buffers.
    // Return triple `(indices, owners, colors)` is the natural compaction result.
    #[allow(
        clippy::unused_async,
        clippy::too_many_arguments,
        clippy::type_complexity
    )]
    pub async fn compact_sparse_attributes(
        &self,
        occupancy: &[u32],
        owner_id: &[u32],
        color_rgba: &[u32],
        brick_origins: &[[u32; 3]],
        brick_dim: u32,
        grid_dims: [u32; 3],
        max_entries: u32,
    ) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>), VoxelizeGpuError> {
        if max_entries == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let brick_count = brick_origins.len() as u32;
        if brick_count == 0 || occupancy.is_empty() {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        self.validate_compact_attrs_inputs(
            occupancy,
            owner_id,
            color_rgba,
            brick_origins,
            brick_dim,
            max_entries,
        )?;

        let buffers = self.create_compact_attrs_buffers(
            occupancy,
            owner_id,
            color_rgba,
            brick_origins,
            brick_dim,
            grid_dims,
            max_entries,
        );

        let count = self.dispatch_compact_attrs(&buffers, brick_count, max_entries)?;

        if count == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        let (indices, owners, colors) = self.readback_attrs(&buffers, count, max_entries)?;
        Ok((indices, owners, colors))
    }
}

// === Validation ===

impl GpuVoxelizer {
    fn validate_compact_attrs_inputs(
        &self,
        occupancy: &[u32],
        owner_id: &[u32],
        color_rgba: &[u32],
        brick_origins: &[[u32; 3]],
        brick_dim: u32,
        max_entries: u32,
    ) -> Result<(), VoxelizeGpuError> {
        let brick_count = brick_origins.len() as u32;

        if brick_count > self.max_compute_workgroups_per_dimension {
            return Err(VoxelizeGpuError::WorkgroupsExceeded {
                label: "compact attrs",
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
        if owner_id.len() < expected_attrs || color_rgba.len() < expected_attrs {
            return Err(VoxelizeGpuError::PipelineValidation(
                "owner/color buffer too small for brick list".to_string(),
            ));
        }

        let occupancy_bytes = (occupancy.len() as u64).saturating_mul(4);
        self.ensure_storage_fits(occupancy_bytes, "compact attrs occupancy")?;

        let brick_origins_bytes = (brick_origins.len() as u64).saturating_mul(16);
        self.ensure_storage_fits(brick_origins_bytes, "compact attrs brick origins")?;

        let owner_bytes = (owner_id.len() as u64).saturating_mul(4);
        self.ensure_storage_fits(owner_bytes, "compact attrs owner")?;

        let color_bytes = (color_rgba.len() as u64).saturating_mul(4);
        self.ensure_storage_fits(color_bytes, "compact attrs color")?;

        let out_entries_bytes = u64::from(max_entries).saturating_mul(4);
        if out_entries_bytes > 0 {
            self.ensure_storage_fits(out_entries_bytes, "compact attrs out_indices")?;
        }

        Ok(())
    }
}

// === Buffer Creation ===

struct CompactAttrsBuffers {
    occupancy: wgpu::Buffer,
    brick_origins: wgpu::Buffer,
    owner: wgpu::Buffer,
    color: wgpu::Buffer,
    out_indices: wgpu::Buffer,
    out_owner: wgpu::Buffer,
    out_color: wgpu::Buffer,
    counter: wgpu::Buffer,
    params: wgpu::Buffer,
}

impl GpuVoxelizer {
    // Many arguments: builds the full set of GPU buffers for the attrs dispatch.
    #[allow(clippy::too_many_arguments)]
    fn create_compact_attrs_buffers(
        &self,
        occupancy: &[u32],
        owner_id: &[u32],
        color_rgba: &[u32],
        brick_origins: &[[u32; 3]],
        brick_dim: u32,
        grid_dims: [u32; 3],
        max_entries: u32,
    ) -> CompactAttrsBuffers {
        let brick_origin_data: Vec<[u32; 4]> = brick_origins
            .iter()
            .map(|o| [o[0], o[1], o[2], 0])
            .collect();

        let occupancy_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.compact_attrs.occupancy"),
                contents: bytemuck::cast_slice(occupancy),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let brick_origins_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.compact_attrs.brick_origins"),
                contents: bytemuck::cast_slice(&brick_origin_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let owner_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.compact_attrs.owner"),
                contents: bytemuck::cast_slice(owner_id),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let color_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.compact_attrs.color"),
                contents: bytemuck::cast_slice(color_rgba),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let out_entries_bytes = u64::from(max_entries).saturating_mul(4).max(4);
        let out_indices_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer.compact_attrs.indices"),
            size: out_entries_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let out_owner_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer.compact_attrs.out_owner"),
            size: out_entries_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let out_color_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer.compact_attrs.out_color"),
            size: out_entries_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let counter_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.compact_attrs.counter"),
                contents: bytemuck::cast_slice(&[0u32]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let params = CompactAttrsParams {
            brick_dim,
            brick_count: brick_origins.len() as u32,
            max_entries,
            _pad0: 0,
            grid_dims: [grid_dims[0], grid_dims[1], grid_dims[2], 0],
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.compact_attrs.params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        CompactAttrsBuffers {
            occupancy: occupancy_buf,
            brick_origins: brick_origins_buf,
            owner: owner_buf,
            color: color_buf,
            out_indices: out_indices_buf,
            out_owner: out_owner_buf,
            out_color: out_color_buf,
            counter: counter_buf,
            params: params_buf,
        }
    }
}

// === Dispatch ===

impl GpuVoxelizer {
    fn dispatch_compact_attrs(
        &self,
        buffers: &CompactAttrsBuffers,
        brick_count: u32,
        max_entries: u32,
    ) -> Result<u32, VoxelizeGpuError> {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxelizer.compact_attrs.bind_group"),
            layout: &self.compact_attrs_bind_group_layout,
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
                    resource: buffers.color.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.out_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.out_owner.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buffers.out_color.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buffers.counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buffers.params.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxelizer.compact_attrs.encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voxelizer.compact_attrs.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compact_attrs_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(brick_count, 1, 1);
        }

        let read_counter = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer.compact_attrs.read_counter"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&buffers.counter, 0, &read_counter, 0, 4);
        self.queue.submit([encoder.finish()]);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let counter = map_buffer_u32(&read_counter, &self.device)?;
        Ok(counter.first().copied().unwrap_or(0).min(max_entries))
    }
}

// === Readback ===

impl GpuVoxelizer {
    // Return triple `(indices, owners, colors)` mirrors the public entry point.
    #[allow(clippy::type_complexity)]
    fn readback_attrs(
        &self,
        buffers: &CompactAttrsBuffers,
        count: u32,
        max_entries: u32,
    ) -> Result<(Vec<u32>, Vec<u32>, Vec<u32>), VoxelizeGpuError> {
        let size = u64::from(max_entries) * 4;

        let read_indices = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer.compact_attrs.read_indices"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let read_owner = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer.compact_attrs.read_owner"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let read_color = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer.compact_attrs.read_color"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxelizer.compact_attrs.readback"),
            });
        encoder.copy_buffer_to_buffer(&buffers.out_indices, 0, &read_indices, 0, size);
        encoder.copy_buffer_to_buffer(&buffers.out_owner, 0, &read_owner, 0, size);
        encoder.copy_buffer_to_buffer(&buffers.out_color, 0, &read_color, 0, size);
        self.queue.submit([encoder.finish()]);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let indices = map_buffer_u32(&read_indices, &self.device)?;
        let owners = map_buffer_u32(&read_owner, &self.device)?;
        let colors = map_buffer_u32(&read_color, &self.device)?;

        let count = count as usize;
        Ok((
            indices[..count].to_vec(),
            owners[..count].to_vec(),
            colors[..count].to_vec(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::error::VoxelizeGpuError;
    use crate::gpu::validate_brick_dim;

    /// `brick_dim = 1626` overflows a `u32` cube; the attribute-compaction
    /// validator must return a typed error, not panic. Pure, no GPU.
    /// (`VoxelizeGpuError` isn't `PartialEq`, so we match the variant + field.)
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
