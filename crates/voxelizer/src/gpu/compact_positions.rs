//! Sparse voxel compaction - extracts world-space positions from occupancy data.

use wgpu::util::DeviceExt;

use super::{CompactParams, GpuVoxelizer, map_buffer_f32, map_buffer_u32};
use crate::error::VoxelizeGpuError;

impl GpuVoxelizer {
    /// Compacts sparse voxel occupancy into world-space positions.
    ///
    /// Returns a flat `Vec<f32>` of xyz positions (3 floats per voxel).
    pub async fn compact_sparse_positions(
        &self,
        occupancy: &[u32],
        brick_origins: &[[u32; 3]],
        brick_dim: u32,
        voxel_size: f32,
        origin_world: [f32; 3],
        max_positions: u32,
    ) -> Result<Vec<f32>, VoxelizeGpuError> {
        let (buffer, count) = self
            .compact_sparse_positions_buffer(
                occupancy,
                brick_origins,
                brick_dim,
                voxel_size,
                origin_world,
                max_positions,
            )
            .await?;

        if count == 0 {
            return Ok(Vec::new());
        }

        let positions = self.readback_positions(&buffer, count)?;
        Ok(positions)
    }

    /// Compacts sparse voxel occupancy into a GPU buffer of world-space positions.
    ///
    /// Returns the buffer and the number of positions written.
    pub async fn compact_sparse_positions_buffer(
        &self,
        occupancy: &[u32],
        brick_origins: &[[u32; 3]],
        brick_dim: u32,
        voxel_size: f32,
        origin_world: [f32; 3],
        max_positions: u32,
    ) -> Result<(wgpu::Buffer, u32), VoxelizeGpuError> {
        if max_positions == 0 {
            return Ok((self.empty_position_buffer(), 0));
        }

        let brick_count = brick_origins.len() as u32;
        if brick_count == 0 || occupancy.is_empty() {
            return Ok((self.empty_position_buffer(), 0));
        }

        self.validate_compact_pos_inputs(occupancy, brick_origins, brick_dim, max_positions)?;

        let buffers = self.create_compact_pos_buffers(
            occupancy,
            brick_origins,
            brick_dim,
            voxel_size,
            origin_world,
            max_positions,
        );

        let (count, had_workgroups) = self
            .dispatch_compact_positions(&buffers, brick_count, max_positions)
            .await?;

        if !had_workgroups {
            return Err(VoxelizeGpuError::PipelineValidation(format!(
                "compact pass produced no workgroups (brick_count={}, max_workgroups={})",
                brick_count, self.max_compute_workgroups_per_dimension
            )));
        }

        Ok((buffers.out_positions, count))
    }
}

// === Validation ===

impl GpuVoxelizer {
    fn validate_compact_pos_inputs(
        &self,
        occupancy: &[u32],
        brick_origins: &[[u32; 3]],
        brick_dim: u32,
        max_positions: u32,
    ) -> Result<(), VoxelizeGpuError> {
        let brick_count = brick_origins.len() as u32;

        if brick_count > self.max_compute_workgroups_per_dimension {
            return Err(VoxelizeGpuError::WorkgroupsExceeded {
                label: "compact positions",
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

        let occupancy_bytes = (occupancy.len() as u64).saturating_mul(4);
        self.ensure_storage_fits(occupancy_bytes, "compact occupancy")?;

        let brick_origins_bytes = (brick_origins.len() as u64).saturating_mul(16);
        self.ensure_storage_fits(brick_origins_bytes, "compact brick origins")?;

        let out_positions_bytes = u64::from(max_positions).saturating_mul(16);
        if out_positions_bytes > 0 {
            self.ensure_storage_fits(out_positions_bytes, "compact positions")?;
        }

        Ok(())
    }
}

// === Buffer Creation ===

pub(super) struct CompactPosBuffers {
    pub occupancy: wgpu::Buffer,
    pub brick_origins: wgpu::Buffer,
    pub out_positions: wgpu::Buffer,
    pub counter: wgpu::Buffer,
    pub debug: wgpu::Buffer,
    pub params: wgpu::Buffer,
}

impl GpuVoxelizer {
    fn create_compact_pos_buffers(
        &self,
        occupancy: &[u32],
        brick_origins: &[[u32; 3]],
        brick_dim: u32,
        voxel_size: f32,
        origin_world: [f32; 3],
        max_positions: u32,
    ) -> CompactPosBuffers {
        let brick_origin_data: Vec<[u32; 4]> = brick_origins
            .iter()
            .map(|o| [o[0], o[1], o[2], 0])
            .collect();

        let occupancy_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.compact.occupancy"),
                contents: bytemuck::cast_slice(occupancy),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let brick_origins_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.compact.brick_origins"),
                contents: bytemuck::cast_slice(&brick_origin_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let out_positions_bytes = u64::from(max_positions).saturating_mul(16);
        let out_positions_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer.compact.positions"),
            size: out_positions_bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let counter_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.compact.counter"),
                contents: bytemuck::cast_slice(&[0u32]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let debug_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.compact.debug"),
                contents: bytemuck::cast_slice(&[0u32, 0u32]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let params = CompactParams {
            brick_dim,
            brick_count: brick_origins.len() as u32,
            max_positions,
            origin_world: [
                origin_world[0],
                origin_world[1],
                origin_world[2],
                voxel_size,
            ],
            _pad0: 0,
        };
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("voxelizer.compact.params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        CompactPosBuffers {
            occupancy: occupancy_buf,
            brick_origins: brick_origins_buf,
            out_positions: out_positions_buf,
            counter: counter_buf,
            debug: debug_buf,
            params: params_buf,
        }
    }
}

// === Dispatch ===

impl GpuVoxelizer {
    async fn dispatch_compact_positions(
        &self,
        buffers: &CompactPosBuffers,
        brick_count: u32,
        max_positions: u32,
    ) -> Result<(u32, bool), VoxelizeGpuError> {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxelizer.compact.bind_group"),
            layout: &self.compact_bind_group_layout,
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
                    resource: buffers.out_positions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buffers.counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buffers.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buffers.debug.as_entire_binding(),
                },
            ],
        });

        let error_scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxelizer.compact.encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("voxelizer.compact.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compact_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(brick_count, 1, 1);
        }

        let read_counter = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer.compact.read_counter"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let read_debug = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer.compact.read_debug"),
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&buffers.counter, 0, &read_counter, 0, 4);
        encoder.copy_buffer_to_buffer(&buffers.debug, 0, &read_debug, 0, 8);

        self.queue.submit([encoder.finish()]);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        if let Some(err) = error_scope.pop().await {
            return Err(VoxelizeGpuError::PipelineValidation(format!(
                "compact pass validation error: {err}"
            )));
        }

        let counter = map_buffer_u32(&read_counter, &self.device)?;
        let count = counter.first().copied().unwrap_or(0).min(max_positions);

        let debug = map_buffer_u32(&read_debug, &self.device)?;
        let had_workgroups = debug.first().copied().unwrap_or(0) > 0;
        let had_hits = debug.get(1).copied().unwrap_or(0) > 0;

        if !had_hits {
            return Ok((0, had_workgroups));
        }

        Ok((count, had_workgroups))
    }
}

// === Readback ===

impl GpuVoxelizer {
    fn readback_positions(
        &self,
        buffer: &wgpu::Buffer,
        count: u32,
    ) -> Result<Vec<f32>, VoxelizeGpuError> {
        let read_positions = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxelizer.compact.read_positions"),
            size: u64::from(count) * 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("voxelizer.compact.readback"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &read_positions, 0, read_positions.size());
        self.queue.submit([encoder.finish()]);
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let data = map_buffer_f32(&read_positions, &self.device)?;

        let mut positions = Vec::with_capacity(count as usize * 3);
        for i in 0..count as usize {
            let base = i * 4;
            positions.push(data[base]);
            positions.push(data[base + 1]);
            positions.push(data[base + 2]);
        }
        Ok(positions)
    }
}

#[cfg(test)]
mod tests {
    use crate::error::VoxelizeGpuError;
    use crate::gpu::validate_brick_dim;

    /// `brick_dim = 1626` cubes to ≈4.3e9, overflowing a `u32` multiply. The
    /// shared validator must return a typed error rather than panic — the bug
    /// this fix closes in the position-compaction validator. Pure, no GPU.
    /// (`VoxelizeGpuError` isn't `PartialEq`, so we match the variant + field.)
    #[test]
    fn brick_dim_cube_does_not_overflow_u32() {
        assert!(matches!(
            validate_brick_dim(1626),
            Err(VoxelizeGpuError::InvalidBrickDim { got: 1626 })
        ));
    }

    /// `brick_dim = 0` is rejected up front (a zero brick is degenerate and would
    /// divide-by-zero downstream). Pure, no GPU.
    #[test]
    fn zero_brick_dim_rejected() {
        assert!(matches!(
            validate_brick_dim(0),
            Err(VoxelizeGpuError::InvalidBrickDim { got: 0 })
        ));
    }
}
