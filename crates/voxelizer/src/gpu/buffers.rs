//! Buffer-map readback helpers.
//!
//! Mirrors the workspace pattern (see `voxel-gpu`): map the buffer, drive the
//! device to completion with `device.poll(PollType::wait_indefinitely())`, then
//! receive the map result over a channel. Errors propagate as
//! [`VoxelizeGpuError`] rather than panicking.

use crate::error::VoxelizeGpuError;

/// Maps `buffer` for read, blocks until ready, and returns its contents as `u32`s.
pub(crate) fn map_buffer_u32(
    buffer: &wgpu::Buffer,
    device: &wgpu::Device,
) -> Result<Vec<u32>, VoxelizeGpuError> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|_| VoxelizeGpuError::Poll)?;
    receiver.recv().map_err(|_| VoxelizeGpuError::Poll)??;

    let data = slice.get_mapped_range();
    let result = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buffer.unmap();
    Ok(result)
}

/// Maps `buffer` for read, blocks until ready, and returns its contents as `f32`s.
pub(crate) fn map_buffer_f32(
    buffer: &wgpu::Buffer,
    device: &wgpu::Device,
) -> Result<Vec<f32>, VoxelizeGpuError> {
    let slice = buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|_| VoxelizeGpuError::Poll)?;
    receiver.recv().map_err(|_| VoxelizeGpuError::Poll)??;

    let data = slice.get_mapped_range();
    let result = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buffer.unmap();
    Ok(result)
}
