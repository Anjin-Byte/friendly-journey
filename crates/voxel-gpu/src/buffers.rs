//! Shared GPU resources for both the buffer path ([`crate::GpuTraverser`]) and
//! the render path ([`crate::GpuRenderer`]): structure upload, the concatenated
//! shader source, and bind-group-layout helpers.

use wgpu::util::DeviceExt;

use voxel_core::SchoolBBuffer;

use crate::error::GpuError;

/// Builds a shader module source by concatenating the shared traversal core
/// ahead of an entry-point module, so both kernels call the same
/// `traverse_ray`.
pub(crate) fn shader_source(entry: &str) -> String {
    format!("{}\n{}", include_str!("../shaders/traversal.wgsl"), entry)
}

/// Uploads the node, leaf, and per-leaf-bounds buffers (each padded to be
/// non-zero-sized so the `k = 0` / empty cases are valid). Returns
/// `(nodes, leaves, leaf_bounds)`.
pub(crate) fn upload_structure(
    device: &wgpu::Device,
    structure: &SchoolBBuffer,
    limit: u64,
) -> Result<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer), GpuError> {
    let mut node_bytes = bytemuck::cast_slice::<_, u8>(structure.nodes()).to_vec();
    if node_bytes.is_empty() {
        node_bytes = vec![0u8; std::mem::size_of::<voxel_core::GpuNode>()];
    }
    let mut leaf_words: Vec<u32> = structure
        .leaves()
        .iter()
        .flat_map(voxel_core::LeafBrick::words32)
        .collect();
    if leaf_words.is_empty() {
        leaf_words = vec![0u32; 16];
    }
    // One packed LeafBounds word per leaf (same order as `leaf_words`). The
    // empty-structure padding is the FULL bound (never skip), not 0 — which would
    // decode to a bogus single-voxel box — so a stray read stays conservative.
    let mut bound_words = structure.leaf_bounds_words().to_vec();
    if bound_words.is_empty() {
        bound_words = vec![voxel_core::LeafBounds::FULL.pack(); 1];
    }

    for needed in [
        node_bytes.len() as u64,
        (leaf_words.len() * 4) as u64,
        (bound_words.len() * 4) as u64,
    ] {
        if needed > limit {
            return Err(GpuError::BufferTooLarge { needed, limit });
        }
    }

    let nodes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("nodes"),
        contents: &node_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    // `COPY_DST` on the leaf and bounds buffers so an in-place [`Edit::Leaf`] can
    // be patched with `queue.write_buffer` (one leaf's 64 words-bytes + 4 bounds-
    // bytes) instead of rebuilding the whole structure. The node buffer is never
    // patched in place — a topology edit renumbers indices and recreates all
    // three — so it stays `STORAGE`-only.
    let leaves = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("leaves"),
        contents: bytemuck::cast_slice(&leaf_words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let leaf_bounds = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("leaf_bounds"),
        contents: bytemuck::cast_slice(&bound_words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    Ok((nodes, leaves, leaf_bounds))
}

pub(crate) fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn bind(binding: u32, resource: wgpu::BindingResource) -> wgpu::BindGroupEntry {
    wgpu::BindGroupEntry { binding, resource }
}
