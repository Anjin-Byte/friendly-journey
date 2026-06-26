//! Compute pipeline creation for the voxelizer.

use super::shaders::{COMPACT_ATTRS_WGSL, COMPACT_VOXELS_WGSL, COMPACT_WGSL, VOXELIZER_WGSL};
use crate::error::VoxelizeGpuError;

/// Collection of compute pipelines used by the voxelizer.
pub(crate) struct GpuPipelines {
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) compact_pipeline: wgpu::ComputePipeline,
    pub(crate) compact_bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) compact_attrs_pipeline: wgpu::ComputePipeline,
    pub(crate) compact_attrs_bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) compact_voxels_pipeline: wgpu::ComputePipeline,
    pub(crate) compact_voxels_bind_group_layout: wgpu::BindGroupLayout,
}

/// Creates all compute pipelines for the voxelizer.
pub(crate) async fn create_pipelines(
    device: &wgpu::Device,
    workgroup_size: u32,
    tiles_per_workgroup: u32,
) -> Result<GpuPipelines, VoxelizeGpuError> {
    let voxelizer = create_voxelizer_pipeline(device, workgroup_size, tiles_per_workgroup);
    let compact = create_compact_pipeline(device).await?;
    let compact_attrs = create_compact_attrs_pipeline(device);
    let compact_voxels = create_compact_voxels_pipeline(device);

    Ok(GpuPipelines {
        pipeline: voxelizer.0,
        bind_group_layout: voxelizer.1,
        compact_pipeline: compact.0,
        compact_bind_group_layout: compact.1,
        compact_attrs_pipeline: compact_attrs.0,
        compact_attrs_bind_group_layout: compact_attrs.1,
        compact_voxels_pipeline: compact_voxels.0,
        compact_voxels_bind_group_layout: compact_voxels.1,
    })
}

// === Voxelizer Pipeline ===

fn create_voxelizer_pipeline(
    device: &wgpu::Device,
    workgroup_size: u32,
    tiles_per_workgroup: u32,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxelizer.wgsl"),
        source: wgpu::ShaderSource::Wgsl(VOXELIZER_WGSL.into()),
    });

    let bind_group_layout = create_voxelizer_bind_group_layout(device);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("voxelizer.pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let constants = [
        ("WORKGROUP_SIZE", f64::from(workgroup_size)),
        ("TILES_PER_WORKGROUP", f64::from(tiles_per_workgroup)),
    ];

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("voxelizer.pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &constants,
            ..Default::default()
        },
        cache: None,
    });

    (pipeline, bind_group_layout)
}

fn create_voxelizer_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("voxelizer.bind_group_layout"),
        entries: &[
            storage_buffer_entry(0, true),   // tris
            storage_buffer_entry(3, true),   // tile_offsets
            storage_buffer_entry(4, true),   // tri_indices
            storage_buffer_entry(6, false),  // occupancy
            storage_buffer_entry(7, false),  // owner_id
            storage_buffer_entry(8, false),  // color_rgba
            uniform_buffer_entry(9),         // params
            storage_buffer_entry(10, true),  // brick_origins
            storage_buffer_entry(11, false), // debug_counts
        ],
    })
}

// === Compact Pipeline ===

async fn create_compact_pipeline(
    device: &wgpu::Device,
) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout), VoxelizeGpuError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxelizer.compact.wgsl"),
        source: wgpu::ShaderSource::Wgsl(COMPACT_WGSL.into()),
    });

    let bind_group_layout = create_compact_bind_group_layout(device);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("voxelizer.compact_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("voxelizer.compact_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    if let Some(err) = error_scope.pop().await {
        return Err(VoxelizeGpuError::PipelineValidation(format!(
            "Compact pipeline validation error: {err}"
        )));
    }

    Ok((pipeline, bind_group_layout))
}

fn create_compact_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("voxelizer.compact_bind_group_layout"),
        entries: &[
            storage_buffer_entry(0, true),  // occupancy
            storage_buffer_entry(1, true),  // brick_origins
            storage_buffer_entry(2, false), // out_positions
            storage_buffer_entry(3, false), // counter
            uniform_buffer_entry(4),        // params
            storage_buffer_entry(5, false), // debug
        ],
    })
}

// === Compact Attrs Pipeline ===

fn create_compact_attrs_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxelizer.compact_attrs.wgsl"),
        source: wgpu::ShaderSource::Wgsl(COMPACT_ATTRS_WGSL.into()),
    });

    let bind_group_layout = create_compact_attrs_bind_group_layout(device);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("voxelizer.compact_attrs_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("voxelizer.compact_attrs_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    (pipeline, bind_group_layout)
}

fn create_compact_attrs_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("voxelizer.compact_attrs_bind_group_layout"),
        entries: &[
            storage_buffer_entry(0, true),  // occupancy
            storage_buffer_entry(1, true),  // brick_origins
            storage_buffer_entry(2, true),  // owner_id
            storage_buffer_entry(3, true),  // color_rgba
            storage_buffer_entry(4, false), // out_indices
            storage_buffer_entry(5, false), // out_owner
            storage_buffer_entry(6, false), // out_color
            storage_buffer_entry(7, false), // counter
            uniform_buffer_entry(8),        // params
        ],
    })
}

// === Compact Voxels Pipeline ===

fn create_compact_voxels_pipeline(
    device: &wgpu::Device,
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("voxelizer.compact_voxels.wgsl"),
        source: wgpu::ShaderSource::Wgsl(COMPACT_VOXELS_WGSL.into()),
    });

    let bind_group_layout = create_compact_voxels_bind_group_layout(device);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("voxelizer.compact_voxels_pipeline_layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("voxelizer.compact_voxels_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    (pipeline, bind_group_layout)
}

fn create_compact_voxels_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("voxelizer.compact_voxels_bind_group_layout"),
        entries: &[
            storage_buffer_entry(0, true),  // occupancy
            storage_buffer_entry(1, true),  // brick_origins
            storage_buffer_entry(2, true),  // owner_id
            storage_buffer_entry(3, true),  // material_table
            storage_buffer_entry(4, false), // out_voxels
            storage_buffer_entry(5, false), // counter
            uniform_buffer_entry(6),        // params
        ],
    })
}

// === Layout Entry Helpers ===

fn storage_buffer_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

fn uniform_buffer_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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
