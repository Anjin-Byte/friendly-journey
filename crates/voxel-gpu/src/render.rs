//! The GPU-resident render path: one compute dispatch builds a camera ray per
//! pixel, traverses, shades the hit, and writes color straight to a storage
//! texture — no readback, no CPU ray-gen, no CPU shading. The viewer blits the
//! resulting texture to its surface.

// Unsafe Quarantine: the only `unsafe` is the `bytemuck` derive on the
// `#[repr(C)]` all-scalar camera uniform.
#![allow(unsafe_code)]

use bytemuck::{Pod, Zeroable};

use voxel_core::SchoolBBuffer;

use crate::buffers;
use crate::context::GpuContext;
use crate::error::GpuError;

/// The output storage-texture format the render kernel writes.
pub const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Camera uniform, matching `render.wgsl`'s `Camera` (std140-friendly: every
/// `vec3` is followed by a scalar to fill its 16-byte slot).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuCamera {
    /// Camera world position.
    pub eye: [f32; 3],
    /// `tan(fov/2)`.
    pub tan: f32,
    /// Forward (unit) direction.
    pub forward: [f32; 3],
    /// Width / height.
    pub aspect: f32,
    /// Right (unit) direction.
    pub right: [f32; 3],
    /// Grid resolution `n` as `f32`.
    pub n: f32,
    /// Up (unit) direction.
    pub up: [f32; 3],
    /// Padding to keep the following `dims` 16-byte aligned.
    pub pad: f32,
    /// `[width, height, internal_levels(k), 0]`.
    pub dims: [u32; 4],
}

/// A compiled render pipeline with one uploaded structure.
pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    node_buf: wgpu::Buffer,
    leaf_buf: wgpu::Buffer,
    camera_buf: wgpu::Buffer,
}

impl GpuRenderer {
    /// Compiles the render kernel and uploads `structure`.
    pub fn new(ctx: &GpuContext, structure: &SchoolBBuffer) -> Result<Self, GpuError> {
        let device = ctx.device.clone();
        let queue = ctx.queue.clone();

        let (node_buf, leaf_buf) =
            buffers::upload_structure(&device, structure, ctx.max_storage_binding())?;
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera"),
            size: std::mem::size_of::<GpuCamera>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render"),
            source: wgpu::ShaderSource::Wgsl(
                buffers::shader_source(include_str!("../shaders/render.wgsl")).into(),
            ),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render layout"),
            entries: &[
                buffers::storage_entry(0, true),
                buffers::storage_entry(1, true),
                buffers::uniform_entry(2),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: OUTPUT_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("render_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            layout,
            node_buf,
            leaf_buf,
            camera_buf,
        })
    }

    /// Records the render compute pass into `encoder`, writing the shaded image
    /// to `output` (an [`OUTPUT_FORMAT`] storage-texture view of size
    /// `width × height`). The caller blits `output` to its surface.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera: &GpuCamera,
        output: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(camera));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render bind group"),
            layout: &self.layout,
            entries: &[
                buffers::bind(0, self.node_buf.as_entire_binding()),
                buffers::bind(1, self.leaf_buf.as_entire_binding()),
                buffers::bind(2, self.camera_buf.as_entire_binding()),
                buffers::bind(3, wgpu::BindingResource::TextureView(output)),
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("render pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
}
