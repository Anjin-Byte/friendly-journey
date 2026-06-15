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

/// Reusable compute-pass timestamp resources (present iff the device supports
/// `TIMESTAMP_QUERY`): a 2-slot query set and the resolve/readback buffers.
/// Mirrors the traverser's timing so the viewer's render kernel can be measured
/// on the GPU timeline (readback of two timestamps only, no per-pixel copy).
struct RenderTiming {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// Nanoseconds per timestamp tick.
    period: f32,
}

/// A compiled render pipeline with one uploaded structure.
pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    node_buf: wgpu::Buffer,
    leaf_buf: wgpu::Buffer,
    bounds_buf: wgpu::Buffer,
    camera_buf: wgpu::Buffer,
    /// Max storage-buffer binding size, kept so [`reupload`](Self::reupload) can
    /// rebuild the structure buffers after a topology edit without the context.
    max_binding: u64,
    timing: Option<RenderTiming>,
}

impl GpuRenderer {
    /// Compiles the render kernel and uploads `structure`.
    pub fn new(ctx: &GpuContext, structure: &SchoolBBuffer) -> Result<Self, GpuError> {
        let device = ctx.device.clone();
        let queue = ctx.queue.clone();

        let max_binding = ctx.max_storage_binding();
        let (node_buf, leaf_buf, bounds_buf) =
            buffers::upload_structure(&device, structure, max_binding)?;
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
                buffers::storage_entry(0, true), // nodes
                buffers::storage_entry(1, true), // leaf_words
                buffers::storage_entry(2, true), // leaf_bounds
                buffers::uniform_entry(3),       // camera
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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

        let timing = ctx.supports_timestamps().then(|| {
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("render timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("render ts resolve"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("render ts readback"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            RenderTiming {
                query_set,
                resolve,
                readback,
                period: queue.get_timestamp_period(),
            }
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            layout,
            node_buf,
            leaf_buf,
            bounds_buf,
            camera_buf,
            max_binding,
            timing,
        })
    }

    /// Patches a single leaf onto the GPU after an in-place [`Edit::Leaf`].
    ///
    /// `structure` must already have had [`SchoolBBuffer::patch_leaf`] applied
    /// for `leaf_idx`; this copies that one leaf's 16 occupancy words (64 bytes
    /// at `leaf_idx * 64`) and its packed bounds word (4 bytes at `leaf_idx * 4`)
    /// into the resident buffers via `queue.write_buffer`. That is an `O(1)`
    /// upload flushed by the next queue submission (the next rendered frame),
    /// instead of rebuilding the whole structure. To force it through
    /// synchronously — e.g. when timing an edit headlessly — call
    /// [`flush_and_wait`](Self::flush_and_wait).
    ///
    /// [`Edit::Leaf`]: voxel_core::Edit::Leaf
    pub fn update_leaf(&self, structure: &SchoolBBuffer, leaf_idx: u32) {
        let words = structure.leaf_at(leaf_idx).words32();
        self.queue.write_buffer(
            &self.leaf_buf,
            u64::from(leaf_idx) * 64,
            bytemuck::cast_slice(&words),
        );
        let bounds = structure.leaf_bounds_words()[leaf_idx as usize];
        self.queue.write_buffer(
            &self.bounds_buf,
            u64::from(leaf_idx) * 4,
            bytemuck::bytes_of(&bounds),
        );
    }

    /// Replaces the resident structure after a topology edit
    /// ([`Edit::Topology`]), which renumbers leaf indices and invalidates the
    /// node buffer's `subtree_base` offsets. Rebuilds all three buffers from
    /// `structure` (a fresh [`SchoolBBuffer::from_sparse`] of the edited tree)
    /// and swaps them in; the per-frame bind group picks them up on the next
    /// render.
    ///
    /// [`Edit::Topology`]: voxel_core::Edit::Topology
    pub fn reupload(&mut self, structure: &SchoolBBuffer) -> Result<(), GpuError> {
        let (node_buf, leaf_buf, bounds_buf) =
            buffers::upload_structure(&self.device, structure, self.max_binding)?;
        self.node_buf = node_buf;
        self.leaf_buf = leaf_buf;
        self.bounds_buf = bounds_buf;
        Ok(())
    }

    /// Forces any staged buffer writes (from [`update_leaf`](Self::update_leaf))
    /// through to the GPU and blocks until the device is idle. The render loop
    /// does not need this — the next frame's submit flushes staged writes — but a
    /// headless caller timing an edit's full round-trip can use it.
    pub fn flush_and_wait(&self) -> Result<(), GpuError> {
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flush"),
            });
        self.queue.submit(std::iter::once(encoder.finish()));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| GpuError::Poll)?;
        Ok(())
    }

    /// Whether this renderer can report a GPU-timeline kernel time (i.e. the
    /// device supports compute-pass timestamp queries).
    pub fn supports_timing(&self) -> bool {
        self.timing.is_some()
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
        self.record(encoder, camera, output, width, height, false);
    }

    /// Like [`render`](Self::render), but brackets the compute pass with
    /// timestamp queries and appends their resolve+copy to `encoder` (when the
    /// device supports timestamps). After the caller submits `encoder` and the
    /// GPU completes, [`last_kernel_ns`](Self::last_kernel_ns) returns the
    /// traverse+shade time in nanoseconds. With no timestamp support this is
    /// identical to [`render`](Self::render) and `last_kernel_ns` yields `None`.
    pub fn render_timed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera: &GpuCamera,
        output: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        self.record(encoder, camera, output, width, height, true);
    }

    /// Shared pass recorder; when `timed` and timestamps are available, writes
    /// the begin/end timestamps and resolves them into the readback buffer.
    fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        camera: &GpuCamera,
        output: &wgpu::TextureView,
        width: u32,
        height: u32,
        timed: bool,
    ) {
        self.queue
            .write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(camera));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render bind group"),
            layout: &self.layout,
            entries: &[
                buffers::bind(0, self.node_buf.as_entire_binding()),
                buffers::bind(1, self.leaf_buf.as_entire_binding()),
                buffers::bind(2, self.bounds_buf.as_entire_binding()),
                buffers::bind(3, self.camera_buf.as_entire_binding()),
                buffers::bind(4, wgpu::BindingResource::TextureView(output)),
            ],
        });

        let timing = if timed { self.timing.as_ref() } else { None };
        let timestamp_writes = timing.map(|t| wgpu::ComputePassTimestampWrites {
            query_set: &t.query_set,
            beginning_of_pass_write_index: Some(0),
            end_of_pass_write_index: Some(1),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("render pass"),
                timestamp_writes,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }

        if let Some(t) = timing {
            encoder.resolve_query_set(&t.query_set, 0..2, &t.resolve, 0);
            encoder.copy_buffer_to_buffer(&t.resolve, 0, &t.readback, 0, 16);
        }
    }

    /// Maps the most recent [`render_timed`](Self::render_timed) timestamp pair
    /// and returns the compute-pass duration in nanoseconds, or `None` when the
    /// device lacks timestamp support. Call after the encoder has been submitted
    /// and the device polled to completion.
    #[allow(clippy::cast_precision_loss)]
    pub fn last_kernel_ns(&self) -> Result<Option<f64>, GpuError> {
        let Some(t) = self.timing.as_ref() else {
            return Ok(None);
        };
        let slice = t.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| GpuError::Poll)?;
        rx.recv().map_err(|_| GpuError::Poll)??;

        let data = slice.get_mapped_range();
        let begin = u64::from_le_bytes(data[0..8].try_into().expect("8 bytes"));
        let end = u64::from_le_bytes(data[8..16].try_into().expect("8 bytes"));
        drop(data);
        t.readback.unmap();
        Ok(Some(end.saturating_sub(begin) as f64 * f64::from(t.period)))
    }
}
