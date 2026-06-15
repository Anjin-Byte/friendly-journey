//! The GPU traversal: upload a School-B structure once, then dispatch batches of
//! rays through the WGSL kernel and read back hits.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use voxel_core::{NodeLayout, Ray, SchoolBBuffer, VoxelCoord};

use crate::buffers;
use crate::context::GpuContext;
use crate::error::GpuError;

/// Workgroup size; mirrors `@workgroup_size(64)` in the shader.
const WORKGROUP: u32 = 64;

// Pod upload structs. The `unsafe` here is only the `bytemuck` derive on
// `#[repr(C)]` all-scalar data (Unsafe Quarantine); none is hand-written.
#[allow(unsafe_code)]
mod pod {
    use super::{Pod, Zeroable};

    /// A ray in the shader's std430 layout (`vec3` is 16-byte aligned).
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct GpuRay {
        pub(crate) origin: [f32; 3],
        pub(crate) _pad0: f32,
        pub(crate) dir: [f32; 3],
        pub(crate) _pad1: f32,
    }

    /// Kernel parameters (uniform buffer).
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct GpuParams {
        pub(crate) n: u32,
        pub(crate) k: u32,
        pub(crate) ray_count: u32,
        pub(crate) _pad: u32,
    }
}
use pod::{GpuParams, GpuRay};

/// A compiled traversal pipeline with one uploaded structure.
pub struct GpuTraverser {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_layout: wgpu::BindGroupLayout,
    node_buf: wgpu::Buffer,
    leaf_buf: wgpu::Buffer,
    bounds_buf: wgpu::Buffer,
    n: u32,
    k: u32,
}

impl GpuTraverser {
    /// Compiles the kernel and uploads `structure` to the GPU.
    pub fn new(ctx: &GpuContext, structure: &SchoolBBuffer) -> Result<Self, GpuError> {
        let device = ctx.device.clone();
        let queue = ctx.queue.clone();
        let res = structure.resolution();

        let (node_buf, leaf_buf, bounds_buf) =
            buffers::upload_structure(&device, structure, ctx.max_storage_binding())?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hdda"),
            source: wgpu::ShaderSource::Wgsl(
                buffers::shader_source(include_str!("../shaders/hdda.wgsl")).into(),
            ),
        });

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hdda layout"),
            entries: &[
                buffers::storage_entry(0, true),  // nodes
                buffers::storage_entry(1, true),  // leaf_words
                buffers::storage_entry(2, true),  // leaf_bounds
                buffers::storage_entry(3, true),  // rays
                buffers::uniform_entry(4),        // params
                buffers::storage_entry(5, false), // hits
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hdda pipeline layout"),
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hdda pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_layout,
            node_buf,
            leaf_buf,
            bounds_buf,
            n: res.voxels_per_axis(),
            k: res.internal_levels(),
        })
    }

    /// Traverses `rays` on the GPU, returning the first occupied voxel per ray
    /// (`None` for a miss). Order matches `rays`.
    pub fn traverse(&self, rays: &[Ray]) -> Result<Vec<Option<VoxelCoord>>, GpuError> {
        if rays.is_empty() {
            return Ok(Vec::new());
        }
        let ray_count = u32::try_from(rays.len()).expect("ray batch exceeds u32::MAX");

        let gpu_rays: Vec<GpuRay> = rays
            .iter()
            .map(|r| {
                let o = r.origin.as_vec3().to_array();
                let d = r.dir.as_vec3().to_array();
                GpuRay {
                    origin: o,
                    _pad0: 0.0,
                    dir: d,
                    _pad1: 0.0,
                }
            })
            .collect();

        let ray_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rays"),
                contents: bytemuck::cast_slice(&gpu_rays),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck::bytes_of(&GpuParams {
                    n: self.n,
                    k: self.k,
                    ray_count,
                    _pad: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let hits_bytes = u64::from(ray_count) * 16; // vec4<u32>
        let hits_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hits"),
            size: hits_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hits readback"),
            size: hits_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hdda bind group"),
            layout: &self.bind_layout,
            entries: &[
                buffers::bind(0, self.node_buf.as_entire_binding()),
                buffers::bind(1, self.leaf_buf.as_entire_binding()),
                buffers::bind(2, self.bounds_buf.as_entire_binding()),
                buffers::bind(3, ray_buf.as_entire_binding()),
                buffers::bind(4, params_buf.as_entire_binding()),
                buffers::bind(5, hits_buf.as_entire_binding()),
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hdda"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hdda pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(ray_count.div_ceil(WORKGROUP), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&hits_buf, 0, &readback, 0, hits_bytes);
        self.queue.submit(Some(encoder.finish()));

        self.read_hits(&readback)
    }

    /// Blocks until the dispatch completes, then maps `readback` and decodes the
    /// `vec4<u32>` hits into `Option<VoxelCoord>` (`w == 1` ⇒ hit).
    fn read_hits(&self, readback: &wgpu::Buffer) -> Result<Vec<Option<VoxelCoord>>, GpuError> {
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|_| GpuError::Poll)?;
        rx.recv().map_err(|_| GpuError::Poll)??;

        let data = slice.get_mapped_range();
        let hits: &[[u32; 4]] = bytemuck::cast_slice(&data);
        let out = hits
            .iter()
            .map(|h| {
                if h[3] == 1 {
                    Some(VoxelCoord::new(h[0], h[1], h[2]))
                } else {
                    None
                }
            })
            .collect();
        drop(data);
        readback.unmap();
        Ok(out)
    }
}
