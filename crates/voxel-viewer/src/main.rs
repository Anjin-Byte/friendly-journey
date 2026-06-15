//! `voxel-viewer` — an interactive viewer for the sparse MIP voxel structure.
//!
//! Fully GPU-resident: a compute pass builds a camera ray per pixel, traverses
//! the structure, shades the hit, and writes color to a texture (the same
//! `traverse_ray` the `voxel-gpu` differential validates); a fullscreen blit
//! draws that texture to the window. No per-frame readback, CPU ray-gen, or CPU
//! shading. This binary is the only holder of a windowing dependency, keeping
//! the headless `voxel` cli UI-free (Engineering Codex: *Headless First*).
//!
//! Prints build time and a rolling per-frame profile (encode / GPU / present)
//! to stderr.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use glam::Vec3;
use voxel_core::fixtures::{Checkerboard, Dust, OctantFractal, WireLattice};
use voxel_core::{Resolution, SchoolBBuffer, SparseTree};
use voxel_gpu::{GpuCamera, GpuContext, GpuRenderer};
use wgpu::CurrentSurfaceTexture::{Suboptimal, Success};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

#[derive(Parser, Clone, Copy)]
#[command(
    name = "voxel-viewer",
    about = "Render the sparse MIP voxel structure on the GPU"
)]
struct Args {
    /// Grid resolution per axis (`8·4^k`: 8, 32, 128, 512, 2048).
    #[arg(long, default_value_t = 512)]
    res: u32,
    /// Window width in pixels (one primary ray per pixel).
    #[arg(long, default_value_t = 800)]
    width: u32,
    /// Window height in pixels.
    #[arg(long, default_value_t = 600)]
    height: u32,
    /// Occupancy fixture to render.
    #[arg(long, value_enum, default_value_t = Fixture::Sierpinski)]
    fixture: Fixture,
    /// Cap the frame rate to the display refresh (off by default, so the
    /// profile reflects raw GPU throughput).
    #[arg(long)]
    vsync: bool,
    /// Exit after rendering this many frames (`0` = run until closed). Lets the
    /// viewer be driven non-interactively for profiling.
    #[arg(long, default_value_t = 0)]
    frames: u32,
}

#[derive(Clone, Copy, ValueEnum)]
enum Fixture {
    /// Sierpinski tetrahedron, `D = 2`.
    Sierpinski,
    /// Cantor dust, `D = 1`.
    Cantor,
    /// 3-D checkerboard, `D ≈ 3`.
    Checkerboard,
    /// Thin 3-D wireframe lattice — traversal-pathology stress.
    WireLattice,
    /// Sparse hashed noise — warp-divergence stress.
    Dust,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App { args, state: None };
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct App {
    args: Args,
    state: Option<Viewer>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            match Viewer::new(event_loop, self.args) {
                Ok(v) => self.state = Some(v),
                Err(e) => {
                    eprintln!("viewer init failed: {e:?}");
                    event_loop.exit();
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(viewer) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => viewer.resize(size.width.max(1), size.height.max(1)),
            WindowEvent::RedrawRequested => {
                viewer.render();
                if viewer.done() {
                    event_loop.exit();
                } else {
                    viewer.window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

/// Rolling per-frame timing, reported every `REPORT_EVERY` frames.
#[derive(Default)]
struct Profile {
    frames: u32,
    encode: f64,
    gpu: f64,
    present: f64,
    total: f64,
}

const REPORT_EVERY: u32 = 120;

struct Viewer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    ctx: GpuContext,
    renderer: GpuRenderer,
    output_view: wgpu::TextureView,
    blit_pipeline: wgpu::RenderPipeline,
    blit_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    blit_bind: wgpu::BindGroup,
    resolution: Resolution,
    angle: f32,
    profile: Profile,
    max_frames: u32,
    frames_total: u32,
}

impl Viewer {
    fn new(event_loop: &ActiveEventLoop, args: Args) -> Result<Self> {
        let attrs = Window::default_attributes()
            .with_title("voxel-viewer — sparse MIP HDDA on the GPU")
            .with_inner_size(winit::dpi::LogicalSize::new(args.width, args.height));
        let window = Arc::new(event_loop.create_window(attrs)?);

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone())?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .map_err(|_| anyhow::anyhow!("no GPU adapter compatible with the window surface"))?;
        eprintln!("adapter: {}", adapter.get_info().name);

        let adapter_limits = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            ..wgpu::Limits::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("viewer device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        }))
        .context("request_device")?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: if args.vsync {
                wgpu::PresentMode::AutoVsync
            } else {
                wgpu::PresentMode::AutoNoVsync
            },
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // Build + upload the structure (timed; the scan is parallel in core).
        let resolution = Resolution::new(args.res)?;
        eprintln!("building {}³ structure…", resolution.voxels_per_axis());
        let t = Instant::now();
        let tree = match args.fixture {
            Fixture::Sierpinski => {
                SparseTree::build(&OctantFractal::sierpinski_tetrahedron(resolution))
            }
            Fixture::Cantor => SparseTree::build(&OctantFractal::cantor_dust(resolution)),
            Fixture::Checkerboard => SparseTree::build(&Checkerboard { resolution }),
            Fixture::WireLattice => SparseTree::build(&WireLattice::new(resolution)),
            Fixture::Dust => SparseTree::build(&Dust::new(resolution)),
        };
        let structure = SchoolBBuffer::from_sparse(&tree);
        eprintln!(
            "built {}³: {} nodes, {} leaves in {:.2?}",
            resolution.voxels_per_axis(),
            tree.node_count(),
            tree.leaf_count(),
            t.elapsed(),
        );

        let ctx = GpuContext { device, queue };
        let renderer = GpuRenderer::new(&ctx, &structure)?;

        // Blit pipeline + the intermediate render-output texture.
        let (blit_pipeline, blit_layout, sampler) = build_blit(&ctx.device, format);
        let output_view = make_output(&ctx.device, config.width, config.height);
        let blit_bind = make_blit_bind(&ctx.device, &blit_layout, &output_view, &sampler);

        window.request_redraw();
        Ok(Self {
            window,
            surface,
            config,
            ctx,
            renderer,
            output_view,
            blit_pipeline,
            blit_layout,
            sampler,
            blit_bind,
            resolution,
            angle: 0.0,
            profile: Profile::default(),
            max_frames: args.frames,
            frames_total: 0,
        })
    }

    /// Whether the configured frame budget (if any) has been rendered.
    fn done(&self) -> bool {
        self.max_frames != 0 && self.frames_total >= self.max_frames
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.ctx.device, &self.config);
        self.output_view = make_output(&self.ctx.device, width, height);
        self.blit_bind = make_blit_bind(
            &self.ctx.device,
            &self.blit_layout,
            &self.output_view,
            &self.sampler,
        );
    }

    fn render(&mut self) {
        let (w, h) = (self.config.width, self.config.height);
        self.angle += 0.01;

        let t0 = Instant::now();
        let camera = self.camera(w, h);
        let (Success(frame) | Suboptimal(frame)) = self.surface.get_current_texture() else {
            self.surface.configure(&self.ctx.device, &self.config);
            return;
        };
        let target = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        // Compute: traverse + shade → output texture.
        self.renderer
            .render(&mut encoder, &camera, &self.output_view, w, h);
        // Blit: output texture → surface.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.blit_pipeline);
            pass.set_bind_group(0, &self.blit_bind, &[]);
            pass.draw(0..3, 0..1);
        }
        let t_encode = t0.elapsed();

        self.ctx.queue.submit([encoder.finish()]);
        let gpu_start = Instant::now();
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
        let t_gpu = gpu_start.elapsed();

        let present_start = Instant::now();
        frame.present();
        let t_present = present_start.elapsed();

        self.record(w, h, t_encode, t_gpu, t_present, t0.elapsed());
    }

    fn record(
        &mut self,
        w: u32,
        h: u32,
        encode: std::time::Duration,
        gpu: std::time::Duration,
        present: std::time::Duration,
        total: std::time::Duration,
    ) {
        self.frames_total = self.frames_total.saturating_add(1);
        let p = &mut self.profile;
        p.frames += 1;
        p.encode += encode.as_secs_f64();
        p.gpu += gpu.as_secs_f64();
        p.present += present.as_secs_f64();
        p.total += total.as_secs_f64();
        if p.frames >= REPORT_EVERY {
            let f = f64::from(p.frames);
            eprintln!(
                "{n}³ {w}x{h} {kr}k px | {ms:.2} ms/frame ({fps:.0} fps) | \
                 encode {en:.2} · gpu(traverse+shade+blit) {gp:.2} · present {pr:.2} ms",
                n = self.resolution.voxels_per_axis(),
                kr = (w * h) / 1000,
                ms = p.total / f * 1000.0,
                fps = f / p.total,
                en = p.encode / f * 1000.0,
                gp = p.gpu / f * 1000.0,
                pr = p.present / f * 1000.0,
            );
            self.profile = Profile::default();
        }
    }

    /// The orbiting camera as a [`GpuCamera`] uniform.
    fn camera(&self, w: u32, h: u32) -> GpuCamera {
        let nf = self.resolution.voxels_per_axis() as f32;
        let centre = Vec3::splat(nf * 0.5);
        let radius = nf * 1.6;
        let eye = centre
            + Vec3::new(
                self.angle.cos() * radius,
                nf * 0.35,
                self.angle.sin() * radius,
            );
        let forward = (centre - eye).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward);
        GpuCamera {
            eye: eye.to_array(),
            tan: (60f32.to_radians() * 0.5).tan(),
            forward: forward.to_array(),
            aspect: w as f32 / h as f32,
            right: right.to_array(),
            n: nf,
            up: up.to_array(),
            pad: 0.0,
            dims: [w, h, self.resolution.internal_levels(), 0],
        }
    }
}

/// Builds the fullscreen-blit render pipeline, its bind-group layout, and a
/// nearest-filter sampler.
fn build_blit(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout, wgpu::Sampler) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("blit"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/blit.wgsl").into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("blit layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("blit pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("blit pipeline"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("blit sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    (pipeline, layout, sampler)
}

fn make_output(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("render output"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: voxel_gpu::OUTPUT_FORMAT,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn make_blit_bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    output: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blit bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(output),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}
