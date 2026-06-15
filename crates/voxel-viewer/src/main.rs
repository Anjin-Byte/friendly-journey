//! `voxel-viewer` — an interactive viewer for the sparse MIP voxel structure.
//!
//! Fully GPU-resident: a compute pass builds a camera ray per pixel, traverses
//! the structure, shades the hit, and writes color to a texture (the same
//! `traverse_ray` the `voxel-gpu` differential validates); a fullscreen blit
//! draws that texture to the window, and a small overlay draws the HUD. No
//! per-frame readback of the image, CPU ray-gen, or CPU shading. This binary is
//! the only holder of a windowing dependency, keeping the headless `voxel` cli
//! UI-free (Engineering Codex: *Headless First*).
//!
//! Two cameras share the renderer: a deterministic orbit (the default, and the
//! path the scripted `--frames` profiling run uses) and an interactive free-fly
//! camera (`WASD`+`QE` to move, drag to look, scroll for speed). The first
//! manual input hands control to the fly camera; `Tab` toggles back. The HUD
//! (`H` to toggle) shows FPS, frame/kernel times, scene info, and a frame-time
//! sparkline — so the orientation cost swing is visible while orbiting.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

mod camera;
mod hud;
mod input;

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use glam::Vec3;
use voxel_core::fixtures::{Checkerboard, Dust, NoiseField, OctantFractal, WireLattice};
use voxel_core::{Resolution, SchoolBBuffer, SparseTree};
use voxel_gpu::{GpuCamera, GpuContext, GpuRenderer};
use wgpu::CurrentSurfaceTexture::{Suboptimal, Success};
use winit::application::ApplicationHandler;
use winit::event::{
    DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use camera::{FlyCamera, orbit_camera, orbit_eye_forward};
use hud::{Hud, HudBuilder};
use input::Input;

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
    /// Perlin fBm isosurface — smooth organic clouds/caves.
    Perlin,
    /// Domain-warped ridged multifractal — interconnected veins/caverns.
    Caves,
}

impl Fixture {
    /// Short upper-case label for the HUD.
    fn label(self) -> &'static str {
        match self {
            Fixture::Sierpinski => "SIERPINSKI",
            Fixture::Cantor => "CANTOR",
            Fixture::Checkerboard => "CHECKERBOARD",
            Fixture::WireLattice => "WIRE-LATTICE",
            Fixture::Dust => "DUST",
            Fixture::Perlin => "PERLIN",
            Fixture::Caves => "CAVES",
        }
    }
}

/// Which camera is driving the view.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CamMode {
    /// Deterministic turntable (default; used by scripted profiling).
    Orbit,
    /// Interactive free-fly camera.
    Free,
}

impl CamMode {
    fn label(self) -> &'static str {
        match self {
            CamMode::Orbit => "ORBIT",
            CamMode::Free => "FREE",
        }
    }
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
            WindowEvent::KeyboardInput { event, .. } => viewer.on_key(&event, event_loop),
            WindowEvent::MouseInput { state, button, .. } => viewer.on_mouse_button(state, button),
            WindowEvent::MouseWheel { delta, .. } => viewer.on_scroll(delta),
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

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let (Some(viewer), DeviceEvent::MouseMotion { delta }) = (self.state.as_mut(), event) {
            if viewer.mouse_look {
                viewer.input.look_dx += delta.0 as f32;
                viewer.input.look_dy += delta.1 as f32;
            }
        }
    }
}

/// Frame-time samples kept for the HUD sparkline and min/avg/max.
const HISTORY_CAP: usize = 120;
/// Target frame budget (60 Hz) used to tint sparkline spikes.
const BUDGET_MS: f32 = 1000.0 / 60.0;

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
    hud: Hud,
    show_hud: bool,
    resolution: Resolution,
    fixture: Fixture,
    node_count: usize,
    leaf_count: usize,
    // Camera state.
    mode: CamMode,
    fly: FlyCamera,
    input: Input,
    mouse_look: bool,
    angle: f32,
    // Timing.
    last_instant: Instant,
    history: Vec<f32>,
    kernel_ms: f64,
    encode_ms: f64,
    present_ms: f64,
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
        // Request timestamp queries when available so the HUD can show the true
        // traverse+shade kernel time (readback-free); harmless if unsupported.
        let mut features = wgpu::Features::empty();
        if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("viewer device"),
            required_features: features,
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

        // Build + upload the structure (the occupancy scan is parallel in core).
        let resolution = Resolution::new(args.res)?;
        eprintln!("building {}³ structure…", resolution.voxels_per_axis());
        let (structure, node_count, leaf_count) = build_structure(resolution, args.fixture);

        let ctx = GpuContext { device, queue };
        let renderer = GpuRenderer::new(&ctx, &structure)?;

        // Blit pipeline + the intermediate render-output texture + the HUD.
        let (blit_pipeline, blit_layout, sampler) = build_blit(&ctx.device, format);
        let output_view = make_output(&ctx.device, config.width, config.height);
        let blit_bind = make_blit_bind(&ctx.device, &blit_layout, &output_view, &sampler);
        let hud = Hud::new(&ctx.device, &ctx.queue, format);

        let n = resolution.voxels_per_axis() as f32;
        let (eye, fwd) = orbit_eye_forward(0.0, n);

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
            hud,
            show_hud: true,
            resolution,
            fixture: args.fixture,
            node_count,
            leaf_count,
            mode: CamMode::Orbit,
            fly: FlyCamera::from_eye_forward(eye, fwd, n),
            input: Input::default(),
            mouse_look: false,
            angle: 0.0,
            last_instant: Instant::now(),
            history: Vec::with_capacity(HISTORY_CAP),
            kernel_ms: 0.0,
            encode_ms: 0.0,
            present_ms: 0.0,
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

    /// Switches to the free camera if currently orbiting, seeding it from the
    /// orbit pose so the view does not jump.
    fn ensure_free(&mut self) {
        if self.mode == CamMode::Orbit {
            let n = self.resolution.voxels_per_axis() as f32;
            let (eye, fwd) = orbit_eye_forward(self.angle, n);
            self.fly = FlyCamera::from_eye_forward(eye, fwd, n);
            self.mode = CamMode::Free;
        }
    }

    fn on_key(&mut self, event: &winit::event::KeyEvent, event_loop: &ActiveEventLoop) {
        let pressed = event.state == ElementState::Pressed;
        let PhysicalKey::Code(code) = event.physical_key else {
            return;
        };
        // Movement keys hand control to the free camera on press; other keys do
        // not. `is_movement` is set false in the non-movement arms.
        let mut is_movement = true;
        match code {
            KeyCode::KeyW => self.input.forward = pressed,
            KeyCode::KeyS => self.input.back = pressed,
            KeyCode::KeyA => self.input.left = pressed,
            KeyCode::KeyD => self.input.right = pressed,
            KeyCode::KeyE | KeyCode::Space => self.input.up = pressed,
            KeyCode::KeyQ | KeyCode::ControlLeft => self.input.down = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                self.input.boost = pressed;
                is_movement = false;
            }
            KeyCode::Tab if pressed => {
                self.toggle_mode();
                is_movement = false;
            }
            KeyCode::KeyH if pressed => {
                self.show_hud = !self.show_hud;
                is_movement = false;
            }
            KeyCode::Escape if pressed => {
                event_loop.exit();
                is_movement = false;
            }
            _ => is_movement = false,
        }
        if is_movement && pressed {
            self.ensure_free();
        }
    }

    fn toggle_mode(&mut self) {
        match self.mode {
            CamMode::Orbit => self.ensure_free(),
            CamMode::Free => self.mode = CamMode::Orbit,
        }
    }

    fn on_mouse_button(&mut self, state: ElementState, button: MouseButton) {
        if button == MouseButton::Left {
            self.mouse_look = state == ElementState::Pressed;
            if self.mouse_look {
                self.ensure_free();
            }
        }
    }

    fn on_scroll(&mut self, delta: MouseScrollDelta) {
        let notches = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(p) => p.y as f32 / 120.0,
        };
        if self.mode == CamMode::Free {
            self.input.scroll += notches;
        }
    }

    /// Advances the active camera by `dt` and returns its GPU uniform.
    fn update_camera(&mut self, dt: f32, w: u32, h: u32) -> GpuCamera {
        let n = self.resolution.voxels_per_axis() as f32;
        let k = self.resolution.internal_levels();
        match self.mode {
            CamMode::Orbit => {
                // Fixed per-frame increment keeps scripted --frames runs
                // reproducible regardless of frame rate.
                self.angle += 0.01;
                orbit_camera(self.angle, n, w, h, k)
            }
            CamMode::Free => {
                self.fly.apply(dt, &self.input);
                self.fly.to_gpu(w, h, n, k)
            }
        }
    }

    fn render(&mut self) {
        let (w, h) = (self.config.width, self.config.height);
        let now = Instant::now();
        let dt = (now - self.last_instant).as_secs_f32();
        self.last_instant = now;

        let camera = self.update_camera(dt, w, h);

        let t0 = Instant::now();
        let (Success(frame) | Suboptimal(frame)) = self.surface.get_current_texture() else {
            self.surface.configure(&self.ctx.device, &self.config);
            return;
        };
        let target = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Build the HUD for this frame before recording the pass.
        if self.show_hud {
            let builder = self.build_hud(&camera, w);
            self.hud.prepare(&self.ctx.queue, w, h, &builder);
        }

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        // Compute: traverse + shade → output texture (timed when supported).
        self.renderer
            .render_timed(&mut encoder, &camera, &self.output_view, w, h);
        // Blit the image, then draw the HUD over it, in one render pass.
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
            if self.show_hud {
                self.hud.draw(&mut pass);
            }
        }
        let t_encode = t0.elapsed();

        self.ctx.queue.submit([encoder.finish()]);
        // Wait for the GPU so the kernel timestamps and present are ready.
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

        // Read the GPU-timeline kernel time (None without timestamp support).
        if let Ok(Some(ns)) = self.renderer.last_kernel_ns() {
            self.kernel_ms = ns / 1.0e6;
        }

        let present_start = Instant::now();
        frame.present();
        let t_present = present_start.elapsed();

        // Per-frame stats feed the on-screen HUD; the terminal stays quiet.
        self.encode_ms = t_encode.as_secs_f64() * 1000.0;
        self.present_ms = t_present.as_secs_f64() * 1000.0;
        self.push_history(t0.elapsed().as_secs_f32() * 1000.0);
        self.frames_total = self.frames_total.saturating_add(1);
        self.input.end_frame();
    }

    /// Appends a frame time to the bounded history ring.
    fn push_history(&mut self, ms: f32) {
        if self.history.len() >= HISTORY_CAP {
            self.history.remove(0);
        }
        self.history.push(ms);
    }

    /// Assembles the HUD quads for this frame: a padded info panel (accent
    /// header, stats, a dimmed controls section) plus a frame-time sparkline.
    fn build_hud(&self, camera: &GpuCamera, w: u32) -> HudBuilder {
        const HEADER: [f32; 4] = [1.0, 0.82, 0.45, 1.0]; // warm accent
        const STAT: [f32; 4] = [0.86, 0.92, 1.0, 1.0]; // bright
        const CTRL: [f32; 4] = [0.55, 0.63, 0.75, 1.0]; // dimmed
        const PANEL: [f32; 4] = [0.02, 0.03, 0.06, 0.66];

        let mut b = HudBuilder::new(2.0);
        let lh = b.line_height();
        let margin = 14.0; // panel inset from the window corner
        let pad = 14.0; // inner padding
        let gap = lh * 0.5; // breathing room between sections
        let spark_h = 46.0;

        let eye = Vec3::from_array(camera.eye);
        let dir = Vec3::from_array(camera.forward);
        let (mn, avg, mx) = history_stats(&self.history);
        let fps = if avg > 0.0 { 1000.0 / avg } else { 0.0 };
        let kernel = if self.renderer.supports_timing() {
            format!("KERNEL {:>6.2} MS", self.kernel_ms)
        } else {
            "KERNEL     N/A".to_string()
        };

        let header = format!(
            "{} {}^3",
            self.fixture.label(),
            self.resolution.voxels_per_axis()
        );
        let stats = [
            format!("FPS {fps:>4.0}     {avg:>6.2} MS"),
            kernel,
            format!(
                "ENC {:>5.2}   PRES {:>5.2} MS",
                self.encode_ms, self.present_ms
            ),
            format!("MIN {mn:>5.2}   MAX {mx:>5.2} MS"),
            format!(
                "NODES {}   LEAVES {}",
                compact_count(self.node_count),
                compact_count(self.leaf_count)
            ),
            format!("EYE {:>6.0} {:>6.0} {:>6.0}", eye.x, eye.y, eye.z),
            format!("DIR {:>+6.2} {:>+6.2} {:>+6.2}", dir.x, dir.y, dir.z),
            format!("MODE {}     SPD {:>4.0}", self.mode.label(), self.fly.speed),
        ];
        let controls = [
            "WASD QE  MOVE",
            "DRAG  LOOK",
            "SCROLL  SPEED",
            "TAB MODE    H HUD",
        ];

        // Size the panel to its widest line so nothing overflows; clamp to window.
        let content_w = std::iter::once(b.text_width(&header))
            .chain(stats.iter().map(|l| b.text_width(l)))
            .chain(controls.iter().map(|l| b.text_width(l)))
            .fold(0.0_f32, f32::max);
        let panel_w = (content_w + pad * 2.0).min(w as f32 - margin * 2.0);

        let n_text = 1 + stats.len() + controls.len();
        let panel_h = pad * 2.0 + n_text as f32 * lh + gap * 2.0 + spark_h;
        b.solid(margin, margin, panel_w, panel_h, PANEL);

        let x = margin + pad;
        let mut y = margin + pad;
        b.text(x, y, &header, HEADER);
        y += lh;
        for line in &stats {
            b.text(x, y, line, STAT);
            y += lh;
        }
        y += gap;
        for line in &controls {
            b.text(x, y, line, CTRL);
            y += lh;
        }
        y += gap;
        let spark_max = mx.max(BUDGET_MS) * 1.1;
        b.sparkline(
            [x, y, panel_w - pad * 2.0, spark_h],
            &self.history,
            spark_max,
            BUDGET_MS,
        );
        b
    }
}

/// Builds and uploads the sparse structure for the chosen fixture, returning it
/// with its node and leaf counts.
fn build_structure(resolution: Resolution, fixture: Fixture) -> (SchoolBBuffer, usize, usize) {
    let t = Instant::now();
    let tree = match fixture {
        Fixture::Sierpinski => {
            SparseTree::build(&OctantFractal::sierpinski_tetrahedron(resolution))
        }
        Fixture::Cantor => SparseTree::build(&OctantFractal::cantor_dust(resolution)),
        Fixture::Checkerboard => SparseTree::build(&Checkerboard { resolution }),
        Fixture::WireLattice => SparseTree::build(&WireLattice::new(resolution)),
        Fixture::Dust => SparseTree::build(&Dust::new(resolution)),
        Fixture::Perlin => SparseTree::build(&NoiseField::perlin(resolution)),
        Fixture::Caves => SparseTree::build(&NoiseField::caves(resolution)),
    };
    let structure = SchoolBBuffer::from_sparse(&tree);
    let (node_count, leaf_count) = (tree.node_count(), tree.leaf_count());
    eprintln!(
        "built {}³: {node_count} nodes, {leaf_count} leaves in {:.2?}",
        resolution.voxels_per_axis(),
        t.elapsed(),
    );
    (structure, node_count, leaf_count)
}

/// Formats a count compactly for the HUD (`5585909` → `5.59M`, `142078` →
/// `142.1K`), keeping node/leaf lines short.
fn compact_count(n: usize) -> String {
    let f = n as f64;
    if f >= 1.0e6 {
        format!("{:.2}M", f / 1.0e6)
    } else if f >= 1.0e3 {
        format!("{:.1}K", f / 1.0e3)
    } else {
        n.to_string()
    }
}

/// Min, mean, and max of a frame-time history (all `0.0` when empty).
fn history_stats(h: &[f32]) -> (f32, f32, f32) {
    if h.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    let mut sum = 0.0;
    for &v in h {
        mn = mn.min(v);
        mx = mx.max(v);
        sum += v;
    }
    (mn, sum / h.len() as f32, mx)
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
