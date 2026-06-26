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

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use glam::{Mat4, Vec3};
use voxel_core::fixtures::{Checkerboard, Dust, NoiseField, OctantFractal, WireLattice};
use voxel_core::{
    Edit, Ray, Resolution, SchoolBBuffer, SparseTree, VoxelCoord, brush_voxels, traverse,
};
use voxel_gpu::{GpuCamera, GpuContext, GpuRenderer};
use voxelizer::loader::{load_mesh, rotation_degrees};
use voxelizer::{GpuVoxelizer, GpuVoxelizerConfig, TileSpec, VoxelGrid, VoxelizeOpts};
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

#[derive(Parser, Clone)]
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
    /// Occupancy fixture to render (used when `--mesh` is not given).
    #[arg(long, value_enum, default_value_t = Fixture::Sierpinski)]
    fixture: Fixture,
    /// Mesh file to voxelize and render (`.gltf`/`.glb`, `.obj`, `.stl`). When
    /// set, the mesh is voxelized into the grid instead of building `--fixture`.
    #[arg(long)]
    mesh: Option<PathBuf>,
    /// Voxels of margin around the mesh's bounding box when fitting it into the
    /// grid (only used with `--mesh`).
    #[arg(long, default_value_t = 2.0)]
    padding: f32,
    /// Corrective rotation about X in degrees, applied to `--mesh` before fitting
    /// the grid. Re-orients transform-less formats (OBJ/STL) whose exporter used
    /// a different up-axis (e.g. `--rotate-x -90` for a Z-up model in this Y-up
    /// view).
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    rotate_x: f32,
    /// Corrective rotation about Y in degrees (see `--rotate-x`).
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    rotate_y: f32,
    /// Corrective rotation about Z in degrees (see `--rotate-x`).
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    rotate_z: f32,
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
            match Viewer::new(event_loop, &self.args) {
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
            WindowEvent::CursorMoved { position, .. } => {
                viewer.cursor = (position.x as f32, position.y as f32);
            }
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
/// Largest edit-brush radius (voxels); a radius-`r` sphere is `~(2r+1)³` voxels.
const MAX_BRUSH_RADIUS: u32 = 12;

/// The outcome of the last edit, shown in the HUD: how the brush was applied and
/// what the incremental GPU sync cost.
#[derive(Clone, Copy)]
struct EditFeedback {
    /// Voxels actually changed (not counting no-ops within the brush).
    voxels: u32,
    /// Whether the stroke changed topology (forcing a full re-upload) vs only
    /// in-place leaf patches.
    topology: bool,
    /// Wall-clock of the edit + GPU sync (CPU side), milliseconds.
    ms: f64,
}

// Several independent UI toggles (HUD visibility, mouse-look, brush mode, GPU
// resync flag) — distinct flags, not a state machine, so a struct is right.
#[allow(clippy::struct_excessive_bools)]
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
    /// HUD label for the current scene: the fixture name, or the mesh filename.
    scene_label: String,
    node_count: usize,
    leaf_count: usize,
    // Editing: the live tree + its School-B buffer, kept CPU-side so an edit can
    // `set_voxel` the tree, `patch_leaf` the buffer, and sync the GPU (leaf patch
    // for an in-place edit, full re-upload on topology change).
    tree: SparseTree,
    structure: SchoolBBuffer,
    cursor: (f32, f32),
    brush_radius: u32,
    brush_add: bool,
    last_camera: GpuCamera,
    last_edit: Option<EditFeedback>,
    /// Set if a topology re-upload failed: the renderer's buffers are stale, so
    /// the next edit must do a full re-upload (not an in-place leaf patch into
    /// wrong-sized buffers) to resync.
    needs_full_upload: bool,
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
    // Inherently long: one-shot setup of window, surface, adapter, device,
    // structure, render pipeline, blit pipeline, and HUD.
    #[allow(clippy::too_many_lines)]
    fn new(event_loop: &ActiveEventLoop, args: &Args) -> Result<Self> {
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

        // Build the structure: either voxelize a mesh file (sharing this GPU via
        // `from_device`) or generate the chosen fixture. Noise fixtures generate
        // their occupancy on the GPU (≈17× faster than the CPU build at 512³); the
        // tree is kept alongside its buffer so edits can patch both.
        let ctx = GpuContext { device, queue };
        let resolution = Resolution::new(args.res)?;
        let (scene_label, (tree, structure)) = if let Some(mesh_path) = args.mesh.as_deref() {
            let label = mesh_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("MESH")
                .to_string();
            eprintln!(
                "voxelizing {} into a {}³ grid…",
                mesh_path.display(),
                resolution.voxels_per_axis()
            );
            (
                label,
                build_from_mesh(
                    &ctx,
                    resolution,
                    mesh_path,
                    args.padding,
                    rotation_degrees(args.rotate_x, args.rotate_y, args.rotate_z),
                )?,
            )
        } else {
            eprintln!(
                "building {}³ {} structure…",
                resolution.voxels_per_axis(),
                args.fixture.label()
            );
            (
                args.fixture.label().to_string(),
                build_structure(&ctx, resolution, args.fixture),
            )
        };
        let (node_count, leaf_count) = (tree.node_count(), tree.leaf_count());

        let renderer = GpuRenderer::new(&ctx, &structure)?;

        // Blit pipeline + the intermediate render-output texture + the HUD.
        let (blit_pipeline, blit_layout, sampler) = build_blit(&ctx.device, format);
        let output_view = make_output(&ctx.device, config.width, config.height);
        let blit_bind = make_blit_bind(&ctx.device, &blit_layout, &output_view, &sampler);
        let hud = Hud::new(&ctx.device, &ctx.queue, format);

        let n = resolution.voxels_per_axis() as f32;
        let (eye, fwd) = orbit_eye_forward(0.0, n);
        let last_camera = orbit_camera(
            0.0,
            n,
            config.width,
            config.height,
            resolution.internal_levels(),
        );

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
            scene_label,
            node_count,
            leaf_count,
            tree,
            structure,
            cursor: (0.0, 0.0),
            brush_radius: 3,
            brush_add: true,
            last_camera,
            last_edit: None,
            needs_full_upload: false,
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
        // Keep last_camera (used by edit picks) consistent with the new viewport,
        // so a click landing before the next frame re-casts with matching dims.
        let n = self.resolution.voxels_per_axis() as f32;
        let k = self.resolution.internal_levels();
        self.last_camera = match self.mode {
            CamMode::Orbit => orbit_camera(self.angle, n, width, height, k),
            CamMode::Free => self.fly.to_gpu(width, height, n, k),
        };
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
            KeyCode::KeyB if pressed => {
                self.brush_add = !self.brush_add; // toggle add ↔ remove
                is_movement = false;
            }
            KeyCode::BracketLeft if pressed => {
                self.brush_radius = self.brush_radius.saturating_sub(1);
                is_movement = false;
            }
            KeyCode::BracketRight if pressed => {
                self.brush_radius = (self.brush_radius + 1).min(MAX_BRUSH_RADIUS);
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
        // Right-click stamps the brush at the cursor (left stays camera-look).
        if button == MouseButton::Right && state == ElementState::Pressed {
            self.edit_at_cursor();
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

    /// Casts a ray from the cursor through the on-screen camera (reproducing the
    /// render kernel's ray-gen exactly so the edit lands where it is drawn), and
    /// if it hits a voxel, stamps the current brush there.
    fn edit_at_cursor(&mut self) {
        let (w, h) = (self.config.width as f32, self.config.height as f32);
        if w <= 0.0 || h <= 0.0 {
            return;
        }
        let cam = self.last_camera;
        let (px, py) = self.cursor;
        // Identical to render.wgsl: NDC → camera-basis direction.
        let ndc_x = ((px + 0.5) / w * 2.0 - 1.0) * cam.tan * cam.aspect;
        let ndc_y = (1.0 - (py + 0.5) / h * 2.0) * cam.tan;
        let dir = (Vec3::from_array(cam.forward)
            + Vec3::from_array(cam.right) * ndc_x
            + Vec3::from_array(cam.up) * ndc_y)
            .normalize();
        let ray = Ray::new(Vec3::from_array(cam.eye).as_dvec3(), dir.as_dvec3());
        if let Some(hit) = traverse(&self.structure, &ray) {
            self.apply_brush(hit.voxel);
        }
    }

    /// Applies the current brush (add or remove per `brush_add`) centred on
    /// `center`, then syncs the GPU the cheap way: in-place leaf edits patch only
    /// the touched leaves (`update_leaf`), and a stroke that changed topology
    /// rebuilds and re-uploads the structure once. Records timing for the HUD.
    fn apply_brush(&mut self, center: VoxelCoord) {
        let t = Instant::now();
        let mut touched: Vec<u32> = Vec::new();
        let mut any_topology = false;
        let mut changed = 0u32;
        for c in brush_voxels(center, self.brush_radius) {
            match self.tree.set_voxel(c, self.brush_add) {
                Edit::Unchanged => {}
                Edit::Leaf(idx) => {
                    touched.push(idx);
                    changed += 1;
                }
                Edit::Topology => {
                    any_topology = true;
                    changed += 1;
                }
            }
        }

        if changed > 0 {
            // `structure` is always brought into step with `tree` first (so the
            // CPU buffer is never topology-stale — `patch_leaf` can't panic);
            // only the GPU upload may fail, and that is handled below.
            if any_topology {
                // Topology renumbers leaf indices and invalidates node offsets:
                // re-serialize and re-upload the whole structure once.
                self.structure = SchoolBBuffer::from_sparse(&self.tree);
                self.sync_renderer_full();
            } else {
                // No topology change → indices are stable across the stroke; patch
                // each touched leaf's words+bounds in the CPU buffer.
                touched.sort_unstable();
                touched.dedup();
                for &idx in &touched {
                    self.structure.patch_leaf(&self.tree, idx);
                }
                if self.needs_full_upload {
                    // A prior topology re-upload failed, so the renderer's buffers
                    // are stale/wrong-sized: a full re-upload (carrying these leaf
                    // patches too) resyncs, rather than patching into bad buffers.
                    self.sync_renderer_full();
                } else {
                    for &idx in &touched {
                        self.renderer.update_leaf(&self.structure, idx);
                    }
                }
            }
            self.node_count = self.tree.node_count();
            self.leaf_count = self.tree.leaf_count();
        }

        self.last_edit = Some(EditFeedback {
            voxels: changed,
            topology: any_topology,
            ms: t.elapsed().as_secs_f64() * 1000.0,
        });
    }

    /// Re-uploads the whole structure to the renderer, tracking whether it
    /// succeeded so a failure (the unreachable-in-practice `BufferTooLarge`)
    /// leaves the renderer flagged stale and retried on the next edit, never
    /// patched into mismatched buffers.
    fn sync_renderer_full(&mut self) {
        match self.renderer.reupload(&self.structure) {
            Ok(()) => self.needs_full_upload = false,
            Err(e) => {
                eprintln!("edit GPU re-upload failed ({e:?}); retrying on next edit");
                self.needs_full_upload = true;
            }
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
        self.last_camera = camera; // kept so an edit-click can re-cast the on-screen ray

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
        let brush = format!(
            "BRUSH {}  R{}",
            if self.brush_add { "ADD" } else { "REM" },
            self.brush_radius
        );
        let edit = match self.last_edit {
            Some(e) if e.voxels == 0 => "EDIT  no-op".to_string(),
            Some(e) => format!(
                "EDIT {} {:>4}v {:>6.3} MS",
                if e.topology { "TOPO" } else { "LEAF" },
                e.voxels,
                e.ms
            ),
            None => "EDIT  —".to_string(),
        };

        let header = format!(
            "{} {}^3",
            self.scene_label,
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
            brush,
            edit,
        ];
        let controls = [
            "WASD QE  MOVE",
            "DRAG  LOOK",
            "SCROLL  SPEED",
            "RCLICK EDIT   B ADD/REM",
            "[ ] BRUSH RADIUS",
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

/// Voxelizes a mesh file into the sparse structure, sharing the viewer's GPU
/// device via [`GpuVoxelizer::from_device`] (one device for both voxelizing and
/// rendering). Returns the live tree + its School-B buffer, like
/// [`build_structure`]. Occupancy-only (the renderer needs no per-voxel
/// material), so even a 512³ grid stays within the storage limit.
fn build_from_mesh(
    ctx: &GpuContext,
    resolution: Resolution,
    path: &Path,
    padding: f32,
    rotation: Mat4,
) -> Result<(SparseTree, SchoolBBuffer)> {
    let mut mesh = load_mesh(path).with_context(|| format!("loading mesh {}", path.display()))?;
    eprintln!("  {} triangles", mesh.triangles.len());
    // Re-orient transform-less formats before measuring the bounding box.
    if rotation != Mat4::IDENTITY {
        mesh.transform(rotation);
    }
    let grid = VoxelGrid::fit_mesh(resolution, &mesh, padding);
    let tiles = TileSpec::new([4, 4, 4], grid.dims())?;
    let opts = VoxelizeOpts {
        epsilon: 1e-4,
        store_owner: false,
        store_color: false,
    };
    let vox = pollster::block_on(GpuVoxelizer::from_device(
        &ctx.device,
        &ctx.queue,
        GpuVoxelizerConfig::default(),
    ))?;
    let out = pollster::block_on(vox.voxelize_surface(&mesh, &grid, &tiles, &opts))?;
    let tree = out.occupancy.to_sparse_tree();
    let structure = SchoolBBuffer::from_sparse(&tree);
    Ok((tree, structure))
}

/// Builds the [`SparseTree`] for a noise fixture, evaluating the occupancy on the
/// GPU when possible (≈17× faster than the parallel CPU build at 512³) and
/// falling back to the CPU build otherwise (no adapter, or `n³` past the GPU
/// generator's `u32` index cap at 2048³).
fn build_noise(ctx: &GpuContext, field: &NoiseField) -> SparseTree {
    match voxel_gpu::generate_noise_tree(ctx, field) {
        Ok(tree) => tree,
        Err(e) => {
            eprintln!("GPU noise generation unavailable ({e}); building on CPU");
            SparseTree::build(field)
        }
    }
}

/// Builds the sparse structure for the chosen fixture, returning the live
/// [`SparseTree`] (kept for editing) and its School-B buffer.
fn build_structure(
    ctx: &GpuContext,
    resolution: Resolution,
    fixture: Fixture,
) -> (SparseTree, SchoolBBuffer) {
    let t = Instant::now();
    let tree = match fixture {
        Fixture::Sierpinski => {
            SparseTree::build(&OctantFractal::sierpinski_tetrahedron(resolution))
        }
        Fixture::Cantor => SparseTree::build(&OctantFractal::cantor_dust(resolution)),
        Fixture::Checkerboard => SparseTree::build(&Checkerboard { resolution }),
        Fixture::WireLattice => SparseTree::build(&WireLattice::new(resolution)),
        Fixture::Dust => SparseTree::build(&Dust::new(resolution)),
        Fixture::Perlin => build_noise(ctx, &NoiseField::perlin(resolution)),
        Fixture::Caves => build_noise(ctx, &NoiseField::caves(resolution)),
    };
    let structure = SchoolBBuffer::from_sparse(&tree);
    eprintln!(
        "built {}³: {} nodes, {} leaves in {:.2?}",
        resolution.voxels_per_axis(),
        tree.node_count(),
        tree.leaf_count(),
        t.elapsed(),
    );
    (tree, structure)
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
