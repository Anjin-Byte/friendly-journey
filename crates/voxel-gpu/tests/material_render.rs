//! GPU material-render differential (`docs/materials/03-gpu-read.md`).
//!
//! Renders a coloured scene to a texture and reads the framebuffer back, proving
//! the WGSL hit-time material read (`render.wgsl::read_material`) produces the
//! table colour the CPU packer and `read_slot` agree on — the **on-hardware** end
//! of the bit-exact contract (the CPU end is pinned by
//! `voxel_core::palette`'s `wgsl_bit_layout_matches_pack`). It also naga-validates
//! `render.wgsl` (the shader only compiles when `GpuRenderer::new` runs).
//!
//! Gated like `differential.rs`: with no adapter it skips (passes), unless
//! `VOXEL_REQUIRE_GPU=1` forces a hard failure so a GPU lane can't silently skip.

#![allow(clippy::cast_precision_loss)]

use voxel_core::fixtures::Solid;
use voxel_core::{MaterialTable, Resolution, SchoolBBuffer, SparseTree};
use voxel_gpu::{GpuCamera, GpuContext, GpuError, GpuRenderer, OUTPUT_FORMAT};

fn require_gpu() -> bool {
    std::env::var_os("VOXEL_REQUIRE_GPU").is_some()
}

fn context_or_skip() -> Option<GpuContext> {
    match GpuContext::try_new() {
        Ok(ctx) => Some(ctx),
        Err(GpuError::NoAdapter) if !require_gpu() => {
            eprintln!("skip: no GPU adapter present (set VOXEL_REQUIRE_GPU=1 to require one)");
            None
        }
        Err(e) => panic!("GPU unavailable: {e}"),
    }
}

/// RGBA8 little-endian (`unpack4x8unorm`, R in the low byte): opaque red.
const RED: u32 = 0xFF00_00FF;
const RED_PX: [u8; 4] = [255, 0, 0, 255];
const MAGENTA_PX: [u8; 4] = [255, 0, 255, 255];

/// A perspective camera looking straight down +Z at the centre of an `n³` grid
/// from well in front of it, with a wide enough FOV that the cube fills a large
/// central region of the `dim×dim` framebuffer.
fn front_camera(r: Resolution, dim: u32) -> GpuCamera {
    let n = r.voxels_per_axis() as f32;
    let half = n * 0.5;
    GpuCamera {
        eye: [half, half, -40.0],
        tan: 1.0,
        forward: [0.0, 0.0, 1.0],
        aspect: 1.0,
        right: [1.0, 0.0, 0.0],
        n,
        up: [0.0, 1.0, 0.0],
        pad: 0.0,
        dims: [dim, dim, r.internal_levels(), 0],
    }
}

/// Renders `structure` + `table` from `camera` into a `dim×dim` framebuffer and
/// reads it back as RGBA8 pixels (row-major). `dim` must keep `dim*4` a multiple
/// of 256 (wgpu's copy row alignment); `dim = 64` ⇒ 256.
fn render_pixels(
    ctx: &GpuContext,
    structure: &SchoolBBuffer,
    table: &MaterialTable,
    camera: &GpuCamera,
    dim: u32,
) -> Vec<[u8; 4]> {
    let renderer = GpuRenderer::new(ctx, structure, table).unwrap();
    let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("material test output"),
        size: wgpu::Extent3d {
            width: dim,
            height: dim,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: OUTPUT_FORMAT,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

    let bytes = u64::from(dim * dim * 4);
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("material test readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    renderer.render(&mut encoder, camera, &view, dim, dim);
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(dim * 4),
                rows_per_image: Some(dim),
            },
        },
        wgpu::Extent3d {
            width: dim,
            height: dim,
            depth_or_array_layers: 1,
        },
    );
    ctx.queue.submit(std::iter::once(encoder.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range();
    let px: Vec<[u8; 4]> = data
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect();
    drop(data);
    readback.unmap();
    px
}

#[test]
fn gpu_renders_assigned_material_colour() {
    let Some(ctx) = context_or_skip() else { return };
    let r = Resolution::new(32).unwrap();
    let mut tree = SparseTree::build(&Solid { resolution: r });
    tree.fill_materials(|_| 1); // every occupied voxel → global id 1
    let mut table = MaterialTable::missing_only();
    assert_eq!(table.push(RED).unwrap(), 1, "first colour gets global id 1");
    let structure = SchoolBBuffer::from_sparse(&tree);

    let dim = 64u32;
    let px = render_pixels(&ctx, &structure, &table, &front_camera(r, dim), dim);

    let red = px.iter().filter(|p| **p == RED_PX).count();
    let magenta = px.iter().filter(|p| **p == MAGENTA_PX).count();
    assert!(
        red > 200,
        "the red cube should fill a large central region; only {red}/{} pixels were red",
        px.len()
    );
    assert_eq!(
        magenta, 0,
        "every voxel is colour 1, so none must render the magenta sentinel"
    );
}

#[test]
fn gpu_unassigned_material_falls_back_to_position_not_magenta() {
    // With no materials assigned, every hit reads global-0; the shader shades
    // those by position (the prior fixture look), NOT magenta. This guards the
    // gid==0 fallback so adding materials never regresses fixture rendering.
    let Some(ctx) = context_or_skip() else { return };
    let r = Resolution::new(32).unwrap();
    let tree = SparseTree::build(&Solid { resolution: r }); // no fill_materials ⇒ all gid 0
    let table = MaterialTable::missing_only();
    let structure = SchoolBBuffer::from_sparse(&tree);

    let dim = 64u32;
    let px = render_pixels(&ctx, &structure, &table, &front_camera(r, dim), dim);

    let magenta = px.iter().filter(|p| **p == MAGENTA_PX).count();
    assert_eq!(magenta, 0, "global-0 hits shade by position, never magenta");
    // Position shading varies across the cube — many distinct hit colours, unlike
    // the single-colour material case.
    let distinct: std::collections::HashSet<[u8; 4]> = px.iter().copied().collect();
    assert!(
        distinct.len() > 3,
        "position shading should produce a varied hit region, got {} colours",
        distinct.len()
    );
}
