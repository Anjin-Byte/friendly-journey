//! GPU truecolor-render differential (`docs/materials/11-truecolor-design.md`, P4).
//!
//! Renders a per-voxel **truecolor**-baked scene and reads the framebuffer back,
//! proving the WGSL hit-time colour read (`render_truecolor.wgsl`:
//! `leaf_color_rank` → chunk-select → `unpack4x8unorm`) reproduces the compact
//! `leaf_color` the CPU assembler wrote. It also **naga-validates**
//! `render_truecolor.wgsl` (the shader only compiles when the truecolor
//! `GpuRenderer::new` branch runs) and exercises the **chunk-select** path on real
//! hardware via a forced-tiny `per_chunk` (so `N > 1` needs only kilobytes of VRAM,
//! not the production 285 MiB).
//!
//! Gated like `differential.rs`/`material_render.rs`: with no adapter it skips,
//! unless `VOXEL_REQUIRE_GPU=1` forces a hard failure.

#![allow(clippy::cast_precision_loss)]

use voxel_core::fixtures::Solid;
use voxel_core::{MaterialTable, Resolution, SchoolBBuffer, SparseTree, VoxelCoord};
use voxel_gpu::{GpuCamera, GpuContext, GpuError, GpuRenderer, OUTPUT_FORMAT};

/// Four distinct mid-range RGBA8 bytes (R low) — unreachable by the palette read
/// or the position-shade fallback, so the N=1 test's match is load-bearing.
const FLAT_COLOR: [u8; 4] = [0x12, 0x34, 0x56, 0xFF];
/// The unique colour of the cross-chunk (g=2) voxel in the chunk-select test.
const HI_COLOR: [u8; 4] = [0xAA, 0xBB, 0xCC, 0xFF];

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

/// A perspective camera looking straight down +Z at the centre of an `n³` grid.
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

/// Renders a pre-built `renderer` from `camera` into a `dim×dim` framebuffer and
/// reads it back as RGBA8 pixels (row-major). `dim*4` must be a multiple of 256.
fn read_render(
    ctx: &GpuContext,
    renderer: &GpuRenderer,
    camera: &GpuCamera,
    dim: u32,
) -> Vec<[u8; 4]> {
    let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("truecolor test output"),
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
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("truecolor test readback"),
        size: u64::from(dim * dim * 4),
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
fn truecolor_renders_baked_colour_matching_the_assembler() {
    // N=1: a Solid cube baked to a CONSTANT colour. The exact bytes are unreachable
    // by both the palette read (magenta / table) and the position-shade fallback
    // (which varies monotonically with world coords), so a large solid region of
    // the exact colour proves the truecolor read ran — a mis-wire to either
    // fallback yields a varying or magenta region.
    let Some(ctx) = context_or_skip() else { return };
    // sRGB byte-order guard: a future flip to an sRGB target would shift these
    // mid-range bytes on store and fail the equality below.
    assert_eq!(OUTPUT_FORMAT, wgpu::TextureFormat::Rgba8Unorm);

    let r = Resolution::new(32).unwrap();
    let tree = SparseTree::build(&Solid { resolution: r });
    let mut structure = SchoolBBuffer::from_sparse(&tree);
    structure.assemble_leaf_color(&tree, |_| FLAT_COLOR);
    assert!(
        structure.has_leaf_color(),
        "scene must route through truecolor"
    );

    let renderer = GpuRenderer::new(&ctx, &structure, &MaterialTable::missing_only()).unwrap();
    let dim = 64u32;
    let px = read_render(&ctx, &renderer, &front_camera(r, dim), dim);

    let exact = px.iter().filter(|p| **p == FLAT_COLOR).count();
    let magenta = px.iter().filter(|p| **p == [255, 0, 255, 255]).count();
    assert!(
        exact > 200,
        "the baked cube should fill a large central region; only {exact}/{} pixels matched the baked colour",
        px.len()
    );
    assert_eq!(
        magenta, 0,
        "truecolor must not fall back to the magenta sentinel"
    );
}

#[test]
fn truecolor_chunk_select_reads_a_high_chunk() {
    // Forced-tiny per_chunk=2 drives the N>1 cross-chunk path on a 3-voxel scene
    // (no 285 MiB needed). Three voxels in ONE brick get g = 0,1,2; the visible
    // voxel at local (3,3,0) (morton 27) has rank 2 ⇒ g=2 ⇒ chunk 1. Its unique
    // colour appearing in the readback proves read_leaf_color's chunk==1 arm works.
    let Some(ctx) = context_or_skip() else { return };
    let r = Resolution::new(32).unwrap();

    // Brick (2,2,0): locals (0,0,0)=m0, (1,0,0)=m1, (3,3,0)=m27 — ascending morton,
    // so the (19,19,0) voxel ranks last (g=2). Near grid centre ⇒ clearly visible.
    let vox_g0 = VoxelCoord::new(16, 16, 0);
    let vox_g1 = VoxelCoord::new(17, 16, 0);
    let vox_g2 = VoxelCoord::new(19, 19, 0); // the cross-chunk voxel
    let color_of = move |coord: VoxelCoord| -> [u8; 4] {
        if coord == vox_g0 {
            [0x11, 0x22, 0x33, 0xFF]
        } else if coord == vox_g1 {
            [0x44, 0x55, 0x66, 0xFF]
        } else if coord == vox_g2 {
            HI_COLOR
        } else {
            [0, 0, 0, 0xFF]
        }
    };

    let tree = SparseTree::from_voxels(r, [vox_g0, vox_g1, vox_g2].map(|v| (v, 0u16)));
    let mut structure = SchoolBBuffer::from_sparse(&tree);
    structure.assemble_leaf_color(&tree, color_of);
    assert_eq!(
        structure.leaf_color_words().len(),
        3,
        "exactly 3 occupied voxels"
    );

    // CPU precondition (R2 #1): the visible voxel actually crosses a chunk boundary,
    // else the test is vacuous (chunk-select never leaves chunk 0).
    let slot = tree.leaf_slot_of(vox_g2).unwrap() as usize;
    let morton = voxel_core::morton::encode_brick(vox_g2.x & 7, vox_g2.y & 7, vox_g2.z & 7);
    let g =
        structure.leaf_color_base_words()[slot] + structure.leaves()[slot].occupied_rank(morton);
    assert_eq!(g, 2, "the (19,19,0) voxel must be global index 2");
    let per_chunk = 2u32;
    assert!(
        g / per_chunk >= 1,
        "geometry did not cross a chunk boundary; test is vacuous"
    );

    let renderer = GpuRenderer::new_with_per_chunk(
        &ctx,
        &structure,
        &MaterialTable::missing_only(),
        per_chunk,
    )
    .unwrap();
    let dim = 64u32;
    let px = read_render(&ctx, &renderer, &front_camera(r, dim), dim);

    let hits_c = px.iter().filter(|p| **p == HI_COLOR).count();
    assert!(
        hits_c > 0,
        "the chunk-1 voxel's colour must render (proves read_leaf_color crossed into chunk 1)"
    );
}

#[test]
fn truecolor_blend_composites_front_over_back() {
    // Phase 2 BLEND: a semi-transparent RED voxel (α=128) in front of an OPAQUE BLUE
    // voxel along the same ray must read back as the front-to-back composite
    // (~½red + ½blue), proving: (1) `has_transparency` routed to the blend pipeline,
    // (2) `traverse_and_composite` did NOT stop at the first hit, (3) the opaque
    // backdrop (bit-18 clear) terminated the accumulation, (4) the blend WGSL is
    // naga-valid (it only compiles when this pipeline is built).
    let Some(ctx) = context_or_skip() else { return };
    let r = Resolution::new(32).unwrap();
    // A 5×5 patch centred on the camera axis (eye is at grid centre 16,16). Front
    // layer z=8 (semi-transparent red), back layer z=10 (opaque blue) — same z-brick
    // (8..15), so the leaf carries the transparency bit and the +Z ray hits red then
    // blue with negligible perspective divergence (2 voxels apart).
    let mut voxels = Vec::new();
    for dy in 0..5u32 {
        for dx in 0..5u32 {
            voxels.push(VoxelCoord::new(14 + dx, 14 + dy, 8));
            voxels.push(VoxelCoord::new(14 + dx, 14 + dy, 10));
        }
    }
    let tree = SparseTree::from_voxels(r, voxels.iter().map(|&v| (v, 0u16)));
    let mut structure = SchoolBBuffer::from_sparse(&tree);
    structure.assemble_leaf_color(&tree, |c| {
        if c.z == 8 {
            [255, 0, 0, 128]
        } else {
            [0, 0, 255, 255]
        }
    });
    assert!(
        structure.has_transparency(),
        "the α=128 front voxels must flag the scene transparent (→ blend pipeline)"
    );

    let renderer = GpuRenderer::new(&ctx, &structure, &MaterialTable::missing_only()).unwrap();
    let dim = 64u32;
    let px = read_render(&ctx, &renderer, &front_camera(r, dim), dim);

    // The composite ≈ (128, 0, 127): R from the front blend, B from the back opaque.
    let composited = px
        .iter()
        .filter(|p| p[0] > 90 && p[0] < 165 && p[2] > 90 && p[2] < 165 && p[1] < 40)
        .count();
    assert!(
        composited > 4,
        "front blend must composite over the back opaque; got {composited} composite pixels"
    );
    // A first-hit-stop (no compositing) would render the front voxel as opaque red.
    let opaque_red = px.iter().filter(|p| p[0] > 200 && p[2] < 40).count();
    assert_eq!(
        opaque_red, 0,
        "the front BLEND voxel must not render as opaque red (that would be first-hit-stop)"
    );
}

#[test]
fn device_grants_enough_storage_buffers_for_truecolor() {
    // The pure probe can't see the device's GRANTED limit; this confirms the real
    // adapter supplies the 7 storage buffers the truecolor layout binds (3 carried +
    // base + 3 chunks). Stock wgpu default is 8, so this passes everywhere the
    // palette path already runs.
    let Some(ctx) = context_or_skip() else { return };
    assert!(
        ctx.max_storage_buffers() >= 7,
        "truecolor binds 7 storage buffers but the device grants only {}",
        ctx.max_storage_buffers()
    );
}
