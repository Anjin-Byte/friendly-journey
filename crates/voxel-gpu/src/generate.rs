//! GPU occupancy generation for the procedural noise fixtures.
//!
//! [`generate_noise_tree`] evaluates a [`NoiseField`] on the GPU — one invocation
//! per 8³ brick, building a register-resident Morton leaf and atomic-appending
//! only the non-empty bricks — and reads back just the occupied leaves to
//! assemble a [`SparseTree`] via [`SparseTree::from_bricks`]. Noise evaluation is
//! the entire cost of building these fixtures (the CPU build saturates every core
//! yet still takes seconds at 512³ and minutes at 2048³), and it is pure,
//! per-voxel, divergence-free ALU work — the GPU's home turf (≈20-25× faster).
//!
//! The CPU [`SparseTree::build`] over the f64 [`NoiseField`] stays the
//! reference/oracle and the A/B baseline; this is the fast path. f32 will not
//! bit-match f64 at threshold-grazing voxels, but the field is statistically and
//! visually identical (sub-1% of occupied voxels differ) and the differential
//! never uses this path.
//!
//! [`SparseTree::build`]: voxel_core::SparseTree::build

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use voxel_core::fixtures::NoiseField;
use voxel_core::{LeafBrick, SparseTree, morton};

use crate::buffers;
use crate::context::GpuContext;
use crate::error::GpuError;

/// Workgroup size; mirrors `@workgroup_size(64)` in `noise_gen_bricks.wgsl`.
const BRICK_WORKGROUP: u32 = 64;

/// The shared noise core, concatenated ahead of the kernel (the same pattern
/// `buffers::shader_source` uses for the traversal core).
const NOISE_CORE: &str = include_str!("../shaders/noise_core.wgsl");

/// Builds a noise-generation shader source: the shared core + one kernel.
fn noise_shader(kernel: &str) -> String {
    format!("{NOISE_CORE}\n{kernel}")
}

// Pod uniform struct (std140-friendly: all 4-byte scalars; 12 fields = 48 B, a
// multiple of 16). The `unsafe` is only the `bytemuck` derive on `#[repr(C)]`
// all-scalar data (Unsafe Quarantine).
#[allow(unsafe_code)]
mod pod {
    use super::{Pod, Zeroable};

    /// Mirrors `Params` in `noise_core.wgsl` (field order is significant).
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct GenParams {
        pub(crate) n: u32,
        pub(crate) seed: u32,
        pub(crate) octaves: u32,
        pub(crate) ridged: u32,
        pub(crate) _reserved: u32, // was the dense path's total_words; kept for layout
        pub(crate) frequency: f32,
        pub(crate) lacunarity: f32,
        pub(crate) gain: f32,
        pub(crate) warp: f32,
        pub(crate) threshold: f32,
        pub(crate) _pad0: u32,
        pub(crate) _pad1: u32,
    }
}
use pod::GenParams;

/// Maps a `MAP_READ` buffer, blocks until ready, and copies its bytes out.
fn read_buffer(device: &wgpu::Device, buf: &wgpu::Buffer) -> Result<Vec<u8>, GpuError> {
    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|_| GpuError::Poll)?;
    rx.recv().map_err(|_| GpuError::Poll)??;
    let data = slice.get_mapped_range();
    let bytes = data.to_vec();
    drop(data);
    buf.unmap();
    Ok(bytes)
}

/// Generates `field`'s occupancy on the GPU and returns the assembled
/// [`SparseTree`]. Each invocation evaluates one 8³ brick into a register-resident
/// Morton leaf and atomic-appends only the non-empty bricks, so the host reads
/// back just the occupied leaves (≈⅓ at 2048³) and runs [`SparseTree::from_bricks`]
/// — no dense readback, no CPU re-scan.
///
/// # Errors
/// - [`GpuError::Unsupported`] if the brick count / packed coordinate / worst-case
///   leaf buffer exceeds a GPU limit (everything through `2048³` is supported);
///   the caller falls back to the CPU build.
/// - [`GpuError::Poll`] / [`GpuError::BufferMap`] on a device/readback failure.
#[allow(clippy::cast_possible_truncation)] // counts are bounded < u32::MAX by the guards
#[allow(clippy::too_many_lines)] // one-shot GPU setup: pipeline + buffers + 2 dispatches + readback
pub fn generate_noise_tree(ctx: &GpuContext, field: &NoiseField) -> Result<SparseTree, GpuError> {
    let res = field.resolution;
    let n = res.voxels_per_axis();
    let bpa = n / 8; // bricks per axis
    let total_bricks_u128 = u128::from(bpa).pow(3);
    let leaf_bytes = total_bricks_u128 * 16 * 4; // worst case: every brick non-empty
    // The append slot and brick index are u32; the coord packs 10 bits/axis
    // (bpa ≤ 1024); the worst-case leaf buffer must fit the storage binding.
    if total_bricks_u128 > u128::from(u32::MAX)
        || bpa > 1024
        || leaf_bytes > u128::from(ctx.max_storage_binding())
    {
        return Err(GpuError::Unsupported {
            n,
            reason: "noise-gen brick count / coord / buffer exceeds a GPU limit",
        });
    }
    let total_bricks = total_bricks_u128 as u32;

    let device = &ctx.device;
    let queue = &ctx.queue;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("noise_gen_bricks"),
        source: wgpu::ShaderSource::Wgsl(
            noise_shader(include_str!("../shaders/noise_gen_bricks.wgsl")).into(),
        ),
    });
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("noise_gen_bricks layout"),
        entries: &[
            buffers::uniform_entry(0),        // params
            buffers::storage_entry(1, false), // counter (atomic)
            buffers::storage_entry(2, false), // out_coords
            buffers::storage_entry(3, false), // out_leaves
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("noise_gen_bricks pipeline layout"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("noise_gen_bricks pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("generate_bricks"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let params = GenParams {
        n,
        seed: field.seed,
        octaves: field.octaves,
        ridged: u32::from(field.ridged),
        _reserved: 0,
        frequency: field.frequency as f32,
        lacunarity: field.lacunarity as f32,
        gain: field.gain as f32,
        warp: field.warp as f32,
        threshold: field.threshold as f32,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("noise_gen_bricks params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let counter_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("noise_gen_bricks counter"),
        contents: bytemuck::bytes_of(&0u32),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let coords_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("noise_gen_bricks coords"),
        size: u64::from(total_bricks) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let leaves_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("noise_gen_bricks leaves"),
        size: leaf_bytes as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let counter_rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("noise_gen_bricks counter readback"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("noise_gen_bricks bind group"),
        layout: &bind_layout,
        entries: &[
            buffers::bind(0, params_buf.as_entire_binding()),
            buffers::bind(1, counter_buf.as_entire_binding()),
            buffers::bind(2, coords_buf.as_entire_binding()),
            buffers::bind(3, leaves_buf.as_entire_binding()),
        ],
    });

    // Pass 1: evaluate + compact, then read the occupied-brick count.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("noise_gen_bricks"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("noise_gen_bricks pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // 2D dispatch: the brick count exceeds the 65535-per-dimension cap at high
        // resolution (2048³ → 256³/64 ≈ 262k workgroups). The shader flattens
        // (x, y) back to a 1D brick index via `num_workgroups`.
        let total_wg = total_bricks.div_ceil(BRICK_WORKGROUP);
        let max_dim = device.limits().max_compute_workgroups_per_dimension;
        let wg_x = total_wg.min(max_dim);
        let wg_y = total_wg.div_ceil(wg_x);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    encoder.copy_buffer_to_buffer(&counter_buf, 0, &counter_rb, 0, 4);
    queue.submit(Some(encoder.finish()));

    let count_bytes = read_buffer(device, &counter_rb)?;
    let count = u32::from_le_bytes(count_bytes[0..4].try_into().expect("4 bytes"));
    if count == 0 {
        return Ok(SparseTree::from_bricks(res, Vec::new()));
    }

    // Pass 2: read back only the `count` occupied bricks (coords + leaves).
    let coords_rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("noise_gen_bricks coords readback"),
        size: u64::from(count) * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let leaves_rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("noise_gen_bricks leaves readback"),
        size: u64::from(count) * 64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("noise_gen_bricks readback"),
    });
    encoder.copy_buffer_to_buffer(&coords_buf, 0, &coords_rb, 0, u64::from(count) * 4);
    encoder.copy_buffer_to_buffer(&leaves_buf, 0, &leaves_rb, 0, u64::from(count) * 64);
    queue.submit(Some(encoder.finish()));

    let coord_data = read_buffer(device, &coords_rb)?;
    let leaf_data = read_buffer(device, &leaves_rb)?;
    let coords: &[u32] = bytemuck::cast_slice(&coord_data);
    let leaf_words: &[u32] = bytemuck::cast_slice(&leaf_data);

    let bricks: Vec<(u64, LeafBrick)> = (0..count as usize)
        .map(|i| {
            let c = coords[i];
            let (bx, by, bz) = (c & 0x3ff, (c >> 10) & 0x3ff, (c >> 20) & 0x3ff);
            let mut w = [0u32; 16];
            w.copy_from_slice(&leaf_words[i * 16..i * 16 + 16]);
            (morton::encode(bx, by, bz), LeafBrick::from_words32(w))
        })
        .collect();

    Ok(SparseTree::from_bricks(res, bricks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::{OccupancyField, Resolution, VoxelCoord};

    /// The GPU tree agrees with the CPU `NoiseField` (the reference) on the vast
    /// majority of voxels — only threshold-grazing voxels differ (f32 vs f64).
    /// Skipped with no GPU adapter (CPU-only CI), like the differential tests.
    #[test]
    fn gpu_tree_matches_cpu_field_closely() {
        let Ok(ctx) = GpuContext::try_new() else {
            return;
        };
        let res = Resolution::new(128).unwrap();
        let field = NoiseField::caves(res);
        let tree = generate_noise_tree(&ctx, &field).expect("gpu gen");

        let n = res.voxels_per_axis();
        let mut diffs = 0u64;
        let mut occupied = 0u64;
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let c = VoxelCoord::new(x, y, z);
                    let g = tree.is_occupied(c);
                    if g {
                        occupied += 1;
                    }
                    if g != field.is_occupied(c) {
                        diffs += 1;
                    }
                }
            }
        }
        assert!(occupied > 0, "gpu field is empty");
        assert!(
            diffs * 100 < occupied,
            "gpu/cpu disagree on {diffs} voxels vs {occupied} occupied (>1%) — port mismatch?"
        );
    }

    /// Manual A/B: GPU brick gen vs CPU build for Caves. Run with
    /// `cargo test -p voxel-gpu --release gpu_vs_cpu_build_timing -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing benchmark; run manually in --release"]
    fn gpu_vs_cpu_build_timing() {
        let Ok(ctx) = GpuContext::try_new() else {
            eprintln!("no GPU; skipping");
            return;
        };

        // 512³: CPU build is bearable (~3 s), so compare directly.
        let res = Resolution::new(512).unwrap();
        let field = NoiseField::caves(res);
        let t = std::time::Instant::now();
        let cpu = SparseTree::build(&field);
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        let gpu = generate_noise_tree(&ctx, &field).expect("gpu gen");
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "caves 512³ (leaves cpu {} gpu {}): CPU {cpu_ms:.1} ms | GPU bricks {gpu_ms:.1} ms ({:.1}x)",
            cpu.leaf_count(),
            gpu.leaf_count(),
            cpu_ms / gpu_ms
        );

        // 2048³: the case the CPU build takes ~154 s for. GPU-only here.
        let res = Resolution::new(2048).unwrap();
        let field = NoiseField::caves(res);
        let t = std::time::Instant::now();
        let gpu = generate_noise_tree(&ctx, &field).expect("gpu gen 2048");
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "caves 2048³ (leaves gpu {}): GPU bricks {gpu_ms:.1} ms (CPU build was ~153640 ms → {:.0}x)",
            gpu.leaf_count(),
            153_640.0 / gpu_ms
        );
    }
}
