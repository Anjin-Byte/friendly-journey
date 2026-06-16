//! GPU occupancy generation for the procedural noise fixtures.
//!
//! [`generate_noise_occupancy`] evaluates a [`NoiseField`] on the GPU — one
//! invocation per output word, no atomics — and reads the dense occupancy back
//! into a [`BitGrid`] the caller hands to [`SparseTree::build`]. Noise evaluation
//! is the entire cost of building these fixtures (the CPU build saturates every
//! core yet still takes seconds at 512³ and minutes at 2048³), and it is pure,
//! per-voxel, divergence-free ALU work — the GPU's home turf.
//!
//! The CPU [`NoiseField`] stays the f64 reference/oracle; this is the fast path.
//! f32 will not bit-match f64 at threshold-grazing voxels, but the field is
//! statistically and visually identical and the differential never uses this.
//!
//! [`SparseTree::build`]: voxel_core::SparseTree::build

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use voxel_core::fixtures::NoiseField;
use voxel_core::{BitGrid, LeafBrick, SparseTree, morton};

use crate::buffers;
use crate::context::GpuContext;
use crate::error::GpuError;

/// Workgroup size for the dense bitset kernel (`@workgroup_size(256)`).
const WORKGROUP: u32 = 256;
/// Workgroup size for the brick kernel (`@workgroup_size(64)`).
const BRICK_WORKGROUP: u32 = 64;

/// The shared noise core, concatenated ahead of each kernel (see
/// `buffers::shader_source` for the same pattern on the traversal core).
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

    /// Mirrors `Params` in `noise_gen.wgsl` (field order is significant).
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    pub(crate) struct GenParams {
        pub(crate) n: u32,
        pub(crate) seed: u32,
        pub(crate) octaves: u32,
        pub(crate) ridged: u32,
        pub(crate) total_words: u32,
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

/// Generates `field`'s dense occupancy on the GPU and returns it as a
/// [`BitGrid`], ready for [`SparseTree::build`](voxel_core::SparseTree::build).
///
/// # Errors
/// - [`GpuError::Unsupported`] if `n³/32` exceeds `u32::MAX` (the word index /
///   storage buffer would overflow). All standard resolutions through `2048³`
///   (a 1 GiB bitset) are supported; `8192³` and up fall back to the CPU build.
/// - [`GpuError::Poll`] / [`GpuError::BufferMap`] on a device/readback failure.
#[allow(clippy::cast_possible_truncation)] // counts are bounded < u32::MAX by the guard
#[allow(clippy::too_many_lines)] // one-shot GPU setup: pipeline + buffers + dispatch + readback
pub fn generate_noise_occupancy(ctx: &GpuContext, field: &NoiseField) -> Result<BitGrid, GpuError> {
    let res = field.resolution;
    let n = res.voxels_per_axis();
    let total_voxels = res.total_voxels(); // u128
    // The dense bitset is one u32 per 32 voxels; the word index (and the storage
    // buffer) must fit u32 / the binding limit. n³/32 fits u32 through 2048³
    // (268M words = 1 GiB, within the 4 GiB storage-binding limit); 8192³ and up
    // overflow it → the caller falls back to the CPU build.
    let total_words_u128 = total_voxels / 32; // n is a multiple of 8 → n³ a multiple of 512
    if total_words_u128 > u128::from(u32::MAX) {
        return Err(GpuError::Unsupported {
            n,
            reason: "noise-gen word count exceeds u32 (resolution past 2048³)",
        });
    }
    let total_words = total_words_u128 as u32;
    let word_bytes = u64::from(total_words) * 4;

    let device = &ctx.device;
    let queue = &ctx.queue;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("noise_gen"),
        source: wgpu::ShaderSource::Wgsl(
            noise_shader(include_str!("../shaders/noise_gen.wgsl")).into(),
        ),
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("noise_gen layout"),
        entries: &[
            buffers::uniform_entry(0),        // params
            buffers::storage_entry(1, false), // bits (read_write)
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("noise_gen pipeline layout"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("noise_gen pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("generate"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let params = GenParams {
        n,
        seed: field.seed,
        octaves: field.octaves,
        ridged: u32::from(field.ridged),
        total_words,
        frequency: field.frequency as f32,
        lacunarity: field.lacunarity as f32,
        gain: field.gain as f32,
        warp: field.warp as f32,
        threshold: field.threshold as f32,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("noise_gen params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bits_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("noise_gen bits"),
        size: word_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("noise_gen readback"),
        size: word_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("noise_gen bind group"),
        layout: &bind_layout,
        entries: &[
            buffers::bind(0, params_buf.as_entire_binding()),
            buffers::bind(1, bits_buf.as_entire_binding()),
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("noise_gen"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("noise_gen pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // 2D dispatch: the workgroup count exceeds the 65535-per-dimension cap at
        // high resolution (2048³ → ~1.05M workgroups). The shader flattens (x, y)
        // back to a 1D word index via `num_workgroups`.
        let total_wg = total_words.div_ceil(WORKGROUP);
        let max_dim = device.limits().max_compute_workgroups_per_dimension;
        let wg_x = total_wg.min(max_dim);
        let wg_y = total_wg.div_ceil(wg_x);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
    encoder.copy_buffer_to_buffer(&bits_buf, 0, &readback, 0, word_bytes);
    queue.submit(Some(encoder.finish()));

    // Map the readback and recombine u32 pairs into the BitGrid's u64 words
    // (little-endian: word[2k] is the low half of u64 word k — the exact bit
    // order BitGrid::set uses, so it drops straight in).
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|_| GpuError::Poll)?;
    rx.recv().map_err(|_| GpuError::Poll)??;

    let data = slice.get_mapped_range();
    let u32s: &[u32] = bytemuck::cast_slice(&data);
    let words: Vec<u64> = u32s
        .chunks_exact(2)
        .map(|c| u64::from(c[0]) | (u64::from(c[1]) << 32))
        .collect();
    drop(data);
    readback.unmap();

    Ok(BitGrid::from_raw(res, words))
}

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
/// [`SparseTree`] directly — the *optimized* path. Each invocation evaluates one
/// 8³ brick into a register-resident Morton leaf and atomic-appends only the
/// non-empty bricks, so the host reads back just the occupied leaves (≈⅓ at
/// 2048³) and runs [`SparseTree::from_bricks`] — no dense readback and no CPU
/// re-scan, unlike [`generate_noise_occupancy`].
///
/// # Errors
/// - [`GpuError::Unsupported`] if the brick count / packed coordinate / worst-case
///   leaf buffer exceeds a GPU limit (everything through `2048³` is supported).
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
        total_words: 0, // unused by the brick path
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
    use voxel_core::{OccupancyField, Resolution, SparseTree, VoxelCoord};

    /// The optimized brick path produces exactly the same structure as the dense
    /// bitset path (both run the same f32 noise) — every voxel agrees. Skipped
    /// with no GPU adapter (CPU-only CI).
    #[test]
    fn gpu_brick_path_matches_bitset_path() {
        let Ok(ctx) = GpuContext::try_new() else {
            return;
        };
        let res = Resolution::new(128).unwrap();
        let field = NoiseField::caves(res);

        let bitset = SparseTree::build(&generate_noise_occupancy(&ctx, &field).expect("bitset"));
        let bricks = generate_noise_tree(&ctx, &field).expect("bricks");

        assert_eq!(
            bricks.leaf_count(),
            bitset.leaf_count(),
            "brick vs bitset leaf count"
        );
        let n = res.voxels_per_axis();
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let c = VoxelCoord::new(x, y, z);
                    assert_eq!(
                        bricks.is_occupied(c),
                        bitset.is_occupied(c),
                        "voxel {c:?} differs between brick and bitset paths"
                    );
                }
            }
        }
    }

    /// Manual A/B: GPU-gen + build vs CPU build for Caves 512³. Run with
    /// `cargo test -p voxel-gpu --release gpu_vs_cpu_build_timing -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing benchmark; run manually in --release"]
    #[allow(clippy::similar_names)] // cpu_/gpu_ pairs are deliberately parallel
    fn gpu_vs_cpu_build_timing() {
        let Ok(ctx) = GpuContext::try_new() else {
            eprintln!("no GPU; skipping");
            return;
        };
        let res = Resolution::new(512).unwrap();
        let field = NoiseField::caves(res);

        let t = std::time::Instant::now();
        let from_cpu = SparseTree::build(&field);
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t = std::time::Instant::now();
        let grid = generate_noise_occupancy(&ctx, &field).expect("gpu gen");
        let gen_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        let from_gpu = SparseTree::build(&grid);
        let rescan_ms = t.elapsed().as_secs_f64() * 1000.0;

        let t = std::time::Instant::now();
        let brick = generate_noise_tree(&ctx, &field).expect("gpu bricks");
        let brick_ms = t.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
            "caves 512³ (leaves cpu {} bitset {} brick {}):",
            from_cpu.leaf_count(),
            from_gpu.leaf_count(),
            brick.leaf_count()
        );
        eprintln!("  CPU build:    {cpu_ms:8.1} ms (1.0x)");
        eprintln!(
            "  GPU bitset:   {:8.1} ms ({:.1}x)  [gen {gen_ms:.1} + rescan {rescan_ms:.1}]",
            gen_ms + rescan_ms,
            cpu_ms / (gen_ms + rescan_ms)
        );
        eprintln!(
            "  GPU bricks:   {brick_ms:8.1} ms ({:.1}x)",
            cpu_ms / brick_ms
        );

        // 2048³: the case the CPU build takes ~154 s for. GPU-only (CPU ≈153640 ms).
        let res = Resolution::new(2048).unwrap();
        let field = NoiseField::caves(res);
        let t = std::time::Instant::now();
        let grid = generate_noise_occupancy(&ctx, &field).expect("gpu gen 2048");
        let gen_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        let bitset = SparseTree::build(&grid);
        let rescan_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        let brick = generate_noise_tree(&ctx, &field).expect("gpu bricks 2048");
        let brick_ms = t.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "caves 2048³ (leaves bitset {} brick {}; CPU build ≈153640 ms):",
            bitset.leaf_count(),
            brick.leaf_count()
        );
        eprintln!(
            "  GPU bitset:   {:8.1} ms ({:.0}x)  [gen {gen_ms:.1} + rescan {rescan_ms:.1}]",
            gen_ms + rescan_ms,
            153_640.0 / (gen_ms + rescan_ms)
        );
        eprintln!(
            "  GPU bricks:   {brick_ms:8.1} ms ({:.0}x)",
            153_640.0 / brick_ms
        );
    }

    /// The GPU generator agrees with the CPU `NoiseField` on the vast majority of
    /// voxels — only threshold-grazing voxels differ (f32 vs f64). Skipped with no
    /// GPU adapter (CPU-only CI), like the differential tests.
    #[test]
    fn gpu_noise_matches_cpu_field_closely() {
        let Ok(ctx) = GpuContext::try_new() else {
            return;
        };
        let res = Resolution::new(128).unwrap();
        let field = NoiseField::caves(res);
        let grid = generate_noise_occupancy(&ctx, &field).expect("gpu gen");

        let n = res.voxels_per_axis();
        let mut diffs = 0u64;
        let mut occupied = 0u64;
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let c = VoxelCoord::new(x, y, z);
                    let g = grid.is_occupied(c);
                    let f = field.is_occupied(c);
                    if g {
                        occupied += 1;
                    }
                    if g != f {
                        diffs += 1;
                    }
                }
            }
        }
        // Both report a comparable fill, and disagreements (f32 vs f64 at the
        // isosurface) are a small fraction of the occupied set.
        assert!(occupied > 0, "gpu field is empty");
        assert!(
            diffs * 100 < occupied,
            "gpu/cpu disagree on {diffs} voxels vs {occupied} occupied (>1%) — port mismatch?"
        );
    }
}
