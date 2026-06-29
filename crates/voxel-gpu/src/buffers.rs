//! Shared GPU resources for both the buffer path ([`crate::GpuTraverser`]) and
//! the render path ([`crate::GpuRenderer`]): structure upload, the concatenated
//! shader source, and bind-group-layout helpers.

use wgpu::util::DeviceExt;

use voxel_core::SchoolBBuffer;

use crate::error::GpuError;

/// Builds a shader module source by concatenating the shared traversal core
/// ahead of an entry-point module, so both kernels call the same
/// `traverse_ray`.
pub(crate) fn shader_source(entry: &str) -> String {
    format!("{}\n{}", include_str!("../shaders/traversal.wgsl"), entry)
}

/// Uploads the node, leaf, and per-leaf-bounds buffers (each padded to be
/// non-zero-sized so the `k = 0` / empty cases are valid). Returns
/// `(nodes, leaves, leaf_bounds)`.
pub(crate) fn upload_structure(
    device: &wgpu::Device,
    structure: &SchoolBBuffer,
    limit: u64,
) -> Result<(wgpu::Buffer, wgpu::Buffer, wgpu::Buffer), GpuError> {
    let mut node_bytes = bytemuck::cast_slice::<_, u8>(structure.nodes()).to_vec();
    if node_bytes.is_empty() {
        node_bytes = vec![0u8; std::mem::size_of::<voxel_core::GpuNode>()];
    }
    let mut leaf_words: Vec<u32> = structure
        .leaves()
        .iter()
        .flat_map(voxel_core::LeafBrick::words32)
        .collect();
    if leaf_words.is_empty() {
        leaf_words = vec![0u32; 16];
    }
    // One packed LeafBounds word per leaf (same order as `leaf_words`). The
    // empty-structure padding is the FULL bound (never skip), not 0 — which would
    // decode to a bogus single-voxel box — so a stray read stays conservative.
    let mut bound_words = structure.leaf_bounds_words().to_vec();
    if bound_words.is_empty() {
        bound_words = vec![voxel_core::LeafBounds::FULL.pack(); 1];
    }

    for needed in [
        node_bytes.len() as u64,
        (leaf_words.len() * 4) as u64,
        (bound_words.len() * 4) as u64,
    ] {
        if needed > limit {
            return Err(GpuError::BufferTooLarge { needed, limit });
        }
    }

    let nodes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("nodes"),
        contents: &node_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    // `COPY_DST` on the leaf and bounds buffers so an in-place [`Edit::Leaf`] can
    // be patched with `queue.write_buffer` (one leaf's 64 words-bytes + 4 bounds-
    // bytes) instead of rebuilding the whole structure. The node buffer is never
    // patched in place — a topology edit renumbers indices and recreates all
    // three — so it stays `STORAGE`-only.
    let leaves = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("leaves"),
        contents: bytemuck::cast_slice(&leaf_words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let leaf_bounds = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("leaf_bounds"),
        contents: bytemuck::cast_slice(&bound_words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    Ok((nodes, leaves, leaf_bounds))
}

/// Uploads the render-path material buffers: the per-leaf packed material slots
/// (`leaf_mat`, `STRIDE_W` words/leaf, `COPY_DST` for the in-place
/// [`crate::GpuRenderer::update_leaf_mat`] patch) and the global
/// `global_id → colour` table (`material_table`). Both padded non-zero-sized for
/// the empty/fixture cases. The headless traverser does not read materials, so
/// this is render-only.
pub(crate) fn upload_materials(
    device: &wgpu::Device,
    structure: &SchoolBBuffer,
    table: &voxel_core::MaterialTable,
    limit: u64,
) -> Result<(wgpu::Buffer, wgpu::Buffer), GpuError> {
    let mut mat_words = structure.leaf_mat_words().to_vec();
    if mat_words.is_empty() {
        // One uniform slot (header 0 ⇒ bits 0 ⇒ every voxel reads global-0) so
        // the binding is non-zero-sized for the empty structure.
        mat_words = vec![0u32; voxel_core::palette::STRIDE_W];
    }
    let table_words = table.words().to_vec(); // always >= 1 (the magenta slot 0)

    for needed in [(mat_words.len() * 4) as u64, (table_words.len() * 4) as u64] {
        if needed > limit {
            return Err(GpuError::BufferTooLarge { needed, limit });
        }
    }

    let leaf_mat = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("leaf_mat"),
        contents: bytemuck::cast_slice(&mat_words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let material_table = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("material_table"),
        contents: bytemuck::cast_slice(&table_words),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    Ok((leaf_mat, material_table))
}

pub(crate) fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(crate) fn bind(binding: u32, resource: wgpu::BindingResource) -> wgpu::BindGroupEntry {
    wgpu::BindGroupEntry { binding, resource }
}

/// The write-only `D2` storage-texture layout entry (the render output@4), shared
/// by the palette and truecolor bind-group layouts.
pub(crate) fn storage_texture_entry(
    binding: u32,
    format: wgpu::TextureFormat,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

// ---- truecolor (per-voxel baked colour) — docs/materials/11, P4 ---------------

/// Logical colour-chunk size in `u32` entries (`= 128 MiB / 4`). The compact
/// `leaf_color` array is physically split into `N = ceil(len / COLOR_PER_CHUNK)`
/// sub-buffers because the full 2048³ colour buffer (~285 MiB) exceeds both the
/// 128 MiB storage-binding cap and the 256 MiB `max_buffer_size`. Pinned to the
/// 128 MiB floor (not the raised Metal cap) so production and the forced-tiny test
/// share the same `N > 1` shape. **Single source of truth** — injected into the
/// WGSL by [`color_shader_source`] (drift-pinned by a test), never hand-typed.
pub(crate) const COLOR_PER_CHUNK: u32 = 33_554_432;

/// The fixed number of colour-chunk bindings the truecolor shader/layout declares.
/// Truecolor storage-buffer count = 3 carried (`nodes`, `leaf_words`,
/// `leaf_bounds`) + `leaf_color_base` + `N_MAX_CHUNKS` = 7 ≤ the stock 8-per-stage
/// ceiling. `3 × 128 MiB = 384 MiB > 285 MiB`, so 3 chunks cover 2048³.
pub(crate) const N_MAX_CHUNKS: u32 = 3;

/// The maximum number of occupied voxels a truecolor scene can carry:
/// `N_MAX_CHUNKS × COLOR_PER_CHUNK` colour entries (one per occupied voxel).
/// A scene above this is rejected by `probe_truecolor` at renderer
/// construction; exposed so a caller (the viewer) can reject it *before* running
/// the multi-hundred-MB CPU colour bake instead of after.
pub const MAX_TRUECOLOR_VOXELS: usize = (N_MAX_CHUNKS as usize) * (COLOR_PER_CHUNK as usize);

/// Max composited voxels per ray in the BLEND path (`docs/materials/11` Phase 2): a
/// hard cap on colour reads per transparent ray so deep glass can't unbound the
/// cost. Injected into the blend shader by [`color_blend_shader_source`]. Tunable.
pub(crate) const MAX_BLEND: u32 = 8;

/// Builds the truecolor shader source: the injected `PER_CHUNK` const, then the
/// shared `traversal.wgsl` core, then the truecolor entry. Injecting the const
/// here (rather than declaring it in the `.wgsl`) keeps CPU and WGSL in lockstep.
pub(crate) fn color_shader_source(per_chunk: u32) -> String {
    format!(
        "const PER_CHUNK: u32 = {per_chunk}u;\n{}\n{}",
        include_str!("../shaders/traversal.wgsl"),
        include_str!("../shaders/render_truecolor.wgsl"),
    )
}

/// Builds the truecolor **BLEND** shader source (`PER_CHUNK` + `MAX_BLEND` consts,
/// then the shared traversal core, then the front-to-back compositing entry). Same
/// 7 bindings as [`color_shader_source`]; selected only when the scene has
/// transparent leaves.
pub(crate) fn color_blend_shader_source(per_chunk: u32, max_blend: u32) -> String {
    format!(
        "const PER_CHUNK: u32 = {per_chunk}u;\nconst MAX_BLEND: u32 = {max_blend}u;\n{}\n{}",
        include_str!("../shaders/traversal.wgsl"),
        include_str!("../shaders/render_truecolor_blend.wgsl"),
    )
}

/// Re-tiles the flat `leaf_color` array into `ceil(len / per_chunk)` contiguous
/// slices of ≤ `per_chunk` entries (the last partial). This is a pure bijection:
/// `chunks[g / per_chunk][g % per_chunk] == leaf_color[g]` for every `g`, so a
/// leaf's colour block straddling a chunk boundary is harmless (each voxel is read
/// by its own global index). `per_chunk` must be ≥ 1.
pub(crate) fn split_color_chunks(leaf_color: &[u32], per_chunk: u32) -> Vec<&[u32]> {
    leaf_color.chunks(per_chunk as usize).collect()
}

/// The up-front truecolor capability probe (pure over numbers, so unit-testable).
/// Run **before** any colour buffer is created so a failure leaves no partial GPU
/// state. Returns the chunk count `N` on success. `n` is the scene resolution (for
/// the error message), `len` the `leaf_color` length, `base_bytes` the
/// `leaf_color_base` buffer size.
pub(crate) fn probe_truecolor(
    n: u32,
    len: usize,
    base_bytes: u64,
    per_chunk: u32,
    max_storage_buffers: u32,
    binding_cap: u64,
    buffer_cap: u64,
) -> Result<u32, GpuError> {
    // 3 carried storage buffers + leaf_color_base + N_MAX chunks.
    if 4 + N_MAX_CHUNKS > max_storage_buffers {
        return Err(GpuError::Unsupported {
            n,
            reason: "truecolor needs 7 storage buffers but the adapter caps lower",
        });
    }
    let n_chunks = u32::try_from(len.div_ceil(per_chunk as usize)).unwrap_or(u32::MAX);
    if n_chunks > N_MAX_CHUNKS {
        return Err(GpuError::Unsupported {
            n,
            reason: "scene exceeds the compiled truecolor colour-chunk count",
        });
    }
    let cap = binding_cap.min(buffer_cap);
    let chunk_bytes = u64::from(per_chunk) * 4;
    if chunk_bytes > cap {
        return Err(GpuError::BufferTooLarge {
            needed: chunk_bytes,
            limit: cap,
        });
    }
    if base_bytes > cap {
        return Err(GpuError::BufferTooLarge {
            needed: base_bytes,
            limit: cap,
        });
    }
    Ok(n_chunks)
}

/// Uploads the truecolor render buffers: `N = ceil(len / per_chunk)` colour chunks
/// (flat slices of `leaf_color`), the `leaf_color_base` prefix-sum buffer, and one
/// shared 1-`u32` dummy bound into the unused chunk slots `[N, N_MAX)`. All
/// `STORAGE`-only (build-once / static — no `COPY_DST`). Returns
/// `(chunks /* len N */, base, dummy)`. The caller must have run [`probe_truecolor`].
pub(crate) fn upload_color_chunks(
    device: &wgpu::Device,
    leaf_color: &[u32],
    base_words: &[u32],
    per_chunk: u32,
) -> (Vec<wgpu::Buffer>, wgpu::Buffer, wgpu::Buffer) {
    let chunks: Vec<wgpu::Buffer> = split_color_chunks(leaf_color, per_chunk)
        .into_iter()
        .enumerate()
        .map(|(i, slice)| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("leaf_color_{i}")),
                contents: bytemuck::cast_slice(slice),
                usage: wgpu::BufferUsages::STORAGE,
            })
        })
        .collect();
    let base = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("leaf_color_base"),
        contents: bytemuck::cast_slice(base_words),
        usage: wgpu::BufferUsages::STORAGE,
    });
    // wgpu rejects a zero-sized storage binding, so the unused-slot filler is 1 u32.
    let dummy = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("leaf_color_dummy"),
        contents: bytemuck::cast_slice(&[0u32]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    (chunks, base, dummy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_color_chunks_is_a_bijective_retiling() {
        // The straddle-harmless property: chunks[g/per][g%per] == words[g] for all g,
        // so a leaf's block crossing a chunk boundary never corrupts a read.
        let words: Vec<u32> = (100..120u32).collect(); // len 20
        let per = 7u32;
        let chunks = split_color_chunks(&words, per);
        assert_eq!(chunks.len(), 20usize.div_ceil(7), "3 chunks");
        let cat: Vec<u32> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(
            cat, words,
            "concatenation must round-trip (no entry lost/dup)"
        );
        for c in &chunks[..chunks.len() - 1] {
            assert_eq!(c.len(), per as usize, "non-last chunks are full");
        }
        assert_eq!(
            chunks.last().unwrap().len(),
            20 % 7,
            "last chunk is the remainder"
        );
        for g in 0..words.len() {
            assert_eq!(
                chunks[g / per as usize][g % per as usize],
                words[g],
                "bijection broke at g={g}"
            );
        }
    }

    #[test]
    fn color_shader_source_injects_per_chunk() {
        // CPU↔WGSL PER_CHUNK sync: the emitted source must declare exactly the Rust
        // const, ahead of the modules that use it.
        let src = color_shader_source(COLOR_PER_CHUNK);
        assert!(
            src.starts_with("const PER_CHUNK: u32 = "),
            "const must lead"
        );
        assert!(
            src.contains(&format!("const PER_CHUNK: u32 = {COLOR_PER_CHUNK}u;")),
            "injected PER_CHUNK drifted from the Rust const"
        );
        assert!(
            src.contains("fn read_leaf_color"),
            "the truecolor entry is present"
        );
        assert!(
            src.contains("fn traverse_ray"),
            "the shared traversal core is present"
        );
    }

    #[test]
    fn color_blend_shader_source_injects_per_chunk_and_max_blend() {
        // CPU↔WGSL sync for BLEND: both consts declared up front, then the traversal
        // core + the compositing entry.
        let src = color_blend_shader_source(COLOR_PER_CHUNK, MAX_BLEND);
        assert!(
            src.contains(&format!("const PER_CHUNK: u32 = {COLOR_PER_CHUNK}u;")),
            "injected PER_CHUNK drifted from the Rust const"
        );
        assert!(
            src.contains(&format!("const MAX_BLEND: u32 = {MAX_BLEND}u;")),
            "injected MAX_BLEND drifted from the Rust const"
        );
        assert!(
            src.contains("fn traverse_and_composite"),
            "the compositing traversal is present"
        );
        assert!(
            src.contains("fn traverse_ray"),
            "the shared traversal core is present"
        );
    }

    #[test]
    fn probe_truecolor_accepts_fit_and_rejects_each_overflow() {
        let per = COLOR_PER_CHUNK;
        let cap = u64::from(per) * 4; // one chunk's worth of bytes
        // Fits: 1 chunk, small base, 8 storage buffers.
        assert_eq!(
            probe_truecolor(512, 1000, 4000, per, 8, cap, cap).unwrap(),
            1
        );
        // N > N_MAX (needs 4 chunks).
        let big = (N_MAX_CHUNKS as usize + 1) * per as usize;
        assert!(matches!(
            probe_truecolor(2048, big, 4000, per, 8, cap, cap),
            Err(GpuError::Unsupported { .. })
        ));
        // Degraded adapter: fewer than 7 storage buffers.
        assert!(matches!(
            probe_truecolor(512, 1000, 4000, per, 4, cap, cap),
            Err(GpuError::Unsupported { .. })
        ));
        // Per-chunk byte size exceeds the device cap.
        assert!(matches!(
            probe_truecolor(512, 1000, 4000, per, 8, cap - 1, cap),
            Err(GpuError::BufferTooLarge { .. })
        ));
        // Base buffer exceeds the device cap.
        assert!(matches!(
            probe_truecolor(512, 1000, cap + 1, per, 8, cap, cap),
            Err(GpuError::BufferTooLarge { .. })
        ));
    }

    #[test]
    fn max_truecolor_voxels_is_the_probe_accept_reject_boundary() {
        // The exposed pre-bake ceiling (C1) must equal the probe's accept→reject
        // edge to the voxel, so the viewer's early reject and the GPU probe at
        // renderer construction agree exactly.
        let per = COLOR_PER_CHUNK;
        let cap = u64::from(per) * 4;
        assert_eq!(
            MAX_TRUECOLOR_VOXELS,
            (N_MAX_CHUNKS as usize) * (per as usize)
        );
        // Exactly the ceiling → exactly N_MAX_CHUNKS chunks → accepted.
        assert_eq!(
            probe_truecolor(2048, MAX_TRUECOLOR_VOXELS, 4000, per, 8, cap, cap).unwrap(),
            N_MAX_CHUNKS
        );
        // One voxel past the ceiling → a 4th chunk → rejected.
        assert!(matches!(
            probe_truecolor(2048, MAX_TRUECOLOR_VOXELS + 1, 4000, per, 8, cap, cap),
            Err(GpuError::Unsupported { .. })
        ));
    }
}
