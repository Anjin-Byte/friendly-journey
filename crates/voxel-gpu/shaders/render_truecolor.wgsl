// Truecolor render-path entry point (docs/materials/11, P4). Concatenated *after*
// `traversal.wgsl`, so the shared structure bindings (`nodes`@0, `leaf_words`@1,
// `leaf_bounds`@2) and `traverse_ray` are already in scope. Unlike the palette
// `render.wgsl`, this reads the per-voxel baked colour directly: it DROPS the
// palette `leaf_mat`@5 / `material_table`@6 and instead binds `leaf_color_base`@5
// plus N_MAX colour chunks@6..8 — keeping the truecolor storage-buffer count
// (3 carried + base + 3 chunks = 7) under the stock 8-per-stage ceiling.
//
// `const PER_CHUNK: u32` is INJECTED ahead of this module by
// `buffers::color_shader_source` (single source of truth = Rust `COLOR_PER_CHUNK`,
// pinned by `color_shader_source_injects_per_chunk`). Do NOT hand-declare it here.

struct Camera {
    eye: vec3<f32>,
    tan: f32,
    forward: vec3<f32>,
    aspect: f32,
    right: vec3<f32>,
    n: f32,
    up: vec3<f32>,
    _pad: f32,
    dims: vec4<u32>, // width, height, k, _
}

@group(0) @binding(3) var<uniform> camera: Camera;
@group(0) @binding(4) var output: texture_storage_2d<rgba8unorm, write>;

// Per-leaf colour base (prefix sum of count_occupied) + N_MAX=3 colour chunks.
// One cold read at the hit only — never on the hot occupancy probe loop.
@group(0) @binding(5) var<storage, read> leaf_color_base: array<u32>;
@group(0) @binding(6) var<storage, read> leaf_color_0: array<u32>;
@group(0) @binding(7) var<storage, read> leaf_color_1: array<u32>;
@group(0) @binding(8) var<storage, read> leaf_color_2: array<u32>;

// Frozen transcription of `LeafBrick::occupied_rank` (== leaf.rs `wgsl_rank`,
// parity-pinned): a 16-word masked popcount over the SAME `leaf_words` view
// `leaf_bit` reads (stride 16 u32/leaf). Counts occupied voxels with intra-brick
// Morton STRICTLY < `m`. NOT one `countOneBits` — a single-word mask is UB for
// `m >= 32`. The `rem == 0` skip is load-bearing: for `m` a multiple of 32 the
// partial word index would be `full` (16 at m=512, OOB); `m < 512` keeps
// `full <= 15`. Do NOT optimise it away.
fn leaf_color_rank(slot: u32, m: u32) -> u32 {
    let wbase = slot * 16u;
    let full = m >> 5u;
    var rank: u32 = 0u;
    for (var w: u32 = 0u; w < full; w = w + 1u) {
        rank = rank + countOneBits(leaf_words[wbase + w]);
    }
    let rem = m & 31u;
    if (rem > 0u) {
        rank = rank + countOneBits(leaf_words[wbase + full] & ((1u << rem) - 1u));
    }
    return rank;
}

// Chunk-select over N_MAX=3. The capability probe guarantees a valid global index
// `g < N * PER_CHUNK <= N_MAX * PER_CHUNK`, so `g / PER_CHUNK < N <= N_MAX` — the
// final arm is reached only for a real chunk 2, never a dummy-bound unused slot.
fn read_leaf_color(g: u32) -> u32 {
    let chunk = g / PER_CHUNK;
    let local = g % PER_CHUNK;
    if (chunk == 0u) {
        return leaf_color_0[local];
    }
    if (chunk == 1u) {
        return leaf_color_1[local];
    }
    return leaf_color_2[local];
}

@compute @workgroup_size(8, 8)
fn render_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = camera.dims.x;
    let height = camera.dims.y;
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    let w = f32(width);
    let h = f32(height);
    let ndc_x = ((f32(gid.x) + 0.5) / w * 2.0 - 1.0) * camera.tan * camera.aspect;
    let ndc_y = (1.0 - (f32(gid.y) + 0.5) / h * 2.0) * camera.tan;
    let dir = normalize(camera.forward + camera.right * ndc_x + camera.up * ndc_y);

    let hit = traverse_ray(camera.eye, dir, camera.n, camera.dims.z);

    var color: vec4<f32>;
    if (hit.hit == 1u) {
        // One cold colour read at the hit (hit.leaf = slot, hit.vox = morton).
        // g = leaf_color_base[slot] + rank(morton); the stored u32 is sRGB RGBA8
        // (R low), unpacked verbatim into the rgba8unorm store (no re-encode).
        let g = leaf_color_base[hit.leaf] + leaf_color_rank(hit.leaf, hit.vox);
        color = unpack4x8unorm(read_leaf_color(g));
    } else {
        let t = f32(gid.y) / h;
        color = vec4<f32>(0.08 * (1.0 - t), 0.10 * (1.0 - t), 0.16 + 0.12 * t, 1.0);
    }
    textureStore(output, vec2<u32>(gid.x, gid.y), color);
}
