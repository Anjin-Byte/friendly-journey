// Render-path entry point. Concatenated *after* traversal.wgsl, so the shared
// structure bindings (`nodes`@0, `leaf_words`@1, `leaf_bounds`@2) and
// `traverse_ray` are already in scope; per-call data starts at binding 3.
//
// One invocation per pixel: builds the camera ray, traverses, shades the hit by
// voxel position, and writes the color straight to a storage texture — no
// readback, no CPU ray-gen, no CPU shading.

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
// Per-leaf packed material slots (STRIDE_W u32 each) and the global colour table.
// Read once cold at the hit only — never on the hot occupancy probe.
@group(0) @binding(5) var<storage, read> leaf_mat: array<u32>;
@group(0) @binding(6) var<storage, read> material_table: array<u32>;

// Mirrors voxel_core::palette (STRIDE_W=73, PAL_OFF=1, IDX_OFF=9). Pinned to the
// CPU packer by the wgsl_bit_layout_matches_pack parity test.
const MAT_STRIDE_W: u32 = 73u;
const MAT_PAL_OFF: u32 = 1u;
const MAT_IDX_OFF: u32 = 9u;

// The global material id at leaf slot `s`, intra-brick morton voxel `m`. This is
// the second implementation of the bit layout the CPU `pack_leaf` writes (see
// docs/materials/03-gpu-read.md §2 and the parity test). A `bits == 0` slot — a
// single-material leaf OR the deferred uniform-magenta spill — reads palette
// slot 0 for every voxel; the `bits == 0` short-circuit is mandatory because the
// mask `(1u << 0u) - 1u` is degenerate.
fn read_material(s: u32, m: u32) -> u32 {
    let base = s * MAT_STRIDE_W;
    let bits = leaf_mat[base] & 0xFu; // bits_per_voxel
    var pi: u32 = 0u;
    if (bits != 0u) {
        let off = m * bits;
        let iw = base + MAT_IDX_OFF + (off >> 5u);
        let pos = off & 31u;
        pi = leaf_mat[iw] >> pos;
        if (pos + bits > 32u) { // straddles a 32-bit word (vs the reference's 64)
            pi = pi | (leaf_mat[iw + 1u] << (32u - pos));
        }
        pi = pi & ((1u << bits) - 1u);
    }
    let pal_word = leaf_mat[base + MAT_PAL_OFF + (pi >> 1u)];
    return (pal_word >> (16u * (pi & 1u))) & 0xFFFFu; // u16 low/high half-select
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
        // One cold material read at the hit (hit.leaf = slot, hit.vox = morton).
        let mat_id = read_material(hit.leaf, hit.vox);
        if (mat_id == 0u) {
            // global-0 = no material (occupancy-only fixture, or an unresolved
            // voxel): fall back to position shading, the prior look.
            color = vec4<f32>(f32(hit.world.x) / camera.n, f32(hit.world.y) / camera.n, f32(hit.world.z) / camera.n, 1.0);
        } else {
            // Real material: RGBA8 colour from the global table (R in low byte).
            color = unpack4x8unorm(material_table[mat_id]);
        }
    } else {
        let t = f32(gid.y) / h;
        color = vec4<f32>(0.08 * (1.0 - t), 0.10 * (1.0 - t), 0.16 + 0.12 * t, 1.0);
    }
    textureStore(output, vec2<u32>(gid.x, gid.y), color);
}
