// Truecolor BLEND render-path entry (docs/materials/11, Phase 2). Concatenated
// *after* `traversal.wgsl`, so the shared bindings (`nodes`@0, `leaf_words`@1,
// `leaf_bounds`@2) and the DDA helpers (`make_frame`/`walker_step`/`leaf_bit`/
// `leaf_reaches`/`morton8`) are already in scope. Same 7 storage buffers + camera +
// output as `render_truecolor.wgsl`; it ADDS front-to-back alpha compositing.
//
// This is chosen at scene-build time ONLY when the scene has transparent leaves
// (`SchoolBBuffer::has_transparency`); pure-opaque / MASK-only scenes get the
// byte-identical opaque `render_truecolor.wgsl` instead — so the opaque path never
// pays for this kernel.
//
// `PER_CHUNK` and `MAX_BLEND` are INJECTED ahead of this module by
// `buffers::color_blend_shader_source` (single source of truth = the Rust consts).

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

// Per-leaf colour base (prefix sum of count_occupied) + N_MAX=3 colour chunks —
// identical layout to render_truecolor.wgsl.
@group(0) @binding(5) var<storage, read> leaf_color_base: array<u32>;
@group(0) @binding(6) var<storage, read> leaf_color_0: array<u32>;
@group(0) @binding(7) var<storage, read> leaf_color_1: array<u32>;
@group(0) @binding(8) var<storage, read> leaf_color_2: array<u32>;

// === colour read (copied verbatim from render_truecolor.wgsl) ===

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

// The unpacked colour (sRGB RGBA8 → [0,1], A in .w) of the occupied voxel at this
// leaf slot + intra-leaf morton — the SAME `g = base + rank` read the opaque path does.
fn voxel_color(slot: u32, vox: u32) -> vec4<f32> {
    let g = leaf_color_base[slot] + leaf_color_rank(slot, vox);
    return unpack4x8unorm(read_leaf_color(g));
}

// === front-to-back compositing traversal ===
//
// A copy of `traverse_ray`'s DDA (reusing every shared helper) that, instead of
// returning the first occupancy hit, alpha-composites occupied voxels front-to-back
// (the DDA is already front-to-back, so no sort). Returns PREMULTIPLIED accumulated
// colour in `.rgb` and accumulated coverage in `.a`; the caller composites the sky
// under `(1 - .a)`. A hit in a leaf WITHOUT the transparency bit is the opaque
// backdrop (composite at α=1 and stop). Bounded by `acc_a >= 0.99` and `MAX_BLEND`.
fn traverse_and_composite(o: vec3<f32>, d: vec3<f32>, n: f32, k: u32) -> vec4<f32> {
    var acc = vec3<f32>(0.0, 0.0, 0.0); // premultiplied
    var acc_a = 0.0;
    var depth = 0u;

    // Grid-clip (f32 slab) against [0, n]³ — identical to traverse_ray.
    var t_near = -BIG;
    var t_far = BIG;
    var missed = false;
    if (d.x == 0.0) { if (o.x < 0.0 || o.x > n) { missed = true; } }
    else { let inv = 1.0 / d.x; var a = (0.0 - o.x) * inv; var b = (n - o.x) * inv; if (a > b) { let t = a; a = b; b = t; } t_near = max(t_near, a); t_far = min(t_far, b); }
    if (d.y == 0.0) { if (o.y < 0.0 || o.y > n) { missed = true; } }
    else { let inv = 1.0 / d.y; var a = (0.0 - o.y) * inv; var b = (n - o.y) * inv; if (a > b) { let t = a; a = b; b = t; } t_near = max(t_near, a); t_far = min(t_far, b); }
    if (d.z == 0.0) { if (o.z < 0.0 || o.z > n) { missed = true; } }
    else { let inv = 1.0 / d.z; var a = (0.0 - o.z) * inv; var b = (n - o.z) * inv; if (a > b) { let t = a; a = b; b = t; } t_near = max(t_near, a); t_far = min(t_far, b); }

    if (missed || t_near > t_far || t_far < 0.0) {
        return vec4<f32>(acc, acc_a);
    }
    let t_entry = max(t_near, 0.0);

    var root_level = 1u;
    if (k > 0u) {
        root_level = k + 1u;
    }
    var cur = make_frame(o, d, 0u, root_level, vec3<u32>(0u, 0u, 0u), t_entry);
    var stack: array<Frame, 6>;
    var sp = 0u;

    for (var iter = 0u; iter < 200000u; iter = iter + 1u) {
        if (cur.level == 1u) {
            let v = cur.cell;
            if (leaf_bit(cur.node, v)) {
                let c = voxel_color(cur.node, morton8(v & vec3<u32>(7u)));
                let is_blend = ((leaf_bounds[cur.node] >> 18u) & 1u) == 1u;
                if (!is_blend) {
                    // Opaque backdrop: composite at α=1 and stop.
                    acc = acc + (1.0 - acc_a) * c.rgb;
                    return vec4<f32>(acc, 1.0);
                }
                // Semi-transparent voxel: premultiplied OVER, then continue the DDA.
                let wgt = (1.0 - acc_a) * c.a;
                acc = acc + wgt * c.rgb;
                acc_a = acc_a + wgt;
                depth = depth + 1u;
                if (acc_a >= 0.99 || depth >= MAX_BLEND) {
                    return vec4<f32>(acc, acc_a);
                }
                // fall through to step past this voxel
            }
            if (walker_step(&cur)) { continue; }
            loop {
                if (sp == 0u) { return vec4<f32>(acc, acc_a); }
                sp = sp - 1u;
                cur = stack[sp];
                if (walker_step(&cur)) { break; }
            }
        } else {
            let c = cur.cell;
            let bit = child_bit(c);
            let node = nodes[cur.node];
            let child_level = cur.level - 1u;
            let size = cell_size_of(cur.level);
            let child_origin = cur.origin + c * size;
            var descend = has_child(node, bit);
            var slot = 0u;
            if (descend) {
                slot = child_slot(node, bit);
                if (child_level == 1u) {
                    descend = leaf_reaches(slot, o, d, child_origin, cur.t_entry);
                }
            }
            if (descend) {
                stack[sp] = cur;
                sp = sp + 1u;
                cur = make_frame(o, d, slot, child_level, child_origin, cur.t_entry);
            } else if (!walker_step(&cur)) {
                loop {
                    if (sp == 0u) { return vec4<f32>(acc, acc_a); }
                    sp = sp - 1u;
                    cur = stack[sp];
                    if (walker_step(&cur)) { break; }
                }
            }
        }
    }
    return vec4<f32>(acc, acc_a);
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

    let acc = traverse_and_composite(camera.eye, dir, camera.n, camera.dims.z);

    // The same sky gradient as the opaque path, composited under residual
    // transmittance (1 - acc.a). Opaque backdrop (acc.a == 1) → just acc.rgb.
    let t = f32(gid.y) / h;
    let sky = vec3<f32>(0.08 * (1.0 - t), 0.10 * (1.0 - t), 0.16 + 0.12 * t);
    let rgb = acc.rgb + (1.0 - acc.a) * sky;
    textureStore(output, vec2<u32>(gid.x, gid.y), vec4<f32>(rgb, 1.0));
}
