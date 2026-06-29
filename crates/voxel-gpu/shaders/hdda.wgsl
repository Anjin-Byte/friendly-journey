// Buffer-path entry point. Concatenated *after* traversal.wgsl, so the shared
// structure bindings (`nodes`@0, `leaf_words`@1, `leaf_bounds`@2) and
// `traverse_ray` are already in scope; per-call data starts at binding 3.
//
// Reads one ray per invocation and writes its hit (vx, vy, vz, hit_flag).

struct Ray {
    origin: vec3<f32>,
    dir: vec3<f32>,
}

struct Params {
    n: u32,
    k: u32,
    ray_count: u32,
    _pad: u32,
}

@group(0) @binding(3) var<storage, read> rays: array<Ray>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read_write> hits: array<vec4<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ri = gid.x;
    if (ri >= params.ray_count) {
        return;
    }
    // Down-convert the widened HitResult to the frozen `vec4<u32>` hits layout
    // (world.xyz + hit flag) so the readback Pod + the GPU↔mirror differential
    // stay byte-identical; the leaf/vox fields are headless-path-irrelevant.
    let h = traverse_ray(rays[ri].origin, rays[ri].dir, f32(params.n), params.k);
    hits[ri] = vec4<u32>(h.world.x, h.world.y, h.world.z, h.hit);
}
