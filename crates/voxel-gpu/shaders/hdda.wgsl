// Buffer-path entry point. Concatenated *after* traversal.wgsl, so the shared
// `nodes`/`leaf_words` bindings and `traverse_ray` are already in scope.
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

@group(0) @binding(2) var<storage, read> rays: array<Ray>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> hits: array<vec4<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ri = gid.x;
    if (ri >= params.ray_count) {
        return;
    }
    hits[ri] = traverse_ray(rays[ri].origin, rays[ri].dir, f32(params.n), params.k);
}
