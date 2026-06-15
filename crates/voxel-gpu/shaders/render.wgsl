// Render-path entry point. Concatenated *after* traversal.wgsl, so the shared
// `nodes`/`leaf_words` bindings and `traverse_ray` are already in scope.
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

@group(0) @binding(2) var<uniform> camera: Camera;
@group(0) @binding(3) var output: texture_storage_2d<rgba8unorm, write>;

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
    if (hit.w == 1u) {
        // Shade by voxel position, with a touch of depth-ish dimming via y.
        color = vec4<f32>(f32(hit.x) / camera.n, f32(hit.y) / camera.n, f32(hit.z) / camera.n, 1.0);
    } else {
        let t = f32(gid.y) / h;
        color = vec4<f32>(0.08 * (1.0 - t), 0.10 * (1.0 - t), 0.16 + 0.12 * t, 1.0);
    }
    textureStore(output, vec2<u32>(gid.x, gid.y), color);
}
