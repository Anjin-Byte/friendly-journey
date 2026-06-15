// HUD overlay: instanced quads drawn over the blitted image. Each instance is a
// pixel-space rectangle that is either a textured glyph (sampled from the 8x8
// bitmap-font atlas) or a solid fill (panel background, sparkline bars). Alpha
// is blended over the surface by the pipeline.

struct HudQuad {
    rect: vec4<f32>,   // x, y, w, h in pixels (top-left origin)
    uv: vec4<f32>,     // u0, v0, u1, v1 in atlas space (0..1)
    color: vec4<f32>,  // rgba, premultiplied by caller intent
    kind: u32,         // 0 = textured glyph, 1 = solid fill
    // Three scalar u32 pads — NOT a vec3<u32>. In std430 a vec3 is 16-byte
    // aligned, which would round the struct up to 80 bytes; the Rust HudQuad
    // uploads at a 64-byte stride, so a vec3 here desyncs every instance after
    // the first (scattered glyphs). Scalars keep the struct at 64 bytes.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct HudUniform {
    screen: vec2<f32>, // viewport size in pixels
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> quads: array<HudQuad>;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<uniform> u: HudUniform;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) kind: u32,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    // Two triangles covering the unit square, as (corner_x, corner_y) in 0..1.
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
    );
    let c = corners[vi];
    let q = quads[ii];

    // Pixel position (top-left origin) → clip space (y points up).
    let px = q.rect.xy + c * q.rect.zw;
    let ndc = vec2<f32>(px.x / u.screen.x * 2.0 - 1.0, 1.0 - px.y / u.screen.y * 2.0);

    var out: VsOut;
    out.pos = vec4<f32>(ndc, 0.0, 1.0);
    out.uv = mix(q.uv.xy, q.uv.zw, c);
    out.color = q.color;
    out.kind = q.kind;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    if (in.kind == 1u) {
        return in.color;
    }
    let coverage = textureSampleLevel(atlas, samp, in.uv, 0.0).r;
    return vec4<f32>(in.color.rgb, in.color.a * coverage);
}
