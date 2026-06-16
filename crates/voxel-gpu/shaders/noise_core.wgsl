// Shared noise core for the GPU occupancy generators — concatenated ahead of a
// kernel that consumes `occupied(x, y, z)` (see `noise_gen.wgsl`, the dense
// bitset, and `noise_gen_bricks.wgsl`, the compacted-leaf path).
//
// A direct f32 port of `voxel-core`'s `noise.rs` (Perlin improved-noise gradient
// → fBm/ridged `fractal` → `domain_warp`) plus `NoiseField::value`. The CPU f64
// field stays the reference/oracle; f32 disagrees only on a sub-1% of
// threshold-grazing voxels, which never reach the differential.

struct Params {
    n: u32,           // voxels per axis
    seed: u32,
    octaves: u32,
    ridged: u32,      // bool
    total_words: u32, // n³ / 32 (dense path) — unused by the brick path
    frequency: f32,
    lacunarity: f32,
    gain: f32,
    warp: f32,
    threshold: f32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;

// Integer-lattice hash for gradient selection (mirrors `lattice_hash`).
fn lattice_hash(seed: u32, ix: i32, iy: i32, iz: i32) -> u32 {
    let x = bitcast<u32>(ix);
    let y = bitcast<u32>(iy);
    let z = bitcast<u32>(iz);
    var h = seed ^ (x * 0x8da6b343u) ^ (y * 0xd8163841u) ^ (z * 0xcb1ab31fu);
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    return h;
}

// Ken Perlin's improved-noise gradient (mirrors `grad`).
fn grad(hash: u32, x: f32, y: f32, z: f32) -> f32 {
    let h = hash & 15u;
    var u = select(y, x, h < 8u);
    var v = select(select(z, x, h == 12u || h == 14u), y, h < 4u);
    if ((h & 1u) != 0u) { u = -u; }
    if ((h & 2u) != 0u) { v = -v; }
    return u + v;
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn perlin3(seed: u32, p: vec3<f32>) -> f32 {
    let fl = floor(p);
    let i = vec3<i32>(fl);
    let f = p - fl;
    let u = fade(f.x);
    let v = fade(f.y);
    let w = fade(f.z);

    let g000 = grad(lattice_hash(seed, i.x, i.y, i.z), f.x, f.y, f.z);
    let g100 = grad(lattice_hash(seed, i.x + 1, i.y, i.z), f.x - 1.0, f.y, f.z);
    let g010 = grad(lattice_hash(seed, i.x, i.y + 1, i.z), f.x, f.y - 1.0, f.z);
    let g110 = grad(lattice_hash(seed, i.x + 1, i.y + 1, i.z), f.x - 1.0, f.y - 1.0, f.z);
    let g001 = grad(lattice_hash(seed, i.x, i.y, i.z + 1), f.x, f.y, f.z - 1.0);
    let g101 = grad(lattice_hash(seed, i.x + 1, i.y, i.z + 1), f.x - 1.0, f.y, f.z - 1.0);
    let g011 = grad(lattice_hash(seed, i.x, i.y + 1, i.z + 1), f.x, f.y - 1.0, f.z - 1.0);
    let g111 = grad(lattice_hash(seed, i.x + 1, i.y + 1, i.z + 1), f.x - 1.0, f.y - 1.0, f.z - 1.0);

    let x00 = mix(g000, g100, u);
    let x10 = mix(g010, g110, u);
    let x01 = mix(g001, g101, u);
    let x11 = mix(g011, g111, u);
    let y0 = mix(x00, x10, v);
    let y1 = mix(x01, x11, v);
    return mix(y0, y1, w);
}

// fBm / ridged multifractal octave sum (mirrors `fractal`).
fn fractal(seed: u32, p: vec3<f32>, octaves: u32, lacunarity: f32, gain: f32, ridged: bool) -> f32 {
    var amp = 1.0;
    var freq = 1.0;
    var sum = 0.0;
    var norm = 0.0;
    for (var o = 0u; o < octaves; o = o + 1u) {
        let off = f32(o);
        let sp = vec3<f32>(
            p.x * freq + off * 17.13,
            p.y * freq - off * 9.71,
            p.z * freq + off * 5.37,
        );
        let n = perlin3(seed + o * 0x9e3779b9u, sp);
        var contrib = n;
        if (ridged) {
            let r = 1.0 - abs(n);
            contrib = r * r;
        }
        sum = sum + amp * contrib;
        norm = norm + amp;
        amp = amp * gain;
        freq = freq * lacunarity;
    }
    var mean = 0.0;
    if (norm > 0.0) { mean = sum / norm; }
    if (ridged) { return mean * 2.0 - 1.0; }
    return mean;
}

// Domain warp on three decorrelated channels (mirrors `domain_warp`).
fn domain_warp(seed: u32, p: vec3<f32>, amp: f32, octaves: u32, lac: f32, gain: f32, ridged: bool) -> vec3<f32> {
    let wx = fractal(seed ^ 0x11111111u, vec3<f32>(p.x + 1.7, p.y + 9.2, p.z + 3.3), octaves, lac, gain, ridged);
    let wy = fractal(seed ^ 0x22222222u, vec3<f32>(p.x - 5.1, p.y + 2.8, p.z - 7.4), octaves, lac, gain, ridged);
    let wz = fractal(seed ^ 0x33333333u, vec3<f32>(p.x + 8.6, p.y - 4.5, p.z + 1.9), octaves, lac, gain, ridged);
    return vec3<f32>(p.x + amp * wx, p.y + amp * wy, p.z + amp * wz);
}

// `NoiseField::is_occupied` for voxel (x, y, z).
fn occupied(x: u32, y: u32, z: u32) -> bool {
    let n = f32(params.n);
    let s = params.frequency / n;
    var p = vec3<f32>(
        (f32(x) + 0.5) * s,
        (f32(y) + 0.5) * s,
        (f32(z) + 0.5) * s,
    );
    let ridged = params.ridged != 0u;
    if (params.warp > 0.0) {
        let warp_oct = min(params.octaves, 3u);
        p = domain_warp(params.seed, p, params.warp, warp_oct, params.lacunarity, params.gain, ridged);
    }
    let v = fractal(params.seed, p, params.octaves, params.lacunarity, params.gain, ridged);
    return v > params.threshold;
}
