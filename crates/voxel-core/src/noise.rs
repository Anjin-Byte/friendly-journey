//! Deterministic 3-D gradient (Perlin-style) noise — pure and dependency-free.
//!
//! Build-time only (like [`morton`](crate::morton)): the noise fixtures sample
//! this to decide occupancy; the GPU never runs it. It is Ken Perlin's
//! improved-noise gradient on the integer lattice, summed into fractional
//! Brownian motion ([`fractal`]), with a ridged variant and [`domain_warp`] for
//! organic, swirling features. All functions are pure and deterministic in
//! `(seed, point)`, so builds and tests reproduce exactly.

// Gradient-noise math is inherently cast-heavy (lattice floor → int, offsets
// back to float) and single-letter (the x/y/z/u/v/w of interpolation); allow the
// pedantic lints that flag the idiomatic form (Engineering Codex: Clippy as
// Discipline — intentional, scoped configuration).
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]

/// Integer-lattice hash for gradient selection. Deterministic; the lattice
/// coordinates are cast to `u32` (wrapping), which is fine — collisions only at
/// a `2³²` period, far outside any grid we build.
fn lattice_hash(seed: u32, ix: i64, iy: i64, iz: i64) -> u32 {
    let x = ix as u32;
    let y = iy as u32;
    let z = iz as u32;
    let mut h = seed
        ^ x.wrapping_mul(0x8da6_b343)
        ^ y.wrapping_mul(0xd816_3841)
        ^ z.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h
}

/// Ken Perlin's improved-noise gradient: dot of `(x, y, z)` with one of 12
/// edge-direction gradients chosen by the low bits of `hash`.
fn grad(hash: u32, x: f64, y: f64, z: f64) -> f64 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    let u = if h & 1 == 0 { u } else { -u };
    let v = if h & 2 == 0 { v } else { -v };
    u + v
}

/// Quintic fade curve `6t⁵ − 15t⁴ + 10t³` (zero 1st/2nd derivative at 0 and 1).
fn fade(t: f64) -> f64 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// 3-D gradient noise at `p`, in roughly `[-1, 1]`. Deterministic in
/// `(seed, p)` and continuous (`C²`) everywhere.
pub(crate) fn perlin3(seed: u32, p: [f64; 3]) -> f64 {
    let [x, y, z] = p;
    let (fx, fy, fz) = (x.floor(), y.floor(), z.floor());
    let (ix, iy, iz) = (fx as i64, fy as i64, fz as i64);
    let (xf, yf, zf) = (x - fx, y - fy, z - fz);
    let (u, v, w) = (fade(xf), fade(yf), fade(zf));

    // Gradient dot at lattice corner (dx, dy, dz) ∈ {0,1}³.
    let g = |dx: i64, dy: i64, dz: i64| {
        grad(
            lattice_hash(seed, ix + dx, iy + dy, iz + dz),
            xf - dx as f64,
            yf - dy as f64,
            zf - dz as f64,
        )
    };

    let x00 = lerp(g(0, 0, 0), g(1, 0, 0), u);
    let x10 = lerp(g(0, 1, 0), g(1, 1, 0), u);
    let x01 = lerp(g(0, 0, 1), g(1, 0, 1), u);
    let x11 = lerp(g(0, 1, 1), g(1, 1, 1), u);
    let y0 = lerp(x00, x10, v);
    let y1 = lerp(x01, x11, v);
    lerp(y0, y1, w)
}

/// Parameters for a fractal (multi-octave) noise sum.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Fractal {
    /// Number of octaves summed.
    pub octaves: u32,
    /// Frequency multiplier between octaves (`> 1`; classically `2.0`).
    pub lacunarity: f64,
    /// Amplitude multiplier between octaves (`< 1`; classically `0.5`).
    pub gain: f64,
    /// Ridged multifractal (`(1 − |n|)²` per octave) instead of plain fBm.
    pub ridged: bool,
}

/// Fractional Brownian motion: a sum of `octaves` gradient-noise octaves at
/// geometrically increasing frequency and decreasing amplitude, normalized to
/// roughly `[-1, 1]`.
///
/// With `ridged`, each octave contributes `(1 − |n|)²` (sharp ridges/veins) and
/// the result is remapped to `[-1, 1]` so the same `threshold` semantics apply.
pub(crate) fn fractal(seed: u32, p: [f64; 3], f: Fractal) -> f64 {
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for o in 0..f.octaves {
        // Offset each octave's domain and seed so octaves decorrelate.
        let off = f64::from(o);
        let sp = [
            p[0] * freq + off * 17.13,
            p[1] * freq - off * 9.71,
            p[2] * freq + off * 5.37,
        ];
        let n = perlin3(seed.wrapping_add(o.wrapping_mul(0x9e37_79b9)), sp);
        let contrib = if f.ridged {
            let r = 1.0 - n.abs();
            r * r
        } else {
            n
        };
        sum += amp * contrib;
        norm += amp;
        amp *= f.gain;
        freq *= f.lacunarity;
    }
    let mean = if norm > 0.0 { sum / norm } else { 0.0 };
    if f.ridged {
        // Ridged octaves are in [0, 1]; recenter to a symmetric [-1, 1].
        mean * 2.0 - 1.0
    } else {
        mean
    }
}

/// Domain warping: displaces `p` by `amp` times a (low-octave) fractal field
/// sampled on three decorrelated channels. Bends the noise into swirls,
/// overhangs, and tunnels — the "interesting features" a plain isosurface lacks.
pub(crate) fn domain_warp(seed: u32, p: [f64; 3], amp: f64, f: Fractal) -> [f64; 3] {
    let wx = fractal(seed ^ 0x1111_1111, [p[0] + 1.7, p[1] + 9.2, p[2] + 3.3], f);
    let wy = fractal(seed ^ 0x2222_2222, [p[0] - 5.1, p[1] + 2.8, p[2] - 7.4], f);
    let wz = fractal(seed ^ 0x3333_3333, [p[0] + 8.6, p[1] - 4.5, p[2] + 1.9], f);
    [p[0] + amp * wx, p[1] + amp * wy, p[2] + amp * wz]
}

#[cfg(test)]
mod tests {
    use super::*;

    const FBM: Fractal = Fractal {
        octaves: 5,
        lacunarity: 2.0,
        gain: 0.5,
        ridged: false,
    };
    const RIDGED: Fractal = Fractal {
        ridged: true,
        ..FBM
    };

    #[test]
    #[allow(clippy::float_cmp)] // exact reproducibility is the property under test
    fn perlin_is_deterministic() {
        let p = [3.25, -7.5, 12.125];
        assert_eq!(perlin3(42, p), perlin3(42, p));
        assert_ne!(perlin3(42, p), perlin3(43, p), "seed must matter");
    }

    #[test]
    fn perlin_is_roughly_unit_range_and_centered() {
        // Sample a grid; values stay within ~[-1,1] and average near zero.
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        let mut sum = 0.0;
        let mut count = 0.0;
        for zi in 0..20 {
            for yi in 0..20 {
                for xi in 0..20 {
                    let p = [
                        f64::from(xi) * 0.37,
                        f64::from(yi) * 0.37,
                        f64::from(zi) * 0.37,
                    ];
                    let n = perlin3(7, p);
                    min = min.min(n);
                    max = max.max(n);
                    sum += n;
                    count += 1.0;
                }
            }
        }
        assert!(
            min > -1.2 && max < 1.2,
            "range [{min}, {max}] escaped ~[-1,1]"
        );
        assert!(
            (sum / count).abs() < 0.15,
            "mean {} not near 0",
            sum / count
        );
    }

    #[test]
    fn perlin_is_continuous() {
        // A small step in the domain makes a small step in value (no jumps).
        let p = [4.2, 1.1, -3.7];
        let q = [p[0] + 1e-4, p[1], p[2]];
        assert!((perlin3(1, p) - perlin3(1, q)).abs() < 1e-2);
    }

    #[test]
    fn perlin_is_zero_on_the_lattice() {
        // Gradient noise vanishes at integer lattice points (the dot is with 0).
        assert!(perlin3(99, [3.0, -2.0, 5.0]).abs() < 1e-12);
    }

    #[test]
    fn fractal_variants_stay_in_range() {
        for zi in 0..12 {
            for yi in 0..12 {
                for xi in 0..12 {
                    let p = [
                        f64::from(xi) * 0.5,
                        f64::from(yi) * 0.5,
                        f64::from(zi) * 0.5,
                    ];
                    let a = fractal(3, p, FBM);
                    let b = fractal(3, p, RIDGED);
                    assert!(a > -1.3 && a < 1.3, "fbm {a} out of range");
                    assert!(b > -1.3 && b < 1.3, "ridged {b} out of range");
                }
            }
        }
    }

    #[test]
    #[allow(clippy::float_cmp)] // zero-amplitude warp is exactly the identity
    fn domain_warp_moves_the_point() {
        let p = [2.0, 3.0, 4.0];
        let w = domain_warp(5, p, 0.5, FBM);
        let moved = (0..3).any(|i| (w[i] - p[i]).abs() > 1e-6);
        assert!(moved, "warp should displace the point");
        // Zero amplitude is the identity.
        assert_eq!(domain_warp(5, p, 0.0, FBM), p);
    }
}
