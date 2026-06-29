//! CPU texture baking: the canonical per-voxel colour sample.
//!
//! Given a voxel centre, its **colour owner** triangle (3 world vertices + 3 UVs),
//! and that triangle's material texture, [`expected_color`] reconstructs where on
//! the surface the voxel sits and reads the texel — the single source of truth the
//! GPU/bridge is validated against (docs/materials/11-truecolor-design.md).
//!
//! Every step is pinned so the result is reproducible and oracle-testable:
//! 1. **closest point** on the triangle to the voxel centre (Ericson's clamped
//!    region test — handles edge/vertex/degenerate cases),
//! 2. **barycentric** weights *recomputed from the clamped point* (so they sum to
//!    1 and stay in `[0,1]`),
//! 3. **UV interpolation**,
//! 4. **wrap** (REPEAT = `fract`, CLAMP = `saturate`) per axis,
//! 5. **bilinear in LINEAR space** — glTF base-colour textures are sRGB-encoded, so
//!    each tap is decoded sRGB→linear before blending,
//! 6. **linear tint** by `base_color_factor` (linear per the glTF spec),
//! 7. **re-encode linear→sRGB** to RGBA8 — matching the renderer's verbatim-sRGB
//!    display path (`unpack4x8unorm` → `rgba8unorm` store).
//!
//! The byte order is R in the low byte (RGBA8 little-endian) to match the GPU
//! `unpack4x8unorm`.

// The closest-point/barycentric solvers are the textbook (Ericson) form with
// `d1..d6` / `va,vb,vc` names — clearer left as-is than renamed.
#![allow(clippy::many_single_char_names)]

use crate::error::VoxelizerError;
use glam::{Vec2, Vec3};

/// glTF sampler wrap mode for one axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrapMode {
    /// `fract(u)` — tile the texture.
    Repeat,
    /// `clamp(u, 0, 1)` — extend the edge texel.
    ClampToEdge,
}

/// glTF `alphaMode` — how a material's base-colour alpha is interpreted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AlphaMode {
    /// Alpha is ignored; the surface is fully opaque. The default.
    #[default]
    Opaque,
    /// Alpha-test: a texel is opaque iff its alpha `>= alpha_cutoff`, else absent.
    Mask,
    /// Alpha compositing (semi-transparent). Rendered, not cut.
    Blend,
}

/// A decoded, **sRGB-encoded** RGBA8 base-colour texture (alpha is linear).
///
/// The fields are private and a `Texture` is only constructible through the
/// checked [`Texture::new`], so the invariant `rgba.len() == width * height`
/// with `width >= 1 && height >= 1` holds for *every* value that exists (Codex:
/// *Make Invalid States Unrepresentable*, *Checked Constructors*). That is what
/// lets the sampling primitives index `rgba` without a bounds check and never
/// panic — a malformed texture cannot be represented, so it cannot reach them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Texture {
    width: u32,
    height: u32,
    rgba: Vec<[u8; 4]>,
}

impl Texture {
    /// Construct a texture, enforcing `width >= 1`, `height >= 1`, and
    /// `rgba.len() == width * height` (computed in `usize`, so huge dims cannot
    /// overflow the check).
    ///
    /// # Errors
    /// Returns [`VoxelizerError::MalformedTexture`] if any invariant is
    /// violated — the boundary at which a corrupt or hand-built texture is
    /// rejected gracefully instead of panicking mid-bake.
    pub fn new(width: u32, height: u32, rgba: Vec<[u8; 4]>) -> Result<Self, VoxelizerError> {
        let expected = (width as usize).checked_mul(height as usize);
        if width == 0 || height == 0 || expected != Some(rgba.len()) {
            return Err(VoxelizerError::MalformedTexture {
                width,
                height,
                len: rgba.len(),
            });
        }
        Ok(Self {
            width,
            height,
            rgba,
        })
    }

    /// Width in texels (always `>= 1`).
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Height in texels (always `>= 1`).
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Row-major `width * height` RGBA8 texels (R in `[0]`).
    #[must_use]
    pub fn rgba(&self) -> &[[u8; 4]] {
        &self.rgba
    }

    /// The texel at `(x, y)` (caller clamps/wraps to valid coords).
    ///
    /// Indexes in `usize`; the [`Texture::new`] invariant guarantees the index
    /// is in range for any in-bounds `(x, y)`, so this cannot panic in practice.
    #[must_use]
    fn texel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = (y as usize) * (self.width as usize) + (x as usize);
        self.rgba[i]
    }
}

/// sRGB → linear for one 0..=255 channel byte.
#[must_use]
fn srgb_to_linear(b: u8) -> f32 {
    let c = f32::from(b) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// linear → sRGB → 0..=255 byte (round-to-nearest).
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn linear_to_srgb_u8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

/// Wraps one UV coordinate into a samplable range per the mode.
#[must_use]
fn wrap(u: f32, mode: WrapMode) -> f32 {
    match mode {
        WrapMode::Repeat => u - u.floor(), // fract, always in [0,1)
        WrapMode::ClampToEdge => u.clamp(0.0, 1.0),
    }
}

/// Wraps an integer texel index per the mode (for the bilinear neighbour taps).
#[must_use]
fn wrap_texel(i: i64, dim: u32, mode: WrapMode) -> u32 {
    let d = i64::from(dim);
    match mode {
        WrapMode::Repeat => i.rem_euclid(d) as u32,
        WrapMode::ClampToEdge => i.clamp(0, d - 1) as u32,
    }
}

/// Bilinear sample in LINEAR space at UV `uv` (already wrapped into range per the
/// mode by the caller is NOT assumed — wrap is applied here), returning linear
/// RGBA (alpha linear). Uses the half-texel-centre convention.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn sample_bilinear_linear(tex: &Texture, uv: Vec2, wrap_s: WrapMode, wrap_t: WrapMode) -> [f32; 4] {
    let u = wrap(uv.x, wrap_s);
    let v = wrap(uv.y, wrap_t);
    // Texel-centre convention: texel centre i maps to (i+0.5)/dim.
    let fx = u * tex.width as f32 - 0.5;
    let fy = v * tex.height as f32 - 0.5;
    let x0 = fx.floor();
    let y0 = fy.floor();
    let tx = fx - x0;
    let ty = fy - y0;
    let (x0, y0) = (x0 as i64, y0 as i64);

    // Four taps, each wrapped to a valid texel, decoded sRGB→linear (alpha linear).
    let tap = |xi: i64, yi: i64| -> [f32; 4] {
        let x = wrap_texel(xi, tex.width, wrap_s);
        let y = wrap_texel(yi, tex.height, wrap_t);
        let t = tex.texel(x, y);
        [
            srgb_to_linear(t[0]),
            srgb_to_linear(t[1]),
            srgb_to_linear(t[2]),
            f32::from(t[3]) / 255.0,
        ]
    };
    let c00 = tap(x0, y0);
    let c10 = tap(x0 + 1, y0);
    let c01 = tap(x0, y0 + 1);
    let c11 = tap(x0 + 1, y0 + 1);

    let mut out = [0.0f32; 4];
    for k in 0..4 {
        let a = c00[k] * (1.0 - tx) + c10[k] * tx;
        let b = c01[k] * (1.0 - tx) + c11[k] * tx;
        out[k] = a * (1.0 - ty) + b * ty;
    }
    out
}

/// The closest point on triangle `(a,b,c)` to `p` (Ericson, *Real-Time Collision
/// Detection* §5.1.5). Handles the three vertex regions, three edge regions, and
/// the interior; a degenerate (zero-area) triangle falls through to `a`. Public so
/// the nearest-surface colour-owner pass can rank candidates by distance.
#[must_use]
pub fn closest_point_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a; // vertex region A
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b; // vertex region B
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v; // edge AB
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c; // vertex region C
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w; // edge AC
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w; // edge BC
    }
    let denom = 1.0 / (va + vb + vc); // interior
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}

/// Barycentric weights `(u, v, w)` of `p` w.r.t. triangle `(a,b,c)` (projected via
/// the standard 2-basis dot solve). Recomputed from the (already-clamped) `p`, so
/// for a point on the triangle the weights sum to 1 and lie in `[0,1]`. A
/// degenerate triangle returns `(1,0,0)`.
#[must_use]
fn barycentric(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (f32, f32, f32) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-20 {
        return (1.0, 0.0, 0.0); // degenerate
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    (1.0 - v - w, v, w)
}

/// Encodes a LINEAR RGBA sample, tinted by the material's linear `factor`, into
/// sRGB RGBA8 with byte order `[R, G, B, A]` (R low, matching `unpack4x8unorm`): the
/// RGB channels go through the sRGB transfer, alpha stays linear.
#[must_use]
fn encode_color(linear: [f32; 4], factor: [f32; 4]) -> [u8; 4] {
    [
        linear_to_srgb_u8(linear[0] * factor[0]),
        linear_to_srgb_u8(linear[1] * factor[1]),
        linear_to_srgb_u8(linear[2] * factor[2]),
        ((linear[3] * factor[3]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
    ]
}

/// The baked sRGB RGBA8 colour for a voxel centred at `centre` whose colour owner is
/// triangle `tri` with per-vertex `uvs`, via a SINGLE bilinear tap at the
/// closest-point UV. When `texture` is `None` the flat linear `factor` is encoded
/// directly. `wrap` is `(wrap_s, wrap_t)`; `factor` is the material's **linear**
/// `base_color_factor`. This is the unfiltered oracle — [`expected_color_filtered`]
/// is the minification-aware sampler the bake actually uses.
///
/// Byte order is `[R, G, B, A]` (R low), matching the renderer's `unpack4x8unorm`.
#[must_use]
pub fn expected_color(
    centre: Vec3,
    tri: [Vec3; 3],
    uvs: [Vec2; 3],
    texture: Option<&Texture>,
    wrap: (WrapMode, WrapMode),
    factor: [f32; 4],
) -> [u8; 4] {
    let Some(tex) = texture else {
        return encode_color([1.0, 1.0, 1.0, 1.0], factor);
    };
    let p = closest_point_on_triangle(centre, tri[0], tri[1], tri[2]);
    let (u, v, w) = barycentric(p, tri[0], tri[1], tri[2]);
    let uv = uvs[0] * u + uvs[1] * v + uvs[2] * w;
    encode_color(sample_bilinear_linear(tex, uv, wrap.0, wrap.1), factor)
}

/// Upper bound on the per-axis supersample count in [`expected_color_filtered`] (so
/// a huge minification footprint can't blow up the bake cost). 8×8 = 64 taps.
const MAX_SUPERSAMPLE: u32 = 8;

/// Clamps a barycentric coordinate to the triangle simplex (each ≥ 0, renormalised
/// to sum 1) so a supersample offset past an edge maps to the nearest in-triangle UV
/// instead of extrapolating into a foreign atlas region.
fn clamp_bary(u: f32, v: f32, w: f32) -> (f32, f32, f32) {
    let (u, v, w) = (u.max(0.0), v.max(0.0), w.max(0.0));
    let sum = u + v + w;
    if sum > 1e-9 {
        (u / sum, v / sum, w / sum)
    } else {
        (1.0, 0.0, 0.0)
    }
}

/// Like [`expected_color`] but **prefilters texture minification**: it box-averages
/// the texture over the voxel's texel footprint instead of taking one tap, removing
/// the moiré/aliasing a 1-tap bake produces when a voxel covers more than one texel
/// (heavy at low resolution; at the texel≈voxel Nyquist point around 2048³). The
/// footprint = the owner triangle's UV-per-grid gradient × the texture size (the
/// voxel cell is one grid unit, so this is texels-per-voxel-edge); the per-axis
/// sample count adapts — `1` when magnified (identical to [`expected_color`]) up to
/// `MAX_SUPERSAMPLE` when minified. Samples are box-distributed across the cell on
/// the triangle plane (barycentric clamped to the triangle) and averaged in LINEAR
/// space before the sRGB encode.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn expected_color_filtered(
    centre: Vec3,
    tri: [Vec3; 3],
    uvs: [Vec2; 3],
    texture: Option<&Texture>,
    wrap: (WrapMode, WrapMode),
    factor: [f32; 4],
) -> [u8; 4] {
    let Some(tex) = texture else {
        return encode_color([1.0, 1.0, 1.0, 1.0], factor);
    };
    let p = closest_point_on_triangle(centre, tri[0], tri[1], tri[2]);

    // Texels-per-voxel-edge: UV change per grid unit (max over the two edges, a cheap
    // isotropic proxy) times the texture size. ≤ 1 ⇒ magnification ⇒ a single tap.
    let e1 = tri[1] - tri[0];
    let e2 = tri[2] - tri[0];
    let l1 = e1.length();
    let l2 = e2.length();
    let d1 = if l1 > 1e-9 {
        (uvs[1] - uvs[0]).length() / l1
    } else {
        0.0
    };
    let d2 = if l2 > 1e-9 {
        (uvs[2] - uvs[0]).length() / l2
    } else {
        0.0
    };
    let footprint = d1.max(d2) * tex.width.max(tex.height) as f32;
    // `+ 0.5` is a half-texel guard band: it pushes the texel≈voxel Nyquist zone
    // (footprint ≈ 1, where the worst moiré lives) up to s=2 rather than leaving it
    // at a single tap, while a clearly magnified footprint (≲ 0.5) still resolves to
    // s=1 (one tap, identical to `expected_color`).
    let s = if footprint.is_finite() {
        ((footprint + 0.5).ceil() as u32).clamp(1, MAX_SUPERSAMPLE)
    } else {
        1
    };

    if s <= 1 {
        let (u, v, w) = barycentric(p, tri[0], tri[1], tri[2]);
        let uv = uvs[0] * u + uvs[1] * v + uvs[2] * w;
        return encode_color(sample_bilinear_linear(tex, uv, wrap.0, wrap.1), factor);
    }

    // In-plane orthonormal axes to spread the samples across the voxel cell.
    let normal = e1.cross(e2).normalize_or_zero();
    let t1 = e1.normalize_or_zero();
    let t2 = normal.cross(t1);
    let inv = 1.0 / s as f32;
    let mut acc = [0.0f32; 4];
    for i in 0..s {
        for j in 0..s {
            // Stratified offsets in [-0.5, 0.5) grid units = the cell extent.
            let oi = (i as f32 + 0.5) * inv - 0.5;
            let oj = (j as f32 + 0.5) * inv - 0.5;
            let sp = p + t1 * oi + t2 * oj;
            let (u, v, w) = barycentric(sp, tri[0], tri[1], tri[2]);
            let (u, v, w) = clamp_bary(u, v, w);
            let uv = uvs[0] * u + uvs[1] * v + uvs[2] * w;
            let texel = sample_bilinear_linear(tex, uv, wrap.0, wrap.1);
            for k in 0..4 {
                acc[k] += texel[k];
            }
        }
    }
    let n = (s * s) as f32;
    encode_color([acc[0] / n, acc[1] / n, acc[2] / n, acc[3] / n], factor)
}

/// A triangle considered as a **colour owner** for a voxel: its global index (the
/// min-index tie-break key), world vertices, UVs, and resolved appearance (the
/// texture/wrap/factor already looked up from the material). The colour owner is
/// chosen *independently* of the occupancy owner (`docs/materials/11`, D2): at
/// 2048³ most occupied voxels are multi-covered, and the occupancy min-index
/// triangle is usually not the surface in the cell.
#[derive(Debug, Clone, Copy)]
pub struct ColorCandidate<'a> {
    /// Global triangle index — the min-index tie-break key for equal distances.
    pub tri_index: usize,
    /// World-space triangle vertices.
    pub verts: [Vec3; 3],
    /// Per-vertex base-colour UVs.
    pub uvs: [Vec2; 3],
    /// The material's base-colour texture (`None` = flat factor).
    pub texture: Option<&'a Texture>,
    /// `(wrap_s, wrap_t)` sampler modes.
    pub wrap: (WrapMode, WrapMode),
    /// The material's linear `base_color_factor`.
    pub factor: [f32; 4],
}

/// Picks the **nearest-surface** colour owner among `candidates` (argmin squared
/// closest-point distance to `centre`, **lowest `tri_index` on an exact tie**) and
/// bakes its colour via [`expected_color`]. Returns `None` only when `candidates`
/// is empty (a voxel with no active triangle — the caller decides the fallback).
///
/// The exact `dist2 == bd` tie test is deliberate: a genuine tie means two
/// coincident triangles produce bit-identical distances (resolve by min index);
/// any real distance difference takes the `<` branch.
#[must_use]
pub fn bake_nearest_color(centre: Vec3, candidates: &[ColorCandidate]) -> Option<[u8; 4]> {
    bake_nearest_owner(centre, candidates).map(|(_, color)| color)
}

/// Like [`bake_nearest_color`], but also returns the **owner's index** into
/// `candidates` — so the caller can recover the owner triangle (its `tri_index`,
/// hence its material) for the MASK alpha-cutout decision and the opaque-alpha force.
/// `None` only when `candidates` is empty.
#[must_use]
#[allow(clippy::float_cmp)]
pub fn bake_nearest_owner(centre: Vec3, candidates: &[ColorCandidate]) -> Option<(usize, [u8; 4])> {
    let mut best: Option<(f32, usize, usize)> = None; // (dist2, tri_index, candidate idx)
    for (i, c) in candidates.iter().enumerate() {
        let p = closest_point_on_triangle(centre, c.verts[0], c.verts[1], c.verts[2]);
        let dist2 = (p - centre).length_squared();
        let better = match best {
            None => true,
            // Nearer wins; on an exact distance tie the lower global index wins —
            // matching the occupancy owner's min-index rule for coincident tris.
            Some((bd, bi, _)) => dist2 < bd || (dist2 == bd && c.tri_index < bi),
        };
        if better {
            best = Some((dist2, c.tri_index, i));
        }
    }
    best.map(|(_, _, i)| {
        let c = &candidates[i];
        (
            i,
            expected_color_filtered(centre, c.verts, c.uvs, c.texture, c.wrap, c.factor),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const WHITE_FACTOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    const REPEAT: (WrapMode, WrapMode) = (WrapMode::Repeat, WrapMode::Repeat);
    const CLAMP: (WrapMode, WrapMode) = (WrapMode::ClampToEdge, WrapMode::ClampToEdge);

    /// 2×2 checker: (0,0)=red (1,0)=green (0,1)=blue (1,1)=white, all opaque.
    fn checker() -> Texture {
        Texture::new(
            2,
            2,
            vec![
                [255, 0, 0, 255],     // (0,0) red
                [0, 255, 0, 255],     // (1,0) green
                [0, 0, 255, 255],     // (0,1) blue
                [255, 255, 255, 255], // (1,1) white
            ],
        )
        .expect("2x2 checker is a valid texture")
    }

    /// `Texture::new` accepts a texture whose `rgba` exactly fills `width*height`
    /// and exposes the dimensions back through the accessors.
    #[test]
    fn texture_new_accepts_exact_grid() {
        let t = Texture::new(2, 3, vec![[0, 0, 0, 255]; 6]).expect("2x3 of 6 texels is valid");
        assert_eq!((t.width(), t.height(), t.rgba().len()), (2, 3, 6));
    }

    /// `Texture::new` rejects an `rgba` shorter (or longer) than `width*height` —
    /// the exact pre-fix panic class (`Texture::texel` indexed past `rgba`).
    #[test]
    fn texture_new_rejects_length_mismatch() {
        let short = Texture::new(8, 8, vec![[1, 2, 3, 4]]); // 1 texel, needs 64
        assert!(
            matches!(short, Err(VoxelizerError::MalformedTexture { .. })),
            "short rgba must be rejected, got {short:?}"
        );
        let long = Texture::new(1, 1, vec![[0; 4]; 2]); // 2 texels, needs 1
        assert!(matches!(long, Err(VoxelizerError::MalformedTexture { .. })));
    }

    /// `Texture::new` rejects a zero dimension (the pre-fix `wrap_texel`
    /// `clamp(0,-1)` / `rem_euclid(0)` panic at `dim == 0`).
    #[test]
    fn texture_new_rejects_zero_dimension() {
        assert!(matches!(
            Texture::new(0, 4, vec![]),
            Err(VoxelizerError::MalformedTexture { .. })
        ));
        assert!(matches!(
            Texture::new(4, 0, vec![]),
            Err(VoxelizerError::MalformedTexture { .. })
        ));
    }

    /// Huge dimensions are rejected at construction (the `width*height` product is
    /// computed in `usize`, so the check itself cannot overflow) — closing the
    /// pre-fix `u32` index-overflow path before any indexing happens.
    #[test]
    fn texture_new_rejects_huge_dimensions() {
        let huge = Texture::new(70_000, 70_000, vec![[0; 4]]); // 4.9e9 texels declared
        assert!(matches!(huge, Err(VoxelizerError::MalformedTexture { .. })));
    }

    /// A unit triangle in the z=0 plane whose UVs map directly to the unit square,
    /// so a voxel-centre query at world (x,y,0) samples UV (x,y).
    fn flat_tri() -> ([Vec3; 3], [Vec2; 3]) {
        // Large triangle covering the unit square: (0,0),(2,0),(0,2) with matching
        // UVs, so the closest point for any centre inside [0,1]² is the centre and
        // the UV equals (x,y).
        (
            [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
            ],
            [
                Vec2::new(0.0, 0.0),
                Vec2::new(2.0, 0.0),
                Vec2::new(0.0, 2.0),
            ],
        )
    }

    #[test]
    fn closest_point_interior_edge_vertex() {
        let (a, b, c) = (
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        // A point above the interior projects onto it.
        let p = closest_point_on_triangle(Vec3::new(0.25, 0.25, 5.0), a, b, c);
        assert!((p - Vec3::new(0.25, 0.25, 0.0)).length() < 1e-5);
        // A point far past vertex A clamps to A.
        let p = closest_point_on_triangle(Vec3::new(-3.0, -3.0, 0.0), a, b, c);
        assert!((p - a).length() < 1e-5);
        // A point beyond the hypotenuse clamps onto edge BC.
        let p = closest_point_on_triangle(Vec3::new(1.0, 1.0, 0.0), a, b, c);
        assert!((p - Vec3::new(0.5, 0.5, 0.0)).length() < 1e-5);
    }

    #[test]
    fn barycentric_recovers_weights() {
        let (a, b, c) = (
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let (u, v, w) = barycentric(Vec3::new(0.25, 0.5, 0.0), a, b, c);
        assert!((u - 0.25).abs() < 1e-5 && (v - 0.25).abs() < 1e-5 && (w - 0.5).abs() < 1e-5);
        assert!((u + v + w - 1.0).abs() < 1e-5);
    }

    #[test]
    fn samples_texel_centres_exactly() {
        let tex = checker();
        let (tri, uvs) = flat_tri();
        // Texel centres are at UV (0.25,0.25),(0.75,0.25),(0.25,0.75),(0.75,0.75).
        let at = |x, y| {
            expected_color(
                Vec3::new(x, y, 0.0),
                tri,
                uvs,
                Some(&tex),
                CLAMP,
                WHITE_FACTOR,
            )
        };
        assert_eq!(at(0.25, 0.25), [255, 0, 0, 255], "texel (0,0) red");
        assert_eq!(at(0.75, 0.25), [0, 255, 0, 255], "texel (1,0) green");
        assert_eq!(at(0.25, 0.75), [0, 0, 255, 255], "texel (0,1) blue");
        assert_eq!(at(0.75, 0.75), [255, 255, 255, 255], "texel (1,1) white");
    }

    #[test]
    fn bilinear_centre_blends_in_linear_space() {
        let tex = checker();
        let (tri, uvs) = flat_tri();
        // UV (0.5,0.5) is the meeting point of all four texels → equal-weight blend
        // IN LINEAR SPACE. Linear avg of {red,green,blue,white} per channel:
        //   R: (1+0+0+1)/4 = 0.5 ; G: (0+1+0+1)/4 = 0.5 ; B: (0+0+1+1)/4 = 0.5
        // re-encoded to sRGB: linear 0.5 → ~188.
        let got = expected_color(
            Vec3::new(0.5, 0.5, 0.0),
            tri,
            uvs,
            Some(&tex),
            CLAMP,
            WHITE_FACTOR,
        );
        assert_eq!(got[3], 255);
        for (ch, &val) in got.iter().take(3).enumerate() {
            assert!(
                (i32::from(val) - 188).abs() <= 1,
                "channel {ch} = {val} (want ~188)"
            );
        }
        // The naive WRONG sRGB-space average would be (255+0+0+255)/4 = 127 — assert
        // we are NOT that, proving linear blending.
        assert!(
            got[0] > 180,
            "must blend in linear (got {}), not sRGB-space 127",
            got[0]
        );
    }

    #[test]
    fn repeat_wraps_uv_past_one() {
        let tex = checker();
        let (tri, uvs) = flat_tri();
        // UV 1.25 under REPEAT → fract 0.25 → same as the 0.25 texel-centre column.
        let wrapped = expected_color(
            Vec3::new(1.25, 0.25, 0.0),
            tri,
            uvs,
            Some(&tex),
            REPEAT,
            WHITE_FACTOR,
        );
        let base = expected_color(
            Vec3::new(0.25, 0.25, 0.0),
            tri,
            uvs,
            Some(&tex),
            REPEAT,
            WHITE_FACTOR,
        );
        assert_eq!(wrapped, base, "REPEAT must alias UV 1.25 to 0.25");
    }

    #[test]
    fn untextured_uses_linear_factor() {
        // No texture → encode the linear factor. Linear 0.5 → sRGB ~188.
        let got = expected_color(
            Vec3::ZERO,
            flat_tri().0,
            flat_tri().1,
            None,
            CLAMP,
            [0.5, 0.5, 0.5, 1.0],
        );
        for (ch, &val) in got.iter().take(3).enumerate() {
            assert!((i32::from(val) - 188).abs() <= 1, "channel {ch} = {val}");
        }
    }

    #[test]
    fn degenerate_triangle_does_not_panic() {
        let tex = checker();
        let zero = [Vec3::ZERO, Vec3::ZERO, Vec3::ZERO];
        let uvs = [Vec2::new(0.25, 0.25); 3];
        let got = expected_color(Vec3::ZERO, zero, uvs, Some(&tex), CLAMP, WHITE_FACTOR);
        assert_eq!(
            got,
            [255, 0, 0, 255],
            "degenerate → vertex-0 UV (0.25,0.25) = red"
        );
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn filtered_sampler_antialiases_minified_texture() {
        // 16×16 texture, left half black / right half white (features 8 texels wide,
        // so a single bilinear tap reads pure black or pure white in each half — NOT
        // pre-smoothed). UVs lay it at 1.3 uv per grid unit → footprint 1.3 periods
        // (minification) AND non-commensurate with the integer voxel grid, so the
        // point sample's phase DRIFTS across voxels (the moiré) while the box filter
        // averages a full period to the 50% mean (sRGB ≈ 188) at every voxel.
        let n = 16u32;
        let mut rgba = Vec::with_capacity((n * n) as usize);
        for _y in 0..n {
            for x in 0..n {
                let c = if x < n / 2 { 0 } else { 255 };
                rgba.push([c, c, c, 255]);
            }
        }
        let tex = Texture::new(n, n, rgba).expect("n×n checker is a valid texture");
        let tri = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(8.0, 0.0, 0.0),
            Vec3::new(0.0, 8.0, 0.0),
        ];
        let uvs = [
            Vec2::new(0.0, 0.0),
            Vec2::new(10.4, 0.0), // 10.4 / 8 = 1.3 uv per grid unit
            Vec2::new(0.0, 10.4),
        ];

        let mut tap_min = 255u8;
        let mut tap_max = 0u8;
        let mut filt_min = 255u8;
        let mut filt_max = 0u8;
        for y in 0..8u32 {
            for x in 0..8u32 {
                let c = Vec3::new(x as f32 + 0.5, y as f32 + 0.5, 0.0);
                let tap = expected_color(c, tri, uvs, Some(&tex), REPEAT, WHITE_FACTOR);
                let filt = expected_color_filtered(c, tri, uvs, Some(&tex), REPEAT, WHITE_FACTOR);
                tap_min = tap_min.min(tap[0]);
                tap_max = tap_max.max(tap[0]);
                filt_min = filt_min.min(filt[0]);
                filt_max = filt_max.max(filt[0]);
            }
        }
        // Point-sampling aliases: it visits both near-black and near-white extremes.
        assert!(
            tap_min < 40 && tap_max > 215,
            "single tap should alias to extremes, got [{tap_min}, {tap_max}]"
        );
        // The filtered sampler stays in a tight mid band around the grey mean — no
        // swing to the extremes, i.e. the moiré is gone.
        assert!(
            filt_min > 140 && filt_max < 215 && filt_max - filt_min < 60,
            "filtered should stay grey, got [{filt_min}, {filt_max}]"
        );
    }

    /// A flat triangle covering the unit square in the `z = z0` plane (UVs match
    /// the square; used to place colour candidates at controlled distances).
    fn tri_at_z(z: f32) -> [Vec3; 3] {
        [
            Vec3::new(0.0, 0.0, z),
            Vec3::new(2.0, 0.0, z),
            Vec3::new(0.0, 2.0, z),
        ]
    }

    fn flat_candidate(tri_index: usize, z: f32, factor: [f32; 4]) -> ColorCandidate<'static> {
        ColorCandidate {
            tri_index,
            verts: tri_at_z(z),
            uvs: flat_tri().1,
            texture: None, // untextured → colour is the encoded factor (distinct per cand.)
            wrap: CLAMP,
            factor,
        }
    }

    #[test]
    fn nearest_surface_owner_wins() {
        // Two candidates straddling the voxel centre at z=1: A (z=0, dist 1) is
        // nearer than B (z=10, dist 9), so A's colour is baked — even though B has
        // the lower index. This is the blocker the review flagged: nearest-surface,
        // NOT occupancy min-index.
        let centre = Vec3::new(0.5, 0.5, 1.0);
        let near = flat_candidate(7, 0.0, [1.0, 0.0, 0.0, 1.0]); // red, far index
        let far = flat_candidate(2, 10.0, [0.0, 1.0, 0.0, 1.0]); // green, low index
        let got = bake_nearest_color(centre, &[near, far]).expect("non-empty");
        assert_eq!(got, [255, 0, 0, 255], "nearer (red) triangle must win");
    }

    #[test]
    fn exact_distance_tie_breaks_to_min_index() {
        // A (z=0) and B (z=2) are equidistant from centre z=1. The lower tri_index
        // wins regardless of list order.
        let centre = Vec3::new(0.5, 0.5, 1.0);
        let a = flat_candidate(5, 0.0, [1.0, 0.0, 0.0, 1.0]); // red, index 5
        let b = flat_candidate(2, 2.0, [0.0, 1.0, 0.0, 1.0]); // green, index 2
        let got = bake_nearest_color(centre, &[a, b]).expect("non-empty");
        assert_eq!(got, [0, 255, 0, 255], "tie → lower index (green) wins");
        // Order-independent.
        let got_rev = bake_nearest_color(centre, &[b, a]).expect("non-empty");
        assert_eq!(got_rev, [0, 255, 0, 255], "tie-break is order-independent");
    }

    #[test]
    fn no_candidates_returns_none() {
        assert_eq!(bake_nearest_color(Vec3::ZERO, &[]), None);
    }
}
