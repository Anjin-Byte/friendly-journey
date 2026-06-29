//! Import-DTO **appearance vocabulary**: the per-material base-colour types the
//! loaders produce and the per-voxel bake consumes — `Texture` / `WrapMode` /
//! `AlphaMode` (boundary value types, with `Texture`'s checked constructor) plus
//! `MaterialDef` / `MeshAppearance`. This is the IO/compute *boundary* vocabulary,
//! kept out of the compute modules (`bake` owns only the sampling math).
//!
//! `MeshInput` is the sibling import DTO intentionally left in [`crate::core`];
//! a future `voxel-io` extraction lifts both this module and `MeshInput`.

use crate::error::VoxelizerError;

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
}

/// One material's base-colour appearance: an optional sRGB texture (an index into
/// [`MeshAppearance::textures`]), the **linear** base-colour tint, and the
/// sampler's wrap modes. Indexed by `material_id`.
#[derive(Debug, Clone)]
pub struct MaterialDef {
    /// The source material name (glTF `material.name`, MTL `newmtl`), lower-cased
    /// matching is used to spot toon **outline** hulls in [`crate::core::MeshInput::drop_outline_triangles`].
    /// `None` for unnamed materials.
    pub name: Option<String>,
    /// Index into [`MeshAppearance::textures`], or `None` for an untextured
    /// (flat `base_color_factor`) material.
    pub base_color_texture: Option<usize>,
    /// Linear `base_color_factor` tint (multiplies the sampled texel).
    pub base_color_factor: [f32; 4],
    /// Wrap mode for the U axis.
    pub wrap_s: WrapMode,
    /// Wrap mode for the V axis.
    pub wrap_t: WrapMode,
    /// glTF `alphaMode` (OBJ/STL default `Opaque`). MASK voxels below
    /// [`alpha_cutoff`](Self::alpha_cutoff) are cut at bake time.
    pub alpha_mode: AlphaMode,
    /// glTF `alphaCutoff` (default `0.5`); only meaningful for `alpha_mode == Mask`.
    pub alpha_cutoff: f32,
}

/// A mesh's base-colour appearance: the decoded textures plus one [`MaterialDef`]
/// per `material_id`. Carried alongside the geometry so the per-voxel texture bake
/// (`docs/materials/11`) can resolve `triangle → material → texture + UV → texel`.
#[derive(Debug, Clone)]
pub struct MeshAppearance {
    /// Decoded sRGB base-colour textures, indexed by [`MaterialDef::base_color_texture`].
    pub textures: Vec<Texture>,
    /// Per-`material_id` appearance.
    pub materials: Vec<MaterialDef>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::VoxelizerError;

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
}
