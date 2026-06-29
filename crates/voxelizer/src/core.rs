//! Plain-data types and validators for the voxelizer, in `voxel-core`'s
//! vocabulary.
//!
//! The grid is a cubic [`voxel_core::Resolution`] (`8·4^k`) plus a world
//! placement, so voxelizer output drops straight into the renderer's intake:
//! [`VoxelOccupancy`] implements [`voxel_core::OccupancyField`], which
//! [`voxel_core::SparseTree::build`] consumes to produce a
//! [`voxel_core::SchoolBBuffer`]. Per-voxel `owner_id`/`color_rgba` are carried
//! as auxiliary material outputs (not yet consumed by the renderer).

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};
use voxel_core::{OccupancyField, Resolution, SchoolBBuffer, SparseTree, VoxelCoord};

use crate::bake::{AlphaMode, Texture, WrapMode};
use crate::error::VoxelizerError;

/// A cubic voxel grid: a [`Resolution`]-sized lattice (`n³`, `n = 8·4^k`) placed
/// in world space by an origin + uniform voxel size (or an explicit affine
/// `world_to_grid` override).
///
/// The cubic [`Resolution`] is what lets the output feed the renderer directly —
/// `SparseTree`/`SchoolBBuffer` are defined only over `8·4^k` grids.
#[derive(Debug, Clone)]
pub struct VoxelGrid {
    /// The cubic grid resolution (`n` voxels per axis).
    pub resolution: Resolution,
    /// World-space position of the grid origin (the `(0,0,0)` voxel's min corner).
    pub origin_world: Vec3,
    /// World units per voxel (uniform across axes).
    pub voxel_size: f32,
    /// Optional explicit world→grid affine. When `None`, derived from
    /// `origin_world` + `voxel_size`.
    pub world_to_grid: Option<Mat4>,
}

impl VoxelGrid {
    /// A grid at `resolution` placed by `origin_world` + `voxel_size`.
    #[must_use]
    pub fn new(resolution: Resolution, origin_world: Vec3, voxel_size: f32) -> Self {
        Self {
            resolution,
            origin_world,
            voxel_size,
            world_to_grid: None,
        }
    }

    /// A grid at `resolution` sized + placed to enclose `mesh`'s bounding box,
    /// leaving `padding` voxels of margin on every side. Empty meshes collapse to
    /// a unit-size grid at the origin.
    #[must_use]
    pub fn fit_mesh(resolution: Resolution, mesh: &MeshInput, padding: f32) -> Self {
        if mesh.triangles.is_empty() {
            return Self::new(resolution, Vec3::ZERO, 1.0);
        }
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for tri in &mesh.triangles {
            for v in tri {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
        }
        if !lo.is_finite() || !hi.is_finite() {
            lo = Vec3::ZERO;
            hi = Vec3::ZERO;
        }
        let n = resolution.voxels_per_axis() as f32;
        let extent = (hi - lo).max(Vec3::splat(1e-6));
        let max_extent = extent.x.max(extent.y).max(extent.z);
        let usable = (n - 2.0 * padding).max(1.0);
        let voxel_size = max_extent / usable;
        let origin_world = lo - Vec3::splat(padding * voxel_size);
        Self::new(resolution, origin_world, voxel_size)
    }

    /// Voxels per axis (`n`).
    #[must_use]
    pub fn voxels_per_axis(&self) -> u32 {
        self.resolution.voxels_per_axis()
    }

    /// Grid dimensions `[n, n, n]` — the cubic resolution as a 3-vector, for the
    /// dense-grid math and the GPU `Params`.
    #[must_use]
    pub fn dims(&self) -> [u32; 3] {
        [self.voxels_per_axis(); 3]
    }

    /// Total voxel count `n³`.
    #[must_use]
    pub fn num_voxels(&self) -> u64 {
        let n = u64::from(self.voxels_per_axis());
        n * n * n
    }

    /// The world→grid affine: the explicit override if set, else
    /// `scale(1/voxel_size) · translate(-origin_world)`.
    #[must_use]
    pub fn world_to_grid_matrix(&self) -> Mat4 {
        if let Some(mat) = self.world_to_grid {
            return mat;
        }
        let inv = 1.0 / self.voxel_size;
        Mat4::from_scale(Vec3::splat(inv)) * Mat4::from_translation(-self.origin_world)
    }

    /// Validates the placement is finite and well-formed.
    ///
    /// # Errors
    /// Returns [`VoxelizerError`] if `voxel_size` is non-positive/non-finite, the
    /// origin is non-finite, or the `world_to_grid` override is non-finite. The
    /// resolution itself is illegal-by-type (`Resolution` only constructs valid
    /// `8·4^k` sizes), so no dimension check is needed.
    pub fn validate(&self) -> Result<(), VoxelizerError> {
        if !self.voxel_size.is_finite() || self.voxel_size <= 0.0 {
            return Err(VoxelizerError::NonPositiveVoxelSize(self.voxel_size));
        }
        if self.world_to_grid.is_none() && !(1.0_f32 / self.voxel_size).is_finite() {
            return Err(VoxelizerError::VoxelSizeTooSmall(self.voxel_size));
        }
        if !self.origin_world.is_finite() {
            return Err(VoxelizerError::NonFiniteOrigin);
        }
        if let Some(mat) = self.world_to_grid {
            if !mat.is_finite() {
                return Err(VoxelizerError::NonFiniteTransform);
            }
        }
        Ok(())
    }
}

/// A dense binary occupancy field over a cubic [`Resolution`] grid, packed
/// X-major into 32-bit words (`bit = x + n·(y + n·z)`, LSB-first within a word).
///
/// This is the voxelizer's native, renderer-ready output: it implements
/// [`OccupancyField`], so [`SparseTree::build`] (and thus the renderer's
/// [`SchoolBBuffer`]) consumes it directly with no repack — the wrapped word
/// buffer is exactly what the GPU/CPU paths produce.
#[derive(Debug, Clone)]
pub struct VoxelOccupancy {
    resolution: Resolution,
    words: Vec<u32>,
}

impl VoxelOccupancy {
    /// Wraps a packed occupancy word buffer at `resolution`.
    ///
    /// `words` should hold at least `ceil(n³ / 32)` entries in the X-major packing
    /// described on [`VoxelOccupancy`]; surplus words are ignored. A buffer shorter
    /// than `ceil(n³/32)` reads as all-zero past its end (see
    /// [`OccupancyField::is_occupied`]). Use [`from_words_checked`] to reject an
    /// undersized buffer up front.
    ///
    /// [`from_words_checked`]: VoxelOccupancy::from_words_checked
    #[must_use]
    pub fn from_words(resolution: Resolution, words: Vec<u32>) -> Self {
        Self { resolution, words }
    }

    /// Wraps a packed occupancy word buffer, rejecting one shorter than
    /// `ceil(n³ / 32)`.
    ///
    /// Unlike [`from_words`], which silently treats a short buffer as all-zero past
    /// its end, this validates the length up front.
    ///
    /// # Errors
    /// Returns [`VoxelizerError::OccupancyBufferTooSmall`] when `words` holds fewer
    /// than `ceil(n³ / 32)` entries.
    ///
    /// [`from_words`]: VoxelOccupancy::from_words
    pub fn from_words_checked(
        resolution: Resolution,
        words: Vec<u32>,
    ) -> Result<Self, VoxelizerError> {
        let n = u64::from(resolution.voxels_per_axis());
        let need = (n * n * n).div_ceil(32) as usize;
        if words.len() < need {
            return Err(VoxelizerError::OccupancyBufferTooSmall {
                got: words.len(),
                need,
            });
        }
        Ok(Self::from_words(resolution, words))
    }

    /// The grid resolution.
    #[must_use]
    pub fn resolution(&self) -> Resolution {
        self.resolution
    }

    /// The packed occupancy words (X-major, see type docs).
    #[must_use]
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// Consumes self, returning the packed occupancy words.
    #[must_use]
    pub fn into_words(self) -> Vec<u32> {
        self.words
    }

    /// Number of occupied voxels (population count over the word buffer).
    #[must_use]
    pub fn count_occupied(&self) -> u64 {
        self.words.iter().map(|w| u64::from(w.count_ones())).sum()
    }

    /// Linear bit index `x + n·(y + n·z)` for an in-bounds coordinate.
    fn linear_index(&self, c: VoxelCoord) -> Option<u64> {
        if !c.in_bounds(self.resolution) {
            return None;
        }
        let n = u64::from(self.resolution.voxels_per_axis());
        Some(u64::from(c.x) + n * (u64::from(c.y) + n * u64::from(c.z)))
    }

    /// Builds the renderer's sparse tree from this occupancy.
    #[must_use]
    pub fn to_sparse_tree(&self) -> SparseTree {
        SparseTree::build(self)
    }

    /// Builds the renderer's flat School-B buffer from this occupancy
    /// (`SparseTree::build` → `SchoolBBuffer::from_sparse`).
    #[must_use]
    pub fn to_school_b(&self) -> SchoolBBuffer {
        SchoolBBuffer::from_sparse(&self.to_sparse_tree())
    }
}

impl OccupancyField for VoxelOccupancy {
    fn resolution(&self) -> Resolution {
        self.resolution
    }

    fn is_occupied(&self, c: VoxelCoord) -> bool {
        match self.linear_index(c) {
            Some(i) => {
                let wi = (i >> 5) as usize;
                if wi >= self.words.len() {
                    return false;
                }
                (self.words[wi] >> (i & 31)) & 1 == 1
            }
            None => false,
        }
    }
}

/// A tiling of the grid into compute work-groups (one tile per dispatched
/// work-group region). Internal to the GPU dispatch; sized against device limits.
#[derive(Debug, Clone)]
pub struct TileSpec {
    /// Voxel extent of one tile per axis.
    pub tile_dims: [u32; 3],
    /// Tile count per axis covering the grid.
    pub num_tiles: [u32; 3],
}

impl TileSpec {
    /// Tiles `grid_dims` into `tile_dims`-sized tiles (rounded up).
    ///
    /// # Errors
    /// Returns [`VoxelizerError::ZeroTileDim`] if any tile dimension is zero.
    pub fn new(tile_dims: [u32; 3], grid_dims: [u32; 3]) -> Result<Self, VoxelizerError> {
        if tile_dims.contains(&0) {
            return Err(VoxelizerError::ZeroTileDim);
        }
        let num_tiles = [
            grid_dims[0].div_ceil(tile_dims[0]),
            grid_dims[1].div_ceil(tile_dims[1]),
            grid_dims[2].div_ceil(tile_dims[2]),
        ];
        Ok(Self {
            tile_dims,
            num_tiles,
        })
    }

    /// Total number of tiles. Widened to `u64` so absurd tile counts cannot
    /// overflow the product.
    #[must_use]
    pub fn num_tiles_total(&self) -> u64 {
        u64::from(self.num_tiles[0]) * u64::from(self.num_tiles[1]) * u64::from(self.num_tiles[2])
    }

    /// Validates the tile fits the device's per-workgroup invocation budget.
    ///
    /// # Errors
    /// Returns [`VoxelizerError::ZeroTileDim`] if a tile dimension is zero, or
    /// [`VoxelizerError::TileTooLarge`] if the tile's voxel count exceeds
    /// `max_invocations`.
    pub fn validate(&self, max_invocations: u32) -> Result<(), VoxelizerError> {
        if self.tile_dims.contains(&0) {
            return Err(VoxelizerError::ZeroTileDim);
        }
        let tile_voxels = u64::from(self.tile_dims[0])
            * u64::from(self.tile_dims[1])
            * u64::from(self.tile_dims[2]);
        if tile_voxels > u64::from(max_invocations) {
            return Err(VoxelizerError::TileTooLarge {
                got: u32::try_from(tile_voxels).unwrap_or(u32::MAX),
                limit: max_invocations,
            });
        }
        Ok(())
    }
}

/// One material's base-colour appearance: an optional sRGB texture (an index into
/// [`MeshAppearance::textures`]), the **linear** base-colour tint, and the
/// sampler's wrap modes. Indexed by `material_id`.
#[derive(Debug, Clone)]
pub struct MaterialDef {
    /// The source material name (glTF `material.name`, MTL `newmtl`), lower-cased
    /// matching is used to spot toon **outline** hulls in [`MeshInput::drop_outline_triangles`].
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

/// A triangle-soup mesh in world space, with optional per-triangle material ids,
/// UVs, and base-colour appearance (textures) for per-voxel baking.
///
/// Derives [`Default`] (empty mesh: no triangles, no ids/uvs/appearance) so test
/// and external constructors can use `MeshInput { triangles, ..Default::default() }`
/// and stay source-compatible when new optional fields are added.
#[derive(Debug, Clone, Default)]
pub struct MeshInput {
    /// World-space triangles (three vertices each).
    pub triangles: Vec<[Vec3; 3]>,
    /// Optional per-triangle material id (length must match `triangles`).
    pub material_ids: Option<Vec<u32>>,
    /// Optional per-triangle base-colour UVs (the set the base-colour texture
    /// references), aligned to `triangles`. `None` for formats/loaders without UVs.
    pub uvs: Option<Vec<[Vec2; 3]>>,
    /// Optional base-colour textures + per-material defs for the per-voxel bake.
    /// `None` when the mesh carries no textures.
    pub appearance: Option<MeshAppearance>,
}

impl MeshInput {
    /// Applies an affine transform to every vertex in place.
    ///
    /// Formats without a scene graph (OBJ, STL) deliver vertices in raw model
    /// space, so an exporter's up-axis convention bakes straight into the soup.
    /// This lets a caller re-orient (or scale/translate) the mesh before
    /// [`VoxelGrid::fit_mesh`] measures its bounding box. A pure rotation leaves
    /// the fit unchanged in size and only changes which way the model lies.
    pub fn transform(&mut self, m: Mat4) {
        for tri in &mut self.triangles {
            for v in tri {
                *v = m.transform_point3(*v);
            }
        }
    }

    /// Drops triangles whose material is a toon **outline** hull. These are
    /// inverted-hull silhouette tricks (e.g. `LittlestTokyo`'s "outline" material is
    /// 37% of its triangles); a *surface* voxelizer can't back-face-cull them, so it
    /// turns them into a solid black shell over the real geometry and the textured
    /// surface underneath never shows.
    ///
    /// The signature is deliberately conservative — a material qualifies only if its
    /// **name contains "outline"** (case-insensitive) AND it is untextured AND its
    /// `base_color_factor` is dark (luminance < `0.1`). Name is load-bearing: a
    /// near-black factor alone would also delete legitimate black walls, tyres, and
    /// dark-coloured surfaces, so it is never sufficient by itself.
    ///
    /// Keeps `triangles`/`material_ids`/`uvs` aligned. Returns the number dropped (0
    /// if there is no appearance/material map, or no material matches — the common
    /// case, so a no-op for ordinary meshes).
    pub fn drop_outline_triangles(&mut self) -> usize {
        let keep: Vec<bool> = {
            let (Some(app), Some(mat_ids)) = (self.appearance.as_ref(), self.material_ids.as_ref())
            else {
                return 0;
            };
            // An outline hull: named "outline", untextured, and dark. The name is
            // required — darkness alone would catch real black/dark-coloured surfaces.
            let is_outline: Vec<bool> = app
                .materials
                .iter()
                .map(|m| {
                    let named_outline = m
                        .name
                        .as_deref()
                        .is_some_and(|n| n.to_ascii_lowercase().contains("outline"));
                    let f = m.base_color_factor;
                    named_outline
                        && m.base_color_texture.is_none()
                        && 0.2126 * f[0] + 0.7152 * f[1] + 0.0722 * f[2] < 0.1
                })
                .collect();
            mat_ids
                .iter()
                .map(|&id| {
                    usize::try_from(id)
                        .ok()
                        .and_then(|i| is_outline.get(i))
                        .copied()
                        != Some(true)
                })
                .collect()
        };
        let dropped = keep.iter().filter(|&&k| !k).count();
        if dropped == 0 {
            return 0;
        }
        // `retain` visits in order, so a fresh iterator over `keep` stays aligned with
        // each parallel array.
        let mut it = keep.iter();
        self.triangles.retain(|_| *it.next().unwrap());
        if let Some(ids) = self.material_ids.as_mut() {
            let mut it = keep.iter();
            ids.retain(|_| *it.next().unwrap());
        }
        if let Some(uvs) = self.uvs.as_mut() {
            let mut it = keep.iter();
            uvs.retain(|_| *it.next().unwrap());
        }
        dropped
    }

    /// Validates the mesh: id/UV lengths match the triangle count, and every
    /// vertex *and UV* is finite.
    ///
    /// # Errors
    /// Returns [`VoxelizerError::MaterialIdLenMismatch`] or
    /// [`VoxelizerError::UvLenMismatch`] on a length mismatch, and
    /// [`VoxelizerError::NonFiniteVertex`] / [`VoxelizerError::NonFiniteUv`] on a
    /// non-finite position / texture coordinate (a non-finite UV would sample a
    /// garbage texel in the bake, so it is rejected here at the boundary).
    pub fn validate(&self) -> Result<(), VoxelizerError> {
        if let Some(ids) = &self.material_ids {
            if ids.len() != self.triangles.len() {
                return Err(VoxelizerError::MaterialIdLenMismatch {
                    ids: ids.len(),
                    tris: self.triangles.len(),
                });
            }
        }
        if let Some(uvs) = &self.uvs {
            if uvs.len() != self.triangles.len() {
                return Err(VoxelizerError::UvLenMismatch {
                    uvs: uvs.len(),
                    tris: self.triangles.len(),
                });
            }
            for tri_uv in uvs {
                for uv in tri_uv {
                    if !uv.is_finite() {
                        return Err(VoxelizerError::NonFiniteUv);
                    }
                }
            }
        }
        for tri in &self.triangles {
            for v in tri {
                if !v.is_finite() {
                    return Err(VoxelizerError::NonFiniteVertex);
                }
            }
        }
        Ok(())
    }
}

/// Voxelization options: surface-overlap epsilon and which material channels to
/// store.
#[derive(Debug, Clone)]
pub struct VoxelizeOpts {
    /// AABB padding (grid units) when rasterizing a triangle's voxel span.
    pub epsilon: f32,
    /// Store the per-voxel owning triangle index (`owner_id`).
    pub store_owner: bool,
    /// Store the per-voxel hashed color (`color_rgba`).
    pub store_color: bool,
}

impl Default for VoxelizeOpts {
    fn default() -> Self {
        Self {
            epsilon: 1e-4,
            store_owner: true,
            store_color: true,
        }
    }
}

impl VoxelizeOpts {
    /// Validates the options are internally consistent.
    ///
    /// # Errors
    /// Returns [`VoxelizerError::InvalidEpsilon`] when `epsilon` is non-finite or
    /// negative, or [`VoxelizerError::ColorRequiresOwner`] when `store_color` is
    /// set without `store_owner` (color is hashed from the owning triangle, so it
    /// has no source without owner storage).
    pub fn validate(&self) -> Result<(), VoxelizerError> {
        if !self.epsilon.is_finite() || self.epsilon < 0.0 {
            return Err(VoxelizerError::InvalidEpsilon(self.epsilon));
        }
        if self.store_color && !self.store_owner {
            return Err(VoxelizerError::ColorRequiresOwner);
        }
        Ok(())
    }
}

/// Counters describing one voxelization dispatch.
#[derive(Debug, Clone)]
pub struct DispatchStats {
    /// Triangles processed.
    pub triangles: u32,
    /// Tiles dispatched.
    pub tiles: u32,
    /// Total grid voxels (`n³`; widened to `u64` so large grids cannot overflow).
    pub voxels: u64,
    /// GPU dispatch time in milliseconds, when measured.
    pub gpu_time_ms: Option<f32>,
}

/// The result of a dense voxelization: native occupancy plus auxiliary material
/// channels.
#[derive(Debug, Clone)]
pub struct VoxelizationOutput {
    /// Binary occupancy as a renderer-ready [`OccupancyField`].
    pub occupancy: VoxelOccupancy,
    /// Per-voxel owning triangle index (`u32::MAX` = empty), when `store_owner`.
    pub owner_id: Option<Vec<u32>>,
    /// Per-voxel hashed RGBA color, when `store_color`.
    pub color_rgba: Option<Vec<u32>>,
    /// Dispatch counters.
    pub stats: DispatchStats,
}

/// A compacted voxel with global coordinates and resolved material.
///
/// Produced by the GPU compact pass. Each entry is one occupied voxel with its
/// global voxel-space position and material id. 16 bytes, `AoS` layout matching the
/// GPU output buffer.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct CompactVoxel {
    /// Global voxel X coordinate (can be negative).
    pub vx: i32,
    /// Global voxel Y coordinate (can be negative).
    pub vy: i32,
    /// Global voxel Z coordinate (can be negative).
    pub vz: i32,
    /// Resolved material id (`u16` carried as `u32`). `0xFFFF_FFFF` = unresolved.
    pub material: u32,
}

/// The result of a sparse (brick-based) voxelization: per-brick occupancy +
/// material, allocating storage only for occupied bricks.
#[derive(Debug, Clone)]
pub struct SparseVoxelizationOutput {
    /// Brick edge length in voxels.
    pub brick_dim: u32,
    /// Grid origin of each emitted brick.
    pub brick_origins: Vec<[u32; 3]>,
    /// Packed per-brick occupancy words.
    pub occupancy: Vec<u32>,
    /// Per-voxel owning triangle index, when stored.
    pub owner_id: Option<Vec<u32>>,
    /// Per-voxel hashed RGBA color, when stored.
    pub color_rgba: Option<Vec<u32>>,
    /// Debug flags from the sparse pass.
    pub debug_flags: [u32; 3],
    /// Debug: dispatched work-group count.
    pub debug_workgroups: u32,
    /// Debug: triangle-voxel tests performed.
    pub debug_tested: u32,
    /// Debug: occupancy hits recorded.
    pub debug_hits: u32,
    /// Dispatch counters.
    pub stats: DispatchStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::{OccupancyField, Resolution, VoxelCoord};

    fn res8() -> Resolution {
        Resolution::new(8).unwrap()
    }

    fn mat(name: Option<&str>, tex: Option<usize>, factor: [f32; 4]) -> MaterialDef {
        MaterialDef {
            name: name.map(str::to_owned),
            base_color_texture: tex,
            base_color_factor: factor,
            wrap_s: WrapMode::Repeat,
            wrap_t: WrapMode::Repeat,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
        }
    }

    #[test]
    fn drop_outline_triangles_removes_only_the_named_outline_hull() {
        let tri = [Vec3::ZERO, Vec3::X, Vec3::Y];
        // mat 0: textured surface. mat 1: the toon "outline" (untextured, near-black).
        // mat 2: a legit pure-black wall — SAME darkness as the outline but NOT named
        // "outline", so it must be KEPT (the false-positive guard the audit demanded).
        let mut mesh = MeshInput {
            triangles: vec![tri, tri, tri, tri],
            material_ids: Some(vec![0, 1, 2, 1]),
            uvs: Some(vec![[Vec2::ZERO; 3]; 4]),
            appearance: Some(MeshAppearance {
                textures: vec![],
                materials: vec![
                    mat(Some("paintmat"), Some(0), [1.0, 1.0, 1.0, 1.0]),
                    mat(Some("outline"), None, [0.014, 0.009, 0.006, 1.0]),
                    mat(Some("Wall_Black"), None, [0.0, 0.0, 0.0, 1.0]),
                ],
            }),
        };
        let dropped = mesh.drop_outline_triangles();
        assert_eq!(dropped, 2, "both outline (mat 1) triangles dropped");
        assert_eq!(mesh.triangles.len(), 2);
        // Survivors are the textured wall (0) and the black wall (2) — never mat 1.
        assert_eq!(mesh.material_ids.as_deref(), Some(&[0u32, 2][..]));
        assert_eq!(mesh.uvs.as_ref().unwrap().len(), 2, "uvs stay aligned");
    }

    #[test]
    fn drop_outline_keeps_dark_and_bright_non_outline_materials() {
        let tri = [Vec3::ZERO, Vec3::X, Vec3::Y];
        // None of these are a dark, untextured, "outline"-named hull, so NONE drop:
        // a bright material that happens to be named outline, a dark colored surface,
        // and an unnamed near-black material (the loose old heuristic would have
        // wrongly nuked the last two).
        let mut mesh = MeshInput {
            triangles: vec![tri, tri, tri],
            material_ids: Some(vec![0, 1, 2]),
            uvs: None,
            appearance: Some(MeshAppearance {
                textures: vec![],
                materials: vec![
                    mat(Some("outline_glow"), None, [1.0, 0.9, 0.2, 1.0]), // bright
                    mat(Some("Cloth_Red"), None, [0.172, 0.009, 0.009, 1.0]), // dark red
                    mat(None, None, [0.0, 0.0, 0.0, 1.0]),                 // unnamed black
                ],
            }),
        };
        assert_eq!(
            mesh.drop_outline_triangles(),
            0,
            "nothing qualifies as outline"
        );
        assert_eq!(mesh.triangles.len(), 3);
    }

    #[test]
    fn validate_rejects_non_finite_uv() {
        let tri = [Vec3::ZERO, Vec3::X, Vec3::Y];
        // Finite vertices but a NaN UV: previously slipped through validate and
        // sampled a garbage texel in the bake. Must now be rejected.
        let bad = MeshInput {
            triangles: vec![tri],
            material_ids: None,
            uvs: Some(vec![[Vec2::new(f32::NAN, 0.0), Vec2::ZERO, Vec2::ZERO]]),
            appearance: None,
        };
        assert_eq!(bad.validate(), Err(VoxelizerError::NonFiniteUv));

        // An Inf UV is likewise rejected.
        let inf = MeshInput {
            triangles: vec![tri],
            material_ids: None,
            uvs: Some(vec![[
                Vec2::ZERO,
                Vec2::new(0.0, f32::INFINITY),
                Vec2::ZERO,
            ]]),
            appearance: None,
        };
        assert_eq!(inf.validate(), Err(VoxelizerError::NonFiniteUv));

        // The same triangle with finite UVs validates.
        let good = MeshInput {
            triangles: vec![tri],
            material_ids: None,
            uvs: Some(vec![[Vec2::ZERO; 3]]),
            appearance: None,
        };
        assert!(good.validate().is_ok());
    }

    #[test]
    fn drop_outline_triangles_is_a_noop_without_appearance() {
        let tri = [Vec3::ZERO, Vec3::X, Vec3::Y];
        let mut mesh = MeshInput {
            triangles: vec![tri],
            material_ids: Some(vec![0]),
            uvs: None,
            appearance: None,
        };
        assert_eq!(mesh.drop_outline_triangles(), 0);
        assert_eq!(mesh.triangles.len(), 1);
    }

    #[test]
    fn is_occupied_on_undersized_buffer_returns_false() {
        // res 8 needs 16 words; give it just one. Reads past the end must be
        // `false`, never panic (the `OccupancyField` out-of-bounds contract).
        let occ = VoxelOccupancy::from_words(res8(), vec![u32::MAX]);
        // A coordinate whose word index is past the (length-1) buffer.
        assert!(!occ.is_occupied(VoxelCoord::new(7, 7, 7)));
        // The first word is fully set, so an in-range coordinate still reads true.
        assert!(occ.is_occupied(VoxelCoord::new(0, 0, 0)));
    }

    #[test]
    fn from_words_checked_rejects_short_accepts_exact_and_surplus() {
        // ceil(8^3 / 32) = 16 words. (VoxelOccupancy isn't PartialEq, so we match
        // the Err arm rather than assert_eq! on the whole Result.)
        assert_eq!(
            VoxelOccupancy::from_words_checked(res8(), vec![0u32; 15]).unwrap_err(),
            VoxelizerError::OccupancyBufferTooSmall { got: 15, need: 16 }
        );
        assert!(VoxelOccupancy::from_words_checked(res8(), vec![0u32; 16]).is_ok());
        assert!(VoxelOccupancy::from_words_checked(res8(), vec![0u32; 64]).is_ok());
    }

    #[test]
    fn packing_round_trip_at_top_corner_has_no_aliasing() {
        // Linear index of (7,7,7) at n=8 is 511 → word 15, bit 31.
        let mut words = vec![0u32; 16];
        words[15] = 1 << 31;
        let occ = VoxelOccupancy::from_words(res8(), words);
        assert!(occ.is_occupied(VoxelCoord::new(7, 7, 7)));
        assert_eq!(occ.count_occupied(), 1, "exactly one bit set");
        // No other corner aliases the top corner's bit.
        for &c in &[
            VoxelCoord::new(0, 0, 0),
            VoxelCoord::new(7, 0, 0),
            VoxelCoord::new(0, 7, 0),
            VoxelCoord::new(0, 0, 7),
            VoxelCoord::new(6, 7, 7),
        ] {
            assert!(!occ.is_occupied(c), "voxel {c:?} must not alias (7,7,7)");
        }
    }

    #[test]
    fn to_sparse_tree_on_undersized_buffer_does_not_panic() {
        // Empty buffer at res 8: reads as all-zero, so the tree is empty and the
        // build must not panic on the missing words.
        let occ = VoxelOccupancy::from_words(res8(), Vec::new());
        let tree = occ.to_sparse_tree();
        assert_eq!(tree.leaf_count(), 0, "an all-empty field has no leaves");
    }

    #[test]
    fn tile_spec_validate_rejects_huge_dims_without_panic() {
        // 1626^3 ≈ 4.3e9 overflows a u32 product (the bug); with the u64 widening
        // it reports TileTooLarge instead of panicking/wrapping.
        let grid = VoxelGrid::new(res8(), Vec3::ZERO, 1.0);
        let tiles = TileSpec {
            tile_dims: [1626, 1626, 1626],
            num_tiles: grid.dims(),
        };
        let err = tiles.validate(256).unwrap_err();
        assert!(matches!(err, VoxelizerError::TileTooLarge { .. }));
    }

    #[test]
    fn num_tiles_total_does_not_overflow_at_extremes() {
        let small = TileSpec {
            tile_dims: [1, 1, 1],
            num_tiles: [1, 1, 1],
        };
        assert_eq!(small.num_tiles_total(), 1);
        let big = TileSpec {
            tile_dims: [1, 1, 1],
            num_tiles: [2048, 2048, 2048],
        };
        // 2048^3 = 8_589_934_592 — exceeds u32, fits u64.
        assert_eq!(big.num_tiles_total(), 8_589_934_592);
    }

    #[test]
    fn validate_rejects_subnormal_voxel_size_but_accepts_min_positive() {
        // f32::from_bits(1) is the smallest positive subnormal; its reciprocal is
        // +inf, so the derived world→grid matrix would be non-finite.
        let bad = VoxelGrid::new(res8(), Vec3::ZERO, f32::from_bits(1));
        assert_eq!(
            bad.validate(),
            Err(VoxelizerError::VoxelSizeTooSmall(f32::from_bits(1)))
        );
        // The smallest *normal* float has a finite reciprocal → valid.
        let ok = VoxelGrid::new(res8(), Vec3::ZERO, f32::MIN_POSITIVE);
        assert!(ok.validate().is_ok());
        assert!((1.0_f32 / f32::MIN_POSITIVE).is_finite());
    }

    #[test]
    fn transform_rotates_vertices_in_place() {
        // A +90° rotation about X sends +Y → +Z (right-handed): a vertex on the
        // Y axis lands on the Z axis. Tolerant compare — the rotation is f32.
        let mut mesh = MeshInput {
            triangles: vec![[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ]],
            material_ids: None,
            uvs: None,
            appearance: None,
        };
        mesh.transform(Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2));
        let rotated = mesh.triangles[0][2];
        assert!(
            (rotated - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-6,
            "+90° about X must map +Y → +Z, got {rotated:?}"
        );
        // The other two vertices (origin, +X) are on/parallel to the axis.
        assert!((mesh.triangles[0][0]).length() < 1e-6, "origin is fixed");
        assert!(
            (mesh.triangles[0][1] - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-6,
            "+X is fixed under an X rotation"
        );
    }

    #[test]
    fn fit_mesh_empty_is_unit_grid_at_origin() {
        let mesh = MeshInput {
            triangles: Vec::new(),
            material_ids: None,
            uvs: None,
            appearance: None,
        };
        for padding in [0.0_f32, 4.0] {
            let grid = VoxelGrid::fit_mesh(res8(), &mesh, padding);
            // Exact bit-compare: fit_mesh returns the literal 1.0 for empty meshes.
            assert_eq!(
                grid.voxel_size.to_bits(),
                1.0_f32.to_bits(),
                "empty mesh → unit voxel size"
            );
            assert_eq!(grid.origin_world, Vec3::ZERO, "empty mesh → origin");
            assert!(grid.validate().is_ok());
        }
    }

    #[test]
    fn fit_mesh_large_padding_stays_finite_and_valid() {
        // padding >= n/2 would drive `usable` non-positive; the `.max(1.0)` clamp
        // keeps voxel_size finite/positive and validate-Ok.
        let mesh = MeshInput {
            triangles: vec![[
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(4.0, 0.0, 0.0),
                Vec3::new(0.0, 4.0, 0.0),
            ]],
            material_ids: None,
            uvs: None,
            appearance: None,
        };
        for padding in [4.0_f32, 8.0, 100.0] {
            let grid = VoxelGrid::fit_mesh(res8(), &mesh, padding);
            assert!(
                grid.voxel_size.is_finite() && grid.voxel_size > 0.0,
                "voxel_size finite/positive at padding {padding}"
            );
            assert!(grid.validate().is_ok());
        }
    }

    #[test]
    fn voxelize_opts_validate_rejects_bad_epsilon_and_color_combo() {
        let mut opts = VoxelizeOpts::default();
        assert!(opts.validate().is_ok(), "defaults are valid");

        opts.epsilon = -1.0;
        assert_eq!(opts.validate(), Err(VoxelizerError::InvalidEpsilon(-1.0)));

        opts.epsilon = f32::NAN;
        assert!(matches!(
            opts.validate(),
            Err(VoxelizerError::InvalidEpsilon(_))
        ));

        let bad_combo = VoxelizeOpts {
            epsilon: 1e-4,
            store_owner: false,
            store_color: true,
        };
        assert_eq!(
            bad_combo.validate(),
            Err(VoxelizerError::ColorRequiresOwner)
        );
    }
}
