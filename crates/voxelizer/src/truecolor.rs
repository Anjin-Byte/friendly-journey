//! Per-voxel truecolor bake: the CPU pass that fills a [`SchoolBBuffer`]'s compact
//! `leaf_color` from a textured mesh (`docs/materials/11`, P2).
//!
//! For each occupied voxel it picks the **nearest-surface** colour owner (design
//! `docs/materials/11` **D2**): among the brick's candidate triangles
//! ([`build_brick_csr`]) **restricted to the voxel's own occupancy material**, the
//! one with the smallest closest-point distance (min `tri_index` tie-break), baked
//! via `bake_nearest_color`. The material restriction is load-bearing: it stops
//! the owner from jumping to a *different primitive that merely shares a packed
//! atlas texture* — the `LittlestTokyo` failure mode, where a wrong-region atlas fetch
//! reads a different object's pixels. D2 explicitly rejects the occupancy min-index
//! owner ("usually not the surface in the cell" under multi-coverage). The work is
//! done in **grid space**; the `world_to_grid` scale is uniform, so closest-point
//! distance argmin and the interpolated UV are both invariant vs world space.
//!
//! Because every voxel in a leaf shares one brick, the candidate list is rebuilt
//! only when the brick changes — [`SchoolBBuffer::assemble_leaf_color`] walks
//! voxels leaf-by-leaf, so the memo hits for all but the first voxel of each leaf.

use std::collections::HashMap;

use glam::{Mat4, Vec2, Vec3};
use voxel_core::{MISSING_MAGENTA, SchoolBBuffer, SparseTree};

use crate::bake::{AlphaMode, ColorCandidate, Texture, WrapMode, bake_nearest_owner};
use crate::core::{CompactVoxel, MeshAppearance, MeshInput, VoxelGrid};
use crate::csr::{BrickTriangleCsr, build_brick_csr};

/// Resolves a material id to its base-colour texture, wrap modes, and linear
/// factor. An out-of-range id (e.g. the glTF default `u32::MAX`) or a mesh with no
/// appearance falls back to a flat white tint.
fn resolve_material(
    appearance: Option<&MeshAppearance>,
    mat_id: u32,
) -> (Option<&Texture>, (WrapMode, WrapMode), [f32; 4]) {
    if let Some(app) = appearance {
        if let Some(def) = usize::try_from(mat_id)
            .ok()
            .and_then(|i| app.materials.get(i))
        {
            let tex = def.base_color_texture.and_then(|ti| app.textures.get(ti));
            return (tex, (def.wrap_s, def.wrap_t), def.base_color_factor);
        }
    }
    (
        None,
        (WrapMode::Repeat, WrapMode::Repeat),
        [1.0, 1.0, 1.0, 1.0],
    )
}

/// The `(alpha_mode, alpha_cutoff)` of material `mat_id`. An out-of-range id or a
/// mesh with no appearance is treated as `Opaque` (the conservative default — alpha
/// ignored, voxel kept). Used by the MASK-cutout cull and the bake's opaque-alpha
/// force; kept separate from [`resolve_material`] so the per-candidate colour path
/// stays a 3-tuple.
fn material_alpha(appearance: Option<&MeshAppearance>, mat_id: u32) -> (AlphaMode, f32) {
    appearance
        .and_then(|app| {
            usize::try_from(mat_id)
                .ok()
                .and_then(|i| app.materials.get(i))
        })
        .map_or((AlphaMode::Opaque, 0.5), |def| {
            (def.alpha_mode, def.alpha_cutoff)
        })
}

/// The global material id of triangle `ti` from the `packed` stream (2 tris/`u32`,
/// from `material_table_for_sparse`), or 0 when absent.
fn tri_global_mat(packed: Option<&[u32]>, ti: usize) -> u16 {
    packed.and_then(|p| p.get(ti >> 1)).map_or(0, |&w| {
        u16::try_from((w >> ((ti & 1) * 16)) & 0xFFFF).unwrap_or(0)
    })
}

/// Gathers the owner-candidate triangles for the 8³ `brick` into the reusable
/// `cand_buf` (GRID-space verts + appearance) and `cand_mat` (per-candidate global
/// material) — the per-brick step shared by [`bake_leaf_colors`] and the MASK cull.
/// Pure of occupancy: it depends only on the mesh, grid, and brick CSR.
#[allow(clippy::too_many_arguments)]
fn gather_brick_candidates<'a>(
    brick: [u32; 3],
    mesh: &MeshInput,
    to_grid: &Mat4,
    csr: &BrickTriangleCsr,
    appearance: Option<&'a MeshAppearance>,
    packed: Option<&[u32]>,
    cand_buf: &mut Vec<ColorCandidate<'a>>,
    cand_mat: &mut Vec<u16>,
) {
    cand_buf.clear();
    cand_mat.clear();
    // brick_origins is sorted by (z, y, x); search with the same key.
    let key = [brick[2], brick[1], brick[0]];
    let Ok(bi) = csr
        .brick_origins
        .binary_search_by(|o| [o[2], o[1], o[0]].cmp(&key))
    else {
        return;
    };
    let lo = csr.brick_offsets[bi] as usize;
    let hi = csr.brick_offsets[bi + 1] as usize;
    let mat_ids = mesh.material_ids.as_deref();
    let uvs = mesh.uvs.as_deref();
    for &ti in &csr.tri_indices[lo..hi] {
        let ti = ti as usize;
        // The CSR is built from this mesh, so `ti` is always in range; guard with
        // `.get` (matching the uv/material lookups below) so a malformed CSR index
        // would skip the candidate rather than panic. Defense-in-depth.
        let Some(&w) = mesh.triangles.get(ti) else {
            continue;
        };
        let verts = [
            to_grid.transform_point3(w[0]),
            to_grid.transform_point3(w[1]),
            to_grid.transform_point3(w[2]),
        ];
        let uv = uvs
            .and_then(|u| u.get(ti))
            .copied()
            .unwrap_or([Vec2::ZERO; 3]);
        let mat_id = mat_ids.and_then(|m| m.get(ti)).copied().unwrap_or(u32::MAX);
        let (texture, wrap, factor) = resolve_material(appearance, mat_id);
        cand_buf.push(ColorCandidate {
            tri_index: ti,
            verts,
            uvs: uv,
            texture,
            wrap,
            factor,
        });
        cand_mat.push(tri_global_mat(packed, ti));
    }
}

/// Picks the nearest-surface owner among the brick candidates, constrained to the
/// voxel's own global material when `vox_mat` is `Some` (the wrong-region atlas
/// guard). Returns `(owner triangle index, baked sRGB RGBA8)`, or `None` when the
/// brick has no candidate. `filtered` is a reusable same-material scratch buffer.
fn pick_owner<'a>(
    centre: Vec3,
    cand_buf: &[ColorCandidate<'a>],
    cand_mat: &[u16],
    vox_mat: Option<u16>,
    filtered: &mut Vec<ColorCandidate<'a>>,
) -> Option<(usize, [u8; 4])> {
    let pick: &[ColorCandidate] = if let Some(vm) = vox_mat {
        filtered.clear();
        for (cand, &m) in cand_buf.iter().zip(cand_mat.iter()) {
            if m == vm {
                filtered.push(*cand);
            }
        }
        // No same-material candidate (a binning edge) → fall back to all candidates.
        if filtered.is_empty() {
            cand_buf
        } else {
            filtered
        }
    } else {
        cand_buf
    };
    bake_nearest_owner(centre, pick).map(|(i, color)| (pick[i].tri_index, color))
}

/// Bakes `buffer`'s compact per-voxel truecolor from `mesh` (which must carry UVs +
/// appearance for a textured result; otherwise voxels resolve to flat factors).
/// `grid` places the voxels in world space and `epsilon` matches the voxelization
/// overlap padding used to bin triangles to bricks.
///
/// Each voxel's colour owner is the **nearest-surface** triangle (design D2): the
/// smallest closest-point distance among the brick candidates, with min `tri_index`
/// tie-break (`bake_nearest_color`). `packed` (the voxelizer's per-triangle
/// global-material stream, 2 tris/`u32`), when `Some`, restricts the search to the
/// voxel's **own occupancy material** — so the owner can never jump to a different
/// primitive that shares a packed atlas texture (a wrong-region fetch). Pass `None`
/// for single-material fixtures where no constraint is needed.
///
/// A voxel whose brick has no candidate triangle bakes to [`MISSING_MAGENTA`] (an
/// occupied voxel with no nearby surface is a binning/voxelization inconsistency,
/// surfaced as magenta rather than silently black).
///
/// # Panics
/// Panics (via [`SchoolBBuffer::assemble_leaf_color`]) if `tree` is topology-stale
/// relative to `buffer`.
pub fn bake_leaf_colors(
    buffer: &mut SchoolBBuffer,
    tree: &SparseTree,
    mesh: &MeshInput,
    grid: &VoxelGrid,
    epsilon: f32,
    packed: Option<&[u32]>,
) {
    let csr = build_brick_csr(mesh, grid, 8, epsilon);
    // Work in GRID space (voxel centres are integer+0.5). `world_to_grid` is a
    // uniform scale, so closest-point distance argmin and the interpolated UV are
    // identical to working in world space — only the absolute distances scale.
    let to_grid = grid.world_to_grid_matrix();
    let appearance = mesh.appearance.as_ref();
    let mat_ids = mesh.material_ids.as_deref();
    let magenta = MISSING_MAGENTA.to_le_bytes();

    // Memoized per-brick candidates (GRID-space verts) + global materials. The
    // assembler walks voxels leaf-by-leaf, so the memo hits within each brick.
    let mut cand_buf: Vec<ColorCandidate> = Vec::new();
    let mut cand_mat: Vec<u16> = Vec::new();
    let mut filtered: Vec<ColorCandidate> = Vec::new(); // same-material candidate scratch
    let mut last_brick: Option<[u32; 3]> = None;

    buffer.assemble_leaf_color(tree, |c| {
        let brick = [c.x & !7u32, c.y & !7u32, c.z & !7u32];
        if last_brick != Some(brick) {
            gather_brick_candidates(
                brick,
                mesh,
                &to_grid,
                &csr,
                appearance,
                packed,
                &mut cand_buf,
                &mut cand_mat,
            );
            last_brick = Some(brick);
        }
        let centre = Vec3::new(c.x as f32 + 0.5, c.y as f32 + 0.5, c.z as f32 + 0.5);
        // Constrain the owner to the voxel's own occupancy material (D2) when a
        // packed stream is present; empty brick → magenta.
        let vox_mat = packed.is_some().then(|| tree.material_at(c));
        match pick_owner(centre, &cand_buf, &cand_mat, vox_mat, &mut filtered) {
            Some((owner_ti, mut color)) => {
                // Force opaque alpha unless the owner material is BLEND. OPAQUE
                // ignores alpha, a kept MASK voxel is binary-opaque (above cutoff);
                // only BLEND keeps its real baked alpha (for Phase 2 compositing).
                let owner_mat = mat_ids
                    .and_then(|m| m.get(owner_ti))
                    .copied()
                    .unwrap_or(u32::MAX);
                if material_alpha(appearance, owner_mat).0 != AlphaMode::Blend {
                    color[3] = 255;
                }
                color
            }
            None => magenta,
        }
    });
}

/// Removes voxels that land on a glTF **MASK** material's transparent texels (the
/// alpha-cutout): a voxel whose nearest-surface owner is [`AlphaMode::Mask`] with
/// sampled alpha below the owner's `alpha_cutoff` is dropped, so it never becomes
/// occupied — killing the "transparent texel baked as an opaque colour" artifact.
/// **OPAQUE and BLEND voxels are always kept** (BLEND is composited later, not cut).
///
/// Must run on the compact voxel list **before** `tree_from_compact`: clearing an
/// occupancy bit after the tree is built would shift every higher-Morton voxel's
/// `occupied_rank` and corrupt the colour assembler. Owner resolution mirrors
/// [`bake_leaf_colors`] exactly (same brick CSR, same material constraint), so a
/// surviving voxel's bake is unchanged. A no-op (clones `voxels`) when the mesh has
/// no MASK material, or no appearance/`packed` to resolve owners with.
#[must_use]
pub fn cull_mask_cutout(
    voxels: &[CompactVoxel],
    mesh: &MeshInput,
    grid: &VoxelGrid,
    epsilon: f32,
    packed: Option<&[u32]>,
) -> Vec<CompactVoxel> {
    let appearance = mesh.appearance.as_ref();
    let mat_ids = mesh.material_ids.as_deref();
    // Map each dense global id → its (alpha_mode, cutoff) via the (global, glTF)
    // pairing the `packed` stream + per-triangle material ids give. The occupancy
    // material is 1:1 with the owner's glTF material (the D2 same-material owner), so
    // a voxel's global id alone tells us whether it can be a MASK cutout — letting us
    // skip owner resolution for the OPAQUE/BLEND majority.
    let (Some(pk), Some(mids), Some(app)) = (packed, mat_ids, appearance) else {
        return voxels.to_vec();
    };
    let mut gid_alpha: HashMap<u16, (AlphaMode, f32)> = HashMap::new();
    for ti in 0..mesh.triangles.len() {
        let global_id = tri_global_mat(Some(pk), ti);
        let gltf = mids.get(ti).copied().unwrap_or(u32::MAX);
        gid_alpha
            .entry(global_id)
            .or_insert_with(|| material_alpha(Some(app), gltf));
    }
    if !gid_alpha.values().any(|(m, _)| *m == AlphaMode::Mask) {
        return voxels.to_vec(); // no MASK material anywhere → nothing to cull
    }

    let csr = build_brick_csr(mesh, grid, 8, epsilon);
    let to_grid = grid.world_to_grid_matrix();
    let mut cand_buf: Vec<ColorCandidate> = Vec::new();
    let mut cand_mat: Vec<u16> = Vec::new();
    let mut filtered: Vec<ColorCandidate> = Vec::new();
    let mut last_brick: Option<[u32; 3]> = None;
    let mut out: Vec<CompactVoxel> = Vec::with_capacity(voxels.len());

    for &v in voxels {
        // Out-of-bounds coords are kept; `tree_from_compact` drops them.
        if v.vx < 0 || v.vy < 0 || v.vz < 0 {
            out.push(v);
            continue;
        }
        let global_id = u16::try_from(v.material & 0xFFFF).unwrap_or(0);
        let (mode, cutoff) = gid_alpha
            .get(&global_id)
            .copied()
            .unwrap_or((AlphaMode::Opaque, 0.5));
        if mode != AlphaMode::Mask {
            out.push(v); // OPAQUE / BLEND → keep without resolving the owner
            continue;
        }
        let coord = [v.vx as u32, v.vy as u32, v.vz as u32];
        let brick = [coord[0] & !7u32, coord[1] & !7u32, coord[2] & !7u32];
        if last_brick != Some(brick) {
            gather_brick_candidates(
                brick,
                mesh,
                &to_grid,
                &csr,
                appearance,
                packed,
                &mut cand_buf,
                &mut cand_mat,
            );
            last_brick = Some(brick);
        }
        let centre = Vec3::new(
            coord[0] as f32 + 0.5,
            coord[1] as f32 + 0.5,
            coord[2] as f32 + 0.5,
        );
        // Keep if the owner texel is at/above the MASK cutoff (or there's no owner —
        // a binning edge the bake would magenta, not our call to cull).
        let keep = match pick_owner(centre, &cand_buf, &cand_mat, Some(global_id), &mut filtered) {
            Some((_, color)) => f32::from(color[3]) >= cutoff * 255.0,
            None => true,
        };
        if keep {
            out.push(v);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::{WrapMode, expected_color};
    use crate::core::{MaterialDef, MeshAppearance};
    use voxel_core::{Resolution, VoxelCoord};

    /// A 2×2 checker (red / green / blue / white), as in the bake oracle tests.
    fn checker() -> Texture {
        Texture::new(
            2,
            2,
            vec![
                [255, 0, 0, 255],
                [0, 255, 0, 255],
                [0, 0, 255, 255],
                [255, 255, 255, 255],
            ],
        )
        .expect("2x2 checker is a valid texture")
    }

    #[test]
    fn bakes_textured_voxels_matching_the_oracle() {
        // World == grid (origin 0, voxel_size 1). One big triangle in the z=0 plane
        // covering the grid, with UVs mapping grid (x,y) → (x/8, y/8). Several
        // occupied voxels sit on the plane; each must bake to exactly what
        // expected_color computes directly (proving brick lookup + material
        // resolution + rank placement all line up).
        let r = Resolution::new(8).unwrap();
        let grid = VoxelGrid::new(r, Vec3::ZERO, 1.0);
        let tri = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(16.0, 0.0, 0.0),
            Vec3::new(0.0, 16.0, 0.0),
        ];
        let uv = [
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(0.0, 2.0),
        ];
        let mesh = MeshInput {
            triangles: vec![tri],
            material_ids: Some(vec![0]),
            uvs: Some(vec![uv]),
            appearance: Some(MeshAppearance {
                textures: vec![checker()],
                materials: vec![MaterialDef {
                    name: None,
                    base_color_texture: Some(0),
                    base_color_factor: [1.0, 1.0, 1.0, 1.0],
                    wrap_s: WrapMode::ClampToEdge,
                    wrap_t: WrapMode::ClampToEdge,
                    alpha_mode: AlphaMode::Opaque,
                    alpha_cutoff: 0.5,
                }],
            }),
        };

        // Occupied voxels on the z=0 layer, spread across the texture.
        let voxels = [
            VoxelCoord::new(1, 1, 0),
            VoxelCoord::new(6, 1, 0),
            VoxelCoord::new(1, 6, 0),
            VoxelCoord::new(6, 6, 0),
            VoxelCoord::new(3, 4, 0),
        ];
        let tree = SparseTree::from_voxels(r, voxels.iter().map(|&c| (c, 0u16)));
        let mut buffer = SchoolBBuffer::from_sparse(&tree);

        bake_leaf_colors(&mut buffer, &tree, &mesh, &grid, 0.0, None);
        assert!(buffer.has_leaf_color());

        // For every occupied voxel, the stored colour must equal the oracle.
        for &c in &voxels {
            let slot = tree.leaf_slot_of(c).expect("occupied voxel has a leaf") as usize;
            let base = buffer.leaf_color_base_words()[slot];
            // rank = occupied voxels with smaller morton in this leaf.
            let leaf = buffer.leaves()[slot];
            let target_m = voxel_core::morton::encode_brick(c.x & 7, c.y & 7, c.z & 7);
            let mut rank = 0u32;
            for m in 0..target_m {
                let l = voxel_core::morton::decode(u64::from(m));
                if leaf.get_local(l.x, l.y, l.z) {
                    rank += 1;
                }
            }
            let got = buffer.leaf_color_words()[(base + rank) as usize];

            let centre = Vec3::new(c.x as f32 + 0.5, c.y as f32 + 0.5, c.z as f32 + 0.5);
            let want = expected_color(
                centre,
                tri,
                uv,
                Some(&checker()),
                (WrapMode::ClampToEdge, WrapMode::ClampToEdge),
                [1.0, 1.0, 1.0, 1.0],
            );
            assert_eq!(
                got,
                u32::from_le_bytes(want),
                "voxel {c:?} colour mismatch (slot {slot}, rank {rank})"
            );
        }
    }

    #[test]
    fn voxel_with_no_candidate_triangle_bakes_magenta() {
        // A voxel far from any triangle (empty brick) must bake to MISSING_MAGENTA.
        let r = Resolution::new(8).unwrap();
        let grid = VoxelGrid::new(r, Vec3::ZERO, 1.0);
        // Triangle only in brick (0,0,0); the occupied voxel is in a different brick
        // is impossible at res 8 (one brick). Instead: a tiny triangle in one corner
        // and an occupied voxel whose brick has it binned, but place the triangle far
        // so the brick has NO candidate — use a mesh whose triangle's AABB misses.
        let tri = [
            Vec3::new(100.0, 100.0, 100.0),
            Vec3::new(101.0, 100.0, 100.0),
            Vec3::new(100.0, 101.0, 100.0),
        ];
        let mesh = MeshInput {
            triangles: vec![tri],
            material_ids: Some(vec![0]),
            uvs: Some(vec![[Vec2::ZERO; 3]]),
            appearance: None,
        };
        let voxels = [VoxelCoord::new(2, 2, 2)];
        let tree = SparseTree::from_voxels(r, voxels.iter().map(|&c| (c, 0u16)));
        let mut buffer = SchoolBBuffer::from_sparse(&tree);
        bake_leaf_colors(&mut buffer, &tree, &mesh, &grid, 0.0, None);
        assert_eq!(
            buffer.leaf_color_words()[0],
            MISSING_MAGENTA,
            "no-candidate voxel must be magenta"
        );
    }

    #[test]
    fn cull_drops_transparent_mask_keeps_opaque_and_blend() {
        let r = Resolution::new(8).unwrap();
        let grid = VoxelGrid::new(r, Vec3::ZERO, 1.0);
        // One big triangle in the z=0 plane covering the grid; a few occupied voxels
        // sit on it, all carrying global material id 1 (the single real material).
        let tri = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(16.0, 0.0, 0.0),
            Vec3::new(0.0, 16.0, 0.0),
        ];
        let uv = [
            Vec2::new(0.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(0.0, 2.0),
        ];
        let voxels: Vec<CompactVoxel> = [(1, 1), (3, 4), (6, 1)]
            .iter()
            .map(|&(x, y)| CompactVoxel {
                vx: x,
                vy: y,
                vz: 0,
                material: 1,
            })
            .collect();

        // Build a single-material textured mesh whose one texture is uniformly `alpha`.
        let make = |alpha_mode: AlphaMode, alpha: u8| {
            let tex = Texture::new(2, 2, vec![[200, 200, 200, alpha]; 4])
                .expect("2x2 uniform texture is valid");
            let mesh = MeshInput {
                triangles: vec![tri],
                material_ids: Some(vec![0]),
                uvs: Some(vec![uv]),
                appearance: Some(MeshAppearance {
                    textures: vec![tex],
                    materials: vec![MaterialDef {
                        name: None,
                        base_color_texture: Some(0),
                        base_color_factor: [1.0, 1.0, 1.0, 1.0],
                        wrap_s: WrapMode::Repeat,
                        wrap_t: WrapMode::Repeat,
                        alpha_mode,
                        alpha_cutoff: 0.5,
                    }],
                }),
            };
            let (_, packed) = crate::materials::material_table_for_sparse(&mesh).unwrap();
            (mesh, packed)
        };
        let cull = |mesh: &MeshInput, packed: &[u32]| {
            cull_mask_cutout(&voxels, mesh, &grid, 0.0, Some(packed))
        };

        // MASK + fully transparent → every voxel is cut.
        let (m, p) = make(AlphaMode::Mask, 0);
        assert!(cull(&m, &p).is_empty(), "transparent MASK voxels are cut");
        // MASK + opaque → kept (texel alpha >= cutoff).
        let (m, p) = make(AlphaMode::Mask, 255);
        assert_eq!(cull(&m, &p).len(), voxels.len(), "opaque MASK kept");
        // BLEND + fully transparent → kept (composited later, never cut).
        let (m, p) = make(AlphaMode::Blend, 0);
        assert_eq!(cull(&m, &p).len(), voxels.len(), "transparent BLEND kept");
        // OPAQUE → alpha ignored, kept.
        let (m, p) = make(AlphaMode::Opaque, 0);
        assert_eq!(cull(&m, &p).len(), voxels.len(), "OPAQUE kept");
    }

    #[test]
    fn cull_is_a_noop_without_appearance_or_packed() {
        let r = Resolution::new(8).unwrap();
        let grid = VoxelGrid::new(r, Vec3::ZERO, 1.0);
        let mesh = MeshInput {
            triangles: vec![[Vec3::ZERO, Vec3::X, Vec3::Y]],
            material_ids: Some(vec![0]),
            uvs: None,
            appearance: None,
        };
        let voxels = vec![CompactVoxel {
            vx: 1,
            vy: 1,
            vz: 0,
            material: 1,
        }];
        assert_eq!(
            cull_mask_cutout(&voxels, &mesh, &grid, 0.0, None).len(),
            1,
            "no appearance/packed → nothing cut"
        );
    }
}
