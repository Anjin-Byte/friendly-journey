//! Resolving per-voxel materials from a voxelization's `owner_id` grid and the
//! mesh's per-triangle `material_id`s into the renderer's global colour table.
//!
//! The chain is `owner_id[voxel] → triangle → material_id → global_id → colour`.
//! Voxels with no owner (`owner_id == u32::MAX`) or whose triangle carries no
//! material (`material_id == u32::MAX`) stay **global-0** — the magenta MISSING
//! sentinel, which the renderer shades by position (docs/materials/02 §4, 05
//! hole 1). Distinct real material ids are compacted into dense global ids `1..`
//! so the per-leaf palette stays minimal.

use std::collections::HashMap;

use voxel_core::{MaterialTable, Resolution, SparseTree, VoxelCoord};

use crate::core::{CompactVoxel, MeshInput, VoxelizationOutput};
use crate::error::VoxelizerError;

/// A deterministic opaque RGBA8 colour for a material id, packed little-endian
/// (R in the low byte) for the renderer's `unpack4x8unorm`. Uses the same
/// Numerical-Recipes LCG spread as `reference_cpu::hash_color` does for owners.
fn material_color(id: u32) -> u32 {
    let mut x = id.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let r = (x & 0xff) as u8;
    x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let g = (x & 0xff) as u8;
    x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let b = (x & 0xff) as u8;
    u32::from_le_bytes([r, g, b, 255])
}

/// Resolves and assigns per-voxel materials to `tree` from `output`'s `owner_id`
/// grid and `mesh`'s per-triangle `material_id`s, returning the global colour
/// table to upload alongside the structure.
///
/// Returns the magenta-only table (no colouring) when the voxelization carried
/// no `owner_id` (`store_owner` was off) or the mesh had no `material_id`s (e.g.
/// an STL): those voxels render position-shaded as global-0, not magenta.
///
/// Occupancy is never touched, so this cannot change the tree's topology.
///
/// # Errors
/// [`VoxelizerError::TooManyMaterials`] if the mesh has more than 65535 distinct
/// materials (id 0 reserved magenta; real global ids are `1..=u16::MAX`).
pub fn apply_mesh_materials(
    tree: &mut SparseTree,
    output: &VoxelizationOutput,
    mesh: &MeshInput,
) -> Result<MaterialTable, VoxelizerError> {
    let (Some(owner_id), Some(material_ids)) =
        (output.owner_id.as_ref(), mesh.material_ids.as_ref())
    else {
        return Ok(MaterialTable::missing_only());
    };

    let (table, to_global) = build_global_table(material_ids)?;

    let n = u64::from(tree.resolution().voxels_per_axis());
    tree.fill_materials(|c: VoxelCoord| {
        // Linear index matches VoxelOccupancy::linear_index: x + n·(y + n·z).
        let lin = u64::from(c.x) + n * (u64::from(c.y) + n * u64::from(c.z));
        let Some(&owner) = usize::try_from(lin).ok().and_then(|i| owner_id.get(i)) else {
            return 0;
        };
        if owner == u32::MAX {
            return 0; // unresolved voxel → global-0 (magenta / position)
        }
        let mid = material_ids
            .get(owner as usize)
            .copied()
            .unwrap_or(u32::MAX);
        to_global.get(&mid).copied().unwrap_or(0) // u32::MAX (no material) → 0
    });

    Ok(table)
}

/// Builds the global colour table + the `material_id → dense global_id` map from a
/// mesh's per-triangle ids: distinct REAL ids (dropping the `u32::MAX` "no
/// material" sentinel) sorted → dense global `1..`, id `0` reserved magenta. Both
/// the dense ([`apply_mesh_materials`]) and sparse ([`material_table_for_sparse`])
/// paths go through this, so they emit byte-identical tables (the differential
/// oracle depends on it).
///
/// # Errors
/// [`VoxelizerError::TooManyMaterials`] if there are more than 65535 distinct
/// real materials (id 0 reserved magenta; real ids are `1..=u16::MAX`).
fn build_global_table(
    material_ids: &[u32],
) -> Result<(MaterialTable, HashMap<u32, u16>), VoxelizerError> {
    let mut distinct: Vec<u32> = material_ids
        .iter()
        .copied()
        .filter(|&m| m != u32::MAX)
        .collect();
    distinct.sort_unstable();
    distinct.dedup();

    let mut table = MaterialTable::missing_only();
    let mut to_global: HashMap<u32, u16> = HashMap::with_capacity(distinct.len());
    for mid in distinct {
        let gid = table
            .push(material_color(mid))
            .map_err(|_| VoxelizerError::TooManyMaterials)?;
        to_global.insert(mid, gid);
    }
    Ok((table, to_global))
}

/// Builds the global colour table **and** the per-triangle global-id table the
/// sparse compact shader (`compact_voxels.wgsl`) consumes: `packed[t>>1]` holds
/// `to_global[material_ids[t]]` (`u32::MAX`/unmapped → 0) as a `u16`, two per
/// `u32` word. Pass the returned `Vec<u32>` as the `material_table` of
/// [`GpuVoxelizer::compact_surface_sparse`](crate::GpuVoxelizer::compact_surface_sparse)
/// so each `CompactVoxel.material` comes back as the renderer global id (matching
/// the dense path bit-for-bit).
///
/// `mesh.material_ids == None` (e.g. STL) → `(missing_only(), [])`: the empty
/// table makes the shader leave every voxel at global-0 (magenta).
///
/// # Errors
/// [`VoxelizerError::TooManyMaterials`] (see `build_global_table`).
pub fn material_table_for_sparse(
    mesh: &MeshInput,
) -> Result<(MaterialTable, Vec<u32>), VoxelizerError> {
    let Some(material_ids) = mesh.material_ids.as_ref() else {
        return Ok((MaterialTable::missing_only(), Vec::new()));
    };
    let (table, to_global) = build_global_table(material_ids)?;
    let mut packed = vec![0u32; material_ids.len().div_ceil(2)];
    for (t, &mid) in material_ids.iter().enumerate() {
        let gid = if mid == u32::MAX {
            0
        } else {
            to_global.get(&mid).copied().unwrap_or(0)
        };
        packed[t >> 1] |= u32::from(gid) << ((t & 1) * 16);
    }
    Ok((table, packed))
}

/// Assembles a renderer [`SparseTree`] (occupancy **and** materials) from the
/// sparse compact pass's per-occupied-voxel output — the GPU-free half of the
/// `2048³` mesh path (`docs/materials/09-sparse-material-bridge.md`). Each
/// [`CompactVoxel`] carries an absolute `i32` global coord and a renderer global
/// material id in its low 16 bits (`0` = magenta). Absolute coords are re-binned
/// into fixed `8³` leaves (independent of the voxelizer's internal `brick_dim`);
/// any voxel outside `[0, n)` is dropped.
///
/// Returns `(tree, dropped)` where `dropped` is the out-of-range count — a
/// non-zero value means geometry fell outside the fitted grid and the caller
/// should surface it.
#[must_use]
pub fn tree_from_compact(resolution: Resolution, voxels: &[CompactVoxel]) -> (SparseTree, usize) {
    let n = i64::from(resolution.voxels_per_axis());
    let mut dropped = 0usize;
    // Stream `(coord, gid)` straight into `from_voxels` — no intermediate `pairs`
    // Vec, so the only large allocation is `from_voxels`'s per-brick map (the
    // input `voxels` slice is the caller's). `dropped` is updated as the iterator
    // drains, then read once the borrow ends.
    let tree = {
        let pairs = voxels.iter().filter_map(|v| {
            let (x, y, z) = (i64::from(v.vx), i64::from(v.vy), i64::from(v.vz));
            if x < 0 || y < 0 || z < 0 || x >= n || y >= n || z >= n {
                dropped += 1;
                return None;
            }
            let coord = VoxelCoord::new(
                u32::try_from(v.vx).expect("checked >= 0"),
                u32::try_from(v.vy).expect("checked >= 0"),
                u32::try_from(v.vz).expect("checked >= 0"),
            );
            // The renderer global id is the low 16 bits; any high bits are masked
            // off by design (a documented, tested contract — see
            // `tree_from_compact_masks_material_to_u16`), so this is total, not a
            // panic site.
            Some((
                coord,
                u16::try_from(v.material & 0xFFFF).expect("masked to 16 bits"),
            ))
        });
        SparseTree::from_voxels(resolution, pairs)
    };
    (tree, dropped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{DispatchStats, VoxelOccupancy};
    use glam::Vec3;
    use voxel_core::Resolution;

    fn stats() -> DispatchStats {
        DispatchStats {
            triangles: 0,
            tiles: 0,
            voxels: 512,
            gpu_time_ms: None,
        }
    }

    #[test]
    fn resolves_owner_to_material_and_unresolved_to_global_zero() {
        let r = Resolution::new(8).unwrap();
        // Occupy voxels (0..4, 0, 0) in brick (0,0,0): linear bits 0,1,2,3.
        let mut words = vec![0u32; 16];
        words[0] = 0b1111;
        let occupancy = VoxelOccupancy::from_words(r, words);

        // owner: voxel0→tri0, voxel1→tri1, voxel2→tri0, voxel3→tri2, rest empty.
        let mut owner = vec![u32::MAX; 8 * 8 * 8];
        owner[0] = 0;
        owner[1] = 1;
        owner[2] = 0;
        owner[3] = 2;
        let output = VoxelizationOutput {
            occupancy,
            owner_id: Some(owner),
            color_rgba: None,
            stats: stats(),
        };
        // tri0→material 10, tri1→material 20, tri2→u32::MAX (no material).
        let mesh = MeshInput {
            triangles: vec![[Vec3::ZERO; 3]; 3],
            material_ids: Some(vec![10, 20, u32::MAX]),
            uvs: None,
            appearance: None,
        };

        let mut tree = output.occupancy.to_sparse_tree();
        let table = apply_mesh_materials(&mut tree, &output, &mesh).unwrap();

        // Two distinct REAL materials → dense global ids 1, 2 (0 stays magenta).
        assert_eq!(table.words().len(), 3);
        assert_eq!(table.color(1), material_color(10));
        assert_eq!(table.color(2), material_color(20));

        // Each voxel takes its owning triangle's material; the u32::MAX-material
        // voxel stays global-0 (magenta / position).
        assert_eq!(tree.material_at(VoxelCoord::new(0, 0, 0)), 1); // mat 10
        assert_eq!(tree.material_at(VoxelCoord::new(1, 0, 0)), 2); // mat 20
        assert_eq!(tree.material_at(VoxelCoord::new(2, 0, 0)), 1); // mat 10
        assert_eq!(tree.material_at(VoxelCoord::new(3, 0, 0)), 0); // no material
    }

    #[test]
    fn no_owner_grid_yields_magenta_only_table() {
        let r = Resolution::new(8).unwrap();
        let mut words = vec![0u32; 16];
        words[0] = 0b1;
        let output = VoxelizationOutput {
            occupancy: VoxelOccupancy::from_words(r, words),
            owner_id: None, // store_owner was off
            color_rgba: None,
            stats: stats(),
        };
        let mesh = MeshInput {
            triangles: vec![[Vec3::ZERO; 3]],
            material_ids: Some(vec![10]),
            uvs: None,
            appearance: None,
        };
        let mut tree = output.occupancy.to_sparse_tree();
        let table = apply_mesh_materials(&mut tree, &output, &mesh).unwrap();
        assert_eq!(table.words().len(), 1, "no owner grid ⇒ magenta-only table");
        assert_eq!(tree.material_at(VoxelCoord::new(0, 0, 0)), 0);
    }

    #[test]
    fn tree_from_compact_drops_out_of_range_and_carries_materials() {
        let r = Resolution::new(32).unwrap(); // n = 32
        let voxels = vec![
            CompactVoxel {
                vx: 0,
                vy: 0,
                vz: 0,
                material: 3,
            },
            CompactVoxel {
                vx: 5,
                vy: 5,
                vz: 5,
                material: 7,
            },
            CompactVoxel {
                vx: -1,
                vy: 0,
                vz: 0,
                material: 9,
            }, // negative → drop (B4)
            CompactVoxel {
                vx: 32,
                vy: 0,
                vz: 0,
                material: 9,
            }, // >= n → drop
            CompactVoxel {
                vx: 0,
                vy: 40,
                vz: 0,
                material: 9,
            }, // >= n → drop
        ];
        let (tree, dropped) = tree_from_compact(r, &voxels);
        assert_eq!(dropped, 3, "negative + 2 over-range coords dropped");
        assert!(tree.is_occupied(VoxelCoord::new(0, 0, 0)));
        assert!(tree.is_occupied(VoxelCoord::new(5, 5, 5)));
        assert_eq!(tree.material_at(VoxelCoord::new(0, 0, 0)), 3);
        assert_eq!(tree.material_at(VoxelCoord::new(5, 5, 5)), 7);
    }

    #[test]
    fn tree_from_compact_masks_material_to_u16() {
        let r = Resolution::new(8).unwrap();
        // High bits set → only the low 16 bits become the global id.
        let voxels = vec![CompactVoxel {
            vx: 1,
            vy: 2,
            vz: 3,
            material: 0xFFFF_0005,
        }];
        let (tree, dropped) = tree_from_compact(r, &voxels);
        assert_eq!(dropped, 0);
        assert_eq!(tree.material_at(VoxelCoord::new(1, 2, 3)), 5);
    }

    #[test]
    fn material_table_for_sparse_builds_table_and_per_triangle_globals() {
        // 4 triangles: materials [10, MAX, 20, 10]. Distinct real {10,20} → 1,2.
        let mesh = MeshInput {
            triangles: vec![[Vec3::ZERO; 3]; 4],
            material_ids: Some(vec![10, u32::MAX, 20, 10]),
            uvs: None,
            appearance: None,
        };
        let (table, packed) = material_table_for_sparse(&mesh).unwrap();
        assert_eq!(table.words().len(), 3, "magenta + 2 real materials");
        // packed[t>>1] holds gid << ((t&1)*16): tri0=10→1 (low w0), tri1=MAX→0
        // (high w0), tri2=20→2 (low w1), tri3=10→1 (high w1).
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], 1);
        assert_eq!(packed[1], 2 | (1 << 16));
    }

    #[test]
    fn material_table_for_sparse_rejects_more_than_65535_distinct() {
        // Exactly 65535 distinct real materials fit (dense ids 1..=65535, id 0
        // reserved magenta → 65536 table entries). The 65536th overflows. Pins
        // the ceiling the reworded TooManyMaterials error now describes (C2).
        // (`material_table_for_sparse` reads only `material_ids`, so the empty
        // `triangles` is fine for this table-building unit.)
        let ok_ids: Vec<u32> = (0..65_535u32).collect();
        let mesh_ok = MeshInput {
            triangles: vec![],
            material_ids: Some(ok_ids),
            uvs: None,
            appearance: None,
        };
        let (table, _) = material_table_for_sparse(&mesh_ok).expect("65535 distinct must fit");
        assert_eq!(table.words().len(), 65_536, "magenta + 65535 reals");

        let over_ids: Vec<u32> = (0..65_536u32).collect();
        let mesh_over = MeshInput {
            triangles: vec![],
            material_ids: Some(over_ids),
            uvs: None,
            appearance: None,
        };
        assert!(
            matches!(
                material_table_for_sparse(&mesh_over),
                Err(VoxelizerError::TooManyMaterials)
            ),
            "65536 distinct real materials must be rejected"
        );
    }

    #[test]
    fn material_table_for_sparse_none_is_magenta_only() {
        let mesh = MeshInput {
            triangles: vec![[Vec3::ZERO; 3]],
            material_ids: None,
            uvs: None,
            appearance: None,
        };
        let (table, packed) = material_table_for_sparse(&mesh).unwrap();
        assert_eq!(table.words().len(), 1, "magenta only");
        assert!(packed.is_empty(), "no per-triangle table");
    }
}
