//! glTF / GLB input adapter: load a glTF document into the crate's world-space
//! triangle-soup [`MeshInput`].
//!
//! This is the first of several planned input adapters (OBJ / STL will follow),
//! so the public surface is intentionally narrow and source-agnostic: every
//! adapter funnels into the same [`MeshInput`] the voxelizer already consumes.
//! The whole module is gated behind the `gltf` cargo feature so the heavy
//! `gltf` dependency is droppable via `--no-default-features`.
//!
//! # What it reads
//! - The **default scene** if the document declares one ([`gltf::Document::default_scene`]),
//!   else the first scene, else *all* scenes when none is marked default.
//! - Each scene's node tree, recursively, accumulating the full **world
//!   transform** (`parent · node.transform()`) as a [`glam::Mat4`].
//! - For every node carrying a mesh, every primitive whose mode is
//!   [`gltf::mesh::Mode::Triangles`]: its `POSITION` attribute and (optional)
//!   index buffer.
//!
//! # What it skips (without erroring)
//! A scene can legitimately mix geometry kinds, so the loader is permissive:
//! - Non-`Triangles` primitives (points / lines / strips / fans) are skipped.
//! - Primitives with no `POSITION` accessor are skipped.
//!
//! Only a structural parse failure (malformed bytes, an accessor the `gltf`
//! crate cannot resolve) or a failed [`MeshInput::validate`] surfaces as a
//! [`VoxelizerError::MeshLoad`].
//!
//! # Material ids
//! Each emitted triangle records its primitive's material index
//! (`primitive.material().index()`), or [`u32::MAX`] for the glTF default
//! material. The resulting `material_ids` length always equals `triangles`, so
//! the returned [`MeshInput`] passes [`MeshInput::validate`].

use glam::{Mat4, Vec3};

use crate::core::MeshInput;
use crate::error::VoxelizerError;

/// Load a glTF or GLB document from an in-memory byte slice into a world-space
/// [`MeshInput`].
///
/// Accepts either form `gltf::import_slice` understands: a `.glb` binary
/// container, or a `.gltf` JSON document **whose buffers are self-contained**
/// (embedded as data-URIs or as the GLB `BIN` chunk). A plain `.gltf` that
/// references external `.bin` files by relative URI cannot be resolved from a
/// bare slice — use [`load_gltf_path`] for that, or embed the buffers.
///
/// See the [module docs](self) for scene selection, transform accumulation,
/// triangulation, and the skip rules.
///
/// # Errors
/// Returns [`VoxelizerError::MeshLoad`] if the bytes fail to parse as glTF/GLB,
/// or if the assembled mesh fails [`MeshInput::validate`].
pub fn load_gltf_slice(bytes: &[u8]) -> Result<MeshInput, VoxelizerError> {
    let (document, buffers, _images) =
        gltf::import_slice(bytes).map_err(|e| VoxelizerError::MeshLoad(e.to_string()))?;

    let mut triangles: Vec<[Vec3; 3]> = Vec::new();
    let mut material_ids: Vec<u32> = Vec::new();

    // Default scene, else the first scene, else every scene (covers documents
    // that declare scenes without marking one default).
    if let Some(scene) = document
        .default_scene()
        .or_else(|| document.scenes().next())
    {
        for node in scene.nodes() {
            walk_node(
                &node,
                Mat4::IDENTITY,
                &buffers,
                &mut triangles,
                &mut material_ids,
            );
        }
    } else {
        for scene in document.scenes() {
            for node in scene.nodes() {
                walk_node(
                    &node,
                    Mat4::IDENTITY,
                    &buffers,
                    &mut triangles,
                    &mut material_ids,
                );
            }
        }
    }

    let mesh = MeshInput {
        triangles,
        material_ids: Some(material_ids),
    };
    // Surfaces a length mismatch or a non-finite vertex (e.g. a degenerate
    // transform) as MeshLoad rather than panicking downstream.
    mesh.validate()
        .map_err(|e| VoxelizerError::MeshLoad(e.to_string()))?;
    Ok(mesh)
}

/// Load a glTF or GLB document from a filesystem path into a world-space
/// [`MeshInput`].
///
/// Reads the file then delegates to [`load_gltf_slice`]. Unlike a bare slice,
/// this still cannot follow external buffer URIs (it does not resolve relative
/// `.bin` paths); for multi-file glTF, prefer a GLB or a buffer-embedded glTF.
///
/// # Errors
/// Returns [`VoxelizerError::MeshLoad`] if the file cannot be read, fails to
/// parse, or the assembled mesh fails [`MeshInput::validate`].
pub fn load_gltf_path(path: impl AsRef<std::path::Path>) -> Result<MeshInput, VoxelizerError> {
    let bytes = std::fs::read(path).map_err(|e| VoxelizerError::MeshLoad(e.to_string()))?;
    load_gltf_slice(&bytes)
}

/// Recursively accumulate world-space triangles from `node` and its children.
///
/// `parent` is the accumulated world transform of `node`'s parent; this node's
/// world transform is `parent · local`. Each primitive's triangles are
/// transformed into world space and appended, one `material_id` per emitted
/// triangle.
fn walk_node(
    node: &gltf::Node,
    parent: Mat4,
    buffers: &[gltf::buffer::Data],
    triangles: &mut Vec<[Vec3; 3]>,
    material_ids: &mut Vec<u32>,
) {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent * local;

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            // Skip anything that isn't a triangle list — a scene may mix
            // points/lines/strips, which this surface voxelizer ignores.
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = primitive.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
            // No positions → nothing to voxelize; skip (not an error).
            let Some(positions) = reader.read_positions() else {
                continue;
            };
            let positions: Vec<Vec3> = positions
                .map(|p| world.transform_point3(Vec3::from(p)))
                .collect();

            let material_id = primitive.material().index().map_or(u32::MAX, |i| i as u32);

            if let Some(indices) = reader.read_indices() {
                let indices: Vec<u32> = indices.into_u32().collect();
                for tri in indices.chunks_exact(3) {
                    let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                    // A bad index would be a malformed document; guard rather
                    // than panic, dropping only the offending triangle.
                    if let (Some(&v0), Some(&v1), Some(&v2)) =
                        (positions.get(a), positions.get(b), positions.get(c))
                    {
                        triangles.push([v0, v1, v2]);
                        material_ids.push(material_id);
                    }
                }
            } else {
                // Non-indexed: sequential triples of positions form triangles.
                for tri in positions.chunks_exact(3) {
                    triangles.push([tri[0], tri[1], tri[2]]);
                    material_ids.push(material_id);
                }
            }
        }
    }

    for child in node.children() {
        walk_node(&child, world, buffers, triangles, material_ids);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Assemble a minimal in-memory GLB from a JSON chunk and a BIN chunk,
    /// following the binary-glTF container spec. No external assets, no base64.
    fn make_glb(json: &str, bin: &[u8]) -> Vec<u8> {
        // Chunk payloads must be 4-byte aligned: JSON pads with spaces (0x20),
        // BIN pads with zeros.
        let mut json_bytes = json.as_bytes().to_vec();
        while !json_bytes.len().is_multiple_of(4) {
            json_bytes.push(0x20);
        }
        let mut bin_bytes = bin.to_vec();
        while !bin_bytes.len().is_multiple_of(4) {
            bin_bytes.push(0x00);
        }

        let total_len = 12 // header
            + 8 + json_bytes.len() // JSON chunk header + payload
            + 8 + bin_bytes.len(); // BIN chunk header + payload

        let mut out = Vec::with_capacity(total_len);
        // Header: magic "glTF", version 2, total length.
        out.extend_from_slice(&0x4654_6C67_u32.to_le_bytes());
        out.extend_from_slice(&2_u32.to_le_bytes());
        out.extend_from_slice(&(total_len as u32).to_le_bytes());
        // JSON chunk: length, type "JSON", payload.
        out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes());
        out.extend_from_slice(&json_bytes);
        // BIN chunk: length, type "BIN\0", payload.
        out.extend_from_slice(&(bin_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x004E_4942_u32.to_le_bytes());
        out.extend_from_slice(&bin_bytes);
        out
    }

    /// The three vertices of the test triangle, as raw little-endian f32 bytes.
    fn triangle_bin() -> Vec<u8> {
        [
            0.0_f32, 0.0, 0.0, // v0
            1.0, 0.0, 0.0, // v1
            0.0, 1.0, 0.0, // v2
        ]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
    }

    /// Build the single-triangle GLB, injecting an optional node-level
    /// `translation` (e.g. `,"translation":[10.0,0.0,0.0]`).
    fn single_triangle_glb(node_extra: &str) -> Vec<u8> {
        let json = format!(
            r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0{node_extra}}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"mode":4}}]}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0.0,0.0,0.0],"max":[1.0,1.0,0.0]}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":36}}],"buffers":[{{"byteLength":36}}]}}"#
        );
        make_glb(&json, &triangle_bin())
    }

    /// Epsilon compare for a single transformed vertex.
    fn vec_approx(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-5
    }

    #[test]
    fn loads_single_triangle() {
        let glb = single_triangle_glb("");
        let mesh = load_gltf_slice(&glb).expect("single-triangle GLB must load");

        assert_eq!(mesh.triangles.len(), 1, "exactly one triangle");
        let [v0, v1, v2] = mesh.triangles[0];
        assert!(vec_approx(v0, Vec3::new(0.0, 0.0, 0.0)), "v0 = {v0:?}");
        assert!(vec_approx(v1, Vec3::new(1.0, 0.0, 0.0)), "v1 = {v1:?}");
        assert!(vec_approx(v2, Vec3::new(0.0, 1.0, 0.0)), "v2 = {v2:?}");
        // Default material → u32::MAX, one entry per triangle.
        assert_eq!(mesh.material_ids, Some(vec![u32::MAX]));
    }

    #[test]
    fn applies_node_world_transform() {
        // The node translates +10 in x; every loaded vertex must shift +10 x,
        // proving the world-transform accumulation runs.
        let glb = single_triangle_glb(r#","translation":[10.0,0.0,0.0]"#);
        let mesh = load_gltf_slice(&glb).expect("translated GLB must load");

        assert_eq!(mesh.triangles.len(), 1);
        let [v0, v1, v2] = mesh.triangles[0];
        assert!(vec_approx(v0, Vec3::new(10.0, 0.0, 0.0)), "v0 = {v0:?}");
        assert!(vec_approx(v1, Vec3::new(11.0, 0.0, 0.0)), "v1 = {v1:?}");
        assert!(vec_approx(v2, Vec3::new(10.0, 1.0, 0.0)), "v2 = {v2:?}");
    }

    #[test]
    fn rejects_garbage_bytes() {
        let err = load_gltf_slice(b"not a gltf").unwrap_err();
        assert!(
            matches!(err, VoxelizerError::MeshLoad(_)),
            "garbage bytes must yield MeshLoad, got {err:?}"
        );
    }
}
