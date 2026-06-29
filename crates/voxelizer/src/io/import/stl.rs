//! STL input adapter: load an STL document into the crate's world-space
//! triangle-soup [`MeshInput`].
//!
//! A sibling of the [glTF adapter](super::gltf) and [OBJ adapter](super::obj):
//! every adapter funnels into the same [`MeshInput`] the voxelizer consumes. The
//! whole module is gated behind the `stl` cargo feature so the `stl_io`
//! dependency is droppable.
//!
//! # How STL differs from glTF / OBJ
//! STL is the simplest of the three formats: it has **no scene graph, no node
//! transforms, and no materials**. Its vertices are already in model space, so
//! there is NO transform accumulation (like OBJ, unlike the glTF walk).
//!
//! # Triangulation
//! STL facets are *already* triangles — `stl_io` returns an `IndexedMesh` with a
//! shared `vertices` list and a `faces` list of `IndexedTriangle`, each holding
//! three indices into `vertices`. This adapter therefore never triangulates by
//! hand; it emits one triangle per face.
//!
//! # Material ids
//! STL has no material concept whatsoever, so the returned [`MeshInput`] sets
//! `material_ids = None` — honest and valid. No ids are fabricated. (`None`
//! always passes [`MeshInput::validate`], which only length-checks ids when they
//! are present.)
//!
//! # Errors
//! Any `stl_io` parse failure, file IO error, or a failed
//! [`MeshInput::validate`] is surfaced as [`VoxelizerError::MeshLoad`] (the
//! generic loader error) via `.to_string()`. Out-of-range face indices in an
//! otherwise-parseable document drop only the offending triangle rather than
//! erroring or panicking.

use glam::Vec3;

use crate::core::MeshInput;
use crate::error::VoxelizerError;

/// Assemble a [`MeshInput`] from a parsed `stl_io` mesh.
///
/// STL vertices are already in model space, so positions are taken verbatim (no
/// transform). One triangle is emitted per face; out-of-range indices drop only
/// the offending triangle. STL carries no materials, so `material_ids` is
/// [`None`].
fn indexed_mesh_to_input(mesh: &stl_io::IndexedMesh) -> Result<MeshInput, VoxelizerError> {
    let mut triangles: Vec<[Vec3; 3]> = Vec::with_capacity(mesh.faces.len());

    for face in &mesh.faces {
        // `face.vertices` is `[usize; 3]` indexing into the shared vertex list.
        let [a, b, c] = face.vertices;
        // A bad index would be a malformed document; guard rather than panic,
        // dropping only the offending triangle.
        if let (Some(v0), Some(v1), Some(v2)) = (
            mesh.vertices.get(a),
            mesh.vertices.get(b),
            mesh.vertices.get(c),
        ) {
            // `stl_io::Vertex` is `Vector<f32>([f32; 3])`, indexable as `v[0..3]`.
            triangles.push([
                Vec3::new(v0[0], v0[1], v0[2]),
                Vec3::new(v1[0], v1[1], v1[2]),
                Vec3::new(v2[0], v2[1], v2[2]),
            ]);
        }
    }

    let mesh = MeshInput {
        // STL has no material/UV concept; `None` is honest and valid.
        triangles,
        material_ids: None,
        uvs: None,
        appearance: None,
    };
    // Surfaces a non-finite vertex as MeshLoad rather than panicking downstream.
    mesh.validate()
        .map_err(|e| VoxelizerError::MeshLoad(e.to_string()))?;
    Ok(mesh)
}

/// Load an STL document from an in-memory byte slice into a world-space
/// [`MeshInput`].
///
/// `stl_io::read_stl` requires a `Read + Seek` source and auto-detects ASCII vs
/// binary STL; a [`std::io::Cursor`] over the slice satisfies both traits.
///
/// See the [module docs](self) for the no-transform, no-material, one-triangle-
/// per-face rules.
///
/// # Errors
/// Returns [`VoxelizerError::MeshLoad`] if the bytes fail to parse as STL, or if
/// the assembled mesh fails [`MeshInput::validate`].
pub fn load_stl_slice(bytes: &[u8]) -> Result<MeshInput, VoxelizerError> {
    let mesh = stl_io::read_stl(&mut std::io::Cursor::new(bytes))
        .map_err(|e| VoxelizerError::MeshLoad(e.to_string()))?;
    indexed_mesh_to_input(&mesh)
}

/// Load an STL document from a filesystem path into a world-space [`MeshInput`].
///
/// Opens the file and hands it (a `Read + Seek` source) to
/// `stl_io::read_stl`, which auto-detects ASCII vs binary STL.
///
/// # Errors
/// Returns [`VoxelizerError::MeshLoad`] if the file cannot be read, fails to
/// parse, or the assembled mesh fails [`MeshInput::validate`].
pub fn load_stl_path(path: impl AsRef<std::path::Path>) -> Result<MeshInput, VoxelizerError> {
    let mut file =
        std::fs::File::open(path.as_ref()).map_err(|e| VoxelizerError::MeshLoad(e.to_string()))?;
    let mesh = stl_io::read_stl(&mut file).map_err(|e| VoxelizerError::MeshLoad(e.to_string()))?;
    indexed_mesh_to_input(&mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Epsilon compare for a single vertex.
    fn vec_approx(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-5
    }

    /// Build a binary STL document in memory from a list of triangles (each a
    /// `[v0, v1, v2]` of `[x, y, z]`).
    ///
    /// Layout: 80-byte zero header, `u32` LE triangle count, then per triangle
    /// (50 bytes): normal (`[0,0,0]`), v1, v2, v3 (each 3× `f32` LE), and a
    /// `u16` LE attribute-byte-count of 0.
    fn make_binary_stl(tris: &[[[f32; 3]; 3]]) -> Vec<u8> {
        let mut out = Vec::with_capacity(84 + tris.len() * 50);
        out.extend_from_slice(&[0u8; 80]); // header
        out.extend_from_slice(&(tris.len() as u32).to_le_bytes());
        for tri in tris {
            // Normal [0, 0, 0]: stl_io ignores it for IndexedMesh assembly.
            out.extend_from_slice(&0f32.to_le_bytes());
            out.extend_from_slice(&0f32.to_le_bytes());
            out.extend_from_slice(&0f32.to_le_bytes());
            for v in tri {
                for &c in v {
                    out.extend_from_slice(&c.to_le_bytes());
                }
            }
            out.extend_from_slice(&0u16.to_le_bytes()); // attribute byte count
        }
        out
    }

    #[test]
    fn loads_triangle() {
        let stl = make_binary_stl(&[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]);
        let mesh = load_stl_slice(&stl).expect("single-triangle binary STL must load");

        assert_eq!(mesh.triangles.len(), 1, "exactly one triangle");
        let [v0, v1, v2] = mesh.triangles[0];
        assert!(vec_approx(v0, Vec3::new(0.0, 0.0, 0.0)), "v0 = {v0:?}");
        assert!(vec_approx(v1, Vec3::new(1.0, 0.0, 0.0)), "v1 = {v1:?}");
        assert!(vec_approx(v2, Vec3::new(0.0, 1.0, 0.0)), "v2 = {v2:?}");
        // STL has no material concept → None (not fabricated ids).
        assert_eq!(mesh.material_ids, None);
    }

    #[test]
    fn loads_two_triangles() {
        let stl = make_binary_stl(&[
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ]);
        let mesh = load_stl_slice(&stl).expect("two-triangle binary STL must load");

        assert_eq!(mesh.triangles.len(), 2, "exactly two triangles");
        assert_eq!(mesh.material_ids, None);
    }

    #[test]
    fn rejects_garbage() {
        // `stl_io` auto-detects ASCII vs binary; a short non-STL byte stream must
        // surface as MeshLoad (never panic). Whichever path it takes, the result
        // must be a valid `MeshInput` (possibly empty) or a `MeshLoad` error —
        // both are acceptable; we assert no panic and the error variant when it
        // errors.
        match load_stl_slice(b"\x00\x01 nonsense") {
            Ok(mesh) => {
                // stl_io was lenient: yielded an empty (but valid) mesh. An empty
                // MeshInput is valid by design.
                assert!(
                    mesh.triangles.is_empty(),
                    "garbage input must not synthesize triangles"
                );
            }
            Err(err) => {
                assert!(
                    matches!(err, VoxelizerError::MeshLoad(_)),
                    "garbage bytes must yield MeshLoad, got {err:?}"
                );
            }
        }
    }
}
