//! Wavefront OBJ input adapter: load an OBJ document into the crate's
//! world-space triangle-soup [`MeshInput`].
//!
//! A sibling of the [glTF adapter](super::gltf): every adapter funnels into the
//! same [`MeshInput`] the voxelizer consumes. The whole module is gated behind
//! the `obj` cargo feature so the `tobj` dependency is droppable.
//!
//! # How OBJ differs from glTF
//! OBJ has **no scene graph and no node transforms** — its vertices are already
//! in model space. There is therefore NO transform accumulation (unlike the
//! glTF walk): vertices are taken as-is. All models / groups in the document are
//! flattened into a single [`MeshInput`].
//!
//! # Triangulation
//! `tobj` is asked to fan-triangulate every face (`triangulate: true`), so an
//! n-gon face (quad, …) arrives as a triangle list — this adapter never
//! triangulates by hand. `single_index: true` collapses OBJ's separate
//! position/normal/texcoord index streams into one shared index buffer; only
//! `mesh.positions` and `mesh.indices` are consumed here.
//!
//! # Material ids
//! Each emitted triangle records its mesh's material index
//! (`mesh.material_id`), or [`u32::MAX`] when the model has no material. The
//! resulting `material_ids` length always equals `triangles`, so the returned
//! [`MeshInput`] passes [`MeshInput::validate`].
//!
//! # Errors
//! Any `tobj` parse / load failure, or a failed [`MeshInput::validate`], is
//! surfaced as [`VoxelizerError::MeshLoad`] (the generic loader error) via
//! `.to_string()`. Out-of-range face indices in an otherwise-parseable document
//! drop only the offending triangle rather than erroring or panicking.

use glam::Vec3;

use crate::core::MeshInput;
use crate::error::VoxelizerError;

/// The `tobj` load options shared by both entry points: fan-triangulate every
/// face and collapse OBJ's multi-stream indices into one shared buffer.
fn load_options() -> tobj::LoadOptions {
    tobj::LoadOptions {
        triangulate: true,
        single_index: true,
        ..Default::default()
    }
}

/// Assemble a [`MeshInput`] from the `tobj` model list, flattening every model
/// into one world-space triangle soup.
///
/// OBJ vertices are already in model space, so positions are taken verbatim (no
/// transform). Out-of-range indices drop only the offending triangle.
fn meshes_to_input(models: &[tobj::Model]) -> Result<MeshInput, VoxelizerError> {
    let mut triangles: Vec<[Vec3; 3]> = Vec::new();
    let mut material_ids: Vec<u32> = Vec::new();

    for model in models {
        let mesh = &model.mesh;
        // Flat xyz triples → Vec3 vertex list (no transform: OBJ is model-space).
        let positions: Vec<Vec3> = mesh
            .positions
            .chunks_exact(3)
            .map(|p| Vec3::new(p[0], p[1], p[2]))
            .collect();

        let material_id = mesh.material_id.map_or(u32::MAX, |i| i as u32);

        // Indices are already triangulated (LoadOptions::triangulate).
        for tri in mesh.indices.chunks_exact(3) {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            // A bad index would be a malformed document; guard rather than
            // panic, dropping only the offending triangle.
            if let (Some(&v0), Some(&v1), Some(&v2)) =
                (positions.get(a), positions.get(b), positions.get(c))
            {
                triangles.push([v0, v1, v2]);
                material_ids.push(material_id);
            }
        }
    }

    let mesh = MeshInput {
        triangles,
        material_ids: Some(material_ids),
    };
    // Surfaces a length mismatch or a non-finite vertex as MeshLoad rather than
    // panicking downstream.
    mesh.validate()
        .map_err(|e| VoxelizerError::MeshLoad(e.to_string()))?;
    Ok(mesh)
}

/// Load a Wavefront OBJ document from an in-memory byte slice into a world-space
/// [`MeshInput`].
///
/// Parses the OBJ from memory. A `mtllib` directive cannot be resolved from a
/// bare slice (there is no sibling `.mtl` to read), so the material loader
/// fails and every triangle gets the default material id ([`u32::MAX`]) — which
/// is correct for slice input. To resolve a sibling `.mtl`, use
/// [`load_obj_path`].
///
/// See the [module docs](self) for flattening, triangulation, and material-id
/// rules.
///
/// # Errors
/// Returns [`VoxelizerError::MeshLoad`] if the bytes fail to parse as OBJ, or if
/// the assembled mesh fails [`MeshInput::validate`].
pub fn load_obj_slice(bytes: &[u8]) -> Result<MeshInput, VoxelizerError> {
    let opts = load_options();
    let (models, _materials) = tobj::load_obj_buf(
        &mut std::io::Cursor::new(bytes),
        &opts,
        // A bare slice has no filesystem context to resolve a referenced
        // `.mtl`; report failure so referenced materials default. A document
        // with no `mtllib` never invokes this closure.
        |_mtl_path| Err(tobj::LoadError::OpenFileFailed),
    )
    .map_err(|e| VoxelizerError::MeshLoad(e.to_string()))?;

    meshes_to_input(&models)
}

/// Load a Wavefront OBJ document from a filesystem path into a world-space
/// [`MeshInput`].
///
/// Unlike [`load_obj_slice`], this resolves a sibling `.mtl` referenced via
/// `mtllib`, so triangles carry their real material indices.
///
/// # Errors
/// Returns [`VoxelizerError::MeshLoad`] if the file cannot be read, fails to
/// parse, or the assembled mesh fails [`MeshInput::validate`].
pub fn load_obj_path(path: impl AsRef<std::path::Path>) -> Result<MeshInput, VoxelizerError> {
    let opts = load_options();
    // `tobj::load_obj` requires `P: AsRef<Path> + Debug`; a `&Path` satisfies
    // both, keeping this fn's public signature in step with the glTF adapter.
    let (models, _materials) = tobj::load_obj(path.as_ref(), &opts)
        .map_err(|e| VoxelizerError::MeshLoad(e.to_string()))?;

    meshes_to_input(&models)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Epsilon compare for a single vertex.
    fn vec_approx(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-5
    }

    #[test]
    fn loads_triangle() {
        let obj = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
        let mesh = load_obj_slice(obj).expect("single-triangle OBJ must load");

        assert_eq!(mesh.triangles.len(), 1, "exactly one triangle");
        let [v0, v1, v2] = mesh.triangles[0];
        assert!(vec_approx(v0, Vec3::new(0.0, 0.0, 0.0)), "v0 = {v0:?}");
        assert!(vec_approx(v1, Vec3::new(1.0, 0.0, 0.0)), "v1 = {v1:?}");
        assert!(vec_approx(v2, Vec3::new(0.0, 1.0, 0.0)), "v2 = {v2:?}");
        // No material → u32::MAX, one entry per triangle.
        assert_eq!(mesh.material_ids, Some(vec![u32::MAX]));
    }

    #[test]
    fn triangulates_quad() {
        // A single quad face: tobj must fan-triangulate it into two triangles.
        let obj = b"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n";
        let mesh = load_obj_slice(obj).expect("quad OBJ must load");

        assert_eq!(
            mesh.triangles.len(),
            2,
            "a quad fan-triangulates into exactly two triangles"
        );
        // All four corners must appear somewhere across the two triangles.
        let corners = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        for corner in corners {
            let present = mesh
                .triangles
                .iter()
                .flatten()
                .any(|&v| vec_approx(v, corner));
            assert!(present, "corner {corner:?} missing from triangulated quad");
        }
        assert_eq!(mesh.material_ids, Some(vec![u32::MAX; 2]));
    }

    #[test]
    fn rejects_garbage() {
        // tobj treats unrecognized lines leniently, but a leading non-UTF-8 byte
        // stream that fails to parse must surface as MeshLoad (never panic).
        // Whichever path tobj takes, the result must be a valid `MeshInput`
        // (possibly empty) or a `MeshLoad` error — both are acceptable; we
        // assert no panic and the error variant when it errors.
        match load_obj_slice(b"\xff\xff not an obj") {
            Ok(mesh) => {
                // tobj was lenient: junk lines were ignored, yielding an empty
                // (but valid) mesh. Empty MeshInput is valid by design.
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
