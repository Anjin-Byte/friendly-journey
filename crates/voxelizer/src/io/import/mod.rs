//! Input adapters: read external mesh formats into the crate's world-space
//! triangle-soup [`MeshInput`].
//!
//! Every adapter is a sibling *format module* behind one dispatcher, and every
//! adapter funnels into the same [`MeshInput`] the voxelizer already consumes
//! (world-space triangle soup + optional per-triangle material ids). The public
//! surface is intentionally narrow and source-agnostic.
//!
//! # Format modules
//! - [`gltf`] — glTF / GLB, gated behind the `gltf` feature. Has a scene graph,
//!   so it accumulates node world transforms.
//! - [`obj`] — Wavefront OBJ, gated behind the `obj` feature. No scene graph and
//!   no transforms — vertices are already in model space.
//! - [`stl`] — STL, gated behind the `stl` feature. The simplest format: no
//!   scene graph, no transforms, and no materials — facets are already
//!   triangles in model space.
//!
//! # Dispatch
//! [`load_mesh`] selects a format from the file extension and routes to the
//! per-format `*_path` loader, but only when that format's feature is enabled.
//! An unrecognized, missing, or feature-disabled extension yields
//! [`VoxelizerError::MeshLoad`]. The module compiles under any feature
//! combination, including none of `gltf` / `obj` / `stl`.

#[cfg(feature = "gltf")]
pub mod gltf;
#[cfg(feature = "obj")]
pub mod obj;
#[cfg(feature = "stl")]
pub mod stl;

#[cfg(feature = "gltf")]
pub use gltf::{load_gltf_path, load_gltf_slice};
#[cfg(feature = "obj")]
pub use obj::{load_obj_path, load_obj_slice};
#[cfg(feature = "stl")]
pub use stl::{load_stl_path, load_stl_slice};

use crate::core::MeshInput;
use crate::error::VoxelizerError;
use glam::Mat4;

/// Builds a corrective rotation from per-axis angles in **degrees**.
///
/// Transform-less formats (OBJ, STL) bake an exporter's up-axis convention
/// directly into the vertices, so a model authored Z-up arrives lying on its
/// back when the renderer is Y-up. Feed the result to [`MeshInput::transform`]
/// before fitting the grid to re-orient it.
///
/// Rotations compose **X, then Y, then Z** (`Rz · Ry · Rx`), so the angles read
/// as "roll the model `x` about X, then `y` about Y, then `z` about Z." A single
/// nonzero axis is unambiguous regardless of order. All-zero yields the identity.
#[must_use]
pub fn rotation_degrees(x: f32, y: f32, z: f32) -> Mat4 {
    Mat4::from_rotation_z(z.to_radians())
        * Mat4::from_rotation_y(y.to_radians())
        * Mat4::from_rotation_x(x.to_radians())
}

/// Load a mesh from a filesystem path, dispatching on the file extension.
///
/// Recognized extensions (case-insensitive):
/// - `gltf` / `glb` → glTF loader (requires the `gltf` feature).
/// - `obj` → OBJ loader (requires the `obj` feature).
/// - `stl` → STL loader (requires the `stl` feature).
///
/// Routes to the per-format `*_path` loader only when that format's feature is
/// enabled. An extension that is unrecognized, missing, or matches a
/// feature-disabled format yields [`VoxelizerError::MeshLoad`].
///
/// # Errors
/// Returns [`VoxelizerError::MeshLoad`] if the extension is unsupported or its
/// format feature is disabled, or if the chosen loader fails to read, parse, or
/// validate the mesh.
pub fn load_mesh(path: impl AsRef<std::path::Path>) -> Result<MeshInput, VoxelizerError> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);

    match ext.as_deref() {
        #[cfg(feature = "gltf")]
        Some("gltf" | "glb") => load_gltf_path(path),
        #[cfg(feature = "obj")]
        Some("obj") => load_obj_path(path),
        #[cfg(feature = "stl")]
        Some("stl") => load_stl_path(path),
        other => Err(VoxelizerError::MeshLoad(format!(
            "unsupported or disabled mesh format: {}",
            other.unwrap_or("<none>")
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_degrees_identity_and_x90() {
        use glam::Vec3;

        // All-zero is the identity (exact: from_rotation_*(0) is bit-clean).
        assert_eq!(rotation_degrees(0.0, 0.0, 0.0), Mat4::IDENTITY);

        // +90° about X sends +Y → +Z (matches `MeshInput::transform`'s test).
        let p = rotation_degrees(90.0, 0.0, 0.0).transform_point3(Vec3::Y);
        assert!(
            (p - Vec3::Z).length() < 1e-6,
            "+90° about X must map +Y → +Z, got {p:?}"
        );
    }

    #[test]
    fn load_mesh_rejects_unknown_extension() {
        let err = load_mesh("model.xyz").unwrap_err();
        assert!(
            matches!(err, VoxelizerError::MeshLoad(_)),
            "unknown extension must yield MeshLoad, got {err:?}"
        );
    }

    #[cfg(feature = "obj")]
    #[test]
    fn load_mesh_dispatches_obj() {
        use std::io::Write;

        // Write the triangle OBJ to a uniquely-named temp file with a `.obj`
        // extension, dispatch through `load_mesh`, then clean up. Avoids needing
        // a real GPU or any checked-in asset.
        let mut path = std::env::temp_dir();
        path.push(format!(
            "voxelizer_load_mesh_dispatch_{}_{}.obj",
            std::process::id(),
            // A monotonically-unique-enough suffix for parallel test runs.
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));

        {
            let mut f = std::fs::File::create(&path).expect("create temp .obj");
            f.write_all(b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
                .expect("write temp .obj");
        }

        let result = load_mesh(&path);
        // Best-effort cleanup before asserting so a failure still removes the file.
        let _ = std::fs::remove_file(&path);

        let mesh = result.expect("load_mesh must dispatch the .obj to the OBJ loader");
        assert_eq!(mesh.triangles.len(), 1, "exactly one triangle");
    }

    #[cfg(feature = "stl")]
    #[test]
    fn load_mesh_dispatches_stl() {
        use std::io::Write;

        // Build a single-triangle binary STL (80-byte zero header, count=1, then
        // one 50-byte facet: normal[0,0,0], v0, v1, v2, attr=0).
        let mut stl = Vec::with_capacity(84 + 50);
        stl.extend_from_slice(&[0u8; 80]); // header
        stl.extend_from_slice(&1u32.to_le_bytes()); // triangle count
        for f in [
            0.0f32, 0.0, 0.0, // normal
            0.0, 0.0, 0.0, // v0
            1.0, 0.0, 0.0, // v1
            0.0, 1.0, 0.0, // v2
        ] {
            stl.extend_from_slice(&f.to_le_bytes());
        }
        stl.extend_from_slice(&0u16.to_le_bytes()); // attribute byte count

        // Write the binary STL to a uniquely-named temp file with a `.stl`
        // extension, dispatch through `load_mesh`, then clean up. Mirrors the
        // OBJ dispatcher test; avoids any checked-in asset.
        let mut path = std::env::temp_dir();
        path.push(format!(
            "voxelizer_load_mesh_dispatch_{}_{}.stl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));

        {
            let mut file = std::fs::File::create(&path).expect("create temp .stl");
            file.write_all(&stl).expect("write temp .stl");
        }

        let result = load_mesh(&path);
        // Best-effort cleanup before asserting so a failure still removes the file.
        let _ = std::fs::remove_file(&path);

        let mesh = result.expect("load_mesh must dispatch the .stl to the STL loader");
        assert_eq!(mesh.triangles.len(), 1, "exactly one triangle");
    }
}
