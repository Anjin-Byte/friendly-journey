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
//! position/normal/texcoord index streams into one shared index buffer, so
//! `mesh.indices` index `positions` *and* `texcoords` alike.
//!
//! # Material ids
//! Each emitted triangle records its mesh's material index
//! (`mesh.material_id`), or [`u32::MAX`] when the model has no material. The
//! resulting `material_ids` length always equals `triangles`, so the returned
//! [`MeshInput`] passes [`MeshInput::validate`].
//!
//! # UVs + textures (truecolor)
//! [`load_obj_path`] also extracts per-triangle UVs (`vt`) and the MTL's `map_Kd`
//! textures + `Kd` factors into [`MeshInput::appearance`], so a textured OBJ feeds
//! the per-voxel truecolor bake (`docs/materials/11`). UVs are taken **verbatim**
//! (top-left origin, matching the bake's image-row sampling and the glTF adapter);
//! the textbook OBJ bottom-left flip is intentionally NOT applied (these assets are
//! glTF-derived, and flipping samples the wrong atlas region). Textures resolve
//! relative to the OBJ's directory, so [`load_obj_slice`] (no directory) yields
//! UVs but no appearance (palette path).
//!
//! `map_Kd` images are decoded via the `image` dep, built for **PNG + JPEG**
//! only. A `map_Kd` in any other format (TGA / BMP / TIFF / …) — or a missing
//! file — cannot be decoded, so that material **degrades to flat `Kd`** with a
//! logged warning rather than aborting the load (truecolor → palette). If *no*
//! material yields a usable texture, the whole appearance is dropped.
//!
//! # Errors
//! Any `tobj` parse / load failure, or a failed [`MeshInput::validate`], is
//! surfaced as [`VoxelizerError::MeshLoad`] (the generic loader error) via
//! `.to_string()`. Out-of-range face indices in an otherwise-parseable document
//! drop only the offending triangle rather than erroring or panicking.

use std::collections::HashMap;
use std::path::Path;

use glam::{Vec2, Vec3};

use crate::appearance::{AlphaMode, Texture, WrapMode};
use crate::core::{MaterialDef, MeshAppearance, MeshInput};
use crate::error::VoxelizerError;

/// The `tobj` load options shared by both entry points: fan-triangulate every
/// face and collapse OBJ's multi-stream indices into one shared buffer (so
/// `mesh.indices` index `positions` *and* `texcoords` alike).
fn load_options() -> tobj::LoadOptions {
    tobj::LoadOptions {
        triangulate: true,
        single_index: true,
        ..Default::default()
    }
}

/// Decodes a `map_Kd` image file (PNG/JPEG) into an sRGB RGBA8 [`Texture`], or
/// `None` if it cannot be read/decoded (the material then bakes its flat `Kd`).
fn decode_obj_texture(path: &Path) -> Option<Texture> {
    let img = image::open(path).ok()?.to_rgba8();
    let (width, height) = (img.width(), img.height());
    let rgba: Vec<[u8; 4]> = img.pixels().map(|p| p.0).collect();
    // `to_rgba8` always yields exactly width*height texels, so `new` accepts;
    // `.ok()` keeps the existing "decode failure → flat Kd" contract regardless.
    Texture::new(width, height, rgba).ok()
}

/// Builds the base-colour appearance from the MTL `materials` (indexed by
/// `mesh.material_id`), decoding each unique `map_Kd` texture relative to
/// `base_dir`. Returns `None` when no material carries a usable texture (the
/// palette path is then used). `Kd` becomes the linear `base_color_factor`.
fn build_obj_appearance(materials: &[tobj::Material], base_dir: &Path) -> Option<MeshAppearance> {
    let mut textures: Vec<Texture> = Vec::new();
    let mut by_path: HashMap<String, usize> = HashMap::new();
    let mut defs: Vec<MaterialDef> = Vec::with_capacity(materials.len());

    for mat in materials {
        // Decode the diffuse texture once per unique path; a decode failure leaves
        // the material textureless (flat Kd) rather than aborting the load.
        let base_color_texture = mat.diffuse_texture.as_ref().and_then(|name| {
            if let Some(&i) = by_path.get(name) {
                return Some(i);
            }
            if let Some(tex) = decode_obj_texture(&base_dir.join(name)) {
                let i = textures.len();
                textures.push(tex);
                by_path.insert(name.clone(), i);
                Some(i)
            } else {
                // Missing file, or a format the `image` dep is not built with
                // (only png+jpeg). Warn so the truecolor→flat degrade is visible
                // rather than silent, then render the material flat.
                eprintln!(
                    "voxelizer: OBJ map_Kd texture {name:?} could not be read/decoded \
                     (missing file, or a format other than PNG/JPEG); rendering it flat"
                );
                None
            }
        });
        let base_color_factor = mat
            .diffuse
            .map_or([1.0, 1.0, 1.0, 1.0], |d| [d[0], d[1], d[2], 1.0]);
        defs.push(MaterialDef {
            name: (!mat.name.is_empty()).then(|| mat.name.clone()),
            base_color_texture,
            base_color_factor,
            wrap_s: WrapMode::Repeat,
            wrap_t: WrapMode::Repeat,
            // OBJ/MTL carries no alpha-mode intent — treat every material as opaque.
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
        });
    }

    // No textures → nothing to gain from truecolor; keep the palette path.
    (!textures.is_empty()).then_some(MeshAppearance {
        textures,
        materials: defs,
    })
}

/// Assemble a [`MeshInput`] from the `tobj` model list, flattening every model
/// into one world-space triangle soup. Extracts per-triangle UVs (with the OBJ→
/// image **V-flip**) and, when `base_dir` is given, the MTL textures.
///
/// OBJ vertices are already in model space, so positions are taken verbatim (no
/// transform). Out-of-range indices drop only the offending triangle.
fn meshes_to_input(
    models: &[tobj::Model],
    materials: &[tobj::Material],
    base_dir: Option<&Path>,
) -> Result<MeshInput, VoxelizerError> {
    let mut triangles: Vec<[Vec3; 3]> = Vec::new();
    let mut material_ids: Vec<u32> = Vec::new();
    let mut uvs: Vec<[Vec2; 3]> = Vec::new();
    let mut any_uv = false;

    for model in models {
        let mesh = &model.mesh;
        // Flat xyz triples → Vec3 vertex list (no transform: OBJ is model-space).
        let positions: Vec<Vec3> = mesh
            .positions
            .chunks_exact(3)
            .map(|p| Vec3::new(p[0], p[1], p[2]))
            .collect();

        let material_id = mesh.material_id.map_or(u32::MAX, |i| i as u32);

        // single_index collapses streams, so `indices` index `texcoords` too.
        // UVs are taken **verbatim** (top-left origin, matching the image-row
        // convention the bake samples — same as the glTF adapter). The textbook
        // OBJ/OpenGL bottom-left convention would need `v → 1-v`, but the assets
        // here are glTF-derived (top-left), and flipping samples the WRONG atlas
        // region (an empirically-confirmed washout). A future `--flip-v` flag can
        // serve genuinely bottom-left OBJ exports (e.g. some Blender output).
        let has_uv = !mesh.texcoords.is_empty();
        let uv_at = |i: usize| -> Vec2 {
            match (mesh.texcoords.get(2 * i), mesh.texcoords.get(2 * i + 1)) {
                (Some(&u), Some(&v)) => Vec2::new(u, v),
                _ => Vec2::ZERO,
            }
        };

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
                uvs.push([uv_at(a), uv_at(b), uv_at(c)]);
                any_uv |= has_uv;
            }
        }
    }

    // Textures resolve only with a filesystem base dir (path-resolution context).
    let appearance = base_dir.and_then(|dir| build_obj_appearance(materials, dir));

    let mesh = MeshInput {
        triangles,
        material_ids: Some(material_ids),
        uvs: any_uv.then_some(uvs),
        appearance,
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

    // No base dir ⇒ no texture resolution (UVs are still extracted); a slice is
    // the palette path. Use [`load_obj_path`] for MTL-textured truecolor.
    meshes_to_input(&models, &[], None)
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
    let path = path.as_ref();
    // `tobj::load_obj` requires `P: AsRef<Path> + Debug`; a `&Path` satisfies
    // both, keeping this fn's public signature in step with the glTF adapter.
    let (models, materials) =
        tobj::load_obj(path, &opts).map_err(|e| VoxelizerError::MeshLoad(e.to_string()))?;
    // A missing / unreadable `.mtl` is not fatal — the mesh still loads (palette
    // path); only a parse failure of the OBJ itself errors above.
    let materials = materials.unwrap_or_default();
    // map_Kd paths are relative to the OBJ/MTL directory.
    let base_dir = path.parent();

    meshes_to_input(&models, &materials, base_dir)
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
    fn extracts_uvs_from_vt() {
        // A triangle with texcoords: the loader surfaces per-triangle UVs verbatim
        // (top-left origin, matching the bake's image-row sampling — NOT flipped;
        // the glTF-derived assets here are top-left). No MTL ⇒ no appearance.
        let obj = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0.1 0.2\nvt 0.7 0.3\nvt 0.4 0.9\nf 1/1 2/2 3/3\n";
        let mesh = load_obj_slice(obj).expect("UV triangle must load");
        let uvs = mesh.uvs.expect("vt coords must surface as UVs");
        assert_eq!(uvs.len(), 1);
        assert!((uvs[0][0] - Vec2::new(0.1, 0.2)).length() < 1e-6);
        assert!((uvs[0][1] - Vec2::new(0.7, 0.3)).length() < 1e-6);
        assert!((uvs[0][2] - Vec2::new(0.4, 0.9)).length() < 1e-6);
        assert!(mesh.appearance.is_none(), "slice path has no texture dir");
    }

    /// Real-asset smoke: `littlest-tokyo.obj` + its `.mtl` must load with
    /// per-triangle UVs and a populated appearance whose `map_Kd` PNGs decode —
    /// the path the viewer's `--truecolor` needs. `#[ignore]` (needs the asset);
    /// run: `cargo test -p voxelizer --lib loads_obj_mtl_textures -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs models/littlest-tokyo.obj + .mtl + textures; run with --ignored --nocapture"]
    fn loads_obj_mtl_textures() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/littlest-tokyo.obj"
        );
        let mesh = load_obj_path(path).expect("littlest-tokyo.obj must load");
        let uvs = mesh.uvs.as_ref().expect("the OBJ has vt coords");
        assert_eq!(uvs.len(), mesh.triangles.len(), "UVs aligned to triangles");
        let app = mesh
            .appearance
            .as_ref()
            .expect("the MTL's map_Kd textures must build an appearance");
        let textured = app
            .materials
            .iter()
            .filter(|m| m.base_color_texture.is_some())
            .count();
        let texels: usize = app.textures.iter().map(|t| t.rgba().len()).sum();
        eprintln!(
            "littlest-tokyo.obj: {} tris, {} materials ({textured} textured), {} textures, {texels} texels",
            mesh.triangles.len(),
            app.materials.len(),
            app.textures.len(),
        );
        assert!(!app.textures.is_empty(), "map_Kd PNGs must decode");
        assert!(textured > 0, "at least one material references a texture");
        for (i, t) in app.textures.iter().enumerate() {
            assert_eq!(
                t.rgba().len(),
                (t.width() as usize) * (t.height() as usize),
                "texture {i} ({}×{}) decoded to a full RGBA8 grid",
                t.width(),
                t.height()
            );
        }
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
