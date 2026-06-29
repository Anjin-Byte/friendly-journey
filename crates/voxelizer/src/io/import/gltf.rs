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
//! - Each scene's node tree via an **iterative depth-first walk** (an explicit
//!   work-stack, not recursion — see `walk_node`), accumulating the full
//!   **world transform** (`parent · node.transform()`) as a [`glam::Mat4`]. The
//!   iterative walk is safe to arbitrary node depth; a deep chain cannot
//!   overflow the stack and abort the process.
//! - For every node carrying a mesh, every primitive whose mode is
//!   [`gltf::mesh::Mode::Triangles`]: its `POSITION` attribute and (optional)
//!   index buffer.
//!
//! # What it skips (without erroring)
//! A scene can legitimately mix geometry kinds, so the loader is permissive:
//! - Non-`Triangles` primitives (points / lines / strips / fans) are skipped.
//! - Primitives with no `POSITION` accessor are skipped.
//! - A non-multiple-of-3 trailing index/position remainder is dropped
//!   (`chunks_exact(3)`), so a truncated primitive loses only its partial triangle.
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
//!
//! # Faithfulness limits (what is *not* supported)
//! These are graceful (an error or a documented degrade), never a panic:
//! - **Base-colour image formats:** decoded via the `gltf` crate's `image` dep,
//!   built for **PNG + JPEG (8-bit)** only. A higher-bit-depth base-colour image
//!   degrades to **neutral grey** with a logged warning (truecolor → flat for
//!   that texture); a WebP / KTX2-Basis image is a graceful `MeshLoad`.
//! - **Required extensions** the `gltf` crate cannot honor (`KHR_draco_mesh_compression`,
//!   `KHR_texture_basisu`, `KHR_mesh_quantization`, `KHR_texture_transform`) reject
//!   the whole document with a `MeshLoad("Unsupported extension")`, even though the
//!   base geometry might be loadable — a conservative faithfulness ceiling.
//! - **Data-URI images:** [`load_gltf_slice`] (and therefore [`load_gltf_path`],
//!   which delegates to it) resolves embedded data-URI *buffers* but not data-URI
//!   *images*; a base-colour texture embedded as a data-URI fails the whole load.
//!   GLB-embedded and bufferView images work.

use glam::{Mat4, Vec2, Vec3};

use crate::appearance::{AlphaMode, Texture, WrapMode};
use crate::core::{MaterialDef, MeshAppearance, MeshInput};
use crate::error::VoxelizerError;

/// Maps a glTF sampler wrap mode to ours (mirrored-repeat folds to repeat — the
/// bake's seam handling tiles, not mirrors; acceptable for base-colour).
fn wrap_of(mode: gltf::texture::WrappingMode) -> WrapMode {
    match mode {
        gltf::texture::WrappingMode::ClampToEdge => WrapMode::ClampToEdge,
        gltf::texture::WrappingMode::Repeat | gltf::texture::WrappingMode::MirroredRepeat => {
            WrapMode::Repeat
        }
    }
}

/// Converts a decoded glTF image to RGBA8 (sRGB-encoded as base-colour textures
/// are). 8-bit formats convert exactly; higher-bit-depth base-colour textures are
/// rare and fall back to neutral grey (a logged degrade, see [`warn_grey_image`]).
///
/// # Errors
/// Returns [`VoxelizerError::MalformedTexture`] (via [`Texture::new`]) if the
/// decoded pixels do not fill the declared `width * height` — a corrupt embedded
/// image is rejected gracefully rather than silently truncated or panicked on.
fn decode_texture(d: &gltf::image::Data) -> Result<Texture, VoxelizerError> {
    use gltf::image::Format;
    let n = (d.width as usize) * (d.height as usize);
    let rgba = match d.format {
        Format::R8G8B8A8 => d
            .pixels
            .chunks_exact(4)
            .map(|p| [p[0], p[1], p[2], p[3]])
            .collect(),
        Format::R8G8B8 => d
            .pixels
            .chunks_exact(3)
            .map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        Format::R8G8 => d
            .pixels
            .chunks_exact(2)
            .map(|p| [p[0], p[0], p[0], p[1]])
            .collect(),
        Format::R8 => d.pixels.iter().map(|&p| [p, p, p, 255]).collect(),
        other => {
            // Unsupported high-bit-depth base colour → neutral grey. Warn so the
            // truecolor→flat degrade is visible, not silent (faithfulness gap).
            warn_grey_image(other);
            vec![[128, 128, 128, 255]; n]
        }
    };
    Texture::new(d.width, d.height, rgba)
}

/// Emit a one-line warning that a base-colour image decoded to neutral grey
/// because its pixel format is not one of the supported 8-bit-per-channel forms.
fn warn_grey_image(format: gltf::image::Format) {
    eprintln!(
        "voxelizer: glTF base-colour image has unsupported format {format:?}; \
         rendering it as neutral grey (truecolor degraded to flat for this texture)"
    );
}

/// Builds one [`MaterialDef`] per glTF material, indexed by material index (the
/// same index `primitive.material().index()` records on each triangle). The
/// base-colour texture's *image* index points into the decoded texture list.
fn build_material_defs(document: &gltf::Document) -> Vec<MaterialDef> {
    document
        .materials()
        .map(|mat| {
            let pbr = mat.pbr_metallic_roughness();
            let (texture, wrap_s, wrap_t) = match pbr.base_color_texture() {
                Some(info) => {
                    let tex = info.texture();
                    let sampler = tex.sampler();
                    (
                        Some(tex.source().index()),
                        wrap_of(sampler.wrap_s()),
                        wrap_of(sampler.wrap_t()),
                    )
                }
                None => (None, WrapMode::Repeat, WrapMode::Repeat),
            };
            let alpha_mode = match mat.alpha_mode() {
                gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                gltf::material::AlphaMode::Mask => AlphaMode::Mask,
                gltf::material::AlphaMode::Blend => AlphaMode::Blend,
            };
            MaterialDef {
                name: mat.name().map(str::to_owned),
                base_color_texture: texture,
                base_color_factor: pbr.base_color_factor(),
                wrap_s,
                wrap_t,
                alpha_mode,
                alpha_cutoff: sanitize_cutoff(mat.alpha_cutoff().unwrap_or(0.5)),
            }
        })
        .collect()
}

/// Clamp a glTF `alphaCutoff` into `[0, 1]`, mapping a non-finite value to the
/// spec default `0.5`. A raw NaN cutoff makes every alpha-test `< cutoff` false
/// *and* `>= cutoff` false, dropping (or keeping) **all** MASK voxels; a value
/// outside `[0, 1]` trivially cuts everything or nothing. Sanitizing here means
/// the bake/cull always see a sane threshold.
fn sanitize_cutoff(cutoff: f32) -> f32 {
    if cutoff.is_finite() {
        cutoff.clamp(0.0, 1.0)
    } else {
        0.5
    }
}

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
    let (document, buffers, images) =
        gltf::import_slice(bytes).map_err(|e| VoxelizerError::MeshLoad(e.to_string()))?;

    let mut triangles: Vec<[Vec3; 3]> = Vec::new();
    let mut material_ids: Vec<u32> = Vec::new();
    let mut uvs: Vec<[Vec2; 3]> = Vec::new();

    // Default scene, else the first scene, else every scene (covers documents
    // that declare scenes without marking one default).
    if let Some(scene) = document
        .default_scene()
        .or_else(|| document.scenes().next())
    {
        for node in scene.nodes() {
            walk_node(
                node,
                Mat4::IDENTITY,
                &buffers,
                &mut triangles,
                &mut material_ids,
                &mut uvs,
            );
        }
    } else {
        for scene in document.scenes() {
            for node in scene.nodes() {
                walk_node(
                    node,
                    Mat4::IDENTITY,
                    &buffers,
                    &mut triangles,
                    &mut material_ids,
                    &mut uvs,
                );
            }
        }
    }

    // Per-voxel-bake appearance: decode the base-colour textures + per-material
    // defs. Present whenever the document declares materials (textured or flat);
    // the bake resolves an out-of-range / default material to neutral.
    let materials = build_material_defs(&document);
    let appearance = if materials.is_empty() {
        None
    } else {
        // A corrupt embedded image (decoded pixels not filling width*height)
        // fails the whole load gracefully rather than panicking in the bake.
        let textures = images
            .iter()
            .map(decode_texture)
            .collect::<Result<Vec<_>, _>>()?;
        Some(MeshAppearance {
            textures,
            materials,
        })
    };

    let mesh = MeshInput {
        triangles,
        material_ids: Some(material_ids),
        uvs: Some(uvs),
        appearance,
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
/// # Examples
/// ```no_run
/// use voxelizer::load_gltf_path;
///
/// let mesh = load_gltf_path("model.glb")?;
/// println!("{} world-space triangles", mesh.triangles.len());
/// # Ok::<(), voxelizer::VoxelizerError>(())
/// ```
///
/// # Errors
/// Returns [`VoxelizerError::MeshLoad`] if the file cannot be read, fails to
/// parse, or the assembled mesh fails [`MeshInput::validate`].
pub fn load_gltf_path(path: impl AsRef<std::path::Path>) -> Result<MeshInput, VoxelizerError> {
    let bytes = std::fs::read(path).map_err(|e| VoxelizerError::MeshLoad(e.to_string()))?;
    load_gltf_slice(&bytes)
}

/// Accumulate world-space triangles from `node` and all of its descendants.
///
/// `parent` is the accumulated world transform of `node`'s parent; each node's
/// world transform is `parent · local`. Each primitive's triangles are
/// transformed into world space and appended, one `material_id` per emitted
/// triangle.
///
/// Traversal uses an explicit depth-first work-stack rather than recursion: a
/// glTF node hierarchy is attacker-/exporter-controlled and a deep single-child
/// chain is legal per spec, so recursing once per level would overflow the call
/// stack and `SIGABRT` the process (uncatchable) on a deeply-nested file. The
/// work-stack lives on the heap and handles arbitrary depth. Children are pushed
/// in reverse so they pop in document order, making the emission order identical
/// to the equivalent pre-order recursion.
fn walk_node(
    node: gltf::Node,
    parent: Mat4,
    buffers: &[gltf::buffer::Data],
    triangles: &mut Vec<[Vec3; 3]>,
    material_ids: &mut Vec<u32>,
    uvs: &mut Vec<[Vec2; 3]>,
) {
    // (node, parent-world-transform). A `gltf::Node` is a lightweight
    // (document, index) handle, so pushing owned nodes is cheap.
    let mut stack: Vec<(gltf::Node, Mat4)> = vec![(node, parent)];

    while let Some((node, parent)) = stack.pop() {
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

                // Read the UV set the material's base-colour texture references
                // (set 0 when untextured). Absent UVs → (0,0) per vertex, kept
                // aligned so the `uvs` array always matches `triangles`.
                let uv_set = primitive
                    .material()
                    .pbr_metallic_roughness()
                    .base_color_texture()
                    .map_or(0, |info| info.tex_coord());
                let prim_uvs: Option<Vec<[f32; 2]>> = reader
                    .read_tex_coords(uv_set)
                    .map(|r| r.into_f32().collect());
                let uv_at = |i: usize| -> Vec2 {
                    prim_uvs
                        .as_ref()
                        .and_then(|u| u.get(i))
                        .map_or(Vec2::ZERO, |&u| Vec2::from(u))
                };

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
                            uvs.push([uv_at(a), uv_at(b), uv_at(c)]);
                        }
                    }
                } else {
                    // Non-indexed: sequential triples of positions form triangles.
                    for (t, tri) in positions.chunks_exact(3).enumerate() {
                        triangles.push([tri[0], tri[1], tri[2]]);
                        material_ids.push(material_id);
                        uvs.push([uv_at(t * 3), uv_at(t * 3 + 1), uv_at(t * 3 + 2)]);
                    }
                }
            }
        }

        // Push children in reverse so the first child pops first — this yields
        // the same depth-first pre-order as the equivalent recursion.
        let children: Vec<gltf::Node> = node.children().collect();
        for child in children.into_iter().rev() {
            stack.push((child, world));
        }
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

    /// The iterative work-stack must emit triangles in the same depth-first
    /// pre-order as the natural recursion (node, then each child subtree in
    /// document order). Tree:
    /// ```text
    /// 0 (mat 0)
    /// ├─ 1 (mat 1)
    /// │  └─ 2 (mat 2)
    /// └─ 3 (mat 3)
    /// ```
    /// Pre-order visits 0,1,2,3, so `material_ids` must be `[0,1,2,3]`. A naive
    /// stack that pushed children in forward order would pop them reversed,
    /// yielding `[0,3,1,2]` — this pins the reverse-push that preserves order.
    #[test]
    fn walk_node_preserves_depth_first_preorder() {
        let json = r#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],
            "nodes":[
                {"mesh":0,"children":[1,3]},
                {"mesh":1,"children":[2]},
                {"mesh":2},
                {"mesh":3}],
            "meshes":[
                {"primitives":[{"attributes":{"POSITION":0},"material":0,"mode":4}]},
                {"primitives":[{"attributes":{"POSITION":0},"material":1,"mode":4}]},
                {"primitives":[{"attributes":{"POSITION":0},"material":2,"mode":4}]},
                {"primitives":[{"attributes":{"POSITION":0},"material":3,"mode":4}]}],
            "materials":[
                {"pbrMetallicRoughness":{"baseColorFactor":[0.0,0.0,0.0,1.0]}},
                {"pbrMetallicRoughness":{"baseColorFactor":[0.0,0.0,0.0,1.0]}},
                {"pbrMetallicRoughness":{"baseColorFactor":[0.0,0.0,0.0,1.0]}},
                {"pbrMetallicRoughness":{"baseColorFactor":[0.0,0.0,0.0,1.0]}}],
            "accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0.0,0.0,0.0],"max":[1.0,1.0,0.0]}],
            "bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36}],
            "buffers":[{"byteLength":36}]}"#;
        let glb = make_glb(json, &triangle_bin());
        let mesh = load_gltf_slice(&glb).expect("multi-node GLB must load");
        assert_eq!(
            mesh.material_ids,
            Some(vec![0, 1, 2, 3]),
            "emission order must be depth-first pre-order"
        );
    }

    /// `sanitize_cutoff` clamps into `[0,1]` and maps non-finite to the spec
    /// default `0.5` (NaN cannot be expressed in JSON, so this pure-fn test is
    /// the only place that path is reachable).
    #[test]
    fn sanitize_cutoff_clamps_and_defaults() {
        let approx = |a: f32, b: f32| (a - b).abs() < 1e-9;
        assert!(approx(sanitize_cutoff(0.25), 0.25), "in-range passes");
        assert!(approx(sanitize_cutoff(-1.0), 0.0), "negative clamps to 0");
        assert!(approx(sanitize_cutoff(5.0), 1.0), "above 1 clamps to 1");
        assert!(approx(sanitize_cutoff(f32::NAN), 0.5), "NaN → default 0.5");
        assert!(
            approx(sanitize_cutoff(f32::INFINITY), 0.5),
            "Inf → default 0.5"
        );
    }

    /// A MASK material declaring an out-of-range `alphaCutoff` (5.0) loads with
    /// the cutoff clamped to 1.0 — end-to-end through the public loader, so the
    /// cull never sees a threshold that would drop every MASK voxel.
    #[test]
    fn loader_clamps_out_of_range_alpha_cutoff() {
        let json = r#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"material":0,"mode":4}]}],"materials":[{"pbrMetallicRoughness":{"baseColorFactor":[1.0,1.0,1.0,1.0]},"alphaMode":"MASK","alphaCutoff":5.0}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0.0,0.0,0.0],"max":[1.0,1.0,0.0]}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36}],"buffers":[{"byteLength":36}]}"#;
        let glb = make_glb(json, &triangle_bin());
        let mesh = load_gltf_slice(&glb).expect("MASK GLB must load");
        let app = mesh.appearance.expect("materials present");
        assert_eq!(app.materials[0].alpha_mode, AlphaMode::Mask);
        assert!(
            (app.materials[0].alpha_cutoff - 1.0).abs() < 1e-6,
            "cutoff {} must clamp to 1.0",
            app.materials[0].alpha_cutoff
        );
    }

    #[test]
    fn rejects_garbage_bytes() {
        let err = load_gltf_slice(b"not a gltf").unwrap_err();
        assert!(
            matches!(err, VoxelizerError::MeshLoad(_)),
            "garbage bytes must yield MeshLoad, got {err:?}"
        );
    }

    /// A triangle with `TEXCOORD_0` and a base-colour-factor material must surface
    /// per-triangle UVs (aligned to triangles) and an appearance with that
    /// material's linear factor — the P1b extraction contract (textured path is
    /// the same code, just with a texture index instead of `None`).
    #[test]
    fn extracts_uvs_and_material_factor() {
        // BIN: 3 positions (36 B) then 3 UVs (24 B).
        let mut bin = triangle_bin();
        for uv in [[0.0_f32, 0.0], [1.0, 0.0], [0.0, 1.0]] {
            bin.extend(uv.iter().flat_map(|f| f.to_le_bytes()));
        }
        let json = r#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"TEXCOORD_0":1},"material":0,"mode":4}]}],"materials":[{"pbrMetallicRoughness":{"baseColorFactor":[0.5,0.25,0.75,1.0]},"alphaMode":"MASK","alphaCutoff":0.25}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0.0,0.0,0.0],"max":[1.0,1.0,0.0]},{"bufferView":1,"componentType":5126,"count":3,"type":"VEC2","min":[0.0,0.0],"max":[1.0,1.0]}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":24}],"buffers":[{"byteLength":60}]}"#;
        let glb = make_glb(json, &bin);
        let mesh = load_gltf_slice(&glb).expect("textured-triangle GLB must load");

        assert_eq!(mesh.material_ids, Some(vec![0]), "material 0, one triangle");

        // UVs aligned to the single triangle, in vertex order.
        let uvs = mesh.uvs.expect("UVs must be extracted");
        assert_eq!(uvs.len(), 1, "one UV-triple per triangle");
        assert!((uvs[0][0] - Vec2::new(0.0, 0.0)).length() < 1e-6);
        assert!((uvs[0][1] - Vec2::new(1.0, 0.0)).length() < 1e-6);
        assert!((uvs[0][2] - Vec2::new(0.0, 1.0)).length() < 1e-6);

        // Appearance: one material def, the linear factor, no texture.
        let app = mesh.appearance.expect("appearance must be present");
        assert_eq!(app.materials.len(), 1);
        for (got, want) in app.materials[0]
            .base_color_factor
            .iter()
            .zip([0.5, 0.25, 0.75, 1.0])
        {
            assert!((got - want).abs() < 1e-6, "factor {got} vs {want}");
        }
        assert_eq!(app.materials[0].base_color_texture, None);
        assert!(app.textures.is_empty(), "no embedded images");
        // alphaMode / alphaCutoff are faithfully extracted.
        assert_eq!(app.materials[0].alpha_mode, AlphaMode::Mask);
        assert!((app.materials[0].alpha_cutoff - 0.25).abs() < 1e-6);
    }

    /// Real-asset smoke: `littlest-tokyo.glb` must load with per-triangle UVs and a
    /// populated appearance whose textures actually decode (exercises the real PNG
    /// `decode_texture` format path, which the synthetic GLB above does not). Run:
    /// `cargo test -p voxelizer --lib loads_littlest_tokyo_appearance -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs models/littlest-tokyo.glb; run manually with --ignored --nocapture"]
    fn loads_littlest_tokyo_appearance() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/littlest-tokyo.glb"
        );
        let mesh = load_gltf_path(path).expect("littlest-tokyo must load");
        let uvs = mesh.uvs.as_ref().expect("real asset has UVs");
        assert_eq!(uvs.len(), mesh.triangles.len(), "UVs aligned to triangles");
        let app = mesh.appearance.as_ref().expect("real asset has materials");
        let textured = app
            .materials
            .iter()
            .filter(|m| m.base_color_texture.is_some())
            .count();
        let total_texels: usize = app.textures.iter().map(|t| t.rgba().len()).sum();
        eprintln!(
            "littlest-tokyo: {} tris, {} materials ({textured} textured), {} textures, {total_texels} texels",
            mesh.triangles.len(),
            app.materials.len(),
            app.textures.len(),
        );
        assert!(
            !app.textures.is_empty(),
            "photographic asset must have textures"
        );
        assert!(textured > 0, "at least one material references a texture");
        // Every decoded texture must be non-empty (the format path produced texels).
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
}
