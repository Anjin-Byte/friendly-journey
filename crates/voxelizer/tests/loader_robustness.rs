//! Adversarial-input robustness of the public mesh loaders.
//!
//! These exercise the loader entry points as an external consumer would, with
//! deliberately hostile glTF documents (the *most* exposed external-input
//! surface in the crate — the viewer feeds arbitrary user files straight in).
//! The merge bar is **graceful error or correct load, never a process abort**.
//!
//! The headline case is a deeply-nested node hierarchy: glTF node depth is
//! exporter-/attacker-controlled and a deep single-child chain is legal per
//! spec, so a recursive scene walk overflows the stack and `SIGABRT`s the
//! process (uncatchable — `catch_unwind` does not intercept it). The loader
//! walks the hierarchy with an explicit heap work-stack, so arbitrary depth is
//! safe; this test pins that — were the walk ever to revert to recursion, the
//! depth here overflows a test-harness thread's stack and aborts the binary.

#![cfg(feature = "gltf")]
// GLB chunk lengths are small test fixtures; the usize→u32 casts cannot truncate.
#![allow(clippy::cast_possible_truncation)]

use voxelizer::{VoxelizerError, load_gltf_slice};

/// Assemble a minimal GLB container from a JSON chunk and a BIN chunk, per the
/// binary-glTF spec. Self-contained so this integration test needs no asset.
fn make_glb(json: &str, bin: &[u8]) -> Vec<u8> {
    let mut json_bytes = json.as_bytes().to_vec();
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(0x20); // JSON pads with spaces
    }
    let mut bin_bytes = bin.to_vec();
    while !bin_bytes.len().is_multiple_of(4) {
        bin_bytes.push(0x00); // BIN pads with zeros
    }
    let total_len = 12 + 8 + json_bytes.len() + 8 + bin_bytes.len();
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&0x4654_6C67_u32.to_le_bytes()); // "glTF"
    out.extend_from_slice(&2_u32.to_le_bytes()); // version 2
    out.extend_from_slice(&(total_len as u32).to_le_bytes());
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes()); // "JSON"
    out.extend_from_slice(&json_bytes);
    out.extend_from_slice(&(bin_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x004E_4942_u32.to_le_bytes()); // "BIN\0"
    out.extend_from_slice(&bin_bytes);
    out
}

/// The three vertices of one triangle, as little-endian f32 bytes.
fn triangle_bin() -> Vec<u8> {
    [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

/// A GLB whose scene is a single chain of `depth` nodes — node `i` is the only
/// child of node `i-1`; the deepest node carries the one triangle. Loading it
/// forces the scene walk to descend `depth` levels.
fn deep_chain_glb(depth: usize) -> Vec<u8> {
    let nodes = (0..depth)
        .map(|i| {
            if i == depth - 1 {
                r#"{"mesh":0}"#.to_string()
            } else {
                format!(r#"{{"children":[{}]}}"#, i + 1)
            }
        })
        .collect::<Vec<_>>()
        .join(",");
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{nodes}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"mode":4}}]}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0.0,0.0,0.0],"max":[1.0,1.0,0.0]}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":36}}],"buffers":[{{"byteLength":36}}]}}"#
    );
    make_glb(&json, &triangle_bin())
}

/// A 20 000-deep node chain loads without aborting the process and passes the
/// single buried triangle through. Recursing once per level overflows a
/// test-thread stack well before this depth (reproduced ~2 000 on the 2 MiB
/// harness stack) and `SIGABRT`s the whole binary — so this is a sharp guard
/// against any regression back to a recursive walk.
#[test]
fn deep_node_chain_loads_without_stack_overflow() {
    let glb = deep_chain_glb(20_000);
    let mesh = load_gltf_slice(&glb).expect("deep-chain GLB must load, not abort");
    assert_eq!(
        mesh.triangles.len(),
        1,
        "the single buried triangle survives the deep walk"
    );
}

/// A GLB with one triangle whose `TEXCOORD_0` carries a NaN component (finite
/// positions, non-finite UV). The UV accessor declares no min/max, so the gltf
/// importer accepts it and the non-finite UV reaches our validation.
fn nan_uv_glb() -> Vec<u8> {
    let mut bin = triangle_bin(); // 3 positions, 36 B
    for uv in [[f32::NAN, 0.0_f32], [1.0, 0.0], [0.0, 1.0]] {
        bin.extend(uv.iter().flat_map(|f| f.to_le_bytes()));
    }
    let json = r#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0,"TEXCOORD_0":1},"mode":4}]}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0.0,0.0,0.0],"max":[1.0,1.0,0.0]},{"bufferView":1,"componentType":5126,"count":3,"type":"VEC2"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":24}],"buffers":[{"byteLength":60}]}"#;
    make_glb(json, &bin)
}

/// A non-finite UV must be rejected as a graceful `MeshLoad`, not silently baked
/// into a garbage texel sample. Pins B1 end-to-end through the public loader
/// (the OBJ/STL loaders share the same `MeshInput::validate`).
#[test]
fn non_finite_uv_is_rejected_by_the_loader() {
    let err = load_gltf_slice(&nan_uv_glb()).unwrap_err();
    assert!(
        matches!(err, VoxelizerError::MeshLoad(_)),
        "a NaN UV must yield MeshLoad, got {err:?}"
    );
}
