//! OBJ loader degrade behaviour: an undecodable `map_Kd` texture must not abort
//! the load — the mesh still loads with its UVs, just without an appearance
//! (truecolor silently falls back to the palette path). This regression-locks
//! the C3 degrade so a future change (e.g. widening the image formats, or making
//! the failure hard) is a deliberate, visible decision. The `image` dependency
//! is built with png+jpeg only, so a TGA `map_Kd` exercises the unsupported path.

#![cfg(feature = "obj")]

use std::path::PathBuf;
use voxelizer::load_obj_path;

/// Write `bytes` to `<CARGO_TARGET_TMPDIR>/<name>` and return the path.
fn write_temp(name: &str, bytes: &[u8]) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let path = dir.join(name);
    std::fs::write(&path, bytes).expect("write temp fixture");
    path
}

/// A minimal valid 1×1 uncompressed (BGR) TGA. The `image` dep is built without
/// the `tga` feature, so decoding this fails — exactly the silent-degrade path.
fn minimal_tga() -> Vec<u8> {
    let mut v = vec![
        0, // id length
        0, // no colour map
        2, // uncompressed true-colour
        0, 0, 0, 0, 0, // colour-map spec
        0, 0, 0, 0, // x/y origin
        1, 0, // width = 1
        1, 0,  // height = 1
        24, // 24 bpp
        0,  // descriptor
    ];
    v.extend_from_slice(&[0x10, 0x20, 0x30]); // one BGR pixel
    v
}

#[test]
fn obj_with_undecodable_map_kd_degrades_to_flat() {
    // Unique names so parallel test binaries don't collide in the shared tmpdir.
    write_temp("c3_degrade.tga", &minimal_tga());
    write_temp(
        "c3_degrade.mtl",
        b"newmtl m0\nKd 0.8 0.8 0.8\nmap_Kd c3_degrade.tga\n",
    );
    let obj = write_temp(
        "c3_degrade.obj",
        b"mtllib c3_degrade.mtl\nusemtl m0\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n",
    );

    let mesh = load_obj_path(&obj).expect("OBJ must still load despite the bad texture");
    assert_eq!(mesh.triangles.len(), 1, "geometry loads");
    assert!(mesh.uvs.is_some(), "UVs survive the texture-decode failure");
    assert!(
        mesh.appearance.is_none(),
        "an undecodable map_Kd degrades to flat (no appearance), not a hard error"
    );
}
