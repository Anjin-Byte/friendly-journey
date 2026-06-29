//! Sparse material-bridge differential (`docs/materials/09-sparse-material-bridge.md`).
//!
//! The **B5 acceptance gate**: the sparse path
//! (`material_table_for_sparse` → `compact_surface_sparse` → `tree_from_compact`)
//! must produce the same materials + colour table as the dense path
//! (`voxelize_surface` → `to_sparse_tree` → `apply_mesh_materials`) at a small
//! resolution — proving the id space (B1) and the magenta MISSING sentinel (B2)
//! survive the GPU compact path. The existing occupancy-only differentials cannot
//! see a scrambled palette, which is why this exists. GPU-gated like
//! `differential.rs` (skips with no adapter unless `VOXEL_REQUIRE_GPU=1`).

// The 2048³ gate's material histogram prints percentages; `usize as f64` loses
// precision above 2^52 voxels, which is unreachable and irrelevant to a diagnostic.
#![allow(clippy::cast_precision_loss)]

use glam::Vec3;
use voxel_core::morton::encode_brick;
use voxel_core::palette::STRIDE_W;
use voxel_core::{SchoolBBuffer, read_slot};
use voxelizer::{
    CompactVoxel, GpuVoxelizer, GpuVoxelizerConfig, MeshInput, Resolution, TileSpec, VoxelCoord,
    VoxelGrid, VoxelizeGpuError, VoxelizeOpts, apply_mesh_materials, bake_leaf_colors, load_mesh,
    material_table_for_sparse, tree_from_compact,
};

fn gpu_or_skip() -> Option<GpuVoxelizer> {
    match pollster::block_on(GpuVoxelizer::new_standalone(GpuVoxelizerConfig::default())) {
        Ok(v) => Some(v),
        Err(VoxelizeGpuError::NoAdapter) => {
            assert!(
                std::env::var_os("VOXEL_REQUIRE_GPU").is_none(),
                "VOXEL_REQUIRE_GPU set but no GPU adapter present"
            );
            eprintln!("skip: no GPU adapter present");
            None
        }
        Err(e) => panic!("GPU init failed (not NoAdapter): {e}"),
    }
}

/// Pushes the 12 triangles of the axis-aligned box `[min, max]`, all carrying
/// material id `mat`.
fn push_box(tris: &mut Vec<[Vec3; 3]>, mids: &mut Vec<u32>, min: Vec3, max: Vec3, mat: u32) {
    let v = Vec3::new;
    let (x0, y0, z0, x1, y1, z1) = (min.x, min.y, min.z, max.x, max.y, max.z);
    // Six faces, each a quad (a,b,c,d) → two triangles (a,b,c),(a,c,d).
    let faces = [
        [v(x0, y0, z0), v(x0, y1, z0), v(x0, y1, z1), v(x0, y0, z1)], // -X
        [v(x1, y0, z0), v(x1, y1, z0), v(x1, y1, z1), v(x1, y0, z1)], // +X
        [v(x0, y0, z0), v(x1, y0, z0), v(x1, y0, z1), v(x0, y0, z1)], // -Y
        [v(x0, y1, z0), v(x1, y1, z0), v(x1, y1, z1), v(x0, y1, z1)], // +Y
        [v(x0, y0, z0), v(x1, y0, z0), v(x1, y1, z0), v(x0, y1, z0)], // -Z
        [v(x0, y0, z1), v(x1, y0, z1), v(x1, y1, z1), v(x0, y1, z1)], // +Z
    ];
    for f in faces {
        tris.push([f[0], f[1], f[2]]);
        tris.push([f[0], f[2], f[3]]);
        mids.push(mat);
        mids.push(mat);
    }
}

#[test]
fn sparse_bridge_matches_dense_path() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };

    // Three spatially-separated boxes so each occupied voxel has exactly ONE
    // possible material (robust to owner tie-break differences between the dense
    // and sparse voxelizers): mat 7, mat 13, and mat u32::MAX (no material → must
    // render occupied-but-global-0 magenta, exercising B2/Hole 1).
    let mut tris = Vec::new();
    let mut mids = Vec::new();
    push_box(
        &mut tris,
        &mut mids,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(6.0, 6.0, 6.0),
        7,
    );
    push_box(
        &mut tris,
        &mut mids,
        Vec3::new(11.0, 0.0, 0.0),
        Vec3::new(17.0, 6.0, 6.0),
        13,
    );
    push_box(
        &mut tris,
        &mut mids,
        Vec3::new(22.0, 0.0, 0.0),
        Vec3::new(28.0, 6.0, 6.0),
        u32::MAX,
    );
    let mesh = MeshInput {
        triangles: tris,
        material_ids: Some(mids),
        uvs: None,
        appearance: None,
    };

    let r = Resolution::new(32).unwrap();
    let grid = VoxelGrid::fit_mesh(r, &mesh, 1.0);
    let tiles = TileSpec::new([4, 4, 4], grid.dims()).unwrap();
    let opts = VoxelizeOpts {
        store_owner: true,
        ..Default::default()
    };

    // DENSE reference path.
    let dense_out = pollster::block_on(gpu.voxelize_surface(&mesh, &grid, &tiles, &opts)).unwrap();
    let mut dense_tree = dense_out.occupancy.to_sparse_tree();
    let dense_table = apply_mesh_materials(&mut dense_tree, &dense_out, &mesh).unwrap();

    // SPARSE bridge path.
    let (sparse_table, packed) = material_table_for_sparse(&mesh).unwrap();
    let voxels =
        pollster::block_on(gpu.compact_surface_sparse(&mesh, &grid, &opts, &packed, [0, 0, 0]))
            .unwrap();
    let (sparse_tree, dropped) = tree_from_compact(r, &voxels);

    assert_eq!(dropped, 0, "no voxel should fall outside the fitted grid");
    // The colour table is built by the shared `build_global_table`, so the two
    // paths MUST agree byte-for-byte (magenta + the 2 real materials; the
    // u32::MAX box contributes no table entry).
    assert_eq!(
        sparse_table.words(),
        dense_table.words(),
        "colour table mismatch between sparse and dense"
    );
    assert_eq!(sparse_table.words().len(), 3, "magenta + 2 real materials");

    // Per-voxel: materials must agree wherever both paths mark the voxel occupied
    // (the decisive B1/B2 check), and occupancy should match exactly (both
    // voxelizers are bit-exact vs the CPU oracle).
    let n = r.voxels_per_axis();
    let (mut occ_mismatch, mut mat_mismatch, mut both_occ) = (0u32, 0u32, 0u32);
    let (mut seen1, mut seen2, mut seen0_occ) = (false, false, false);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let c = VoxelCoord::new(x, y, z);
                let (so, dnso) = (sparse_tree.is_occupied(c), dense_tree.is_occupied(c));
                if so != dnso {
                    occ_mismatch += 1;
                }
                if so && dnso {
                    both_occ += 1;
                    if sparse_tree.material_at(c) != dense_tree.material_at(c) {
                        mat_mismatch += 1;
                    }
                }
                if so {
                    match sparse_tree.material_at(c) {
                        1 => seen1 = true,
                        2 => seen2 = true,
                        0 => seen0_occ = true,
                        _ => {}
                    }
                }
            }
        }
    }

    assert!(
        both_occ > 50,
        "test should share many occupied voxels, got {both_occ}"
    );
    assert_eq!(
        mat_mismatch, 0,
        "material disagreement on {mat_mismatch}/{both_occ} shared voxels (B1/B2)"
    );
    assert_eq!(
        occ_mismatch, 0,
        "occupancy disagreement on {occ_mismatch} voxels (sparse vs dense voxelizer)"
    );
    assert!(seen1 && seen2, "both real materials must render");
    assert!(
        seen0_occ,
        "the no-material box must render occupied-but-global-0 (magenta), not an aliased material"
    );
}

/// **The 2048³ hard gate** (docs/materials/09 §test 7): proves the sparse path
/// actually works at the target resolution — it has zero prior production callers
/// and zero 2048³ coverage. Runs the full bridge on `littlest-tokyo.glb` and
/// reports wall-time + peak `CompactVoxel` host RAM. `#[ignore]` (expensive);
/// run manually: `cargo test -p voxelizer --test sparse_material_bridge --
/// --ignored --nocapture` on a GPU.
#[test]
#[ignore = "2048³ validation+perf; run manually with --ignored --nocapture on a GPU"]
fn sparse_path_validates_at_2048() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    // `cargo test` runs with CWD = the crate dir, so resolve against the
    // workspace root (two levels up from this crate's manifest).
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/littlest-tokyo.glb"
    );
    let mesh = match load_mesh(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("skip: {path} not loadable: {e}");
            return;
        }
    };
    eprintln!("mesh: {} triangles", mesh.triangles.len());

    let r = Resolution::new(2048).unwrap();
    let grid = VoxelGrid::fit_mesh(r, &mesh, 2.0);
    let opts = VoxelizeOpts {
        store_owner: true,
        ..Default::default()
    };

    let (table, packed) = material_table_for_sparse(&mesh).unwrap();
    eprintln!("materials: {} colours", table.words().len());

    // DIAGNOSTIC: per-material voxel histogram. If the distribution collapses at
    // 2048³ (multi-chunk) vs 512³ (single chunk), the multi-chunk material path is
    // broken.
    let hist_print = |label: &str, vox: &[CompactVoxel]| {
        let mut hist = std::collections::HashMap::<u32, usize>::new();
        for v in vox {
            *hist.entry(v.material).or_default() += 1;
        }
        let total = vox.len().max(1);
        let mut top: Vec<(u32, usize)> = hist.iter().map(|(k, v)| (*k, *v)).collect();
        top.sort_by_key(|&(_, c)| std::cmp::Reverse(c));
        eprintln!(
            "--- {label}: {total} voxels, {} distinct materials ---",
            hist.len()
        );
        for (mat, count) in top.iter().take(8) {
            eprintln!(
                "    mat {mat:>3}: {:>5.1}%",
                100.0 * *count as f64 / total as f64
            );
        }
    };
    // 512³ reference (single chunk).
    {
        let dr = Resolution::new(512).unwrap();
        let dgrid = VoxelGrid::fit_mesh(dr, &mesh, 2.0);
        let dvox = pollster::block_on(gpu.compact_surface_sparse(
            &mesh,
            &dgrid,
            &opts,
            &packed,
            [0, 0, 0],
        ))
        .unwrap();
        hist_print("res 512", &dvox);
    }

    let t0 = std::time::Instant::now();
    let voxels =
        pollster::block_on(gpu.compact_surface_sparse(&mesh, &grid, &opts, &packed, [0, 0, 0]))
            .expect("compact_surface_sparse must complete at 2048³");
    let t_compact = t0.elapsed();
    hist_print("res 2048", &voxels);

    let t1 = std::time::Instant::now();
    let (tree, dropped) = tree_from_compact(r, &voxels);
    let t_assemble = t1.elapsed();

    let ram_mib = voxels.len() * std::mem::size_of::<CompactVoxel>() / (1024 * 1024);
    eprintln!("=== 2048³ sparse path ===");
    eprintln!("occupied voxels : {}", voxels.len());
    eprintln!("CompactVoxel RAM: {ram_mib} MiB");
    eprintln!("dropped (oob)   : {dropped}");
    eprintln!("leaves          : {}", tree.leaf_count());
    eprintln!("compact (GPU)   : {t_compact:.2?}");
    eprintln!("assemble (CPU)  : {t_assemble:.2?}");

    // Correctness invariants, not just "it ran" (impl-review M4):
    assert_eq!(
        dropped, 0,
        "padding=2 must keep all geometry inside the grid"
    );
    assert!(
        voxels.len() > 1_000_000,
        "expected millions of occupied voxels"
    );
    assert!(
        table.words().len() > 1,
        "the multi-material scene must build a non-trivial colour table"
    );
    // Re-bin completeness: every voxel lands in a leaf (≤512/leaf), none grossly
    // lost or duplicated. The per-chunk no-truncation invariant is the M1 tripwire
    // inside the compact dispatch (`raw <= max_entries`).
    let leaves = tree.leaf_count();
    assert!(
        leaves > 0 && voxels.len() <= leaves * 512 && voxels.len() >= leaves,
        "{} voxels inconsistent with {leaves} leaves (expected [leaves, leaves*512])",
        voxels.len()
    );
}

/// **The multi-material-leaf read-path oracle** (impl-review M2+M3): the
/// separated-box oracle above leaves every bridge leaf single-material, so
/// `pack_leaf` always takes the `bits == 0` uniform fast path and the
/// palette+bit-packed-index machinery (where a morton/straddle bug would live)
/// is never exercised on GPU-sourced data. This forces **two materials into one
/// 8³ leaf** and reads every occupied voxel back through `SchoolBBuffer` →
/// `read_slot` (the WGSL-mirrored decode), matching the dense path.
#[test]
fn multi_material_leaf_round_trips_through_read_slot() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };

    // Two abutting boxes filling ONE leaf (no shared voxel): mat 7 at x∈[0,3],
    // mat 13 at x∈[4,7], both y,z ∈ [0,7] ⇒ all in leaf (0,0,0).
    let mut tris = Vec::new();
    let mut mids = Vec::new();
    push_box(
        &mut tris,
        &mut mids,
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(3.0, 7.0, 7.0),
        7,
    );
    push_box(
        &mut tris,
        &mut mids,
        Vec3::new(4.0, 0.0, 0.0),
        Vec3::new(7.0, 7.0, 7.0),
        13,
    );
    let mesh = MeshInput {
        triangles: tris,
        material_ids: Some(mids),
        uvs: None,
        appearance: None,
    };

    let r = Resolution::new(32).unwrap();
    // Explicit 1:1 grid (world == voxel) so the boxes land where intended.
    let grid = VoxelGrid::new(r, Vec3::ZERO, 1.0);
    let opts = VoxelizeOpts {
        store_owner: true,
        ..Default::default()
    };

    // Dense reference.
    let tiles = TileSpec::new([4, 4, 4], grid.dims()).unwrap();
    let dense_out = pollster::block_on(gpu.voxelize_surface(&mesh, &grid, &tiles, &opts)).unwrap();
    let mut dense_tree = dense_out.occupancy.to_sparse_tree();
    apply_mesh_materials(&mut dense_tree, &dense_out, &mesh).unwrap();

    // Sparse bridge → leaf_mat (the actual GPU read path).
    let (_table, packed) = material_table_for_sparse(&mesh).unwrap();
    let voxels =
        pollster::block_on(gpu.compact_surface_sparse(&mesh, &grid, &opts, &packed, [0, 0, 0]))
            .unwrap();
    let (sparse_tree, dropped) = tree_from_compact(r, &voxels);
    assert_eq!(dropped, 0);
    let buf = SchoolBBuffer::from_sparse(&sparse_tree);

    // The test is only meaningful if SOME leaf is genuinely multi-material
    // (header bits_per_voxel ≥ 1), i.e. not the uniform fast path.
    let multi = buf
        .leaf_mat_words()
        .chunks_exact(STRIDE_W)
        .any(|slot| (slot[0] & 0xF) >= 1);
    assert!(
        multi,
        "expected a multi-material leaf (bits>=1); test would be vacuous"
    );

    // Read every occupied voxel back through the leaf_mat slot and match dense.
    let n = r.voxels_per_axis();
    let mut checked = 0u32;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let c = VoxelCoord::new(x, y, z);
                if !sparse_tree.is_occupied(c) {
                    continue;
                }
                let slot = sparse_tree.leaf_slot_of(c).expect("occupied ⇒ a leaf") as usize;
                let words = &buf.leaf_mat_words()[slot * STRIDE_W..slot * STRIDE_W + STRIDE_W];
                let got = read_slot(words, encode_brick(x & 7, y & 7, z & 7));
                assert_eq!(got, dense_tree.material_at(c), "read_slot vs dense @ {c:?}");
                checked += 1;
            }
        }
    }
    assert!(checked > 20, "expected occupied voxels, got {checked}");
}

/// **The viewer's `--truecolor` path, headless** (`docs/materials/11`, P5): replicate
/// `voxel-viewer::build_from_mesh`'s truecolor branch on `littlest-tokyo.glb` — GPU
/// voxelize → `tree_from_compact` → `SchoolBBuffer::from_sparse` → `bake_leaf_colors`
/// — and prove the per-voxel bake samples **diverse real texels** (not flat / magenta),
/// i.e. the user will see textures. `#[ignore]` (loads the 22 MB asset + GPU); run:
/// `cargo test -p voxelizer --test sparse_material_bridge truecolor_bakes_littlest_tokyo
/// -- --ignored --nocapture` on a GPU.
#[test]
#[ignore = "real-asset truecolor bake; run manually with --ignored --nocapture on a GPU"]
fn truecolor_bakes_littlest_tokyo_textures() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    // Defaults to a clean (non-Draco, PNG/JPEG-textured) three.js asset; override
    // with TRUECOLOR_MODEL=/abs/path.glb. Note: Draco-compressed geometry
    // (KHR_draco_mesh_compression) and WebP/KTX2 textures (EXT_texture_webp /
    // KHR_texture_basisu) are NOT decoded by our loader and will fail to load.
    let default_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../models/gltf/readyplayer.me.glb"
    );
    let path = std::env::var("TRUECOLOR_MODEL").unwrap_or_else(|_| default_path.to_string());
    let mesh = load_mesh(&path).expect("model must load");
    eprintln!("baking {path}");
    assert!(mesh.appearance.is_some(), "the asset must carry textures");

    let r = Resolution::new(512).unwrap();
    let grid = VoxelGrid::fit_mesh(r, &mesh, 2.0);
    let opts = VoxelizeOpts {
        epsilon: 1e-4,
        store_owner: true,
        store_color: false,
    };
    let (_table, packed) = material_table_for_sparse(&mesh).unwrap();
    let voxels =
        pollster::block_on(gpu.compact_surface_sparse(&mesh, &grid, &opts, &packed, [0, 0, 0]))
            .unwrap();
    let (tree, _dropped) = tree_from_compact(r, &voxels);
    let mut structure = SchoolBBuffer::from_sparse(&tree);

    let t = std::time::Instant::now();
    bake_leaf_colors(
        &mut structure,
        &tree,
        &mesh,
        &grid,
        opts.epsilon,
        Some(&packed),
    );
    let secs = t.elapsed().as_secs_f64();

    assert!(
        structure.has_leaf_color(),
        "the bake must populate leaf_color"
    );
    let colors = structure.leaf_color_words();
    // The MISSING sentinel (a voxel with no candidate triangle in its brick).
    let magenta = colors.iter().filter(|&&c| c == 0xFFFF_00FF).count();
    let distinct: std::collections::HashSet<u32> = colors.iter().copied().collect();
    // Near-white = all of R,G,B > 230: a wrong UV flip washes the render to the
    // atlas's pale top tiles, so a low near-white fraction confirms the no-flip fix.
    let near_white = colors
        .iter()
        .filter(|&&c| {
            let [r, g, b, _] = c.to_le_bytes();
            r > 230 && g > 230 && b > 230
        })
        .count();
    let (mut sr, mut sg, mut sb) = (0u64, 0u64, 0u64);
    for &c in colors {
        let [r, g, b, _] = c.to_le_bytes();
        sr += u64::from(r);
        sg += u64::from(g);
        sb += u64::from(b);
    }
    let n = colors.len().max(1) as u64;
    eprintln!(
        "littlest-tokyo.obj @512³: {} colours in {secs:.1}s; {} distinct; {} magenta; \
         near-white {} ({:.1}%); mean RGB ({},{},{})",
        colors.len(),
        distinct.len(),
        magenta,
        near_white,
        100.0 * near_white as f64 / colors.len().max(1) as f64,
        sr / n,
        sg / n,
        sb / n,
    );
    assert!(
        (near_white as f64) < 0.5 * colors.len() as f64,
        "render is washing to white ({near_white} near-white) — the UV flip is wrong"
    );
    assert!(
        distinct.len() > 100,
        "a photographic asset must bake many distinct texel colours, got only {}",
        distinct.len()
    );
    assert!(
        (magenta as f64) < 0.5 * colors.len() as f64,
        "most occupied voxels must sample a real texel, not the magenta MISSING sentinel"
    );
}
