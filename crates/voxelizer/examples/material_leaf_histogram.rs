//! TEMPORARY measurement harness (safe to delete).
//!
//! Measures the per-leaf DISTINCT-material distribution P(leaf) under several
//! synthetic material-assignment schemes, then prints the total leaf-storage
//! bytes under SEPARATE / UNIFIED / GLOBAL palette models. Drives the
//! separate-vs-unified memory verdict for palette compression.
//!
//! Run:  cargo run -p voxelizer --release --example material_leaf_histogram
//!
//! Extended post-materials-impl with a fixed-stride-cap sweep (`cap_sweep`) that
//! models the SHIPPED `leaf_mat` footprint at each candidate leaf bit-width and
//! the spill arena — the data behind the `P_cap` decision.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::doc_lazy_continuation,
    clippy::uninlined_format_args,
    clippy::manual_div_ceil
)]

use std::collections::HashMap;

use glam::Vec3;
use voxel_core::{OccupancyField, Resolution, VoxelCoord, fixtures::NoiseField};
use voxelizer::{
    MeshInput,
    core::{TileSpec, VoxelGrid, VoxelizeOpts},
    load_mesh,
    reference_cpu::voxelize_surface_cpu,
    rotation_degrees,
};

const LEAF: u32 = 8; // 8^3 = 512 voxels per leaf

// ---------- material schemes ----------

/// Numerical-Recipes LCG hash, used for the hash(owner)%K scheme.
fn hash32(mut x: u32) -> u32 {
    x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    x ^= x >> 16;
    x = x.wrapping_mul(2_246_822_519);
    x ^= x >> 13;
    x
}

/// A per-(voxel,scheme) material id (>=1; 0 reserved for "empty").
/// owner = winning triangle index (u32::MAX never reaches here).
fn material_id(scheme: Scheme, owner: u32, x: u32, y: u32, z: u32, n: u32) -> u32 {
    match scheme {
        Scheme::HashOwner(k) => 1 + (hash32(owner) % k),
        // Spatially COHERENT: material chosen by which coarse region the voxel
        // sits in. Region grid is 4x4x4 (=64 regions) over the whole model:
        // a "wall is one material" model. K=64 distinct global materials.
        Scheme::CoherentRegion => {
            let r = n / 4; // region edge in voxels
            let r = r.max(1);
            let rx = x / r;
            let ry = y / r;
            let rz = z / r;
            1 + (rx + 4 * (ry + 4 * rz)).min(63)
        }
        // Spatially coarse-COHERENT: only 8 octants (very few region boundaries
        // cut through any single 8^3 leaf -> most leaves single-material).
        Scheme::CoherentOctant => {
            let h = n / 2;
            let ox = u32::from(x >= h);
            let oy = u32::from(y >= h);
            let oz = u32::from(z >= h);
            1 + (ox + 2 * (oy + 2 * oz))
        }
        // Spatially coherent but with region boundaries DELIBERATELY off the 8^3
        // leaf grid (slab pitch=11 voxels, +3 phase) so seams cut through leaves —
        // the realistic "walls/floors at arbitrary placement" case. Only leaves
        // straddling a seam see P>1.
        Scheme::CoherentSlab => {
            let pitch = 11u32;
            let phase = 3u32;
            let sx = (x + phase) / pitch;
            let sy = (y + phase) / pitch;
            let sz = (z + phase) / pitch;
            1 + ((sx + 7 * sy + 53 * sz) % 24)
        }
        // RANDOM per voxel (worst case): K distinct, hashed off the position.
        Scheme::RandomVoxel(k) => 1 + (hash32(x ^ hash32(y ^ hash32(z))) % k),
    }
}

#[derive(Clone, Copy)]
enum Scheme {
    HashOwner(u32),
    CoherentRegion,
    CoherentOctant,
    CoherentSlab,
    RandomVoxel(u32),
}

impl Scheme {
    fn label(self) -> String {
        match self {
            Scheme::HashOwner(k) => format!("hash(owner)%{k}"),
            Scheme::CoherentRegion => "coherent-region(4^3=64)".to_string(),
            Scheme::CoherentOctant => "coherent-octant(8)".to_string(),
            Scheme::CoherentSlab => "coherent-slab(pitch11)".to_string(),
            Scheme::RandomVoxel(k) => format!("random-voxel%{k}"),
        }
    }
}

// ---------- byte-cost models ----------

fn bits_for(values: u32) -> u32 {
    // ceil(log2(values)), min 1 if values>0. values is the count of code points.
    if values <= 1 {
        // separate P=1 => 0 bits (no index array). caller handles P>=1 framing.
        0
    } else {
        32 - (values - 1).leading_zeros()
    }
}

/// Per-leaf bytes under SEPARATE: 64B bitmask + palette(P*u16) + index array.
/// bits = ceil(log2 P); P=1 => 0 bits => no index array.
fn bytes_separate(p: u32) -> u64 {
    let bits = bits_for(p); // P=1 -> 0
    let index_bytes = ((512u64 * bits as u64) + 7) / 8;
    64 + (p as u64) * 2 + index_bytes
}

/// Per-leaf bytes under UNIFIED: NO bitmask; palette includes EMPTY at index 0.
/// bits = ceil(log2(P+1)); index array covers all 512 voxels.
fn bytes_unified(p: u32) -> u64 {
    let codepoints = p + 1; // +1 empty
    let bits = bits_for(codepoints).max(1); // P+1>=2 always so >=1
    let index_bytes = ((512u64 * bits as u64) + 7) / 8;
    (codepoints as u64) * 2 + index_bytes
}

/// Per-leaf bytes under GLOBAL: 64B bitmask kept? In the global model the
/// classic form drops the per-leaf palette but every occupied voxel pays a
/// fixed global index width. We model the *occupancy-preserving* global variant
/// used for traversal: keep the 64B bitmask (occupancy is free for traversal)
/// + a global index per OCCUPIED voxel at width = ceil(log2(global_mats)).
/// occ = occupied voxel count in this leaf.
fn bytes_global(occ: u32, global_bits: u32) -> u64 {
    let index_bytes = ((occ as u64 * global_bits as u64) + 7) / 8;
    64 + index_bytes
}

// ---------- fixed-stride-cap model (the SHIPPED leaf_mat layout) ----------
//
// This is what we actually store: a fixed-stride slot per leaf sized for `cap`
// distinct materials, REGARDLESS of the leaf's real P. A leaf with P > cap
// SPILLS its full (variable-width) palette+index into a growable arena.
//
// Mirrors crates/voxel-core/src/palette.rs:
//   header   = 1 u32 (4 B)
//   palette  = cap u16 entries           (2*cap B)
//   index    = 512 voxels * bits          (64*bits B, bits = ceil(log2 cap))
//   rounded up to a u32 multiple.
// At cap=16 this is 4 + 32 + 256 = 292 B = STRIDE_W(73)*4. (Material-only:
// occupancy is the separate 64 B `leaves` buffer, not counted here.)

/// Fixed-stride per-leaf `leaf_mat` slot bytes for an inline cap of `cap`
/// distinct materials. Paid by EVERY leaf, whatever its real P.
fn cap_slot_bytes(cap: u32) -> u64 {
    let bits = bits_for(cap); // ceil(log2 cap): cap 16 -> 4
    let header = 4u64;
    let palette = u64::from(cap) * 2;
    let index = (512u64 * u64::from(bits)).div_ceil(8);
    (header + palette + index).div_ceil(4) * 4 // round to u32 multiple
}

/// Material-only minimum for a leaf of `p` materials at its TRUE (variable)
/// width — the per-leaf cost a "bit-width pools" allocator would pay (no fixed
/// stride waste). The theoretical floor for the SEPARATE scheme's material side.
fn cap_min_bytes(p: u32) -> u64 {
    if p == 0 {
        return 0;
    }
    let bits = bits_for(p); // P=1 -> 0 (no index array)
    let index = (512u64 * u64::from(bits)).div_ceil(8);
    4 + u64::from(p) * 2 + index // header + u16 palette + index
}

/// Spill-arena bytes for ONE leaf of `p` materials (only when `p > cap`): its
/// full palette+index at true width, plus a small `{offset,len}` descriptor.
fn cap_arena_bytes(p: u32) -> u64 {
    cap_min_bytes(p) + 8 // + arena descriptor
}

/// Prints the fixed-stride-cap memory sweep for a measured per-leaf P map:
/// for each candidate cap (bit width), the inline `leaf_mat` footprint, the
/// spill count + arena, the grand total, and the ratios vs the occupancy buffer
/// (64 B/leaf) and vs the variable-width pools floor.
fn cap_sweep(label: &str, map: &HashMap<u64, (std::collections::HashSet<u32>, u32)>) {
    let leaves: u64 = map.values().filter(|(m, _)| !m.is_empty()).count() as u64;
    if leaves == 0 {
        return;
    }
    let occ_buf = 64 * leaves; // the separate occupancy buffer (unchanged)
    let pools_floor: u64 = map
        .values()
        .map(|(m, _)| cap_min_bytes(m.len() as u32))
        .sum();
    let mib = |b: u64| b as f64 / (1024.0 * 1024.0);

    println!(
        "    cap-sweep [{label}]  ({leaves} leaves; occupancy buffer = {:.2} MiB; pools-floor leaf_mat = {:.2} MiB)",
        mib(occ_buf),
        mib(pools_floor)
    );
    println!(
        "      bits  cap   slot_B   leaf_mat(MiB)   spill_leaves      arena(MiB)   total(MiB)   ×occ   ×pools"
    );
    for bits in 2u32..=8 {
        let cap = 1u32 << bits;
        let slot = cap_slot_bytes(cap);
        let inline = leaves * slot;
        let (mut spill_leaves, mut arena) = (0u64, 0u64);
        for (m, _) in map.values() {
            let p = m.len() as u32;
            if p > cap {
                spill_leaves += 1;
                arena += cap_arena_bytes(p);
            }
        }
        let total = inline + arena;
        let spill_pct = 100.0 * spill_leaves as f64 / leaves as f64;
        let marker = if bits == 4 { " <-- current" } else { "" };
        println!(
            "      {bits:>4}  {cap:>4}   {slot:>6}   {:>10.2}      {spill_leaves:>6} ({spill_pct:4.1}%)   {:>10.3}   {:>9.2}   {:>4.1}   {:>5.2}{marker}",
            mib(inline),
            mib(arena),
            mib(total),
            total as f64 / occ_buf as f64,
            total as f64 / pools_floor.max(1) as f64,
        );
    }
}

// ---------- the measurement core ----------

struct LeafStats {
    leaves: u64,
    // histogram buckets: P=1,2,3,4,5..8,9..16,17+
    hist: [u64; 7],
    sum_p: u64,
    max_p: u32,
    sep: u64,
    uni: u64,
    glob: u64,
    occ_total: u64,
}

fn bucket(p: u32) -> usize {
    match p {
        1 => 0,
        2 => 1,
        3 => 2,
        4 => 3,
        5..=8 => 4,
        9..=16 => 5,
        _ => 6,
    }
}

/// occ_by_leaf: leaf_key -> (set of distinct materials, occupied voxel count)
fn analyze(
    occ_by_leaf: &HashMap<u64, (std::collections::HashSet<u32>, u32)>,
    global_bits: u32,
) -> LeafStats {
    let mut s = LeafStats {
        leaves: 0,
        hist: [0; 7],
        sum_p: 0,
        max_p: 0,
        sep: 0,
        uni: 0,
        glob: 0,
        occ_total: 0,
    };
    for (mats, occ) in occ_by_leaf.values() {
        let p = mats.len() as u32;
        if p == 0 {
            continue;
        }
        s.leaves += 1;
        s.hist[bucket(p)] += 1;
        s.sum_p += p as u64;
        s.max_p = s.max_p.max(p);
        s.occ_total += *occ as u64;
        s.sep += bytes_separate(p);
        s.uni += bytes_unified(p);
        s.glob += bytes_global(*occ, global_bits);
    }
    s
}

fn report(scheme_label: &str, s: &LeafStats, global_label: &str) {
    let l = s.leaves.max(1) as f64;
    let frac = |i: usize| 100.0 * s.hist[i] as f64 / l;
    println!(
        "  {:<26} leaves={:<6} meanP={:.3} maxP={:<3} | P=1:{:5.1}% 2:{:5.1}% 3:{:5.1}% 4:{:5.1}% 5-8:{:5.1}% 9-16:{:5.1}% 17+:{:5.1}%",
        scheme_label,
        s.leaves,
        s.sum_p as f64 / l,
        s.max_p,
        frac(0),
        frac(1),
        frac(2),
        frac(3),
        frac(4),
        frac(5),
        frac(6),
    );
    let kib = |b: u64| b as f64 / 1024.0;
    let baseline = 64 * s.leaves; // occupancy-only bitmask bytes
    println!(
        "      bytes: separate={:>9.1}KiB  unified={:>9.1}KiB  global[{}]={:>9.1}KiB   (occ-only bitmask baseline={:.1}KiB)",
        kib(s.sep),
        kib(s.uni),
        global_label,
        kib(s.glob),
        kib(baseline),
    );
    let winner = if s.sep <= s.uni && s.sep <= s.glob {
        "SEPARATE"
    } else if s.uni <= s.glob {
        "UNIFIED"
    } else {
        "GLOBAL"
    };
    println!(
        "      => winner: {}   (unified/separate = {:.3}, unified saves {:+.1}KiB vs separate)",
        winner,
        s.uni as f64 / s.sep as f64,
        kib(s.sep) - kib(s.uni),
    );
}

fn leaf_key(x: u32, y: u32, z: u32) -> u64 {
    let lx = (x / LEAF) as u64;
    let ly = (y / LEAF) as u64;
    let lz = (z / LEAF) as u64;
    lx | (ly << 21) | (lz << 42)
}

fn run_mesh(path: &str, res: u32) {
    let mut mesh: MeshInput = match load_mesh(path) {
        Ok(m) => m,
        Err(e) => {
            println!("SKIP {path} @res{res}: load failed: {e:?}");
            return;
        }
    };
    mesh.transform(rotation_degrees(-90.0, 0.0, 0.0));
    let resolution = Resolution::new(res).unwrap();
    let grid = VoxelGrid::fit_mesh(resolution, &mesh, 1.0);
    let tiles = TileSpec::new([8, 8, 8], grid.dims()).unwrap();
    let opts = VoxelizeOpts {
        store_owner: true,
        ..Default::default()
    };
    let out = voxelize_surface_cpu(&mesh, &grid, &tiles, &opts);
    let owner = out.owner_id.expect("owner");
    let n = res;
    println!(
        "\n=== MESH {path} @ res {n} ({} occupied voxels) ===",
        out.occupancy.count_occupied()
    );

    // REAL materials, if the asset carries them (glTF per-primitive material
    // index per triangle). material per voxel = material_ids[owner]; an
    // unresolved owner/material (u32::MAX) is kept as the single global-0 marker,
    // matching the shipped derive (pack_leaf counts uncoloured voxels as 0). This
    // is the measured ground truth that replaces the synthetic schemes below.
    if let Some(mids) = mesh.material_ids.as_ref() {
        let real = mids.iter().filter(|&&m| m != u32::MAX).count();
        if real > 0 {
            let mut map: HashMap<u64, (std::collections::HashSet<u32>, u32)> = HashMap::new();
            for z in 0..n {
                for y in 0..n {
                    for x in 0..n {
                        let lin = (x as usize)
                            + (n as usize) * ((y as usize) + (n as usize) * (z as usize));
                        let o = owner[lin];
                        if o == u32::MAX {
                            continue;
                        }
                        let mid = mids.get(o as usize).copied().unwrap_or(u32::MAX);
                        let e = map.entry(leaf_key(x, y, z)).or_default();
                        e.0.insert(mid);
                        e.1 += 1;
                    }
                }
            }
            let mut global: std::collections::HashSet<u32> = std::collections::HashSet::new();
            for (m, _) in map.values() {
                global.extend(m.iter().copied());
            }
            let gbits = bits_for((global.len() as u32 + 1).max(2)).max(1);
            let s = analyze(&map, gbits);
            println!(
                "  *** REAL glTF materials: {} distinct in the voxelized model ***",
                global.len()
            );
            report("REAL(glTF)", &s, &format!("{gbits}b"));
            cap_sweep("REAL(glTF)", &map);
        }
    }

    let schemes = [
        Scheme::HashOwner(4),
        Scheme::HashOwner(16),
        Scheme::HashOwner(64),
        Scheme::CoherentOctant,
        Scheme::CoherentRegion,
        Scheme::CoherentSlab,
        Scheme::RandomVoxel(16),
        Scheme::RandomVoxel(64),
    ];
    for scheme in schemes {
        let mut map: HashMap<u64, (std::collections::HashSet<u32>, u32)> = HashMap::new();
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let lin =
                        (x as usize) + (n as usize) * ((y as usize) + (n as usize) * (z as usize));
                    let o = owner[lin];
                    if o == u32::MAX {
                        continue;
                    }
                    let mid = material_id(scheme, o, x, y, z, n);
                    let e = map.entry(leaf_key(x, y, z)).or_default();
                    e.0.insert(mid);
                    e.1 += 1;
                }
            }
        }
        // global material count = distinct materials across the whole model
        let mut global: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for (m, _) in map.values() {
            global.extend(m.iter().copied());
        }
        let gbits = bits_for((global.len() as u32 + 1).max(2)).max(1);
        let s = analyze(&map, gbits);
        report(&scheme.label(), &s, &format!("{}b", gbits));
        cap_sweep(&scheme.label(), &map);
    }
}

fn run_field<F: OccupancyField>(name: &str, field: &F, res: u32) {
    let n = res;
    println!("\n=== FIXTURE {name} @ res {n} ===");
    // coherent-region + octant + random for a non-mesh contrast.
    let schemes = [
        Scheme::CoherentOctant,
        Scheme::CoherentRegion,
        Scheme::CoherentSlab,
        Scheme::RandomVoxel(16),
    ];
    let mut occ_total: u64 = 0;
    // Precompute occupancy once: store owner=0 placeholder (mesh-owner not
    // meaningful for fixtures); material is purely spatial for these schemes.
    for scheme in schemes {
        let mut map: HashMap<u64, (std::collections::HashSet<u32>, u32)> = HashMap::new();
        occ_total = 0;
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    if !field.is_occupied(VoxelCoord::new(x, y, z)) {
                        continue;
                    }
                    occ_total += 1;
                    let mid = material_id(scheme, 0, x, y, z, n);
                    let e = map.entry(leaf_key(x, y, z)).or_default();
                    e.0.insert(mid);
                    e.1 += 1;
                }
            }
        }
        let mut global: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for (m, _) in map.values() {
            global.extend(m.iter().copied());
        }
        let gbits = bits_for((global.len() as u32 + 1).max(2)).max(1);
        let s = analyze(&map, gbits);
        report(&scheme.label(), &s, &format!("{}b", gbits));
        cap_sweep(&scheme.label(), &map);
    }
    println!("   ({occ_total} occupied voxels)");
    let _ = Vec3::ZERO;
}

fn main() {
    // Valid resolutions are 8*4^k: 8,32,128,512,2048. 256 is NOT representable,
    // so we use 512 (closest to 512) and 128. Leaf is 8^3, so leaf-grid is n/8.
    // The .glb carries real per-primitive materials (the .obj is geometry-only).
    let model = "models/littlest-tokyo.glb";
    run_mesh(model, 512);
    run_mesh(model, 128);

    run_field(
        "Caves",
        &NoiseField::caves(Resolution::new(512).unwrap()),
        512,
    );
}
