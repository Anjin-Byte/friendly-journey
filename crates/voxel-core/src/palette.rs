//! Per-leaf material palette: the CPU packer for an `8³` leaf's material slot.
//!
//! Materials are stored **separate** from occupancy (the bitmask stays the
//! authoritative traversal driver); this module packs one leaf's `P` distinct
//! materials into a fixed-stride slot — a header word, an inline `u16` palette,
//! and a bit-packed per-voxel index array — that the WGSL hit-read decodes at a
//! hit. See `docs/materials/02-storage-layout.md` and `03-gpu-read.md`.
//!
//! The slot is `STRIDE_W` `u32` words (cap `P_CAP = 16` materials ⇒ 4-bit
//! indices). A `P = 1` leaf needs **0 bits** — the lone palette entry *is* the
//! material, so no index array is read (the `bits == 0` branch is mandatory:
//! the mask `(1 << 0) - 1 == 0` is degenerate).
//!
//! **Drift hazard:** the GPU does not run this packer — `traversal.wgsl`
//! re-reads the bytes. The only thing keeping them bit-identical is the
//! `wgsl_bit_layout_matches_pack` test below (the precedent is the occupancy
//! pin at `leaf.rs`'s `wgsl_bit_layout_matches_pack`). Never bypass it. Indices
//! are addressed by intra-brick **Morton** order ([`crate::morton::encode_brick`]),
//! never a linear `x*64 + y*8 + z` — a linear layout silently transposes the
//! whole material field while occupancy stays correct.

use thiserror::Error;

/// Voxels per `8³` leaf brick.
pub const LEAF_VOXELS: usize = 512;
/// Inline palette cap (materials per leaf before spilling). 4-bit indices.
pub const P_CAP: u32 = 16;
/// Max inline `bits_per_voxel` (`ceil(log2 P_CAP)`).
pub const MAX_BITS: u32 = 4;
/// Header word offset within the slot.
pub const HDR_OFF: usize = 0;
/// Inline-palette base word (`u16` entries, two per word).
pub const PAL_OFF: usize = 1;
/// Per-voxel index-array base word. `PAL_OFF + P_CAP/2 = 1 + 8`.
pub const IDX_OFF: usize = 9;
/// Index-array word count: `512 voxels * 4 bits / 32`.
pub const IDX_WORDS: usize = 64;
/// Total slot width in `u32` words (`MAT_STRIDE / 4`).
pub const STRIDE_W: usize = IDX_OFF + IDX_WORDS;

// Header bit-field shifts: {bits_per_voxel:4 @0, palette_len:10 @4, spill_flag:1 @14}.
const HDR_PLEN_SHIFT: u32 = 4;
const HDR_SPILL_SHIFT: u32 = 14;

/// Why a leaf material slot cannot be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PaletteError {
    /// The palette exceeds the inline cap; the leaf must spill (not yet wired).
    #[error("palette length {len} exceeds the inline cap {cap}; leaf must spill")]
    Overflow {
        /// The rejected palette length.
        len: u32,
        /// The inline cap ([`P_CAP`]).
        cap: u32,
    },
    /// A per-voxel index references a palette slot that does not exist.
    #[error("voxel index {index} references palette slot >= len {len}")]
    BadIndex {
        /// The out-of-range palette index.
        index: u32,
        /// The palette length.
        len: u32,
    },
    /// The per-voxel index array was not exactly [`LEAF_VOXELS`] long.
    #[error("expected {LEAF_VOXELS} per-voxel indices, got {got}")]
    WrongIndexLen {
        /// The wrong length supplied.
        got: usize,
    },
}

/// The reserved MISSING colour: opaque magenta (`R=255, G=0, B=255, A=255`),
/// RGBA8 packed little-endian for WGSL `unpack4x8unorm` (R in the low byte).
pub const MISSING_MAGENTA: u32 = 0xFFFF_00FF;

/// Why the global material table could not accept a colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MaterialError {
    /// The table reached the `u16` ceiling. Ids are dense from `1` with id `0`
    /// reserved for the magenta MISSING sentinel, so at most **65535 real
    /// materials** fit (ids `1..=u16::MAX`).
    #[error(
        "material table full: at most 65535 real materials (id 0 reserved magenta; ids 1..=u16::MAX)"
    )]
    TableFull,
}

/// The global `global_id → colour` table the per-leaf palettes index into
/// (docs/materials/02 §4). `table[0]` is always [`MISSING_MAGENTA`]: an
/// unresolved or occupied-but-uncoloured voxel reads global id 0 and renders a
/// **loud magenta** rather than silently aliasing a real material (docs 05,
/// hole 1). Colours are RGBA8 little-endian for WGSL `unpack4x8unorm`, and the
/// whole table uploads verbatim as the `material_table` storage buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaterialTable {
    colors: Vec<u32>, // colors[0] == MISSING_MAGENTA, always
}

impl MaterialTable {
    /// A table holding only the reserved magenta MISSING slot — every voxel
    /// renders magenta until [`push`](Self::push) adds real materials.
    ///
    /// # Examples
    /// ```
    /// use voxel_core::MaterialTable;
    ///
    /// let mut table = MaterialTable::missing_only();
    /// // Real materials get dense ids from 1 (id 0 stays the magenta sentinel).
    /// let green = table.push(0xFF00_FF00).unwrap();
    /// assert_eq!(green, 1);
    /// assert_eq!(table.color(green), 0xFF00_FF00);
    /// ```
    #[must_use]
    pub fn missing_only() -> Self {
        Self {
            colors: vec![MISSING_MAGENTA],
        }
    }

    /// Appends a colour and returns its `global_id`. Ids are dense from `1`
    /// (id `0` stays the magenta sentinel), so the first `push` returns `1`.
    ///
    /// # Errors
    /// [`MaterialError::TableFull`] once 65535 real materials have been pushed —
    /// the next id would be 65536, which does not fit a `u16` (id 0 is reserved
    /// for the magenta sentinel).
    pub fn push(&mut self, color: u32) -> Result<u16, MaterialError> {
        let id = self.colors.len();
        if id > u16::MAX as usize {
            return Err(MaterialError::TableFull);
        }
        self.colors.push(color);
        Ok(u16::try_from(id).expect("checked against u16::MAX above"))
    }

    /// The colour for `global_id`, or [`MISSING_MAGENTA`] if it is out of range
    /// (the same loud fallback the GPU read uses on an over-range id).
    #[must_use]
    pub fn color(&self, global_id: u16) -> u32 {
        self.colors
            .get(usize::from(global_id))
            .copied()
            .unwrap_or(MISSING_MAGENTA)
    }

    /// The table as `u32` words for the GPU `material_table` storage buffer
    /// (index `global_id` → RGBA8 colour). Always at least one word (slot 0).
    #[must_use]
    pub fn words(&self) -> &[u32] {
        &self.colors
    }
}

impl Default for MaterialTable {
    /// The magenta-only table ([`missing_only`](Self::missing_only)).
    fn default() -> Self {
        Self::missing_only()
    }
}

/// Bits per voxel for a palette of `p` entries: `ceil(log2 p)`, **redefined to
/// `0` for `p <= 1`** (a single-material leaf stores no index array — the
/// SEPARATE-scheme divergence from the reference's min-of-1).
#[must_use]
pub fn bits_required(p: u32) -> u32 {
    if p <= 1 {
        0
    } else {
        32 - (p - 1).leading_zeros() // ceil(log2 p)
    }
}

/// One `8³` leaf's material data, ready to pack into a fixed-stride GPU slot.
///
/// `palette[pi]` is a global material id; `indices[m]` is the palette index of
/// the voxel at intra-brick Morton index `m` (`0..512`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeafMaterials {
    palette: Vec<u16>,
    indices: Vec<u8>, // len LEAF_VOXELS, in Morton order
}

impl LeafMaterials {
    /// Builds from a palette and per-voxel indices **in Morton order**
    /// (`indices[encode_brick(x, y, z)]`). Validates the cap, the index length,
    /// and that every index is a valid palette slot.
    ///
    /// # Errors
    /// [`PaletteError::Overflow`] if `palette.len() > P_CAP`,
    /// [`PaletteError::WrongIndexLen`] if `indices.len() != LEAF_VOXELS`,
    /// [`PaletteError::BadIndex`] if any index `>= palette.len()`.
    pub fn new(palette: Vec<u16>, indices: Vec<u8>) -> Result<Self, PaletteError> {
        if palette.len() > P_CAP as usize {
            return Err(PaletteError::Overflow {
                len: u32::try_from(palette.len()).unwrap_or(u32::MAX),
                cap: P_CAP,
            });
        }
        if indices.len() != LEAF_VOXELS {
            return Err(PaletteError::WrongIndexLen { got: indices.len() });
        }
        let plen = palette.len();
        if let Some(&bad) = indices.iter().find(|&&pi| usize::from(pi) >= plen) {
            return Err(PaletteError::BadIndex {
                index: u32::from(bad),
                len: u32::try_from(plen).unwrap_or(u32::MAX),
            });
        }
        Ok(Self { palette, indices })
    }

    /// Builds from a palette and a per-voxel-coordinate assignment, routing each
    /// `(x, y, z)` (each `0..8`) through [`crate::morton::encode_brick`] so the
    /// index array is Morton-ordered — the **only** correct intra-leaf order.
    ///
    /// # Errors
    /// As [`new`](Self::new).
    pub fn from_local<F>(palette: Vec<u16>, assign: F) -> Result<Self, PaletteError>
    where
        F: Fn(u32, u32, u32) -> u8,
    {
        let mut indices = vec![0u8; LEAF_VOXELS];
        for z in 0..8u32 {
            for y in 0..8u32 {
                for x in 0..8u32 {
                    // NEVER a linear x*64+y*8+z — that transposes the field.
                    indices[crate::morton::encode_brick(x, y, z) as usize] = assign(x, y, z);
                }
            }
        }
        Self::new(palette, indices)
    }

    /// `bits_per_voxel` for this leaf (`0` when single-material).
    #[must_use]
    pub fn bits(&self) -> u32 {
        bits_required(u32::try_from(self.palette.len()).unwrap_or(u32::MAX))
    }

    /// The packed header word: `{bits_per_voxel:4, palette_len:10, spill_flag:1}`
    /// (spill always `0` here — the inline path).
    #[must_use]
    pub fn header_word(&self) -> u32 {
        let plen = u32::try_from(self.palette.len()).unwrap_or(u32::MAX);
        (self.bits() & 0xF) | ((plen & 0x3FF) << HDR_PLEN_SHIFT)
    }

    /// Packs the leaf into its fixed-stride slot of [`STRIDE_W`] `u32` words:
    /// header, inline `u16` palette (two per word, low/high), then the per-voxel
    /// index array — LSB-first, Morton-addressed, straddling a 32-bit word when
    /// `pos + bits > 32`. When `bits == 0` the index region is left zero (the
    /// lone palette entry is the material). Bit-identical to the WGSL hit-read.
    ///
    /// # Errors
    /// [`PaletteError::Overflow`] if the palette exceeds [`P_CAP`].
    // `m < 512` and `pi & 1 ∈ {0,1}` — the `as u32` casts cannot truncate.
    #[allow(clippy::cast_possible_truncation)]
    #[must_use = "the packed slot is the output"]
    pub fn pack(&self) -> [u32; STRIDE_W] {
        let bits = self.bits();
        let mut out = [0u32; STRIDE_W];
        out[HDR_OFF] = self.header_word();

        // Inline palette: u16 two-per-word, even index → low half, odd → high.
        for (pi, &gid) in self.palette.iter().enumerate() {
            out[PAL_OFF + (pi >> 1)] |= u32::from(gid) << (16 * (pi as u32 & 1));
        }

        // Per-voxel index array, skipped entirely when bits == 0.
        if bits > 0 {
            for (m, &pi) in self.indices.iter().enumerate() {
                let off = m as u32 * bits;
                let word = IDX_OFF + (off >> 5) as usize;
                let pos = off & 31;
                out[word] |= u32::from(pi) << pos;
                if pos + bits > 32 {
                    out[word + 1] |= u32::from(pi) >> (32 - pos);
                }
            }
        }
        out
    }
}

/// Derives one leaf's packed material slot from its raw per-voxel materials
/// (`mats`, intra-brick Morton order) and an `occupied(x, y, z)` occupancy test.
///
/// The palette is the **distinct materials among occupied voxels** in first-seen
/// Morton order; unoccupied voxels get index `0` (never read — only a hitting,
/// therefore occupied, voxel issues a material read). A single-material leaf
/// packs to the `bits == 0` uniform fast path (no index array).
///
/// **Spill is deferred but loud.** A leaf with more than [`P_CAP`] distinct
/// occupied materials cannot fit the inline palette; until the spill arena lands
/// it is emitted as **uniform magenta** (global-0) with the `spill_flag` set —
/// a visibly-wrong leaf that flags the unimplemented arena, never a silent
/// mis-color. The result is bit-identical to a fresh derive (deterministic
/// `z,y,x` scan), so an in-place material patch matches a full re-serialization.
#[must_use]
pub fn pack_leaf(
    mats: &[u16; LEAF_VOXELS],
    occupied: impl Fn(u32, u32, u32) -> bool,
) -> [u32; STRIDE_W] {
    let mut palette: Vec<u16> = Vec::new();
    let mut indices = vec![0u8; LEAF_VOXELS];
    let mut spilled = false;
    for z in 0..8u32 {
        for y in 0..8u32 {
            for x in 0..8u32 {
                if !occupied(x, y, z) {
                    continue;
                }
                let m = crate::morton::encode_brick(x, y, z) as usize;
                let gid = mats[m];
                let pi = if let Some(p) = palette.iter().position(|&g| g == gid) {
                    p
                } else if palette.len() < P_CAP as usize {
                    palette.push(gid);
                    palette.len() - 1
                } else {
                    spilled = true; // over cap → handled as uniform magenta below
                    0
                };
                indices[m] = u8::try_from(pi).expect("pi < P_CAP <= 16");
            }
        }
    }

    if spilled {
        // Deferred spill: emit uniform magenta. Header {bits:0, plen:1, spill:1};
        // palette[0] and every index are 0, so all 512 voxels read global-0.
        let mut out = [0u32; STRIDE_W];
        out[HDR_OFF] = (1 << HDR_PLEN_SHIFT) | (1 << HDR_SPILL_SHIFT);
        return out;
    }
    if palette.is_empty() {
        palette.push(0); // defensive: an all-empty leaf reads global-0 (magenta)
    }
    LeafMaterials::new(palette, indices)
        .expect("palette <= P_CAP and indices valid by construction")
        .pack()
}

/// Reads the global material id at intra-brick Morton index `morton` from one
/// leaf's packed `slot` (a slice of at least [`STRIDE_W`] words). This is the
/// **CPU mirror of the WGSL hit-read** (`docs/materials/03-gpu-read.md` §2) — the
/// same literal header → index → palette half-select sequence — and is what the
/// `wgsl_bit_layout_matches_pack` parity test pins against [`pack_leaf`] /
/// [`LeafMaterials::pack`]. A `bits == 0` slot (a single-material leaf, or the
/// deferred uniform-magenta spill) reads palette slot 0 for every voxel.
///
/// The result is a global id into a [`MaterialTable`]; out-of-range ids resolve
/// to magenta there. Drift between this and the WGSL reader is a FATAL silent
/// mis-color — never edit one without the other and the parity test.
#[must_use]
pub fn read_slot(slot: &[u32], morton: u32) -> u16 {
    let h = slot[HDR_OFF];
    let bits = h & 0xF;
    let mut pi: u32 = 0;
    if bits != 0 {
        // MANDATORY guard: bits==0 would make the mask (1<<0)-1 == 0.
        let off = morton * bits;
        let w = slot[IDX_OFF + (off >> 5) as usize];
        let pos = off & 31;
        pi = w >> pos;
        if pos + bits > 32 {
            pi |= slot[IDX_OFF + (off >> 5) as usize + 1] << (32 - pos);
        }
        pi &= (1u32 << bits) - 1;
    }
    let pal_word = slot[PAL_OFF + (pi >> 1) as usize];
    u16::try_from((pal_word >> (16 * (pi & 1))) & 0xFFFF).expect("masked to 16 bits")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morton::encode_brick;

    /// A standalone, literal transcription of the GPU material reader
    /// (`render.wgsl` `read_material`) for a single leaf slot (`base = 0`), using
    /// the WGSL's own offsets (`MAT_IDX_OFF = 9`, `MAT_PAL_OFF = 1`). It shares
    /// **no code** with [`read_slot`], so the round-trip test below is a genuine
    /// independent witness of the GPU decode — not a self-alias. Mirrors the
    /// `wgsl_rank` / `brute_force_rank` / `occupied_rank` triple-oracle in
    /// `leaf.rs`. Keep this in lockstep with `render.wgsl` if that layout changes.
    fn wgsl_unpack(words: &[u32; STRIDE_W], morton: u32) -> u16 {
        const WGSL_IDX_OFF: usize = 9;
        const WGSL_PAL_OFF: usize = 1;
        let bits = words[0] & 0xF; // header word: bits_per_voxel
        let mut pi: u32 = 0;
        if bits != 0 {
            // MANDATORY: bits==0 would make the mask `(1<<0)-1 == 0` degenerate.
            let off = morton * bits;
            let pos = off & 31;
            let iw = WGSL_IDX_OFF + (off >> 5) as usize;
            pi = words[iw] >> pos;
            if pos + bits > 32 {
                // Index straddles two 32-bit words; stitch in the high part.
                pi |= words[iw + 1] << (32 - pos);
            }
            pi &= (1u32 << bits) - 1;
        }
        let pal_word = words[WGSL_PAL_OFF + (pi >> 1) as usize];
        u16::try_from((pal_word >> (16 * (pi & 1))) & 0xFFFF).expect("masked to 16 bits")
    }

    #[test]
    fn material_table_reserves_magenta_and_densifies_from_one() {
        let mut t = MaterialTable::missing_only();
        assert_eq!(t.words(), &[MISSING_MAGENTA]);
        assert_eq!(t.color(0), MISSING_MAGENTA);
        // First real colour gets global id 1 (id 0 is the reserved sentinel).
        assert_eq!(t.push(0xFF00_FF00).unwrap(), 1); // opaque green
        assert_eq!(t.push(0xFFFF_0000).unwrap(), 2); // opaque blue
        assert_eq!(t.color(1), 0xFF00_FF00);
        assert_eq!(t.color(2), 0xFFFF_0000);
        // Out-of-range ids fall back to the loud magenta sentinel.
        assert_eq!(t.color(3), MISSING_MAGENTA);
        assert_eq!(t.words().len(), 3);
    }

    #[test]
    fn bits_required_redefines_minimum_to_zero() {
        assert_eq!(bits_required(0), 0);
        assert_eq!(bits_required(1), 0); // single material ⇒ NO index array
        assert_eq!(bits_required(2), 1);
        assert_eq!(bits_required(3), 2);
        assert_eq!(bits_required(4), 2);
        assert_eq!(bits_required(5), 3);
        assert_eq!(bits_required(8), 3);
        assert_eq!(bits_required(9), 4);
        assert_eq!(bits_required(16), 4);
        assert_eq!(bits_required(17), 5); // over cap ⇒ would spill
    }

    #[test]
    fn cap_overflow_is_rejected() {
        let palette: Vec<u16> = (0..17).collect(); // 17 > P_CAP
        let err = LeafMaterials::new(palette, vec![0u8; LEAF_VOXELS]).unwrap_err();
        assert_eq!(
            err,
            PaletteError::Overflow {
                len: 17,
                cap: P_CAP
            }
        );
    }

    #[test]
    fn bad_index_and_len_are_rejected() {
        // index 2 references slot >= palette.len() == 2.
        let mut idx = vec![0u8; LEAF_VOXELS];
        idx[5] = 2;
        assert_eq!(
            LeafMaterials::new(vec![10, 20], idx).unwrap_err(),
            PaletteError::BadIndex { index: 2, len: 2 }
        );
        assert_eq!(
            LeafMaterials::new(vec![10, 20], vec![0u8; 511]).unwrap_err(),
            PaletteError::WrongIndexLen { got: 511 }
        );
    }

    /// Round-trips EVERY voxel of a leaf through pack → the literal WGSL unpack,
    /// for a palette of `p` materials assigned by `(x,y,z)`. Because the assign
    /// closure and the read both go through `encode_brick`, a morton-vs-linear
    /// packer bug scrambles the field and fails here; the straddle/boundary
    /// mortons are covered automatically by iterating all 512 voxels.
    fn assert_full_leaf_roundtrip(palette: Vec<u16>, assign: impl Fn(u32, u32, u32) -> u8) {
        let plen = palette.len();
        let expected = palette.clone(); // for indexing; `palette` is moved into the leaf
        let leaf = LeafMaterials::from_local(palette, &assign).unwrap();
        let packed = leaf.pack();
        for z in 0..8u32 {
            for y in 0..8u32 {
                for x in 0..8u32 {
                    let m = encode_brick(x, y, z);
                    let pi = assign(x, y, z);
                    assert!(usize::from(pi) < plen);
                    let got = wgsl_unpack(&packed, m);
                    // Triple-witness: the independent WGSL transcription, the
                    // production CPU mirror (`read_slot`), and the brute-force
                    // expected palette entry must all agree. A WGSL↔CPU drift
                    // fails the first assert; a packer bug fails the second.
                    assert_eq!(
                        read_slot(&packed, m),
                        got,
                        "read_slot disagrees with the WGSL transcription at ({x},{y},{z}) morton {m}"
                    );
                    assert_eq!(
                        got,
                        expected[usize::from(pi)],
                        "drift at voxel ({x},{y},{z}) morton {m}: bits={}",
                        leaf.bits()
                    );
                }
            }
        }
    }

    #[test]
    fn wgsl_bit_layout_matches_pack() {
        // bits = 0 (P = 1): the uniform fast path — the lone entry is read for
        // EVERY morton, no index array touched.
        assert_full_leaf_roundtrip(vec![42], |_, _, _| 0);

        // bits = 1 (P = 2): prove the NON-degenerate path (a true pi==1 reads
        // back 1), not just the bits==0 short-circuit passing trivially.
        assert_full_leaf_roundtrip(vec![10, 20], |x, _, _| u8::from(x >= 4));

        // bits = 2 (P = 4): even width — reaches the pos+bits==32 boundary but
        // never straddles (guards the `>` vs `>=` reader bug).
        assert_full_leaf_roundtrip(vec![1, 2, 3, 4], |x, y, _| ((x + y) % 4) as u8);

        // bits = 3 (P = 8): ODD width — the ONLY case with a real two-word
        // stitch (pos+bits>32), e.g. morton 10 → off 30, pos 30, 30+3 = 33.
        assert_full_leaf_roundtrip(vec![1, 2, 3, 4, 5, 6, 7, 8], |x, y, z| {
            ((x + 2 * y + 3 * z) % 8) as u8
        });

        // bits = 4 (P = 16): the inline cap; even width, boundary-only.
        let pal16: Vec<u16> = (100..116).collect();
        assert_full_leaf_roundtrip(pal16, |x, y, z| ((x ^ y ^ z) & 15) as u8);
    }

    #[test]
    fn straddle_and_half_select_are_explicit() {
        // Document the bits=3 real straddler: morton 10 sits at off=30 (pos 30),
        // so its 3-bit index spans words IDX_OFF and IDX_OFF+1.
        let bits = 3u32;
        let m = 10u32;
        assert_eq!(m * bits, 30);
        assert!((30 & 31) + bits > 32, "morton 10 must straddle at bits=3");

        // u16 half-select: even palette index → low 16 bits, odd → high 16.
        // Build a leaf where voxel A has pi=2 (even) and voxel B has pi=3 (odd).
        let palette = vec![0u16, 11, 0x2222, 0x3333]; // P=4 ⇒ bits=2
        let leaf = LeafMaterials::from_local(palette.clone(), |x, _, _| match x {
            0 => 2, // even index → low half of palette word
            1 => 3, // odd index  → high half
            _ => 0,
        })
        .unwrap();
        let packed = leaf.pack();
        // Palette word 1 (pi 2,3) packs 0x2222 low, 0x3333 high.
        assert_eq!(packed[PAL_OFF + 1], 0x3333_2222);
        assert_eq!(wgsl_unpack(&packed, encode_brick(0, 0, 0)), 0x2222);
        assert_eq!(wgsl_unpack(&packed, encode_brick(1, 0, 0)), 0x3333);
    }

    use proptest::prelude::*;

    proptest! {
        /// Random palettes (1..=16 entries) with random per-voxel index patterns
        /// round-trip through `pack` → `read_slot` for every voxel — the property
        /// generalization of `wgsl_bit_layout_matches_pack`, covering arbitrary
        /// index distributions across every bit-width 0..=4 (not just the fixed
        /// `x>=4` / `(x+y)%4` patterns).
        #[test]
        fn pack_read_slot_roundtrips_random_palettes(plen in 1usize..=16, seed in any::<u64>()) {
            let palette: Vec<u16> = (0..plen).map(|i| 1000u16 + u16::try_from(i).unwrap()).collect();
            // Pure per-voxel assignment: a hash of (morton, seed) into 0..plen.
            let assign = |x: u32, y: u32, z: u32| -> u8 {
                let m = u64::from(encode_brick(x, y, z));
                let h = (m.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed)
                    .wrapping_mul(0xBF58_476D_1CE4_E5B9);
                u8::try_from((h >> 33) % plen as u64).expect("index < plen <= 16")
            };
            let leaf = LeafMaterials::from_local(palette.clone(), assign).unwrap();
            let packed = leaf.pack();
            for z in 0..8u32 {
                for y in 0..8u32 {
                    for x in 0..8u32 {
                        let pi = assign(x, y, z);
                        prop_assert_eq!(
                            read_slot(&packed, encode_brick(x, y, z)),
                            palette[usize::from(pi)]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn pack_leaf_uniform_multi_and_spill() {
        // Uniform: a full leaf, all material 7 → bits == 0, every voxel reads 7.
        let mats = [7u16; LEAF_VOXELS];
        let packed = pack_leaf(&mats, |_, _, _| true);
        assert_eq!(
            packed[HDR_OFF] & 0xF,
            0,
            "single-material leaf must be bits=0"
        );
        for m in 0..u32::try_from(LEAF_VOXELS).unwrap() {
            assert_eq!(wgsl_unpack(&packed, m), 7);
        }

        // Multi-material: two ids split by x, respecting (full) occupancy.
        let mut mats = [0u16; LEAF_VOXELS];
        for z in 0..8u32 {
            for y in 0..8u32 {
                for x in 0..8u32 {
                    mats[encode_brick(x, y, z) as usize] = if x < 4 { 11 } else { 22 };
                }
            }
        }
        let packed = pack_leaf(&mats, |_, _, _| true);
        for z in 0..8u32 {
            for y in 0..8u32 {
                for x in 0..8u32 {
                    let m = encode_brick(x, y, z);
                    assert_eq!(wgsl_unpack(&packed, m), if x < 4 { 11 } else { 22 });
                }
            }
        }

        // Spill: 17 distinct occupied materials → uniform magenta + spill flag.
        let mut mats = [0u16; LEAF_VOXELS];
        for i in 0..17u32 {
            mats[encode_brick(i % 8, i / 8, 0) as usize] = u16::try_from(100 + i).unwrap();
        }
        let occ = |x: u32, y: u32, z: u32| z == 0 && y * 8 + x < 17;
        let packed = pack_leaf(&mats, occ);
        assert_eq!(
            (packed[HDR_OFF] >> 14) & 1,
            1,
            "over-cap leaf must set spill_flag"
        );
        for i in 0..17u32 {
            assert_eq!(
                wgsl_unpack(&packed, encode_brick(i % 8, i / 8, 0)),
                0,
                "spilled leaf must read uniform magenta (global-0)"
            );
        }
    }

    #[test]
    fn slot_geometry_is_consistent() {
        assert_eq!(STRIDE_W, 73);
        assert_eq!(IDX_OFF, PAL_OFF + (P_CAP as usize) / 2);
        assert_eq!(IDX_WORDS, LEAF_VOXELS * (MAX_BITS as usize) / 32);
        assert_eq!(MAX_BITS, bits_required(P_CAP));
    }

    /// The material-table ceiling: exactly 65535 real materials fit (ids
    /// `1..=u16::MAX`, id 0 reserved magenta); the 65536th `push` is `TableFull`.
    /// Pins the boundary the off-by-one error wording now describes (C2).
    #[test]
    fn push_accepts_65535_real_materials_then_rejects() {
        let mut table = MaterialTable::missing_only();
        for expected_id in 1..=u16::MAX {
            assert_eq!(
                table.push(0xDEAD_BEEF),
                Ok(expected_id),
                "real material {expected_id} must fit"
            );
        }
        // 65535 reals pushed (ids 1..=65535); the next would be id 65536.
        assert_eq!(
            table.push(0xDEAD_BEEF),
            Err(MaterialError::TableFull),
            "the 65536th real material must be rejected"
        );
    }
}
