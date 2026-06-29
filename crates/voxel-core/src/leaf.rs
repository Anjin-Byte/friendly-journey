//! The `8³` bitmask leaf brick.
//!
//! A leaf holds 512 occupancy bits — one per voxel of an `8³` brick — stored in
//! intra-brick Morton order (`idea.md` §6.1/§6.2). A set bit is the terminal
//! surface (`idea.md` §7.2): an occupied brick is *not* itself a hit, only a
//! set voxel inside it is. Internally 8 × `u64`; the frozen GPU contract (P3)
//! is the bit-identical `16 × u32` view.
//!
//! Each leaf also carries a packed [`LeafBounds`] (the bounding box of its set
//! voxels) — a third GPU storage binding (`leaf_bounds`, group 0 binding 2)
//! alongside the nodes and leaf words. It drives the per-brick early-skip
//! (`idea.md` §8): a ray whose chord misses this box skips the `8³` walk. The
//! WGSL kernel unpacks it with the same shift/mask layout as [`LeafBounds::pack`]
//! (pinned by `wgsl_bit_layout_matches_pack`).

use crate::morton;

/// An `8³` leaf brick: 512 occupancy bits in intra-brick Morton order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeafBrick {
    /// 512 bits, `bits[i>>6] & (1 << (i & 63))` for Morton index `i`.
    bits: [u64; 8],
}

/// The axis-aligned bounding box of a brick's set voxels, in local `0..8`
/// coordinates (inclusive `min`/`max`). The basis for the per-brick early-skip:
/// a ray whose chord through the brick misses this box cannot hit any voxel in
/// it, so the `8³` interior walk is skipped without descending (`idea.md` §8 —
/// the leaf is the one level that otherwise has no intra-cell acceleration).
///
/// Packs into one `u32` (`6 × 3` bits) — the frozen GPU `leaf_bounds` word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeafBounds {
    /// Inclusive lower corner of the occupied region (each axis `0..8`).
    pub min: [u32; 3],
    /// Inclusive upper corner of the occupied region (each axis `0..8`).
    pub max: [u32; 3],
}

impl LeafBounds {
    /// The whole brick — the conservative bound (always descend). Used for an
    /// empty brick, which never occurs in a built structure but keeps the bound
    /// safe by construction.
    pub const FULL: LeafBounds = LeafBounds {
        min: [0, 0, 0],
        max: [7, 7, 7],
    };

    /// A spare flag bit in the packed word, **above** the bounds fields (which use
    /// bits 0..18). The truecolor BLEND path (`docs/materials/11`) ORs this into a
    /// leaf's packed `leaf_bounds` word to mark "this leaf contains a semi-transparent
    /// voxel", routing the GPU into the compositing sub-loop. [`unpack`](Self::unpack)
    /// and the WGSL `leaf_reaches` both mask only bits 0..18, so it never disturbs the
    /// occupancy bounds / early-skip.
    pub const TRANSPARENCY_BIT: u32 = 1 << 18;

    /// Packs the bounds into one `u32`: `min.{x,y,z}` in bits `0,3,6` and
    /// `max.{x,y,z}` in bits `9,12,15`.
    ///
    /// Each field is **masked to 3 bits** (`& 7`) so a field `>= 8` cannot bleed
    /// into a neighbouring field or, worse, into bit 18 — which is the reserved
    /// [`TRANSPARENCY_BIT`](Self::TRANSPARENCY_BIT) and would otherwise be set as
    /// a *phantom* transparency flag. In debug builds a `debug_assert` also trips
    /// on an out-of-range field to surface the programming error; in release the
    /// mask alone keeps the word well-formed by construction.
    #[must_use]
    pub const fn pack(self) -> u32 {
        debug_assert!(
            self.min[0] < 8
                && self.min[1] < 8
                && self.min[2] < 8
                && self.max[0] < 8
                && self.max[1] < 8
                && self.max[2] < 8,
            "LeafBounds field out of the 0..8 range"
        );
        (self.min[0] & 7)
            | ((self.min[1] & 7) << 3)
            | ((self.min[2] & 7) << 6)
            | ((self.max[0] & 7) << 9)
            | ((self.max[1] & 7) << 12)
            | ((self.max[2] & 7) << 15)
    }

    /// Inverse of [`pack`](Self::pack).
    #[must_use]
    pub const fn unpack(p: u32) -> Self {
        Self {
            min: [p & 7, (p >> 3) & 7, (p >> 6) & 7],
            max: [(p >> 9) & 7, (p >> 12) & 7, (p >> 15) & 7],
        }
    }
}

impl LeafBrick {
    /// An all-empty brick.
    pub const EMPTY: LeafBrick = LeafBrick { bits: [0; 8] };

    /// Sets the bit at intra-brick Morton index `i` (`0..512`).
    ///
    /// # Panics
    /// Panics if `i >= 512` — a programmer error in index computation.
    pub fn set_morton(&mut self, i: u32) {
        assert!(i < 512, "leaf Morton index out of range: {i}");
        self.bits[(i >> 6) as usize] |= 1u64 << (i & 63);
    }

    /// Sets the bit for intra-brick voxel coordinate `(x, y, z)` (each `0..8`).
    pub fn set_local(&mut self, x: u32, y: u32, z: u32) {
        self.set_morton(morton::encode_brick(x, y, z));
    }

    /// Clears the bit at intra-brick Morton index `i` (`0..512`).
    ///
    /// # Panics
    /// Panics if `i >= 512` — a programmer error in index computation.
    pub fn clear_morton(&mut self, i: u32) {
        assert!(i < 512, "leaf Morton index out of range: {i}");
        self.bits[(i >> 6) as usize] &= !(1u64 << (i & 63));
    }

    /// Clears the bit for intra-brick voxel coordinate `(x, y, z)` (each `0..8`).
    pub fn clear_local(&mut self, x: u32, y: u32, z: u32) {
        self.clear_morton(morton::encode_brick(x, y, z));
    }

    /// Reads the bit for intra-brick voxel coordinate `(x, y, z)` (each `0..8`).
    #[must_use]
    pub fn get_local(&self, x: u32, y: u32, z: u32) -> bool {
        let i = morton::encode_brick(x, y, z);
        (self.bits[(i >> 6) as usize] >> (i & 63)) & 1 == 1
    }

    /// Whether the brick has no set voxels (the coarse skip signal).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bits == [0; 8]
    }

    /// Number of set voxels.
    #[must_use]
    pub fn count_occupied(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum()
    }

    /// Counts occupied voxels whose intra-brick Morton index is **strictly less
    /// than** `morton` — the position
    /// [`assemble_leaf_color`](crate::SchoolBBuffer::assemble_leaf_color) pushes the
    /// voxel at `morton` to, relative to its leaf's `leaf_color_base`.
    ///
    /// **Truecolor contract (P3, `docs/materials/11`):** for an occupied voxel at
    /// Morton `m` in leaf slot `s`,
    /// `leaf_color[leaf_color_base[s] + occupied_rank(m)]` is that voxel's colour.
    ///
    /// This is the CPU canonical for the frozen WGSL `leaf_color_rank` (a 16-word
    /// masked popcount over the same [`words32`](Self::words32) view the GPU reads,
    /// mirroring `child_slot`'s `countOneBits(mask & ((1<<bit)-1))`). It is pinned
    /// bit-for-bit by the `occupied_rank_*` parity tests + the assembler-link test —
    /// never edit it without the WGSL transcription and those tests (the silent
    /// mis-color hazard; cf. `read_slot`/`wgsl_unpack`).
    #[must_use]
    pub fn occupied_rank(&self, morton: u32) -> u32 {
        debug_assert!(morton < 512, "occupied_rank morton out of range: {morton}");
        let words = self.words32(); // [u32; 16] — the GPU's `leaf_words` view
        let full = (morton >> 5) as usize; // complete 32-bit words strictly below
        let mut rank: u32 = words[..full].iter().map(|w| w.count_ones()).sum();
        let rem = morton & 31;
        if rem > 0 {
            rank += (words[full] & ((1u32 << rem) - 1)).count_ones();
        }
        rank
    }

    /// The bounding box of the set voxels (see [`LeafBounds`]). A dense leaf
    /// almost always spans the whole brick (so it `FULL`-gates the skip) and
    /// walks quickly anyway, so a cheap popcount short-circuits it to `FULL`
    /// without scanning its bits — keeping the build's per-leaf cost bounded
    /// regardless of occupancy. Only sparse/thin leaves (the skip's targets)
    /// pay the bit scan. Returning `FULL` early is conservative: it only
    /// disables the skip for that leaf, never drops a hit. Computed once at
    /// build time, never on the traversal hot path.
    #[must_use]
    // The word index `w` is `0..8`, so `as u32` never truncates.
    #[allow(clippy::cast_possible_truncation)]
    pub fn occupied_bounds(&self) -> LeafBounds {
        // Above this many set voxels the box is overwhelmingly the whole brick;
        // the skip's targets (dust ≤ ~4, wire lines ≤ ~24) are far below it.
        const BOUNDS_SCAN_LIMIT: u32 = 64;
        if self.count_occupied() > BOUNDS_SCAN_LIMIT {
            return LeafBounds::FULL;
        }
        let mut min = [7u32; 3];
        let mut max = [0u32; 3];
        let mut any = false;
        for (w, word) in self.bits.iter().enumerate() {
            let mut bits = *word;
            while bits != 0 {
                any = true;
                let i = (w as u32) * 64 + bits.trailing_zeros();
                let c = morton::decode(u64::from(i));
                for (a, v) in [c.x, c.y, c.z].into_iter().enumerate() {
                    min[a] = min[a].min(v);
                    max[a] = max[a].max(v);
                }
                bits &= bits - 1;
            }
        }
        if any {
            LeafBounds { min, max }
        } else {
            LeafBounds::FULL
        }
    }

    /// The raw words, for building the GPU buffer (P3).
    #[must_use]
    pub fn words(&self) -> [u64; 8] {
        self.bits
    }

    /// The leaf as 16 little-endian `u32` words — the frozen GPU view of the
    /// 512-bit mask (each `u64` splits into `[lo, hi]`). WGSL reads the bit for
    /// Morton index `i` as `(words[i >> 5] >> (i & 31)) & 1`.
    #[must_use]
    // The `as u32` halves intentionally take the low/high 32 bits of each word.
    #[allow(clippy::cast_possible_truncation)]
    pub fn words32(&self) -> [u32; 16] {
        let mut out = [0u32; 16];
        for (i, w) in self.bits.iter().enumerate() {
            out[i * 2] = *w as u32;
            out[i * 2 + 1] = (*w >> 32) as u32;
        }
        out
    }

    /// Rebuilds a brick from 16 little-endian `u32` words — the inverse of
    /// [`words32`](Self::words32). For a GPU generator that packs the 512-bit
    /// Morton mask straight into a `u32` storage buffer (bit for Morton index `i`
    /// at `words[i >> 5] & (1 << (i & 31))`) and reads it back.
    #[must_use]
    pub fn from_words32(words: [u32; 16]) -> Self {
        let mut bits = [0u64; 8];
        for (i, b) in bits.iter_mut().enumerate() {
            *b = u64::from(words[i * 2]) | (u64::from(words[i * 2 + 1]) << 32);
        }
        Self { bits }
    }
}

impl Default for LeafBrick {
    fn default() -> Self {
        Self::EMPTY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_local_round_trip() {
        let mut leaf = LeafBrick::EMPTY;
        assert!(leaf.is_empty());
        leaf.set_local(3, 5, 7);
        assert!(leaf.get_local(3, 5, 7));
        assert!(!leaf.get_local(3, 5, 6));
        assert!(!leaf.is_empty());
        assert_eq!(leaf.count_occupied(), 1);
    }

    #[test]
    fn occupied_bounds_tighten_to_set_voxels() {
        let mut leaf = LeafBrick::EMPTY;
        leaf.set_local(2, 1, 5);
        leaf.set_local(6, 4, 3);
        let b = leaf.occupied_bounds();
        assert_eq!(b.min, [2, 1, 3]);
        assert_eq!(b.max, [6, 4, 5]);
        // Round-trips through the packed GPU word.
        assert_eq!(LeafBounds::unpack(b.pack()), b);
    }

    #[test]
    fn single_voxel_bounds_are_a_point() {
        let mut leaf = LeafBrick::EMPTY;
        leaf.set_local(7, 0, 4);
        let b = leaf.occupied_bounds();
        assert_eq!(b.min, [7, 0, 4]);
        assert_eq!(b.max, [7, 0, 4]);
    }

    #[test]
    fn empty_bounds_are_the_full_brick() {
        assert_eq!(LeafBrick::EMPTY.occupied_bounds(), LeafBounds::FULL);
    }

    #[test]
    fn wgsl_bit_layout_matches_pack() {
        // Pin the GPU contract: `leaf_reaches` in traversal.wgsl unpacks the
        // packed word with this exact shift/mask sequence. Replicate it in Rust
        // and assert it recovers the same min/max as pack/unpack, so the WGSL
        // and Rust bit layouts cannot silently drift (no adapter needed).
        for b in [
            LeafBounds {
                min: [0, 1, 2],
                max: [3, 4, 5],
            },
            LeafBounds {
                min: [7, 0, 7],
                max: [7, 6, 7],
            },
            LeafBounds {
                min: [2, 2, 2],
                max: [2, 2, 2],
            },
            LeafBounds::FULL,
        ] {
            let p = b.pack();
            // The literal traversal.wgsl sequence.
            let mn = [p & 7, (p >> 3) & 7, (p >> 6) & 7];
            let mx = [(p >> 9) & 7, (p >> 12) & 7, (p >> 15) & 7];
            assert_eq!(mn, b.min, "WGSL min unpack drifted");
            assert_eq!(mx, b.max, "WGSL max unpack drifted");
            assert_eq!(LeafBounds::unpack(p), b);

            // The TRANSPARENCY_BIT (bit 18) is above the bounds fields: ORing it in
            // must leave the unpacked min/max — and the WGSL bounds read — unchanged,
            // and the bit must read back via `>> 18 & 1` exactly as the blend shader does.
            let pt = p | LeafBounds::TRANSPARENCY_BIT;
            assert_eq!(
                LeafBounds::unpack(pt),
                b,
                "transparency bit disturbed bounds"
            );
            let min_flagged = [pt & 7, (pt >> 3) & 7, (pt >> 6) & 7];
            let max_flagged = [(pt >> 9) & 7, (pt >> 12) & 7, (pt >> 15) & 7];
            assert_eq!(
                (min_flagged, max_flagged),
                (mn, mx),
                "transparency bit leaked into bounds"
            );
            assert_eq!(
                (pt >> 18) & 1,
                1,
                "transparency bit must read back at bit 18"
            );
            assert_eq!((p >> 18) & 1, 0, "bit 18 clear without the flag");
        }
    }

    /// In a debug build, packing an out-of-range field trips the tripwire that
    /// surfaces the programming error (`occupied_bounds` never emits `>= 8`, so
    /// this is a should-never-happen guard).
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic = "out of the 0..8 range"]
    fn pack_debug_asserts_out_of_range_field() {
        let _ = LeafBounds {
            min: [0, 0, 0],
            max: [7, 7, 8], // 8 is out of range
        }
        .pack();
    }

    /// In release (debug-assertions elided), the `& 7` mask still keeps an
    /// out-of-range field from bleeding into bit 18 — so a stray field can never
    /// set a *phantom* `TRANSPARENCY_BIT`. This is the by-construction guarantee.
    #[cfg(not(debug_assertions))]
    #[test]
    fn pack_masks_out_of_range_field_to_three_bits() {
        let p = LeafBounds {
            min: [0, 0, 0],
            max: [7, 7, 8], // 8 & 7 == 0
        }
        .pack();
        assert_eq!(
            p & LeafBounds::TRANSPARENCY_BIT,
            0,
            "an out-of-range field must not set the phantom transparency bit"
        );
        // The masked field reads back as 8 & 7 == 0, and only bits 0..18 are used.
        assert_eq!(p >> 18, 0, "nothing lands at or above bit 18");
        assert_eq!(LeafBounds::unpack(p).max[2], 0, "8 masked to 0");
    }

    #[test]
    fn all_512_voxels_are_independent() {
        // Set every voxel; expect exactly 512 set bits and all readable.
        let mut leaf = LeafBrick::EMPTY;
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    leaf.set_local(x, y, z);
                }
            }
        }
        assert_eq!(leaf.count_occupied(), 512);
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    assert!(leaf.get_local(x, y, z));
                }
            }
        }
    }

    // ---- occupied_rank parity (truecolor P3, docs/materials/11) --------------
    //
    // The rank is the GPU's `leaf_color_rank` CPU canonical. Two INDEPENDENT
    // oracles pin it (the read_slot/wgsl_unpack precedent): `brute_force_rank` is
    // ground truth (bit-by-bit, no popcount), `wgsl_rank` is a literal
    // transcription of the frozen WGSL (the spec lands physically in P4). A bug
    // shared by `occupied_rank` and the WGSL transcription is exposed by divergence
    // from brute force; the school_b assembler-link test ties `occupied_rank` to
    // the `get_local`/morton order the assembler actually uses.

    /// Ground truth: counts set bits at indices `0..m`, one bit at a time.
    fn brute_force_rank(words: &[u32; 16], m: u32) -> u32 {
        let mut r = 0u32;
        for i in 0..m {
            r += (words[(i >> 5) as usize] >> (i & 31)) & 1;
        }
        r
    }

    /// Literal Rust transcription of the frozen WGSL `leaf_color_rank` — shares no
    /// code with `LeafBrick::occupied_rank`, so it pins the WGSL spec independently.
    fn wgsl_rank(words: &[u32; 16], m: u32) -> u32 {
        let mut rank = 0u32;
        let full = m >> 5;
        let mut w = 0u32;
        while w < full {
            rank += words[w as usize].count_ones();
            w += 1;
        }
        let rem = m & 31;
        if rem > 0 {
            rank += (words[full as usize] & ((1u32 << rem) - 1)).count_ones();
        }
        rank
    }

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// The low 32 random bits — truncation is the intent (random leaf words).
    #[allow(clippy::cast_possible_truncation)]
    fn next_u32(state: &mut u64) -> u32 {
        splitmix64(state) as u32
    }

    fn leaf_from_mortons(set: &[u32]) -> LeafBrick {
        let mut leaf = LeafBrick::EMPTY;
        for &m in set {
            leaf.set_morton(m);
        }
        leaf
    }

    /// The fixed leaf shapes the rank matrix must cover (F1–F7 + a name).
    fn rank_fixtures() -> Vec<(&'static str, LeafBrick)> {
        let full = leaf_from_mortons(&(0..512).collect::<Vec<_>>());
        vec![
            ("empty", LeafBrick::EMPTY),
            ("full", full),
            ("single@0", leaf_from_mortons(&[0])),
            ("single@511", leaf_from_mortons(&[511])),
            (
                "boundary-straddle",
                leaf_from_mortons(&[31, 32, 33, 63, 64]),
            ),
            ("high-half", leaf_from_mortons(&[480, 481, 496, 511])),
            ("first@32", leaf_from_mortons(&[32])),
        ]
    }

    #[test]
    fn occupied_rank_matches_brute_and_wgsl_over_all_positions() {
        // A/B/C: occupied_rank == brute, wgsl_rank == brute, occupied_rank == wgsl,
        // for EVERY morton on the fixed shapes plus 256 random leaves (~25% density).
        let mut leaves: Vec<(String, LeafBrick)> = rank_fixtures()
            .into_iter()
            .map(|(n, l)| (n.to_string(), l))
            .collect();
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        for r in 0..256u32 {
            let mut words = [0u32; 16];
            for w in &mut words {
                // AND of two draws ≈ 25% bit density (sparse-ish, like real leaves).
                *w = next_u32(&mut state) & next_u32(&mut state);
            }
            leaves.push((format!("random#{r}"), LeafBrick::from_words32(words)));
        }

        for (name, leaf) in &leaves {
            let words = leaf.words32();
            for m in 0..512u32 {
                let bf = brute_force_rank(&words, m);
                assert_eq!(
                    leaf.occupied_rank(m),
                    bf,
                    "{name}: occupied_rank({m}) vs brute"
                );
                assert_eq!(wgsl_rank(&words, m), bf, "{name}: wgsl_rank({m}) vs brute");
                assert_eq!(
                    leaf.occupied_rank(m),
                    wgsl_rank(&words, m),
                    "{name}: occupied_rank vs WGSL transcription at {m}"
                );
            }
        }
    }

    #[test]
    fn occupied_rank_telescopes_across_word_boundaries() {
        // D: rank(m+1) == rank(m) + bit(m). An off-by-one in the rem==0 partial-word
        // skip at m∈{32,64,...,480} would break the telescope at that boundary.
        let mut leaves: Vec<LeafBrick> = rank_fixtures().into_iter().map(|(_, l)| l).collect();
        let mut state = 0xDEAD_BEEF_F00D_2026u64;
        for _ in 0..64 {
            let mut words = [0u32; 16];
            for w in &mut words {
                *w = next_u32(&mut state);
            }
            leaves.push(LeafBrick::from_words32(words));
        }
        for leaf in &leaves {
            let words = leaf.words32();
            for m in 0..511u32 {
                let bit = (words[(m >> 5) as usize] >> (m & 31)) & 1;
                assert_eq!(
                    leaf.occupied_rank(m + 1),
                    leaf.occupied_rank(m) + bit,
                    "telescope broke at morton {m}"
                );
            }
        }
    }

    #[test]
    fn occupied_rank_boundary_values() {
        // E: the explicit per-shape boundary assertions the synthesis pinned.
        // empty → 0 everywhere (catches a spurious +1).
        for m in [0, 1, 31, 32, 33, 63, 64, 480, 511] {
            assert_eq!(LeafBrick::EMPTY.occupied_rank(m), 0, "empty rank({m})");
        }
        // full → identity rank(m)==m (catches any scaling / off-by-one).
        let full = leaf_from_mortons(&(0..512).collect::<Vec<_>>());
        for m in 0..512u32 {
            assert_eq!(full.occupied_rank(m), m, "full rank({m})");
        }
        // single@0 / single@511 (highest word, full=15, rem=31 partial mask).
        let s0 = leaf_from_mortons(&[0]);
        assert_eq!(s0.occupied_rank(0), 0);
        assert_eq!(s0.occupied_rank(1), 1);
        let s511 = leaf_from_mortons(&[511]);
        for m in 0..=511u32 {
            assert_eq!(
                s511.occupied_rank(m),
                0,
                "single@511 rank({m}) (nothing below)"
            );
        }
        // boundary-straddle {31,32,33,63,64}: rem==0 reads NO partial word, rem==1
        // reads exactly bit 0 of the next word — pins the full/partial seam.
        let f5 = leaf_from_mortons(&[31, 32, 33, 63, 64]);
        assert_eq!(f5.occupied_rank(31), 0); // nothing below 31
        assert_eq!(f5.occupied_rank(32), 1); // bit31 only (rem=0, no partial)
        assert_eq!(f5.occupied_rank(33), 2); // + bit32
        assert_eq!(f5.occupied_rank(63), 3); // + bit33
        assert_eq!(f5.occupied_rank(64), 4); // + bit63 (rem=0 seam at word 2)
        assert_eq!(f5.occupied_rank(65), 5); // + bit64
        // high-half {480,481,496,511}: words 0..14 zero, only word 15 set.
        let f6 = leaf_from_mortons(&[480, 481, 496, 511]);
        assert_eq!(f6.occupied_rank(480), 0); // full=15, rem=0 ⇒ word15 not read
        assert_eq!(f6.occupied_rank(481), 1); // full=15, rem=1 ⇒ partial reads bit0 of word15
        assert_eq!(f6.occupied_rank(496), 2);
        assert_eq!(f6.occupied_rank(511), 3);
        // first@32: the boundary voxel is NOT pulled into rank 1 at m=32.
        let f7 = leaf_from_mortons(&[32]);
        assert_eq!(f7.occupied_rank(32), 0);
        assert_eq!(f7.occupied_rank(33), 1);
    }
}
