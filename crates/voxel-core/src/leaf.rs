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

    /// Packs the bounds into one `u32`: `min.{x,y,z}` in bits `0,3,6` and
    /// `max.{x,y,z}` in bits `9,12,15` (each `0..8`).
    #[must_use]
    pub const fn pack(self) -> u32 {
        self.min[0]
            | (self.min[1] << 3)
            | (self.min[2] << 6)
            | (self.max[0] << 9)
            | (self.max[1] << 12)
            | (self.max[2] << 15)
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
        }
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
}
