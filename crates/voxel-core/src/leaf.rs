//! The `8³` bitmask leaf brick.
//!
//! A leaf holds 512 occupancy bits — one per voxel of an `8³` brick — stored in
//! intra-brick Morton order (`idea.md` §6.1/§6.2). A set bit is the terminal
//! surface (`idea.md` §7.2): an occupied brick is *not* itself a hit, only a
//! set voxel inside it is. Internally 8 × `u64`; the frozen GPU contract (P3)
//! is the bit-identical `16 × u32` view.

use crate::morton;

/// An `8³` leaf brick: 512 occupancy bits in intra-brick Morton order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeafBrick {
    /// 512 bits, `bits[i>>6] & (1 << (i & 63))` for Morton index `i`.
    bits: [u64; 8],
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
