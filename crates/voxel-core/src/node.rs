//! The internal `4³` node and its frozen GPU layout (`idea.md` §5, §6.4).
//!
//! A node carries a 64-bit child mask (one bit per `4³` child, in Morton order)
//! and a base offset into the child level's array. The packed slot of a stored
//! child is the population count of set mask bits below it — the `popcount`-rank
//! that lets only occupied children be stored yet addressed in O(1)
//! (`idea.md` §5).
//!
//! [`GpuNode`] is the **frozen `bytemuck` contract** shared with the GPU
//! adapter (adversarial review R4): a `#[repr(C)]` triple of `u32`. The 64-bit
//! mask is split `lo`/`hi` because WGSL has no `u64` (review R3); the rank
//! across the 32-bit boundary is mirrored bit-for-bit on CPU and GPU.

// Unsafe Quarantine: this is the one data-layout module. The only `unsafe` is
// the `bytemuck` derive proving `Pod` for a `#[repr(C)]` all-`u32` struct, which
// is trivially sound. No hand-written `unsafe`, and none escapes this module.
#![allow(unsafe_code)]

/// An internal `4³` node in the frozen GPU layout: a 64-bit child mask (split
/// into two `u32` for WGSL) and the base index of this node's children in the
/// child level's array.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuNode {
    /// Child-mask bits 0..32.
    pub mask_lo: u32,
    /// Child-mask bits 32..64.
    pub mask_hi: u32,
    /// Index of this node's first stored child in the child level's array.
    pub child_base: u32,
}

impl GpuNode {
    /// Builds a node from a full 64-bit child mask and a child base index.
    #[must_use]
    // `mask as u32` and `(mask >> 32) as u32` intentionally take the low/high
    // halves of the 64-bit mask.
    #[allow(clippy::cast_possible_truncation)]
    pub const fn new(mask: u64, child_base: u32) -> Self {
        Self {
            mask_lo: mask as u32,
            mask_hi: (mask >> 32) as u32,
            child_base,
        }
    }

    /// The reassembled 64-bit child mask.
    #[must_use]
    pub const fn mask(self) -> u64 {
        ((self.mask_hi as u64) << 32) | self.mask_lo as u64
    }

    /// Whether child `bit` (`0..64`) is present.
    #[must_use]
    pub const fn has_child(self, bit: u32) -> bool {
        (self.mask() >> bit) & 1 == 1
    }

    /// The array index of child `bit` among this node's stored children:
    /// `child_base + popcount(mask & ((1 << bit) − 1))` (`idea.md` §5/§6.4).
    ///
    /// Only meaningful when [`has_child`](Self::has_child) is true.
    #[must_use]
    pub const fn child_slot(self, bit: u32) -> u32 {
        // `bit == 0` ⇒ mask 0 ⇒ rank 0, which is correct.
        let below = if bit == 0 {
            0
        } else {
            self.mask() & ((1u64 << bit) - 1)
        };
        self.child_base + below.count_ones()
    }
}

/// The 6-bit Morton child index of a `4³` child at coordinate `(cx, cy, cz)`,
/// each `0..4`. Matches the low 6 bits of the descendant's brick Morton code.
#[must_use]
pub const fn child_bit(cx: u32, cy: u32, cz: u32) -> u32 {
    // 2 bits/axis interleaved → 6-bit index in 0..64. Identical to
    // `morton::encode(cx, cy, cz)` for 2-bit inputs, inlined to stay const and
    // dependency-free.
    crate::morton::encode_brick(cx, cy, cz) & 63
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_three_u32() {
        assert_eq!(size_of::<GpuNode>(), 12);
        assert_eq!(align_of::<GpuNode>(), 4);
    }

    #[test]
    fn mask_round_trips_through_lo_hi() {
        let m = 0xDEAD_BEEF_F00D_1234u64;
        let n = GpuNode::new(m, 7);
        assert_eq!(n.mask(), m);
        assert_eq!(n.mask_lo, 0xF00D_1234);
        assert_eq!(n.mask_hi, 0xDEAD_BEEF);
    }

    #[test]
    fn child_slot_is_rank_across_the_32_bit_boundary() {
        // Children present at bits 1, 33, 40. Their slots are child_base + rank.
        let mask = (1u64 << 1) | (1u64 << 33) | (1u64 << 40);
        let n = GpuNode::new(mask, 100);
        assert!(n.has_child(1) && n.has_child(33) && n.has_child(40));
        assert!(!n.has_child(0) && !n.has_child(32));
        assert_eq!(n.child_slot(1), 100); // rank 0
        assert_eq!(n.child_slot(33), 101); // one bit (bit 1) below
        assert_eq!(n.child_slot(40), 102); // bits 1 and 33 below
    }

    #[test]
    fn child_slot_at_bit_63() {
        let mask = (1u64 << 63) | 1u64;
        let n = GpuNode::new(mask, 0);
        assert_eq!(n.child_slot(0), 0);
        assert_eq!(n.child_slot(63), 1); // one bit (bit 0) below
    }

    #[test]
    fn child_bit_matches_morton() {
        for cz in 0..4 {
            for cy in 0..4 {
                for cx in 0..4 {
                    assert_eq!(
                        u64::from(child_bit(cx, cy, cz)),
                        crate::morton::encode(cx, cy, cz)
                    );
                    assert!(child_bit(cx, cy, cz) < 64);
                }
            }
        }
    }
}
