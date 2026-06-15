//! Morton (Z-order) codes — **build-time only** (`idea.md` §6.2).
//!
//! A 64-bit code interleaves three 21-bit axis coordinates (`3 × 21 = 63`
//! bits). This is computed on the CPU during the build (sort key, §6.4); the
//! GPU traversal never sees a 64-bit Morton code — it addresses children by
//! `popcount`-rank and 6-bit child indices instead (adversarial review R3).
//! Implementation is the ALU-only "magic bits" `splitBy3` cascade (Baert 2013).
//!
//! Also provides [`encode_brick`]: the 9-bit intra-`8³`-brick Morton index used
//! to order the 512 leaf bits.

use crate::coord::VoxelCoord;

/// Maximum coordinate value per axis that a 64-bit Morton code can hold
/// (`2²¹ − 1`). Covers every supported resolution (max `2³¹` per axis would
/// need... no — supported axis coords are `< 2²¹` only at resolutions up to
/// `2²¹`; the largest *used* resolution is `2048 = 2¹¹`, well within range).
pub const MAX_MORTON_COORD: u32 = (1 << 21) - 1;

/// Spreads the low 21 bits of `v` so bit `i` lands at bit `3·i` (`splitBy3`).
#[inline]
// `u32 → u64` is lossless; `as` is used (not `u64::from`) because `From` is not
// yet const-stable and this is a `const fn`.
#[allow(clippy::cast_lossless)]
const fn split_by_3(v: u32) -> u64 {
    let mut x = (v as u64) & 0x1f_ffff; // keep 21 bits
    x = (x | (x << 32)) & 0x001f_0000_0000_ffff;
    x = (x | (x << 16)) & 0x001f_0000_ff00_00ff;
    x = (x | (x << 8)) & 0x100f_00f0_0f00_f00f;
    x = (x | (x << 4)) & 0x10c3_0c30_c30c_30c3;
    x = (x | (x << 2)) & 0x1249_2492_4924_9249;
    x
}

/// Inverse of [`split_by_3`]: gathers bits `3·i` back into bit `i`.
#[inline]
// The final value is masked to 21 bits (`0x1f_ffff`), so it always fits in u32.
#[allow(clippy::cast_possible_truncation)]
const fn compact_by_3(v: u64) -> u32 {
    let mut x = v & 0x1249_2492_4924_9249;
    x = (x | (x >> 2)) & 0x10c3_0c30_c30c_30c3;
    x = (x | (x >> 4)) & 0x100f_00f0_0f00_f00f;
    x = (x | (x >> 8)) & 0x001f_0000_ff00_00ff;
    x = (x | (x >> 16)) & 0x001f_0000_0000_ffff;
    x = (x | (x >> 32)) & 0x1f_ffff;
    x as u32
}

/// Interleaves `(x, y, z)` into a 64-bit Morton code (`x` in the low bit of
/// each triple). Inputs must be `≤ MAX_MORTON_COORD`; higher bits are dropped
/// (and silently collide), so this codec supports resolutions only up to
/// `k = 10` (`n = 8_388_608`). The `debug_assert` makes a higher-`k` misuse
/// loud in tests rather than corrupting the Morton ordering silently.
#[must_use]
pub const fn encode(x: u32, y: u32, z: u32) -> u64 {
    debug_assert!(
        x <= MAX_MORTON_COORD && y <= MAX_MORTON_COORD && z <= MAX_MORTON_COORD,
        "morton::encode coordinate exceeds the 21-bit-per-axis range"
    );
    split_by_3(x) | (split_by_3(y) << 1) | (split_by_3(z) << 2)
}

/// Morton code of a [`VoxelCoord`].
#[must_use]
pub const fn encode_coord(c: VoxelCoord) -> u64 {
    encode(c.x, c.y, c.z)
}

/// Recovers `(x, y, z)` from a 64-bit Morton code.
#[must_use]
pub const fn decode(code: u64) -> VoxelCoord {
    VoxelCoord::new(
        compact_by_3(code),
        compact_by_3(code >> 1),
        compact_by_3(code >> 2),
    )
}

/// Intra-brick Morton index of a voxel within its `8³` leaf brick.
///
/// `x, y, z` are the low 3 bits/axis (`coord & 7`); the result is in `0..512`
/// and indexes the leaf's 512-bit bitmask in Morton order (`idea.md` §6.2).
#[must_use]
// 3 bits/axis interleave to a 9-bit result (`< 512`), so it always fits in u32.
#[allow(clippy::cast_possible_truncation)]
pub const fn encode_brick(x: u32, y: u32, z: u32) -> u32 {
    // Only 3 bits/axis, so the full 21-bit cascade is overkill but correct.
    encode(x & 7, y & 7, z & 7) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn known_values() {
        assert_eq!(encode(0, 0, 0), 0);
        assert_eq!(encode(1, 0, 0), 1);
        assert_eq!(encode(0, 1, 0), 2);
        assert_eq!(encode(0, 0, 1), 4);
        assert_eq!(encode(1, 1, 1), 7);
    }

    #[test]
    fn brick_index_in_range() {
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..8 {
                    assert!(encode_brick(x, y, z) < 512);
                }
            }
        }
        // The 512 indices are a permutation of 0..512 (a bijection).
        let mut seen = [false; 512];
        for x in 0..8 {
            for y in 0..8 {
                for z in 0..8 {
                    let i = encode_brick(x, y, z) as usize;
                    assert!(!seen[i], "duplicate brick index {i}");
                    seen[i] = true;
                }
            }
        }
        assert!(seen.iter().all(|&b| b));
    }

    proptest! {
        #[test]
        fn round_trip(
            x in 0u32..=MAX_MORTON_COORD,
            y in 0u32..=MAX_MORTON_COORD,
            z in 0u32..=MAX_MORTON_COORD,
        ) {
            let c = VoxelCoord::new(x, y, z);
            prop_assert_eq!(decode(encode_coord(c)), c);
        }

        /// Morton order is consistent with lexicographic order on the
        /// bit-interleaved key — adjacent codes are spatially local.
        #[test]
        fn monotone_in_each_axis(
            x in 0u32..MAX_MORTON_COORD, y in 0u32..=MAX_MORTON_COORD, z in 0u32..=MAX_MORTON_COORD,
        ) {
            // Incrementing x with the same y,z strictly increases the code,
            // because x occupies the lowest bit of each triple.
            prop_assert!(encode(x + 1, y, z) > encode(x, y, z));
        }
    }
}
