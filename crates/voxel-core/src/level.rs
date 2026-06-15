//! Traversal levels and their size formulas (`idea.md` §7).
//!
//! The level index `L` runs finest-to-coarsest: `L = 0` is the **voxel**
//! (terminal, a set bit is a surface hit), `L = 1` is the `8³` **leaf brick**,
//! and `L ≥ 2` are the `4³` **internal** nodes up to `L = k + 1` (the coarsest
//! internal level, `COARSE`). These formulas are the corrected ones from
//! `idea.md` §7.1/§7.2 — the pre-correction draft had two off-by-one bugs that
//! this module's tests pin against.

/// A traversal level index `L` (`idea.md` §7). `0` = voxel, `1` = leaf brick,
/// `≥ 2` = internal `4³` node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Level(u32);

impl Level {
    /// The terminal level: an individual voxel. A set voxel bit is a hit.
    pub const VOXEL: Level = Level(0);
    /// The `8³` leaf brick level.
    pub const LEAF_BRICK: Level = Level(1);

    /// An internal `4³` node level. `l` must be `≥ 2`.
    ///
    /// # Panics
    /// Panics if `l < 2` — internal levels start at 2 by definition, so a
    /// smaller value is a programmer error, not runtime data.
    #[must_use]
    pub const fn internal(l: u32) -> Level {
        assert!(
            l >= 2,
            "internal levels start at L = 2 (0 = voxel, 1 = leaf brick)"
        );
        Level(l)
    }

    /// The coarsest internal level (`COARSE`) for a structure with `k` internal
    /// levels: `L = k + 1` (`idea.md` §4).
    #[must_use]
    pub const fn coarse(k: u32) -> Level {
        Level(k + 1)
    }

    /// The raw level index `L`.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }

    /// `true` iff this is the terminal voxel level.
    #[must_use]
    pub const fn is_voxel(self) -> bool {
        self.0 == 0
    }

    /// The finer level (`L − 1`), reached on descent.
    ///
    /// # Panics
    /// Panics at the voxel level, which has no finer level — descending past a
    /// hit is a traversal bug.
    #[must_use]
    pub const fn finer(self) -> Level {
        assert!(self.0 > 0, "the voxel level (L=0) has no finer level");
        Level(self.0 - 1)
    }

    /// The coarser level (`L + 1`), reached on ascent.
    #[must_use]
    pub const fn coarser(self) -> Level {
        Level(self.0 + 1)
    }

    /// The `align` shift such that `cell_size = 1 << align` (`idea.md` §7.1):
    /// `0` at the voxel, else `2L + 1`. The voxel→brick step is therefore `×8`
    /// (align `0 → 3`) and each internal step is `×4` (align `+2`).
    #[must_use]
    pub const fn align(self) -> u32 {
        if self.0 == 0 { 0 } else { 2 * self.0 + 1 }
    }

    /// Cell edge length in **base voxels** (`idea.md` §7.1): `1` at the voxel,
    /// `8` at the leaf brick, then `32, 128, …`. Equals `2^align`.
    #[must_use]
    pub const fn cell_size(self) -> u64 {
        1u64 << self.align()
    }

    /// Number of low base-voxel address bits that identify the **parent**
    /// cell's origin: `2L + 3` (`idea.md` §7.2). Used by the stackless ascent
    /// test. The parent of the voxel (`L=0`) is the `8³` brick at `3` bits (the
    /// `×8` step); a brick/internal parent is at `2(L+1)+1 = 2L+3` bits.
    #[must_use]
    pub const fn parent_bits(self) -> u32 {
        2 * self.0 + 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_size_pins_idea_md_section_7_1() {
        assert_eq!(Level::VOXEL.cell_size(), 1);
        assert_eq!(Level::LEAF_BRICK.cell_size(), 8);
        assert_eq!(Level::internal(2).cell_size(), 32);
        assert_eq!(Level::internal(3).cell_size(), 128);
        assert_eq!(Level::internal(4).cell_size(), 512);
    }

    #[test]
    fn parent_bits_pins_idea_md_section_7_2() {
        // Unconditional 2L+3; correct at the voxel (3, the ×8 step) too.
        assert_eq!(Level::VOXEL.parent_bits(), 3);
        assert_eq!(Level::LEAF_BRICK.parent_bits(), 5);
        assert_eq!(Level::internal(2).parent_bits(), 7);
        assert_eq!(Level::internal(3).parent_bits(), 9);
    }

    #[test]
    fn parent_extent_is_parent_cell_size() {
        // The parent's cell size is 2^parent_bits, i.e. cell_size of L+1.
        for l in 0u32..6 {
            let lvl = if l == 0 {
                Level::VOXEL
            } else if l == 1 {
                Level::LEAF_BRICK
            } else {
                Level::internal(l)
            };
            assert_eq!(1u64 << lvl.parent_bits(), lvl.coarser().cell_size());
        }
    }

    #[test]
    fn coarse_is_k_plus_one() {
        assert_eq!(Level::coarse(3).index(), 4); // 512³: COARSE = L4
        assert_eq!(Level::coarse(4).index(), 5); // 2048³: COARSE = L5
    }

    #[test]
    fn descend_then_ascend_is_identity() {
        let l = Level::internal(3);
        assert_eq!(l.finer().coarser(), l);
    }
}
