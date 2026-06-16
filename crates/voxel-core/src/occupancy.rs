//! Binary occupancy: the input field the structure is built from.
//!
//! [`OccupancyField`] is queried per voxel, so fixtures can be **procedural**
//! (computed on demand) and need not materialize `n³` bits — essential at
//! `2048³`, where a dense grid would be a gigabyte. [`BitGrid`] is the dense,
//! materialized implementation for small grids and for caching a procedural
//! field.

use crate::{Resolution, VoxelCoord};

/// A binary occupancy field over a [`Resolution`]-sized grid.
///
/// Implementors must return `false` for any out-of-bounds coordinate so callers
/// can probe freely near the grid edge.
pub trait OccupancyField {
    /// The grid resolution this field is defined over.
    fn resolution(&self) -> Resolution;

    /// Whether voxel `c` is occupied. Must be `false` when `c` is out of bounds.
    fn is_occupied(&self, c: VoxelCoord) -> bool;
}

/// A dense bitset occupancy grid: one bit per voxel, `n³` bits total.
///
/// Intended for small resolutions (tests, fixtures, oracle inputs) and for
/// materializing a procedural [`OccupancyField`]. At large resolutions prefer a
/// procedural field — a dense `2048³` grid is ~1 GiB.
#[derive(Debug, Clone)]
pub struct BitGrid {
    resolution: Resolution,
    /// `n³` bits packed into 64-bit words, little-endian within each word.
    words: Vec<u64>,
}

impl BitGrid {
    /// An all-empty grid at `resolution`.
    ///
    /// # Panics
    /// Panics if `n³` bits would not fit in addressable memory (`usize`). This
    /// is a deliberate guard against materializing an absurd dense grid; use a
    /// procedural [`OccupancyField`] instead.
    #[must_use]
    pub fn empty(resolution: Resolution) -> Self {
        let bits = resolution.total_voxels();
        let words = usize::try_from(bits.div_ceil(64))
            .expect("dense BitGrid too large for this platform; use a procedural field");
        Self {
            resolution,
            words: vec![0; words],
        }
    }

    /// Builds a grid directly from pre-computed packed words — e.g. a GPU
    /// occupancy generator that evaluated the field in parallel and read the bits
    /// back. `words` must be exactly `ceil(n³/64)` `u64`s in the same
    /// `x + y·n + z·n²` bit order [`set`](Self::set) uses (bit `i` in word `i/64`,
    /// little-endian within the word).
    ///
    /// # Panics
    /// Panics if `words.len()` is not the exact word count for `resolution` —
    /// a layout-contract violation, caught at the boundary rather than silently
    /// producing a corrupt grid.
    #[must_use]
    pub fn from_raw(resolution: Resolution, words: Vec<u64>) -> Self {
        let expected = usize::try_from(resolution.total_voxels().div_ceil(64))
            .expect("dense BitGrid too large for this platform; use a procedural field");
        assert_eq!(
            words.len(),
            expected,
            "BitGrid::from_raw word count mismatch for {}³",
            resolution.voxels_per_axis()
        );
        Self { resolution, words }
    }

    /// Materializes a procedural field into a dense grid (small resolutions).
    #[must_use]
    pub fn from_field<F: OccupancyField>(field: &F) -> Self {
        let mut grid = Self::empty(field.resolution());
        let n = field.resolution().voxels_per_axis();
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let c = VoxelCoord::new(x, y, z);
                    if field.is_occupied(c) {
                        grid.set(c);
                    }
                }
            }
        }
        grid
    }

    /// Linear bit index `x + y·n + z·n²`, or `None` if out of bounds.
    fn linear_index(&self, c: VoxelCoord) -> Option<u64> {
        if !c.in_bounds(self.resolution) {
            return None;
        }
        let n = u64::from(self.resolution.voxels_per_axis());
        Some(u64::from(c.x) + u64::from(c.y) * n + u64::from(c.z) * n * n)
    }

    /// Sets voxel `c` (no-op if out of bounds).
    pub fn set(&mut self, c: VoxelCoord) {
        if let Some(i) = self.linear_index(c) {
            self.words[(i / 64) as usize] |= 1u64 << (i % 64);
        }
    }

    /// Clears voxel `c` (no-op if out of bounds).
    pub fn clear(&mut self, c: VoxelCoord) {
        if let Some(i) = self.linear_index(c) {
            self.words[(i / 64) as usize] &= !(1u64 << (i % 64));
        }
    }

    /// Number of occupied voxels.
    #[must_use]
    pub fn count_occupied(&self) -> u64 {
        self.words.iter().map(|w| u64::from(w.count_ones())).sum()
    }
}

impl OccupancyField for BitGrid {
    fn resolution(&self) -> Resolution {
        self.resolution
    }

    fn is_occupied(&self, c: VoxelCoord) -> bool {
        match self.linear_index(c) {
            Some(i) => (self.words[(i / 64) as usize] >> (i % 64)) & 1 == 1,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn res(n: u32) -> Resolution {
        Resolution::new(n).unwrap()
    }

    #[test]
    fn set_get_round_trip() {
        let mut g = BitGrid::empty(res(32));
        let c = VoxelCoord::new(3, 17, 31);
        assert!(!g.is_occupied(c));
        g.set(c);
        assert!(g.is_occupied(c));
        assert_eq!(g.count_occupied(), 1);
        g.clear(c);
        assert!(!g.is_occupied(c));
    }

    #[test]
    fn out_of_bounds_reads_false_and_writes_noop() {
        let mut g = BitGrid::empty(res(8));
        let oob = VoxelCoord::new(8, 0, 0);
        g.set(oob);
        assert!(!g.is_occupied(oob));
        assert_eq!(g.count_occupied(), 0);
    }

    #[test]
    fn distinct_voxels_do_not_alias() {
        // A regression guard on the linear-index packing.
        let mut g = BitGrid::empty(res(8));
        g.set(VoxelCoord::new(7, 0, 0));
        assert!(!g.is_occupied(VoxelCoord::new(0, 1, 0)));
        assert!(!g.is_occupied(VoxelCoord::new(0, 0, 1)));
    }
}
