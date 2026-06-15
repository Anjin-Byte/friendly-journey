//! The grid resolution: a per-axis voxel count constrained to `8 · 4^k`.

use thiserror::Error;

/// Smallest representable resolution: `8³` (`k = 0`, a single leaf brick).
const MIN_RES: u32 = 8;
/// Largest resolution whose `8·4^k` value still fits in `u32`: `8 · 4¹⁴ = 2³¹`.
const MAX_RES: u32 = 1 << 31;
/// Largest internal-level count `k` with `8·4^k ≤ u32::MAX`.
const MAX_K: u32 = 14;

/// A valid grid resolution: `voxels_per_axis = 8 · 4^k` for some `k ≥ 0`.
///
/// The sparse MIP structure only represents resolutions of this form
/// (`idea.md` §4): the `8` is the `8³` leaf brick (3 bits/axis) and each factor
/// of `4` is one `4³` internal level (2 bits/axis). `1024³` is famously *not*
/// representable — `1024 / 8 = 128` is not a power of four — and is rejected by
/// [`Resolution::new`].
///
/// Construct via [`Resolution::new`] (checked) or
/// [`Resolution::from_internal_levels`].
///
/// ```
/// use voxel_core::Resolution;
/// assert_eq!(Resolution::new(512).unwrap().voxels_per_axis(), 512);
/// assert!(Resolution::new(1024).is_err()); // not 8·4^k
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Resolution {
    /// Voxels per axis, equal to `8 · 4^k`.
    n: u32,
    /// Number of internal `4³` levels (`k`).
    k: u32,
}

/// Why a requested resolution is not representable by the structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ResolutionError {
    /// Below the minimum `8³` grid (includes zero).
    #[error("resolution {requested} is too small; the minimum is {minimum}")]
    TooSmall {
        /// The rejected value.
        requested: u32,
        /// The smallest representable resolution (`8`).
        minimum: u32,
    },
    /// Above the largest `8·4^k` that fits in `u32`.
    #[error("resolution {requested} is too large; the maximum is {maximum}")]
    TooLarge {
        /// The rejected value.
        requested: u32,
        /// The largest representable resolution (`2³¹`).
        maximum: u32,
    },
    /// In range, but not of the form `8 · 4^k`. Reports the bracketing valid
    /// resolutions so the caller can round (this is how `1024 → 512 | 2048`).
    #[error(
        "resolution {requested} is not representable (must be 8·4^k); \
         nearest valid resolutions are {nearest_below} and {nearest_above}"
    )]
    NotRepresentable {
        /// The rejected value.
        requested: u32,
        /// Largest valid resolution `< requested`.
        nearest_below: u32,
        /// Smallest valid resolution `> requested`.
        nearest_above: u32,
    },
}

impl Resolution {
    /// Checks `n` and constructs a [`Resolution`], or explains why `n` is
    /// invalid.
    ///
    /// Valid iff `n = 8 · 4^k` for some `k ∈ [0, 14]`.
    pub fn new(n: u32) -> Result<Self, ResolutionError> {
        if n < MIN_RES {
            return Err(ResolutionError::TooSmall {
                requested: n,
                minimum: MIN_RES,
            });
        }
        if n > MAX_RES {
            return Err(ResolutionError::TooLarge {
                requested: n,
                maximum: MAX_RES,
            });
        }
        // n = 8 · 4^k  ⇔  m = n/8 is a power of four ⇔ a power of two with an
        // even number of trailing zeros.
        if n.is_multiple_of(8) {
            let m = n / 8;
            if m.is_power_of_two() && m.trailing_zeros().is_multiple_of(2) {
                return Ok(Self {
                    n,
                    k: m.trailing_zeros() / 2,
                });
            }
        }
        // In range but not representable: bracket it with the two nearest valid
        // resolutions. `r` walks 8, 32, 128, … so `below < n < above` once `r`
        // first exceeds `n` (n is not itself valid here, so the inequalities
        // are strict).
        // Every value stays ≤ MAX_RES = 2³¹ (the loop exits before `r` could
        // grow past it), so plain u32 arithmetic never overflows here.
        let (mut below, mut above) = (MIN_RES, MIN_RES);
        let mut r = MIN_RES;
        while r < n {
            below = r;
            r *= 4;
            above = r;
        }
        Err(ResolutionError::NotRepresentable {
            requested: n,
            nearest_below: below,
            nearest_above: above,
        })
    }

    /// Constructs the resolution with exactly `k` internal `4³` levels:
    /// `8 · 4^k`. Errors if `k > 14` (would overflow `u32`).
    ///
    /// ```
    /// use voxel_core::Resolution;
    /// assert_eq!(Resolution::from_internal_levels(3).unwrap().voxels_per_axis(), 512);
    /// assert_eq!(Resolution::from_internal_levels(4).unwrap().voxels_per_axis(), 2048);
    /// ```
    pub fn from_internal_levels(k: u32) -> Result<Self, ResolutionError> {
        if k > MAX_K {
            return Err(ResolutionError::TooLarge {
                requested: u32::MAX,
                maximum: MAX_RES,
            });
        }
        // 8 · 4^k = 2^(3 + 2k).
        Ok(Self {
            n: 1 << (3 + 2 * k),
            k,
        })
    }

    /// Voxels per axis (`n`).
    #[must_use]
    pub const fn voxels_per_axis(self) -> u32 {
        self.n
    }

    /// Number of internal `4³` levels (`k`).
    #[must_use]
    pub const fn internal_levels(self) -> u32 {
        self.k
    }

    /// Number of *storage* levels: `k` internal nodes plus the leaf brick
    /// (`idea.md` §4, "Storage levels = k + 1"). Excludes the voxel terminal.
    #[must_use]
    pub const fn storage_levels(self) -> u32 {
        self.k + 1
    }

    /// Number of *traversal* levels: storage levels plus the voxel terminal
    /// (`L = 0`). Equals `k + 2` (`idea.md` §4).
    #[must_use]
    pub const fn traversal_levels(self) -> u32 {
        self.k + 2
    }

    /// Total voxel count `n³`. Returned as `u128` because `n³` exceeds `u64`
    /// for the largest resolutions.
    #[must_use]
    pub const fn total_voxels(self) -> u128 {
        let n = self.n as u128;
        n * n * n
    }

    /// `log₂(n)` — the number of base-voxel address bits per axis.
    #[must_use]
    pub const fn axis_bits(self) -> u32 {
        self.n.trailing_zeros()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_resolutions_round_trip_through_k() {
        for k in 0..=MAX_K {
            let r = Resolution::from_internal_levels(k).unwrap();
            assert_eq!(r.internal_levels(), k);
            assert_eq!(r.voxels_per_axis(), 8 * 4u32.pow(k));
            // new() agrees with from_internal_levels().
            assert_eq!(Resolution::new(r.voxels_per_axis()).unwrap(), r);
        }
    }

    #[test]
    fn named_resolutions() {
        assert_eq!(Resolution::new(8).unwrap().internal_levels(), 0);
        assert_eq!(Resolution::new(512).unwrap().internal_levels(), 3);
        assert_eq!(Resolution::new(2048).unwrap().internal_levels(), 4);
    }

    #[test]
    fn level_counts_match_idea_md_section_4() {
        // 512³: 4 storage levels (k+1), 5 traversal levels (k+2).
        let r = Resolution::new(512).unwrap();
        assert_eq!(r.storage_levels(), 4);
        assert_eq!(r.traversal_levels(), 5);
        // 2048³: 5 storage, 6 traversal.
        let r = Resolution::new(2048).unwrap();
        assert_eq!(r.storage_levels(), 5);
        assert_eq!(r.traversal_levels(), 6);
    }

    #[test]
    fn rejects_1024_with_512_and_2048_bracket() {
        match Resolution::new(1024) {
            Err(ResolutionError::NotRepresentable {
                requested,
                nearest_below,
                nearest_above,
            }) => {
                assert_eq!(requested, 1024);
                assert_eq!(nearest_below, 512);
                assert_eq!(nearest_above, 2048);
            }
            other => panic!("expected NotRepresentable, got {other:?}"),
        }
    }

    #[test]
    fn rejects_too_small_and_zero() {
        assert!(matches!(
            Resolution::new(0),
            Err(ResolutionError::TooSmall { .. })
        ));
        assert!(matches!(
            Resolution::new(7),
            Err(ResolutionError::TooSmall { .. })
        ));
    }

    #[test]
    fn rejects_non_power_of_four_multiples_of_eight() {
        // 8·2 = 16 is a multiple of 8 but 16/8 = 2 is not a power of four.
        assert!(matches!(
            Resolution::new(16),
            Err(ResolutionError::NotRepresentable { .. })
        ));
        // 24, 40, … likewise.
        assert!(Resolution::new(24).is_err());
    }

    #[test]
    fn total_voxels_does_not_overflow_for_2048() {
        let r = Resolution::new(2048).unwrap();
        assert_eq!(r.total_voxels(), 2048u128.pow(3));
    }
}
