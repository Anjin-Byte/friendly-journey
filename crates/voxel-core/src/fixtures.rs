//! Procedural occupancy fixtures with known properties.
//!
//! These are the inputs for tests, the differential harness, and the §10
//! measurements. All are procedural (no `n³` allocation) so they work at any
//! resolution. The star is [`OctantFractal`], a scale-invariant fractal whose
//! box-counting dimension is exactly `log₂(|keep|)` — a *known* ground truth
//! for the §10 dimension estimator.

use crate::{OccupancyField, Resolution, VoxelCoord};

/// Empty everywhere. The MISS-only fixture.
#[derive(Debug, Clone, Copy)]
pub struct Empty {
    /// Grid resolution.
    pub resolution: Resolution,
}

impl OccupancyField for Empty {
    fn resolution(&self) -> Resolution {
        self.resolution
    }
    fn is_occupied(&self, _c: VoxelCoord) -> bool {
        false
    }
}

/// Solid everywhere in bounds (`D = 3`). The first in-bounds voxel a ray enters
/// is always a hit.
#[derive(Debug, Clone, Copy)]
pub struct Solid {
    /// Grid resolution.
    pub resolution: Resolution,
}

impl OccupancyField for Solid {
    fn resolution(&self) -> Resolution {
        self.resolution
    }
    fn is_occupied(&self, c: VoxelCoord) -> bool {
        c.in_bounds(self.resolution)
    }
}

/// A single occupied voxel. Pins exact-hit behavior in tests.
#[derive(Debug, Clone, Copy)]
pub struct SingleVoxel {
    /// Grid resolution.
    pub resolution: Resolution,
    /// The one occupied voxel.
    pub voxel: VoxelCoord,
}

impl OccupancyField for SingleVoxel {
    fn resolution(&self) -> Resolution {
        self.resolution
    }
    fn is_occupied(&self, c: VoxelCoord) -> bool {
        c == self.voxel && c.in_bounds(self.resolution)
    }
}

/// 3-D checkerboard: occupied where `x + y + z` is even (`D = 3`, half-filled).
/// Adversarial for traversal — dense structure, frequent near-grazing hits.
#[derive(Debug, Clone, Copy)]
pub struct Checkerboard {
    /// Grid resolution.
    pub resolution: Resolution,
}

impl OccupancyField for Checkerboard {
    fn resolution(&self) -> Resolution {
        self.resolution
    }
    fn is_occupied(&self, c: VoxelCoord) -> bool {
        c.in_bounds(self.resolution) && (c.x + c.y + c.z).is_multiple_of(2)
    }
}

/// A scale-invariant octant fractal on the `2³` subdivision.
///
/// A voxel is occupied iff, at **every** bit level `b`, its octant index
/// `oct_b = bit_b(x) | bit_b(y)<<1 | bit_b(z)<<2` is a member of the `keep`
/// mask (an 8-bit set over octants `0..8`). Halving the box size multiplies the
/// occupied count by `|keep|`, so the box-counting dimension is exactly
/// `D = log₂(|keep|)` — known ground truth for the §10 estimator.
///
/// `keep = 0x69` ({0,3,5,6}, the even-parity octants) is a Sierpinski
/// tetrahedron, `D = 2`.
#[derive(Debug, Clone, Copy)]
pub struct OctantFractal {
    /// Grid resolution.
    pub resolution: Resolution,
    /// Bitmask over the 8 octants; bit `i` set ⇒ octant `i` is kept.
    pub keep: u8,
}

impl OctantFractal {
    /// Sierpinski tetrahedron: keep the four even-parity octants. `D = 2`.
    #[must_use]
    pub const fn sierpinski_tetrahedron(resolution: Resolution) -> Self {
        // octants 0 (000), 3 (011), 5 (101), 6 (110): even popcount.
        Self {
            resolution,
            keep: 0b0110_1001,
        }
    }

    /// A sparse "Cantor dust": keep two opposite-corner octants. `D = 1`.
    #[must_use]
    pub const fn cantor_dust(resolution: Resolution) -> Self {
        // octants 0 (000) and 7 (111).
        Self {
            resolution,
            keep: 0b1000_0001,
        }
    }

    /// The fractal's exact box-counting dimension, `log₂(|keep|)`.
    #[must_use]
    pub fn dimension(self) -> f64 {
        f64::from(self.keep.count_ones()).log2()
    }
}

impl OccupancyField for OctantFractal {
    fn resolution(&self) -> Resolution {
        self.resolution
    }

    fn is_occupied(&self, c: VoxelCoord) -> bool {
        if !c.in_bounds(self.resolution) {
            return false;
        }
        let levels = self.resolution.axis_bits();
        for b in 0..levels {
            let oct = ((c.x >> b) & 1) | (((c.y >> b) & 1) << 1) | (((c.z >> b) & 1) << 2);
            if (self.keep >> oct) & 1 == 0 {
                return false;
            }
        }
        true
    }
}

/// A 3-D wireframe lattice: thin wires along the edges of a `period`-spaced
/// grid. **Traversal-pathology stress** (analysis #2): a wire-bearing brick
/// exists almost everywhere, so rays *descend* into it — but the wire is only
/// `thickness` voxels, so they usually *miss* and *ascend*. That descend → walk
/// → miss → ascend churn maximizes the §10 descent-frequency and cell-step
/// counts, and the thin wires maximize grazing. It is *not* self-similar, so
/// the §10 dimension fit shows a kink rather than a clean slope.
#[derive(Debug, Clone, Copy)]
pub struct WireLattice {
    /// Grid resolution.
    pub resolution: Resolution,
    /// Spacing of the wire grid in voxels (power-of-two recommended).
    pub period: u32,
    /// Wire half-extent in voxels.
    pub thickness: u32,
}

impl WireLattice {
    /// A lattice with `period = 16` (two bricks) and 1-voxel-thick wires.
    #[must_use]
    pub const fn new(resolution: Resolution) -> Self {
        Self {
            resolution,
            period: 16,
            thickness: 1,
        }
    }
}

impl OccupancyField for WireLattice {
    fn resolution(&self) -> Resolution {
        self.resolution
    }

    fn is_occupied(&self, c: VoxelCoord) -> bool {
        if !c.in_bounds(self.resolution) {
            return false;
        }
        // "near a grid line" on each axis.
        let nx = c.x % self.period < self.thickness;
        let ny = c.y % self.period < self.thickness;
        let nz = c.z % self.period < self.thickness;
        // A wire runs where two axes are on a grid line (the 12 cube edges).
        u32::from(nx) + u32::from(ny) + u32::from(nz) >= 2
    }
}

/// Uniform sparse "dust": pseudo-random voxels at ~`1/density` occupancy.
/// **Warp-divergence stress** (analysis #4): adjacent pixels' rays hit
/// scattered voxels at unrelated depths (or miss while neighbors hit), so warp
/// lanes desynchronize — the case the coherent-primary assumption breaks. Also
/// near-worst-case for the f32 grazing differential. Deterministic via a hash,
/// so builds and tests are reproducible.
#[derive(Debug, Clone, Copy)]
pub struct Dust {
    /// Grid resolution.
    pub resolution: Resolution,
    /// Inverse density: a voxel is occupied iff `hash % density == 0`.
    pub density: u32,
}

impl Dust {
    /// Dust at ~`1/4096` occupancy — sparse enough to fit at `512³`, dense
    /// enough that most camera rays diverge.
    #[must_use]
    pub const fn new(resolution: Resolution) -> Self {
        Self {
            resolution,
            density: 4096,
        }
    }
}

/// A deterministic, reasonably-uniform 3-D integer hash.
fn hash3(x: u32, y: u32, z: u32) -> u32 {
    let mut h =
        x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841) ^ z.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h
}

impl OccupancyField for Dust {
    fn resolution(&self) -> Resolution {
        self.resolution
    }

    fn is_occupied(&self, c: VoxelCoord) -> bool {
        c.in_bounds(self.resolution) && hash3(c.x, c.y, c.z).is_multiple_of(self.density)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn res(n: u32) -> Resolution {
        Resolution::new(n).unwrap()
    }

    #[test]
    fn wire_lattice_is_wires_not_planes() {
        let w = WireLattice::new(res(128));
        // On a wire (x and y on the grid, any z): occupied.
        assert!(w.is_occupied(VoxelCoord::new(0, 0, 5)));
        // On a single plane (only x on the grid): empty.
        assert!(!w.is_occupied(VoxelCoord::new(0, 5, 5)));
        // Deep interior of a period cell: empty.
        assert!(!w.is_occupied(VoxelCoord::new(8, 8, 8)));
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn dust_density_is_roughly_right() {
        let d = Dust {
            resolution: res(128),
            density: 64,
        };
        let n = 128u32;
        let mut count = 0u64;
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    if d.is_occupied(VoxelCoord::new(x, y, z)) {
                        count += 1;
                    }
                }
            }
        }
        let total = u64::from(n).pow(3);
        let frac = count as f64 / total as f64;
        // Expect ~1/64 ≈ 0.0156; allow a generous band for hash non-uniformity.
        assert!(
            frac > 0.008 && frac < 0.025,
            "dust fraction {frac} off ~1/64"
        );
    }

    #[test]
    fn checkerboard_parity() {
        let cb = Checkerboard { resolution: res(8) };
        assert!(cb.is_occupied(VoxelCoord::new(0, 0, 0)));
        assert!(!cb.is_occupied(VoxelCoord::new(1, 0, 0)));
        assert!(cb.is_occupied(VoxelCoord::new(1, 1, 0)));
    }

    #[test]
    fn fractal_dimension_matches_octant_count() {
        let r = res(512);
        assert!((OctantFractal::sierpinski_tetrahedron(r).dimension() - 2.0).abs() < 1e-12);
        assert!((OctantFractal::cantor_dust(r).dimension() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fractal_occupied_count_is_keep_to_the_levels() {
        // On an n³ grid with L = log2(n) levels, |occupied| = |keep|^L.
        for n in [8u32, 32, 128] {
            let r = res(n);
            let frac = OctantFractal::sierpinski_tetrahedron(r);
            let levels = r.axis_bits();
            let expected = u64::from(frac.keep.count_ones()).pow(levels);
            let mut count = 0u64;
            for z in 0..n {
                for y in 0..n {
                    for x in 0..n {
                        if frac.is_occupied(VoxelCoord::new(x, y, z)) {
                            count += 1;
                        }
                    }
                }
            }
            assert_eq!(count, expected, "n={n}");
        }
    }

    #[test]
    fn fractal_origin_voxel_always_kept_when_octant_zero_in_mask() {
        // (0,0,0) has octant 0 at every level; kept by both presets.
        let r = res(128);
        assert!(OctantFractal::sierpinski_tetrahedron(r).is_occupied(VoxelCoord::new(0, 0, 0)));
        assert!(OctantFractal::cantor_dust(r).is_occupied(VoxelCoord::new(0, 0, 0)));
    }
}
