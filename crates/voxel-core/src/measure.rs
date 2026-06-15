//! P3.5: the §10 measurement harness.
//!
//! Three measurements gate the design (`idea.md` §10, review R5):
//!
//! 1. **Box-counting dimension `D`** — the slope of `ln N(L)` vs `ln(cell_size)`
//!    over the MIP pyramid. `D` near 3 means near-solid (abandon sparsity);
//!    low-to-moderate `D` justifies the sparse structure. The fit's `R²`
//!    reports how scale-invariant the field actually is.
//! 2. **Per-level footprint** — stored bytes per level and the cumulative
//!    coarse footprint, against the GPU L2 cache. This drives the School-A vs
//!    School-B call: if the working set fits in L2, descent jumps are cheap and
//!    School A suffices.
//! 3. **Descent frequency** — how many levels a representative ray descends,
//!    the other half of the School-A vs School-B input.
//!
//! All measurements are pure and read off the built [`SparseTree`].

use std::fmt;

use glam::DVec3;

use crate::Resolution;
use crate::ray::Ray;
use crate::sparse::SparseTree;

/// Storage bytes for one stored `4³` node ([`crate::GpuNode`]): three `u32`.
pub const NODE_BYTES: u64 = 12;
/// Storage bytes for one `8³` leaf brick: 512 bits.
pub const LEAF_BYTES: u64 = 64;
/// A representative GPU L2 slice for the cache-residency verdict (4 MiB).
pub const REPRESENTATIVE_L2: u64 = 4 << 20;

/// Occupancy and storage at one level.
#[derive(Debug, Clone, Copy)]
pub struct LevelFootprint {
    /// Traversal level `L` (`0` = voxel, `1` = leaf brick, `≥2` = internal).
    pub level: u32,
    /// Cell edge in base voxels.
    pub cell_size: u64,
    /// Non-empty cells at this level, `N(L)`.
    pub count: u64,
    /// Stored bytes at this level (`0` at the voxel level — voxels live inside
    /// leaves, not as separate cells).
    pub bytes: u64,
}

/// The box-counting dimension fit.
#[derive(Debug, Clone, Copy)]
pub struct DimensionFit {
    /// Estimated `D` (negated slope of `ln N` vs `ln(cell_size)`).
    pub dimension: f64,
    /// Coefficient of determination `R²` (closer to 1 ⇒ more scale-invariant).
    pub r_squared: f64,
    /// Number of levels used in the fit.
    pub points: usize,
}

/// Per-ray descent-frequency summary.
#[derive(Debug, Clone, Copy)]
pub struct DescentStats {
    /// Rays cast.
    pub rays_cast: u64,
    /// Rays that hit.
    pub rays_hit: u64,
    /// Mean cells descended into per ray.
    pub mean_descents: f64,
    /// Maximum cells descended into by any single ray.
    pub max_descents: u64,
    /// Mean DDA cell-steps per ray.
    pub mean_cell_steps: f64,
}

/// The full §10 report for one built structure.
#[derive(Debug, Clone)]
pub struct Report {
    /// Grid resolution (voxels per axis).
    pub resolution_n: u32,
    /// Per-level occupancy and footprint, coarsest first.
    pub levels: Vec<LevelFootprint>,
    /// Total stored bytes (nodes + leaves).
    pub total_bytes: u64,
    /// The dimension fit.
    pub dimension: DimensionFit,
    /// The descent-frequency summary.
    pub descent: DescentStats,
}

/// Cell edge in base voxels at level `L` (`idea.md` §7.1).
const fn cell_size(level: u32) -> u64 {
    match level {
        0 => 1,
        1 => 8,
        l => 1u64 << (2 * l + 1),
    }
}

/// Ordinary least squares: `(slope, r_squared)` for points `(x, y)`.
#[allow(clippy::cast_precision_loss)]
fn least_squares(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let sx: f64 = points.iter().map(|p| p.0).sum();
    let sy: f64 = points.iter().map(|p| p.1).sum();
    let sxx: f64 = points.iter().map(|p| p.0 * p.0).sum();
    let syy: f64 = points.iter().map(|p| p.1 * p.1).sum();
    let sxy: f64 = points.iter().map(|p| p.0 * p.1).sum();
    let cov = n * sxy - sx * sy;
    let var_x = n * sxx - sx * sx;
    let var_y = n * syy - sy * sy;
    let slope = cov / var_x;
    let r2 = if var_y == 0.0 {
        1.0
    } else {
        (cov * cov) / (var_x * var_y)
    };
    (slope, r2)
}

/// Per-level occupancy and footprint, coarsest first.
#[must_use]
#[allow(clippy::cast_precision_loss)] // counts well below 2^53
pub fn per_level_footprint(tree: &SparseTree) -> Vec<LevelFootprint> {
    let coarse = tree.coarse_level();
    let mut levels = Vec::new();
    for level in (0..=coarse).rev() {
        let (count, bytes) = match level {
            0 => (tree.occupied_voxels(), 0),
            1 => {
                let c = tree.leaf_count() as u64;
                (c, c * LEAF_BYTES)
            }
            l => {
                let c = tree.nodes_at_level(l) as u64;
                (c, c * NODE_BYTES)
            }
        };
        levels.push(LevelFootprint {
            level,
            cell_size: cell_size(level),
            count,
            bytes,
        });
    }
    levels
}

/// Estimates the box-counting dimension from the per-level occupancy.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn estimate_dimension(tree: &SparseTree) -> DimensionFit {
    // Points (ln cell_size, ln N(L)) for every level with occupancy. Slope is
    // −D because finer cells (smaller cell_size) hold proportionally more cells.
    let points: Vec<(f64, f64)> = per_level_footprint(tree)
        .iter()
        .filter(|f| f.count > 0)
        .map(|f| ((f.cell_size as f64).ln(), (f.count as f64).ln()))
        .collect();
    if points.len() < 2 {
        return DimensionFit {
            dimension: f64::NAN,
            r_squared: f64::NAN,
            points: points.len(),
        };
    }
    let (slope, r2) = least_squares(&points);
    DimensionFit {
        dimension: -slope,
        r_squared: r2,
        points: points.len(),
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[allow(clippy::cast_precision_loss)]
fn unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Casts `n_rays` deterministic camera-like primary rays at the structure and
/// summarizes how far they descend.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn descent_stats(tree: &SparseTree, n_rays: u64) -> DescentStats {
    let nf = f64::from(tree.resolution().voxels_per_axis());
    let center = DVec3::splat(nf * 0.5);
    let mut rng = 0x5DEE_CE66_D3A9_7F1Bu64;

    let mut total_descents = 0u64;
    let mut max_descents = 0u64;
    let mut total_steps = 0u64;
    let mut hits = 0u64;

    for _ in 0..n_rays {
        // Origin in a shell around the grid; aim at the centre with jitter so
        // most rays actually enter (representative primary rays).
        let origin = DVec3::new(
            unit(&mut rng) * (2.0 * nf) - 0.5 * nf,
            unit(&mut rng) * (2.0 * nf) - 0.5 * nf,
            unit(&mut rng) * (2.0 * nf) - 0.5 * nf,
        );
        let jitter = DVec3::new(
            unit(&mut rng) - 0.5,
            unit(&mut rng) - 0.5,
            unit(&mut rng) - 0.5,
        ) * (nf * 0.25);
        let dir = (center + jitter) - origin;
        if dir.length() < 1e-6 {
            continue;
        }
        let (hit, stats) = tree.traverse_counted(&Ray::new(origin, dir));
        total_descents += stats.descents;
        max_descents = max_descents.max(stats.descents);
        total_steps += stats.cell_steps;
        if hit.is_some() {
            hits += 1;
        }
    }

    let denom = n_rays.max(1) as f64;
    DescentStats {
        rays_cast: n_rays,
        rays_hit: hits,
        mean_descents: total_descents as f64 / denom,
        max_descents,
        mean_cell_steps: total_steps as f64 / denom,
    }
}

/// One view direction's traversal cost in the anisotropy sweep.
#[derive(Debug, Clone, Copy)]
pub struct DirStat {
    /// The unit view direction.
    pub dir: DVec3,
    /// Mean DDA cell-steps per ray over the orthographic batch (the algorithmic
    /// cost from this direction — DDA crossings plus the early-skip).
    pub mean_cell_steps: f64,
    /// Maximum cell-steps by any single ray.
    pub max_cell_steps: u64,
    /// Fraction of the batch that hit.
    pub hit_frac: f64,
}

/// `n` roughly-uniform unit directions on the sphere via the Fibonacci lattice
/// (deterministic, no clustering at the poles).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn fibonacci_dirs(n: usize) -> Vec<DVec3> {
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let theta = golden * i as f64;
            DVec3::new(r * theta.cos(), y, r * theta.sin())
        })
        .collect()
}

/// An orthographic batch of `side²` parallel rays along `dir`, tiled across the
/// grid's bounding sphere. Using the bounding sphere (not the projected box)
/// keeps the ray count and coverage identical from every direction, so the
/// per-direction costs are directly comparable.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn ortho_rays(resolution: Resolution, dir: DVec3, side: u32) -> Vec<Ray> {
    let n = f64::from(resolution.voxels_per_axis());
    let center = DVec3::splat(n * 0.5);
    let radius = n * 0.5 * 3.0_f64.sqrt();
    let forward = dir.normalize();
    // An up vector not parallel to forward, to build the image plane basis.
    let up_guess = if forward.y.abs() < 0.99 {
        DVec3::Y
    } else {
        DVec3::Z
    };
    let right = forward.cross(up_guess).normalize();
    let up = right.cross(forward);
    let start = center - forward * (radius * 1.5);

    let mut rays = Vec::with_capacity((side as usize) * (side as usize));
    for j in 0..side {
        for i in 0..side {
            let u = (f64::from(i) + 0.5) / f64::from(side) * 2.0 - 1.0;
            let v = (f64::from(j) + 0.5) / f64::from(side) * 2.0 - 1.0;
            let origin = start + right * (u * radius) + up * (v * radius);
            rays.push(Ray::new(origin, forward));
        }
    }
    rays
}

/// Mean traversal cost of an orthographic batch cast along `dir` — one sample of
/// the directional anisotropy profile.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn directional_steps(tree: &SparseTree, dir: DVec3, side: u32) -> DirStat {
    let rays = ortho_rays(tree.resolution(), dir, side);
    let mut total = 0u64;
    let mut max = 0u64;
    let mut hits = 0u64;
    for r in &rays {
        let (hit, s) = tree.traverse_counted(r);
        total += s.cell_steps;
        max = max.max(s.cell_steps);
        if hit.is_some() {
            hits += 1;
        }
    }
    let denom = rays.len().max(1) as f64;
    DirStat {
        dir: dir.normalize(),
        mean_cell_steps: total as f64 / denom,
        max_cell_steps: max,
        hit_frac: hits as f64 / denom,
    }
}

/// Runs all three §10 measurements on a built structure.
#[must_use]
pub fn measure(tree: &SparseTree, n_rays: u64) -> Report {
    let levels = per_level_footprint(tree);
    let total_bytes = levels.iter().map(|l| l.bytes).sum();
    Report {
        resolution_n: tree.resolution().voxels_per_axis(),
        levels,
        total_bytes,
        dimension: estimate_dimension(tree),
        descent: descent_stats(tree, n_rays),
    }
}

impl Report {
    /// The School-A vs School-B suggestion implied by the numbers: School B
    /// only earns its bookkeeping when the working set spills L2 *and* rays
    /// descend often (`idea.md` §6.4 gate). Returns `(prefers_school_b, why)`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn school_b_indicated(&self) -> (bool, String) {
        let spills_l2 = self.total_bytes > REPRESENTATIVE_L2;
        let deep = self.descent.mean_descents >= f64::from(self.coarse_level()) * 0.5;
        let prefers = spills_l2 && deep;
        let why = format!(
            "total {} {} {:.1} MiB L2; mean descents {:.2} {} half of {} levels",
            human_bytes(self.total_bytes),
            if spills_l2 { ">" } else { "≤" },
            REPRESENTATIVE_L2 as f64 / (1u64 << 20) as f64,
            self.descent.mean_descents,
            if deep { "≥" } else { "<" },
            self.coarse_level() + 1,
        );
        (prefers, why)
    }

    fn coarse_level(&self) -> u32 {
        self.levels.first().map_or(0, |l| l.level)
    }
}

#[allow(clippy::cast_precision_loss)]
fn human_bytes(b: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    format!("{v:.2} {}", UNITS[u])
}

impl fmt::Display for Report {
    #[allow(clippy::cast_precision_loss)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "§10 measurements — {n}³", n = self.resolution_n)?;
        writeln!(
            f,
            "  dimension D ≈ {:.3}  (R² = {:.4}, {} levels)",
            self.dimension.dimension, self.dimension.r_squared, self.dimension.points
        )?;
        writeln!(f, "  per-level footprint (coarsest first):")?;
        writeln!(f, "    L  cell³        N(L)         bytes      cumulative")?;
        let mut cumulative = 0u64;
        for lf in &self.levels {
            cumulative += lf.bytes;
            writeln!(
                f,
                "    {:<2} {:>6}³  {:>12}  {:>11}  {:>11}",
                lf.level,
                lf.cell_size,
                lf.count,
                human_bytes(lf.bytes),
                human_bytes(cumulative),
            )?;
        }
        writeln!(f, "  total stored: {}", human_bytes(self.total_bytes))?;
        writeln!(
            f,
            "  descent frequency: mean {:.2}, max {} (over {} rays, {:.0}% hit); mean cell-steps {:.1}",
            self.descent.mean_descents,
            self.descent.max_descents,
            self.descent.rays_cast,
            100.0 * self.descent.rays_hit as f64 / self.descent.rays_cast.max(1) as f64,
            self.descent.mean_cell_steps,
        )?;
        let (prefers_b, why) = self.school_b_indicated();
        writeln!(
            f,
            "  School-{} indicated — {why}",
            if prefers_b { "B" } else { "A" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Resolution;
    use crate::fixtures::{Checkerboard, OctantFractal};

    fn res(n: u32) -> Resolution {
        Resolution::new(n).unwrap()
    }

    #[test]
    fn dimension_recovers_known_fractal_d() {
        // The octant fractal has D = log2(|keep|) exactly; the estimator should
        // recover it to within a small tolerance.
        let tree = SparseTree::build(&OctantFractal::sierpinski_tetrahedron(res(512)));
        let fit = estimate_dimension(&tree);
        assert!(
            (fit.dimension - 2.0).abs() < 0.1,
            "D estimate {} not near 2.0",
            fit.dimension
        );
        assert!(fit.r_squared > 0.99, "fractal should be scale-invariant");
    }

    #[test]
    fn dimension_recovers_near_solid_checkerboard() {
        // Checkerboard fills half of every cell at every scale → D ≈ 3.
        let tree = SparseTree::build(&Checkerboard {
            resolution: res(128),
        });
        let fit = estimate_dimension(&tree);
        assert!(
            fit.dimension > 2.8,
            "checkerboard D {} should be ~3",
            fit.dimension
        );
    }

    #[test]
    fn footprint_levels_are_coarse_first_and_consistent() {
        let tree = SparseTree::build(&OctantFractal::sierpinski_tetrahedron(res(128)));
        let levels = per_level_footprint(&tree);
        assert_eq!(levels.first().unwrap().level, tree.coarse_level());
        assert_eq!(levels.last().unwrap().level, 0);
        // N(L) is non-increasing from fine to coarse.
        for w in levels.windows(2) {
            assert!(
                w[0].count <= w[1].count,
                "occupancy should grow toward fine levels"
            );
        }
    }

    #[test]
    fn fibonacci_dirs_are_unit_and_counted() {
        let dirs = fibonacci_dirs(128);
        assert_eq!(dirs.len(), 128);
        for d in &dirs {
            assert!((d.length() - 1.0).abs() < 1e-9, "dir not unit: {d:?}");
        }
        // Roughly balanced over the sphere: the mean points nowhere in particular.
        let mean: DVec3 = dirs.iter().copied().sum::<DVec3>() / 128.0;
        assert!(mean.length() < 0.1, "directions are lopsided: {mean:?}");
    }

    #[test]
    fn ortho_rays_are_parallel_and_cover_the_grid() {
        let r = res(128);
        let dir = DVec3::new(0.3, 1.0, -0.7);
        let rays = ortho_rays(r, dir, 32);
        assert_eq!(rays.len(), 32 * 32);
        let f = dir.normalize();
        // All rays share the view direction…
        for ray in &rays {
            assert!((ray.dir - f).length() < 1e-12);
        }
        // …and enough of them strike a solid grid to count as covering it.
        let solid = SparseTree::build(&crate::fixtures::Solid { resolution: r });
        let hits = rays
            .iter()
            .filter(|ray| solid.traverse_counted(ray).0.is_some())
            .count();
        assert!(hits > rays.len() / 3, "ortho batch barely covered the grid");
    }

    #[test]
    fn directional_steps_vary_with_orientation() {
        // The whole premise: cost depends on view direction. A face-on and a
        // body-diagonal view of the same structure should not cost the same.
        let tree = SparseTree::build(&OctantFractal::sierpinski_tetrahedron(res(128)));
        let axis = directional_steps(&tree, DVec3::X, 48);
        let diag = directional_steps(&tree, DVec3::new(1.0, 1.0, 1.0), 48);
        assert!(axis.mean_cell_steps > 0.0 && diag.mean_cell_steps > 0.0);
        assert!(
            (axis.mean_cell_steps - diag.mean_cell_steps).abs() > 1e-6,
            "expected orientation-dependent cost, got {} vs {}",
            axis.mean_cell_steps,
            diag.mean_cell_steps
        );
    }

    #[test]
    fn descent_stats_are_sane() {
        let tree = SparseTree::build(&OctantFractal::sierpinski_tetrahedron(res(128)));
        let d = descent_stats(&tree, 2000);
        assert_eq!(d.rays_cast, 2000);
        assert!(d.rays_hit > 0, "some camera rays should hit");
        assert!(d.mean_descents >= 1.0, "every traced ray descends the root");
        assert!(d.max_descents <= u64::from(tree.coarse_level()) + 100);
    }
}
