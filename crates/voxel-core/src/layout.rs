//! The layout-agnostic traversal: a [`NodeLayout`] trait and the generic
//! N-level hierarchical HDDA over it (`idea.md` §7).
//!
//! Both the School-A per-level arrays ([`crate::SparseTree`]) and the School-B
//! single post-order buffer ([`crate::SchoolBBuffer`]) implement
//! [`NodeLayout`], so the *same* traversal runs over both — which is exactly
//! how the A-vs-B differential is set up (review R5). The traversal logic is the
//! `f64` reference; P5's GPU kernel and its `f32` mirror reproduce it over the
//! School-B layout.

use glam::DVec3;

use crate::dda::DdaWalker;
use crate::leaf::{LeafBounds, LeafBrick};
use crate::node::child_bit;
use crate::oracle::Hit;
use crate::ray::{Ray, ray_aabb};
use crate::{Resolution, VoxelCoord};

/// Conservative slack (in `t`) for the leaf early-skip. Far larger than an `f64`
/// ULP at any supported `t`, so rounding never lets the skip drop a real hit —
/// the skip only avoids work, keeping the reference traversal bit-identical to
/// the un-skipped one (and thus to the oracle).
const SKIP_EPS: f64 = 1e-9;

/// Voxel-space dilation of the occupied sub-box before the early-skip slab test.
/// The interior DDA can `floor` a grazing ray into the occupied voxel's cell
/// even when the *box* slab test (different arithmetic) says the razor-thin
/// chord misses; a one-voxel halo dwarfs that disagreement, so the skip is never
/// stricter than the walk it guards. Biasing toward *descend* only ever costs a
/// few extra interior walks, never a dropped hit. Shared by the `f32` mirror and
/// the WGSL kernel, where the rounding error is far larger (review found the
/// `f32` paths dropped up to 5% of grazing hits at 2048³ without this).
pub(crate) const SKIP_MARGIN: f64 = 1.0;

/// A resolved cell during descent: an internal node, a leaf brick, or absent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    /// An internal `4³` node, identified by a layout-specific `u32` handle.
    Node(u32),
    /// A leaf brick, identified by its index in the leaf array.
    Leaf(u32),
    /// No stored cell here (the parent's child-mask bit is clear).
    Empty,
}

/// A traversable node layout: resolves the root and, given a node and a child
/// index, the child cell. Abstracts School A (per-level arrays) from School B
/// (single post-order buffer) so one traversal serves both.
pub trait NodeLayout {
    /// The grid resolution.
    fn resolution(&self) -> Resolution;

    /// The root cell: a [`Cell::Node`] for `k ≥ 1`, a [`Cell::Leaf`] for `k = 0`
    /// (single brick), or [`Cell::Empty`] for an empty structure.
    fn root(&self) -> Cell;

    /// The child of internal `node` (at traversal `level`) for the 6-bit Morton
    /// `child_bit`, or [`Cell::Empty`] if that child is absent.
    fn child(&self, node: u32, level: u32, child_bit: u32) -> Cell;

    /// The leaf brick at array index `idx`.
    fn leaf(&self, idx: u32) -> &LeafBrick;

    /// The occupied-voxel [`LeafBounds`] of leaf `idx`, for the per-brick
    /// early-skip. The default scans the brick; layouts that precompute and
    /// store the packed bounds (the GPU buffer) override this.
    fn leaf_bounds(&self, idx: u32) -> LeafBounds {
        self.leaf(idx).occupied_bounds()
    }
}

/// Whether `ray`'s chord can reach the occupied sub-box of leaf `idx`, whose
/// brick lower corner is `origin` (voxel coords) and which the ray enters at
/// `t_enter`. `false` ⇒ the brick is provably a miss and its `8³` walk is
/// skipped. Conservative: only returns `false` when the chord cannot intersect
/// any set voxel (all set voxels lie inside the returned box).
fn leaf_chord_reaches<L: NodeLayout + ?Sized>(
    layout: &L,
    idx: u32,
    ray: &Ray,
    origin: [u32; 3],
    t_enter: f64,
) -> bool {
    let b = layout.leaf_bounds(idx);
    // Full-brick bound ⇒ the box is the cell the ray already entered, so the
    // test can never skip — descend without paying for the slab test. (Dense
    // fractals like Sierpinski hit this on every leaf; the skip is for the
    // sparse/thin bricks whose box is a strict sub-region.)
    if b == LeafBounds::FULL {
        return true;
    }
    let lo = DVec3::new(
        f64::from(origin[0] + b.min[0]) - SKIP_MARGIN,
        f64::from(origin[1] + b.min[1]) - SKIP_MARGIN,
        f64::from(origin[2] + b.min[2]) - SKIP_MARGIN,
    );
    let hi = DVec3::new(
        f64::from(origin[0] + b.max[0] + 1) + SKIP_MARGIN,
        f64::from(origin[1] + b.max[1] + 1) + SKIP_MARGIN,
        f64::from(origin[2] + b.max[2] + 1) + SKIP_MARGIN,
    );
    match ray_aabb(ray.origin, ray.dir, lo, hi) {
        Some((_, t_far)) => t_far >= t_enter - SKIP_EPS,
        None => false,
    }
}

/// Per-ray traversal counters for the §10 measurements (review R5).
#[derive(Debug, Default, Clone, Copy)]
pub struct TraversalStats {
    /// Cells descended into — one per `walk` invocation, including the root.
    pub descents: u64,
    /// DDA cell-steps taken across all levels.
    pub cell_steps: u64,
}

/// Marches `ray` through any [`NodeLayout`], returning the first occupied voxel.
#[must_use]
pub fn traverse<L: NodeLayout + ?Sized>(layout: &L, ray: &Ray) -> Option<Hit> {
    traverse_counted(layout, ray).0
}

/// Like [`traverse`] but also returns the per-ray [`TraversalStats`].
#[must_use]
pub fn traverse_counted<L: NodeLayout + ?Sized>(
    layout: &L,
    ray: &Ray,
) -> (Option<Hit>, TraversalStats) {
    let mut stats = TraversalStats::default();
    let res = layout.resolution();
    let n_world = f64::from(res.voxels_per_axis());
    let Some((t0, t1)) = ray_aabb(ray.origin, ray.dir, DVec3::ZERO, DVec3::splat(n_world)) else {
        return (None, stats);
    };
    if t1 < 0.0 {
        return (None, stats);
    }
    let root = layout.root();
    let level = match root {
        Cell::Leaf(_) => 1,
        Cell::Node(_) => res.internal_levels() + 1,
        Cell::Empty => return (None, stats),
    };
    let hit = walk(layout, ray, root, level, [0, 0, 0], t0.max(0.0), &mut stats);
    (hit, stats)
}

/// Walks one cell's children, recursing into occupied ones. `origin` is the
/// cell's lower corner in **voxel** coordinates.
fn walk<L: NodeLayout + ?Sized>(
    layout: &L,
    ray: &Ray,
    cell: Cell,
    level: u32,
    origin: [u32; 3],
    t_enter: f64,
    stats: &mut TraversalStats,
) -> Option<Hit> {
    stats.descents += 1;
    let f_origin = [
        f64::from(origin[0]),
        f64::from(origin[1]),
        f64::from(origin[2]),
    ];

    match cell {
        Cell::Empty => None,
        Cell::Leaf(idx) => {
            let leaf = layout.leaf(idx);
            let mut walker = DdaWalker::enter(ray, f_origin, 8, 1.0, t_enter);
            loop {
                stats.cell_steps += 1;
                let v = walker.cell();
                if leaf.get_local(v[0], v[1], v[2]) {
                    return Some(Hit {
                        voxel: VoxelCoord::new(
                            origin[0] + v[0],
                            origin[1] + v[1],
                            origin[2] + v[2],
                        ),
                        t_enter: walker.t_entry(),
                    });
                }
                if !walker.step() {
                    return None;
                }
            }
        }
        Cell::Node(node) => {
            let child_size = 1u32 << (2 * (level - 1) + 1); // cell_size(level-1)
            let mut walker = DdaWalker::enter(ray, f_origin, 4, f64::from(child_size), t_enter);
            loop {
                stats.cell_steps += 1;
                let c = walker.cell();
                let child = layout.child(node, level, child_bit(c[0], c[1], c[2]));
                if child != Cell::Empty {
                    let child_origin = [
                        origin[0] + c[0] * child_size,
                        origin[1] + c[1] * child_size,
                        origin[2] + c[2] * child_size,
                    ];
                    // Per-brick early-skip: if the child is a leaf whose
                    // occupied sub-box the ray's chord misses, treat it as empty
                    // and keep stepping — no descent, no interior walk.
                    let descend = match child {
                        Cell::Leaf(idx) => {
                            leaf_chord_reaches(layout, idx, ray, child_origin, walker.t_entry())
                        }
                        _ => true,
                    };
                    if descend {
                        if let Some(hit) = walk(
                            layout,
                            ray,
                            child,
                            level - 1,
                            child_origin,
                            walker.t_entry(),
                            stats,
                        ) {
                            return Some(hit);
                        }
                    }
                }
                if !walker.step() {
                    return None;
                }
            }
        }
    }
}
