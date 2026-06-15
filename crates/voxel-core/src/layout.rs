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
use crate::leaf::LeafBrick;
use crate::node::child_bit;
use crate::oracle::Hit;
use crate::ray::{Ray, ray_aabb};
use crate::{Resolution, VoxelCoord};

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
                if !walker.step() {
                    return None;
                }
            }
        }
    }
}
