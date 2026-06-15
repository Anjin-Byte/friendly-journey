//! P4: the School-B single-buffer layout (`idea.md` §6.4).
//!
//! School A ([`SparseTree`]) stores internal nodes in a separate array per
//! level. School B re-serializes them into **one** buffer where each node's
//! children occupy a contiguous block addressed by `popcount`-rank from the
//! node's `subtree_base`, and each node is emitted immediately before its
//! descendants (a children-contiguous DFS). This is the form the GPU kernel
//! consumes (one storage binding instead of one per level), and it keeps a node
//! adjacent to its child block for locality.
//!
//! The convention chosen here is **parent-first** (a node precedes its child
//! block), which is simpler to emit than `idea.md`'s post-order suggestion and
//! has the same addressing and locality; the A-vs-B-vs-oracle differential is
//! the safety net against the convention bugs §6.4 warns about (review R5).
//!
//! Leaves are unchanged: the Morton-sorted leaf array already places every
//! subtree's leaves contiguously, so an `L=2` node's `subtree_base` indexes the
//! shared leaf array exactly as in School A (§6.4, Option B).

use crate::layout::{Cell, NodeLayout};
use crate::leaf::LeafBrick;
use crate::node::GpuNode;
use crate::{Resolution, SparseTree};

/// The School-B layout: one post-order-ish node buffer plus the shared leaf
/// array. Built from a [`SparseTree`] via [`SchoolBBuffer::from_sparse`].
#[derive(Debug, Clone)]
pub struct SchoolBBuffer {
    resolution: Resolution,
    /// Single buffer of internal nodes; child blocks are contiguous and
    /// `subtree_base` (the [`GpuNode::child_base`] field) points at them.
    nodes: Vec<GpuNode>,
    /// Morton-sorted leaves, identical to the School-A array.
    leaves: Vec<LeafBrick>,
}

/// Emits the children block of the node already placed at `nodes[pos]`, then
/// recurses into each child. Patches `nodes[pos].child_base` to the block start.
fn emit_subtree(
    tree: &SparseTree,
    nodes: &mut Vec<GpuNode>,
    pos: usize,
    level: u32,
    node: GpuNode,
) {
    if level == 2 {
        // Children are leaves; subtree_base is the (unchanged) leaf base.
        nodes[pos].child_base = node.child_base;
        return;
    }

    let children = tree.level_nodes(level - 1);
    let n_children = node.mask().count_ones();
    let block_start =
        u32::try_from(nodes.len()).expect("School-B node buffer exceeds u32 (resolution ≫ 2048³)");

    // Phase 1: push the children contiguously (subtree_base patched in phase 2).
    for j in 0..n_children {
        let child = children[(node.child_base + j) as usize];
        nodes.push(GpuNode::new(child.mask(), 0));
    }
    nodes[pos].child_base = block_start;

    // Phase 2: recurse into each child to emit its own children block.
    for j in 0..n_children {
        let child = children[(node.child_base + j) as usize];
        emit_subtree(tree, nodes, (block_start + j) as usize, level - 1, child);
    }
}

impl SchoolBBuffer {
    /// Re-serializes a [`SparseTree`] into the School-B single-buffer layout.
    #[must_use]
    pub fn from_sparse(tree: &SparseTree) -> Self {
        let resolution = tree.resolution();
        let leaves = tree.leaves_slice().to_vec();
        let mut nodes = Vec::new();

        let k = resolution.internal_levels();
        if k >= 1 && !leaves.is_empty() {
            let coarse = k + 1;
            let root = tree.level_nodes(coarse)[0];
            nodes.push(GpuNode::new(root.mask(), 0)); // subtree_base patched in emit_subtree
            emit_subtree(tree, &mut nodes, 0, coarse, root);
        }

        Self {
            resolution,
            nodes,
            leaves,
        }
    }

    /// The single node buffer (for GPU upload in P5).
    #[must_use]
    pub fn nodes(&self) -> &[GpuNode] {
        &self.nodes
    }

    /// The leaf array (for GPU upload in P5).
    #[must_use]
    pub fn leaves(&self) -> &[LeafBrick] {
        &self.leaves
    }

    /// Total node-buffer length.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

impl NodeLayout for SchoolBBuffer {
    fn resolution(&self) -> Resolution {
        self.resolution
    }

    fn root(&self) -> Cell {
        if self.leaves.is_empty() {
            Cell::Empty
        } else if self.resolution.internal_levels() == 0 {
            Cell::Leaf(0)
        } else {
            Cell::Node(0) // the root is always emitted first
        }
    }

    fn child(&self, node: u32, level: u32, child_bit: u32) -> Cell {
        let n = self.nodes[node as usize];
        if !n.has_child(child_bit) {
            return Cell::Empty;
        }
        let slot = n.child_slot(child_bit);
        if level == 2 {
            Cell::Leaf(slot) // subtree_base indexed the leaf array
        } else {
            Cell::Node(slot) // subtree_base indexed this same buffer
        }
    }

    fn leaf(&self, idx: u32) -> &LeafBrick {
        &self.leaves[idx as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{Checkerboard, Empty, OctantFractal, SingleVoxel, Solid};
    use crate::layout::traverse;
    use crate::{OccupancyField, Ray, VoxelCoord, oracle};
    use glam::DVec3;

    fn res(n: u32) -> Resolution {
        Resolution::new(n).unwrap()
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

    #[test]
    fn school_b_preserves_node_count() {
        let tree = SparseTree::build(&OctantFractal::sierpinski_tetrahedron(res(512)));
        let b = SchoolBBuffer::from_sparse(&tree);
        assert_eq!(b.node_count(), tree.node_count());
        assert_eq!(b.leaves().len(), tree.leaf_count());
    }

    #[test]
    fn point_query_via_school_b_matches_school_a() {
        // A leaf-level point query through School B must match School A exactly.
        let field = OctantFractal::sierpinski_tetrahedron(res(128));
        let tree = SparseTree::build(&field);
        let b = SchoolBBuffer::from_sparse(&tree);
        let n = field.resolution().voxels_per_axis();
        for y in (0..n).step_by(2) {
            for x in (0..n).step_by(2) {
                let ray = Ray::new(
                    DVec3::new(f64::from(x) + 0.5, f64::from(y) + 0.5, -1.0),
                    DVec3::Z,
                );
                assert_eq!(traverse(&tree, &ray), traverse(&b, &ray), "at ({x},{y})");
            }
        }
    }

    #[test]
    fn edges_single_brick_empty_solid() {
        let ray = Ray::new(DVec3::new(-1.0, 4.0, 4.0), DVec3::X);

        let single = SchoolBBuffer::from_sparse(&SparseTree::build(&SingleVoxel {
            resolution: res(8),
            voxel: VoxelCoord::new(4, 4, 4),
        }));
        assert_eq!(
            traverse(&single, &ray).map(|h| h.voxel),
            Some(VoxelCoord::new(4, 4, 4))
        );

        let empty = SchoolBBuffer::from_sparse(&SparseTree::build(&Empty {
            resolution: res(32),
        }));
        assert!(traverse(&empty, &ray).is_none());

        let solid = SchoolBBuffer::from_sparse(&SparseTree::build(&Solid {
            resolution: res(32),
        }));
        assert_eq!(
            traverse(&solid, &ray).map(|h| h.voxel),
            Some(VoxelCoord::new(0, 4, 4))
        );
    }

    #[test]
    fn school_a_and_b_both_match_oracle_on_random_rays() {
        // The P4 differential: oracle == School A == School B, structurally.
        let r = res(128);
        let nf = f64::from(r.voxels_per_axis());
        let checker = Checkerboard { resolution: r };
        let frac = OctantFractal::sierpinski_tetrahedron(r);
        let checker_a = SparseTree::build(&checker);
        let checker_b = SchoolBBuffer::from_sparse(&checker_a);
        let frac_a = SparseTree::build(&frac);
        let frac_b = SchoolBBuffer::from_sparse(&frac_a);

        let mut state = 0xABCD_1234_5678_9999u64;
        let mut compared = 0u32;
        for _ in 0..4000 {
            let origin = DVec3::new(
                unit(&mut state) * (nf + 8.0) - 4.0,
                unit(&mut state) * (nf + 8.0) - 4.0,
                unit(&mut state) * (nf + 8.0) - 4.0,
            );
            let dir = DVec3::new(
                unit(&mut state) * 2.0 - 1.0,
                unit(&mut state) * 2.0 - 1.0,
                unit(&mut state) * 2.0 - 1.0,
            );
            if dir.length() < 1e-3 {
                continue;
            }
            let ray = Ray::new(origin, dir);

            // School A and School B run the identical f64 traversal, so they
            // must agree *exactly* (same voxel) — not just within tolerance.
            for (field_ref, a, b) in [
                (
                    oracle::first_hit(&checker, &ray),
                    traverse(&checker_a, &ray),
                    traverse(&checker_b, &ray),
                ),
                (
                    oracle::first_hit(&frac, &ray),
                    traverse(&frac_a, &ray),
                    traverse(&frac_b, &ray),
                ),
            ] {
                assert_eq!(a, b, "School A vs B disagree, dir={dir:?}");
                assert_eq!(field_ref.is_some(), a.is_some(), "hit/miss vs oracle");
                if let (Some(o), Some(h)) = (field_ref, a) {
                    assert!((o.t_enter - h.t_enter).abs() < 1e-6);
                }
                compared += 1;
            }
        }
        assert!(compared > 1000);
    }
}
