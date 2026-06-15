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
use crate::leaf::{LeafBounds, LeafBrick};
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
    /// One packed [`LeafBounds`] per leaf (same index as `leaves`), precomputed
    /// for the per-brick early-skip and uploaded as the GPU `leaf_bounds` buffer.
    leaf_bounds: Vec<u32>,
    /// The source tree's [`topology_generation`](SparseTree::topology_generation)
    /// at `from_sparse` time. [`patch_leaf`](Self::patch_leaf) asserts the tree
    /// still matches it, so an in-place patch after a topology edit (which would
    /// silently write to the wrong leaf) panics instead of corrupting the buffer.
    source_gen: u64,
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
        u32::try_from(nodes.len()).expect("School-B node buffer length exceeds u32::MAX");

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
        let leaf_bounds = leaves.iter().map(|l| l.occupied_bounds().pack()).collect();
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
            leaf_bounds,
            source_gen: tree.topology_generation(),
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

    /// The packed per-leaf [`LeafBounds`] words (one per leaf), for the GPU
    /// `leaf_bounds` buffer and the `f32` mirror's early-skip.
    #[must_use]
    pub fn leaf_bounds_words(&self) -> &[u32] {
        &self.leaf_bounds
    }

    /// Total node-buffer length.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Patches leaf `idx` in place after an in-place edit ([`Edit::Leaf`]),
    /// copying the leaf's words from `tree` and recomputing its bounds so the
    /// buffer stays consistent with the edited [`SparseTree`]. Node masks and
    /// indices are untouched (an in-place edit changes neither), so only this
    /// leaf's words and bounds need re-uploading to the GPU afterwards.
    ///
    /// Reading from `tree` (rather than taking caller-supplied words) makes it
    /// impossible to patch with stale or mismatched data.
    ///
    /// # Panics
    /// Panics if `tree` has had a [`Edit::Topology`] change since this buffer was
    /// built (its [`topology_generation`](SparseTree::topology_generation) no
    /// longer matches): a topology edit renumbers leaf indices, so `idx` would
    /// address the wrong leaf. After a topology edit, re-run [`from_sparse`].
    ///
    /// [`Edit::Leaf`]: crate::Edit::Leaf
    /// [`Edit::Topology`]: crate::Edit::Topology
    /// [`from_sparse`]: Self::from_sparse
    pub fn patch_leaf(&mut self, tree: &SparseTree, idx: u32) {
        assert_eq!(
            tree.topology_generation(),
            self.source_gen,
            "patch_leaf on a topology-stale buffer (leaf indices have been \
             renumbered); re-run SchoolBBuffer::from_sparse after a topology edit"
        );
        let leaf = &tree.leaves_slice()[idx as usize];
        let i = idx as usize;
        self.leaves[i] = *leaf;
        self.leaf_bounds[i] = leaf.occupied_bounds().pack();
    }

    /// The leaf brick at `idx` (e.g. to fetch the words to re-upload after a
    /// [`patch_leaf`](Self::patch_leaf)).
    #[must_use]
    pub fn leaf_at(&self, idx: u32) -> &LeafBrick {
        &self.leaves[idx as usize]
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

    fn leaf_bounds(&self, idx: u32) -> LeafBounds {
        // Read the precomputed packed word — the same bytes the GPU uploads.
        LeafBounds::unpack(self.leaf_bounds[idx as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{Checkerboard, Empty, OctantFractal, SingleVoxel, Solid};
    use crate::layout::traverse;
    use crate::{Edit, OccupancyField, Ray, VoxelCoord, oracle};
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
    fn leaf_bounds_are_index_parallel_with_leaves() {
        // The GPU `leaf_bounds` buffer is indexed by the same slot as the leaf
        // words; the early-skip's correctness depends on the bounds array being
        // index-parallel with `leaves`. Assert it explicitly.
        let tree = SparseTree::build(&OctantFractal::sierpinski_tetrahedron(res(128)));
        let b = SchoolBBuffer::from_sparse(&tree);
        assert_eq!(b.leaf_bounds_words().len(), b.leaves().len());
        for (i, leaf) in b.leaves().iter().enumerate() {
            assert_eq!(
                LeafBounds::unpack(b.leaf_bounds_words()[i]),
                leaf.occupied_bounds(),
                "leaf_bounds[{i}] disagrees with the leaf brick",
            );
        }
    }

    #[test]
    fn patch_leaf_matches_a_fresh_build_after_in_place_edit() {
        // After an in-place edit, patching the one leaf must leave the School-B
        // buffer byte-identical to a fresh re-serialization of the edited tree.
        let r = res(32);
        let mut tree = SparseTree::build(&OctantFractal::sierpinski_tetrahedron(r));
        let mut b = SchoolBBuffer::from_sparse(&tree);

        // (7,7,7) is in brick (0,0,0) (occupied as leaf 0) but is itself empty
        // in a Sierpinski tetrahedron, so setting it is an in-place Leaf edit.
        let c = VoxelCoord::new(7, 7, 7);
        assert!(!tree.is_occupied(c));
        let idx = match tree.clone().set_voxel(c, true) {
            Edit::Leaf(i) => i,
            other => panic!("expected an in-place Leaf edit, got {other:?}"),
        };

        tree.set_voxel(c, true);
        b.patch_leaf(&tree, idx);

        let fresh = SchoolBBuffer::from_sparse(&tree);
        assert_eq!(
            b.leaves(),
            fresh.leaves(),
            "leaf words diverged after patch"
        );
        assert_eq!(
            b.leaf_bounds_words(),
            fresh.leaf_bounds_words(),
            "leaf bounds diverged after patch"
        );
    }

    #[test]
    #[should_panic(expected = "topology-stale")]
    fn patch_leaf_panics_on_a_topology_stale_buffer() {
        // Building a buffer, then a topology edit on the tree (which renumbers
        // leaf indices), then patching the stale buffer must PANIC rather than
        // silently write the wrong leaf (the review's MAJOR finding).
        let r = res(32);
        let mut tree = SparseTree::build(&OctantFractal::sierpinski_tetrahedron(r));
        let mut b = SchoolBBuffer::from_sparse(&tree);
        // Corner voxel is empty in a Sierpinski tetrahedron, so setting it adds a
        // brick (topology) and bumps the generation.
        let corner = VoxelCoord::new(r.voxels_per_axis() - 1, 0, 0);
        assert_eq!(tree.set_voxel(corner, true), Edit::Topology);
        b.patch_leaf(&tree, 0); // generations now differ → panic
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
