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
use crate::palette::{STRIDE_W, pack_leaf};
use crate::{Resolution, SparseTree, VoxelCoord};

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
    /// The packed per-leaf material slots, index-parallel with `leaves` at a
    /// fixed [`STRIDE_W`]-word stride — derived from the tree's per-voxel
    /// materials via [`pack_leaf`] and uploaded as the GPU `leaf_mat` buffer
    /// (`leaf_idx * STRIDE_W`). Empty until materials are assigned, in which case
    /// every leaf packs to the `bits == 0` uniform-magenta slot (global-0).
    leaf_mat: Vec<u32>,
    /// Compact per-occupied-voxel **sRGB RGBA8** truecolor (one `u32` per occupied
    /// voxel, R in the low byte) in leaf-slot × intra-brick-Morton order. Empty
    /// until [`assemble_leaf_color`](Self::assemble_leaf_color) runs (the palette
    /// path leaves it empty). Read at the hit as
    /// `leaf_color[leaf_color_base[slot] + rank(morton)]`.
    leaf_color: Vec<u32>,
    /// Per-leaf prefix sum of `count_occupied` (slot-parallel with `leaves`): the
    /// base offset into [`leaf_color`](Self::leaf_color) for each leaf. Empty until
    /// [`assemble_leaf_color`](Self::assemble_leaf_color) runs.
    leaf_color_base: Vec<u32>,
    /// `true` once [`assemble_leaf_color`](Self::assemble_leaf_color) baked at least
    /// one semi-transparent voxel (alpha `< 255`), i.e. some leaf carries
    /// [`LeafBounds::TRANSPARENCY_BIT`]. Routes the renderer to the BLEND pipeline.
    has_transparency: bool,
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
    ///
    /// # Examples
    /// ```
    /// use voxel_core::{Resolution, SchoolBBuffer, SparseTree};
    /// use voxel_core::fixtures::Solid;
    ///
    /// let tree = SparseTree::build(&Solid { resolution: Resolution::new(8).unwrap() });
    /// let buf = SchoolBBuffer::from_sparse(&tree);
    /// assert_eq!(buf.leaves().len(), tree.leaf_count());
    /// ```
    #[must_use]
    pub fn from_sparse(tree: &SparseTree) -> Self {
        let resolution = tree.resolution();
        let leaves = tree.leaves_slice().to_vec();
        let leaf_bounds = leaves.iter().map(|l| l.occupied_bounds().pack()).collect();
        // Derive each leaf's packed material slot from the tree's per-voxel
        // materials (occupancy gates which voxels enter the palette). Concatenated
        // at the fixed STRIDE_W stride so leaf_idx → slot is a multiply, like the
        // occupancy/bounds buffers.
        let leaf_mat = leaves
            .iter()
            .enumerate()
            .flat_map(|(idx, leaf)| {
                pack_leaf(tree.leaf_materials(idx), |x, y, z| leaf.get_local(x, y, z))
            })
            .collect();
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
            leaf_mat,
            // Truecolor is opt-in via assemble_leaf_color (the palette path never
            // populates these); empty means "no per-voxel colour buffer".
            leaf_color: Vec::new(),
            leaf_color_base: Vec::new(),
            has_transparency: false,
            source_gen: tree.topology_generation(),
        }
    }

    /// Bakes the compact per-occupied-voxel truecolor buffer from a colour provider
    /// (`docs/materials/11`, P2). Walks the finalized leaves in slot order and,
    /// within each leaf, the occupied voxels in **intra-brick Morton order** —
    /// exactly the order the GPU `rank` (a 16-word masked popcount) addresses — so
    /// `leaf_color[leaf_color_base[slot] + rank(morton)]` is the voxel's colour.
    /// `color_of(world_coord)` returns sRGB RGBA8 (R low), e.g. the nearest-surface
    /// texture bake.
    ///
    /// `tree` must be the same (topology-unchanged) tree this buffer was built from
    /// — it supplies each leaf's brick origin to reconstruct world coords.
    ///
    /// # Panics
    /// Panics if `tree`'s [`topology_generation`](SparseTree::topology_generation)
    /// no longer matches this buffer's (leaf indices renumbered); re-run
    /// [`from_sparse`](Self::from_sparse) after a topology edit.
    pub fn assemble_leaf_color(
        &mut self,
        tree: &SparseTree,
        mut color_of: impl FnMut(VoxelCoord) -> [u8; 4],
    ) {
        assert_eq!(
            tree.topology_generation(),
            self.source_gen,
            "assemble_leaf_color on a topology-stale buffer (leaf indices have been \
             renumbered); re-run SchoolBBuffer::from_sparse after a topology edit"
        );
        let mut colors: Vec<u32> = Vec::new();
        let mut base: Vec<u32> = Vec::with_capacity(self.leaves.len());
        // Per leaf: did it bake any semi-transparent voxel (alpha < 255)? The bake
        // forces non-BLEND alpha to 255, so alpha < 255 ⟺ a BLEND voxel.
        let mut transparent: Vec<bool> = Vec::with_capacity(self.leaves.len());
        for (idx, leaf) in self.leaves.iter().enumerate() {
            base.push(u32::try_from(colors.len()).expect("leaf_color length exceeds u32::MAX"));
            let origin = tree.leaf_origin(idx);
            let mut saw_transparent = false;
            // Morton order 0..512 → occupied voxels appear in ascending rank.
            for m in 0..512u32 {
                let local = crate::morton::decode(u64::from(m));
                if leaf.get_local(local.x, local.y, local.z) {
                    let world =
                        VoxelCoord::new(origin.x + local.x, origin.y + local.y, origin.z + local.z);
                    let col = color_of(world);
                    saw_transparent |= col[3] < 255;
                    colors.push(u32::from_le_bytes(col));
                }
            }
            transparent.push(saw_transparent);
        }
        // Release-safe `assert!`, NOT `debug_assert!`: `[profile.release]` elides
        // debug-assertions, and this is the silent-mis-color tripwire. The GPU
        // indexes `leaf_color` by a prefix sum of `count_occupied`; if the colour
        // count ever disagreed with the occupancy popcount, every voxel past the
        // divergence would read a neighbour's colour (a silent, not crashing,
        // corruption). The check is O(leaves) popcount — negligible vs the bake.
        // The two sides count the same brick bits two ways, so this holds for any
        // valid `LeafBrick`; it guards against a future change to the Morton walk
        // or `count_occupied` drifting them apart.
        let occupied: u32 = self.leaves.iter().map(LeafBrick::count_occupied).sum();
        assert_eq!(
            u32::try_from(colors.len()).unwrap_or(u32::MAX),
            occupied,
            "leaf_color must hold exactly one entry per occupied voxel"
        );
        // Mark transparent leaves in their packed bound word (bit 18) so the BLEND
        // traversal can route on a cheap per-leaf test. The borrow of `self.leaves`
        // above has ended, so `self.leaf_bounds` is free to mutate here.
        self.has_transparency = false;
        for (idx, &t) in transparent.iter().enumerate() {
            if t {
                self.leaf_bounds[idx] |= crate::leaf::LeafBounds::TRANSPARENCY_BIT;
                self.has_transparency = true;
            }
        }
        self.leaf_color = colors;
        self.leaf_color_base = base;
    }

    /// The compact per-occupied-voxel truecolor words (sRGB RGBA8, R low), or empty
    /// if [`assemble_leaf_color`](Self::assemble_leaf_color) has not run.
    #[must_use]
    pub fn leaf_color_words(&self) -> &[u32] {
        &self.leaf_color
    }

    /// The per-leaf colour base offsets (prefix sum of `count_occupied`), or empty
    /// if [`assemble_leaf_color`](Self::assemble_leaf_color) has not run.
    #[must_use]
    pub fn leaf_color_base_words(&self) -> &[u32] {
        &self.leaf_color_base
    }

    /// Whether a truecolor buffer has been assembled (the renderer selects the
    /// truecolor pipeline when true).
    ///
    /// This is `false` when no colour was baked **or** the bake produced an empty
    /// buffer (a scene with zero occupied voxels). In both cases the renderer
    /// falls back to the palette pipeline — an empty truecolor bake is *not* an
    /// error, it simply routes to palette.
    #[must_use]
    pub fn has_leaf_color(&self) -> bool {
        !self.leaf_color_base.is_empty()
    }

    /// Whether [`assemble_leaf_color`](Self::assemble_leaf_color) baked any
    /// semi-transparent voxel (alpha `< 255`). When true at least one leaf carries
    /// [`LeafBounds::TRANSPARENCY_BIT`] and the renderer selects the BLEND pipeline.
    #[must_use]
    pub fn has_transparency(&self) -> bool {
        self.has_transparency
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

    /// The packed per-leaf material slots ([`STRIDE_W`] `u32` words per leaf, same
    /// index as `leaves`), for the GPU `leaf_mat` buffer. The hit-time WGSL read
    /// decodes `leaf_mat[leaf_idx * STRIDE_W ..]` into a global material id.
    #[must_use]
    pub fn leaf_mat_words(&self) -> &[u32] {
        &self.leaf_mat
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

    /// Re-derives leaf `idx`'s material slot in place after an in-place material
    /// edit ([`Edit::Material`]), so only that leaf's `STRIDE_W` words need
    /// re-uploading. Like [`patch_leaf`](Self::patch_leaf), it reads from `tree`
    /// (no stale caller data) and is gated on the topology generation.
    ///
    /// # Panics
    /// Panics if `tree` has had an [`Edit::Topology`] change since this buffer was
    /// built (leaf indices renumbered) — re-run [`from_sparse`](Self::from_sparse).
    /// A *spilled* material edit ([`Edit::Material { spilled: true, .. }`]) is a
    /// topology-class event that bumps the generation, so it correctly trips this
    /// assert rather than being patched.
    ///
    /// [`Edit::Material`]: crate::Edit::Material
    /// [`Edit::Topology`]: crate::Edit::Topology
    pub fn patch_leaf_mat(&mut self, tree: &SparseTree, idx: u32) {
        assert_eq!(
            tree.topology_generation(),
            self.source_gen,
            "patch_leaf_mat on a topology-stale buffer (leaf indices have been \
             renumbered); re-run SchoolBBuffer::from_sparse after a topology edit"
        );
        let leaf = &tree.leaves_slice()[idx as usize];
        let slot = pack_leaf(tree.leaf_materials(idx as usize), |x, y, z| {
            leaf.get_local(x, y, z)
        });
        let base = idx as usize * STRIDE_W;
        self.leaf_mat[base..base + STRIDE_W].copy_from_slice(&slot);
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

    /// Reads the global material id at `coord` straight from a School-B buffer's
    /// `leaf_mat` (the CPU mirror of the WGSL hit-read), or `0` if `coord` is not
    /// an occupied voxel of a stored leaf.
    fn read_material(b: &SchoolBBuffer, tree: &SparseTree, coord: VoxelCoord) -> u16 {
        use crate::palette::{STRIDE_W, read_slot};
        let Some(idx) = tree.leaf_slot_of(coord) else {
            return 0;
        };
        let base = idx as usize * STRIDE_W;
        let m = crate::morton::encode_brick(coord.x & 7, coord.y & 7, coord.z & 7);
        read_slot(&b.leaf_mat_words()[base..base + STRIDE_W], m)
    }

    #[test]
    fn leaf_mat_is_index_parallel_and_defaults_to_magenta() {
        // With no materials assigned, every leaf packs to the bits==0 uniform
        // slot reading global-0 (magenta). The buffer is index-parallel: one
        // STRIDE_W slot per leaf.
        use crate::palette::STRIDE_W;
        let tree = SparseTree::build(&OctantFractal::sierpinski_tetrahedron(res(128)));
        let b = SchoolBBuffer::from_sparse(&tree);
        assert_eq!(b.leaf_mat_words().len(), b.leaves().len() * STRIDE_W);
        for (i, slot) in b.leaf_mat_words().chunks_exact(STRIDE_W).enumerate() {
            assert_eq!(
                slot[0] & 0xF,
                0,
                "leaf {i} should be the bits==0 uniform slot"
            );
            assert_eq!(
                crate::palette::read_slot(slot, 0),
                0,
                "default leaf {i} must read global-0 (magenta)"
            );
        }
    }

    #[test]
    fn derive_roundtrips_assigned_materials() {
        // Colour occupied voxels by a deterministic function, derive the buffer,
        // and read every occupied voxel back through the leaf_mat → expect the
        // assigned id. A morton-vs-linear transpose in the derive would fail here.
        let r = res(32);
        let mut tree = SparseTree::build(&OctantFractal::sierpinski_tetrahedron(r));
        let colour = |c: VoxelCoord| u16::try_from(1 + (c.x + c.y + c.z) % 5).unwrap();
        tree.fill_materials(colour);
        let b = SchoolBBuffer::from_sparse(&tree);

        let n = r.voxels_per_axis();
        let mut checked = 0u32;
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let c = VoxelCoord::new(x, y, z);
                    if tree.is_occupied(c) {
                        assert_eq!(read_material(&b, &tree, c), colour(c), "at {c:?}");
                        checked += 1;
                    }
                }
            }
        }
        assert!(
            checked > 100,
            "expected many occupied voxels, got {checked}"
        );
    }

    #[test]
    fn material_patch_matches_a_fresh_build() {
        // The cold-side analog of patch_leaf_matches_a_fresh_build: an in-place
        // set_material followed by patch_leaf_mat must leave leaf_mat byte-for-byte
        // identical to a fresh from_sparse of the recoloured tree.
        let r = res(32);
        let mut tree = SparseTree::build(&OctantFractal::sierpinski_tetrahedron(r));
        tree.fill_materials(|_| 1); // uniform colour 1
        let mut b = SchoolBBuffer::from_sparse(&tree);

        // Recolour one occupied voxel in place (not a topology edit).
        let c = VoxelCoord::new(0, 0, 0);
        assert!(tree.is_occupied(c));
        let idx = match tree.set_material(c, 2) {
            Edit::Material {
                leaf,
                spilled: false,
            } => leaf,
            other => panic!("expected an in-place Material edit, got {other:?}"),
        };
        b.patch_leaf_mat(&tree, idx);

        let fresh = SchoolBBuffer::from_sparse(&tree);
        assert_eq!(
            b.leaf_mat_words(),
            fresh.leaf_mat_words(),
            "leaf_mat diverged from a fresh build after an in-place material patch"
        );
        assert_eq!(read_material(&b, &tree, c), 2);
    }

    #[test]
    fn leaf_color_assembles_in_slot_morton_rank_order() {
        // The truecolor assembler must lay colours out as the GPU reads them:
        // leaf-slot order, then occupied voxels in intra-brick Morton order, with a
        // per-leaf base = prefix sum of count_occupied. A unique-per-voxel colour
        // (packed world coord) catches any slot/morton/rank transpose.
        let r = res(32);
        let tree = SparseTree::build(&OctantFractal::sierpinski_tetrahedron(r));
        let mut b = SchoolBBuffer::from_sparse(&tree);
        assert!(!b.has_leaf_color(), "no colour before assembly");

        let color_of = |c: VoxelCoord| {
            [
                u8::try_from(c.x & 0xff).unwrap(),
                u8::try_from(c.y & 0xff).unwrap(),
                u8::try_from(c.z & 0xff).unwrap(),
                255,
            ]
        };
        b.assemble_leaf_color(&tree, color_of);
        assert!(b.has_leaf_color());

        let leaves = b.leaves().len();
        assert_eq!(b.leaf_color_base_words().len(), leaves, "one base per leaf");
        let total_occ: u32 = b.leaves().iter().map(LeafBrick::count_occupied).sum();
        assert_eq!(
            u32::try_from(b.leaf_color_words().len()).unwrap(),
            total_occ,
            "one colour per occupied voxel"
        );

        // base is the prefix sum of count_occupied, and every occupied voxel reads
        // back its own colour at base[slot] + rank(morton).
        let mut acc = 0u32;
        let mut checked = 0u32;
        for idx in 0..leaves {
            assert_eq!(
                b.leaf_color_base_words()[idx],
                acc,
                "base[{idx}] prefix sum"
            );
            let leaf = b.leaves()[idx];
            let origin = tree.leaf_origin(idx);
            let base = b.leaf_color_base_words()[idx];
            let mut rank = 0u32;
            for m in 0..512u32 {
                let local = crate::morton::decode(u64::from(m));
                if leaf.get_local(local.x, local.y, local.z) {
                    let world =
                        VoxelCoord::new(origin.x + local.x, origin.y + local.y, origin.z + local.z);
                    assert_eq!(
                        b.leaf_color_words()[(base + rank) as usize],
                        u32::from_le_bytes(color_of(world)),
                        "leaf {idx} morton {m} (rank {rank})"
                    );
                    rank += 1;
                    checked += 1;
                }
            }
            acc += leaf.count_occupied();
        }
        assert!(
            checked > 100,
            "expected many occupied voxels, got {checked}"
        );
    }

    #[test]
    #[should_panic(expected = "topology-stale")]
    fn assemble_leaf_color_panics_on_a_topology_stale_buffer() {
        // A topology edit renumbers leaf indices; assembling colour against the
        // stale buffer would read the wrong brick origin, so it must panic.
        let r = res(32);
        let mut tree = SparseTree::build(&OctantFractal::sierpinski_tetrahedron(r));
        let mut b = SchoolBBuffer::from_sparse(&tree);
        let corner = VoxelCoord::new(r.voxels_per_axis() - 1, 0, 0);
        assert_eq!(tree.set_voxel(corner, true), Edit::Topology);
        b.assemble_leaf_color(&tree, |_| [0, 0, 0, 255]);
    }

    #[test]
    fn leaf_color_reads_back_via_occupied_rank() {
        // Truecolor P3 FIX-C: the mandatory assembler-link. Reads each occupied
        // voxel's colour back through `leaf_color[base[s] + LeafBrick::occupied_rank(m)]`
        // calling the REAL occupied_rank method (the GPU read's CPU canonical) — the
        // only path that ties the rank to the morton order `get_local` defines (it
        // would catch a words32 lo/hi transpose). Distinct from
        // leaf_color_assembles_in_slot_morton_rank_order, which uses an inline
        // counter; both stand as independent witnesses. Unique-per-voxel colour
        // (packed coord) makes any mis-index observable.
        let r = res(32);
        let tree = SparseTree::build(&OctantFractal::sierpinski_tetrahedron(r));
        let mut b = SchoolBBuffer::from_sparse(&tree);
        let color_of = |c: VoxelCoord| {
            [
                u8::try_from(c.x & 0xff).unwrap(),
                u8::try_from(c.y & 0xff).unwrap(),
                u8::try_from(c.z & 0xff).unwrap(),
                255,
            ]
        };
        b.assemble_leaf_color(&tree, color_of);

        let total = u32::try_from(b.leaf_color_words().len()).unwrap();
        let mut checked = 0u32;
        for slot in 0..b.leaves().len() {
            let leaf = b.leaves()[slot];
            let origin = tree.leaf_origin(slot);
            let base = b.leaf_color_base_words()[slot];
            // Span end: the next slot's base, or the total for the last slot.
            let span_end = b
                .leaf_color_base_words()
                .get(slot + 1)
                .copied()
                .unwrap_or(total);
            for m in 0..512u32 {
                let local = crate::morton::decode(u64::from(m));
                if leaf.get_local(local.x, local.y, local.z) {
                    let idx = base + leaf.occupied_rank(m); // the REAL method
                    assert!(
                        idx < span_end,
                        "slot {slot} morton {m}: index {idx} bleeds past span end {span_end}"
                    );
                    let world =
                        VoxelCoord::new(origin.x + local.x, origin.y + local.y, origin.z + local.z);
                    assert_eq!(
                        b.leaf_color_words()[idx as usize],
                        u32::from_le_bytes(color_of(world)),
                        "slot {slot} morton {m} read back the wrong colour"
                    );
                    checked += 1;
                }
            }
        }
        assert!(
            checked > 100,
            "expected many occupied voxels, got {checked}"
        );

        // A leaf whose ONLY voxel is at intra-brick morton 511 (local 7,7,7) must
        // land at base+0 — occupied_rank(511)==0, the high-word/full=15 boundary.
        let lone = VoxelCoord::new(15, 15, 15); // brick (1,1,1), local (7,7,7) = morton 511
        let single = SparseTree::from_voxels(r, [(lone, 0u16)]);
        let mut sb = SchoolBBuffer::from_sparse(&single);
        sb.assemble_leaf_color(&single, color_of);
        assert_eq!(sb.leaf_color_words().len(), 1, "one occupied voxel");
        assert_eq!(
            sb.leaves()[0].occupied_rank(511),
            0,
            "the lone voxel ranks 0"
        );
        assert_eq!(
            sb.leaf_color_words()[0],
            u32::from_le_bytes(color_of(lone)),
            "single-voxel-at-511 colour at base+0"
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

    #[test]
    fn assemble_marks_transparency_bit_for_sub_255_alpha() {
        use crate::leaf::LeafBounds;
        // Two voxels in one 8³ brick (origin 0). A baked alpha < 255 on one of them
        // must set the leaf's TRANSPARENCY_BIT and `has_transparency()`; an all-255
        // bake must leave both clear and the bounds bits intact.
        let r = res(8);
        let voxels = [VoxelCoord::new(1, 1, 1), VoxelCoord::new(2, 2, 2)];
        let tree = SparseTree::from_voxels(r, voxels.iter().map(|&c| (c, 0u16)));
        let bounds_before = {
            let b = SchoolBBuffer::from_sparse(&tree);
            b.leaf_bounds_words()[0]
        };

        // All opaque → no transparency.
        let mut opaque = SchoolBBuffer::from_sparse(&tree);
        opaque.assemble_leaf_color(&tree, |_| [10, 20, 30, 255]);
        assert!(!opaque.has_transparency(), "all-255 → no transparency");
        assert_eq!(
            opaque.leaf_bounds_words()[0] & LeafBounds::TRANSPARENCY_BIT,
            0,
            "no transparency bit set"
        );
        assert_eq!(
            opaque.leaf_bounds_words()[0],
            bounds_before,
            "bounds word unchanged when opaque"
        );

        // One voxel semi-transparent → leaf flagged, bounds bits preserved.
        let mut blend = SchoolBBuffer::from_sparse(&tree);
        blend.assemble_leaf_color(&tree, |c| {
            let a = if c == VoxelCoord::new(1, 1, 1) {
                128
            } else {
                255
            };
            [10, 20, 30, a]
        });
        assert!(blend.has_transparency(), "sub-255 alpha → has_transparency");
        assert_eq!(
            blend.leaf_bounds_words()[0] & LeafBounds::TRANSPARENCY_BIT,
            LeafBounds::TRANSPARENCY_BIT,
            "transparency bit set on the leaf"
        );
        // Bounds (bits 0-17) survive the OR.
        assert_eq!(
            LeafBounds::unpack(blend.leaf_bounds_words()[0]),
            LeafBounds::unpack(bounds_before),
            "transparency bit must not disturb the occupied bounds"
        );
    }
}
