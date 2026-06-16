//! P3: the sparse `4³` node hierarchy over Morton-ordered leaves.
//!
//! This is `idea.md` §11.3 — sparse leaves addressed by `popcount`-rank,
//! built by the §6.4 pipeline (enumerate → Morton-sort → bottom-up `4³`
//! OR-reduce). Only non-empty cells are stored. The traversal is an N-level
//! hierarchical Amanatides–Woo: a recursive descent that walks each level's
//! `4³` (or `8³` at the leaf) cells and recurses into occupied children — the
//! clean `f64` reference form of `idea.md` §7. P4 re-serializes this into the
//! School-B single buffer and adds the `f32` GPU mirror; the traversal logic is
//! identical.
//!
//! Per-level node arrays (a School-A layout) are used here; the §10 gate (P3.5)
//! decides whether the School-B interleaving is worth it (review R5).

use crate::layout::{Cell, NodeLayout, TraversalStats};
use crate::leaf::LeafBrick;
use crate::node::{self, GpuNode};
use crate::oracle::Hit;
use crate::ray::Ray;
use crate::{OccupancyField, Resolution, VoxelCoord};

/// A sparse hierarchy: Morton-ordered leaf bricks and, per internal level, a
/// packed array of `4³` nodes addressed by `popcount`-rank.
#[derive(Debug, Clone)]
pub struct SparseTree {
    resolution: Resolution,
    /// Internal nodes by traversal level. `nodes[L]` holds level-`L` nodes for
    /// `L ∈ 2..=k+1`; indices `0` and `1` are unused (the leaf array is "level
    /// 1"). The root is `nodes[k+1][0]` (or `leaves[0]` when `k = 0`).
    nodes: Vec<Vec<GpuNode>>,
    /// Non-empty leaf bricks, sorted by brick Morton code.
    leaves: Vec<LeafBrick>,
    /// Brick Morton codes, parallel to and in the same order as `leaves`
    /// (ascending). Retained to support incremental edits ([`SparseTree::set_voxel`]):
    /// a topology change binary-searches and splices this list, then rebuilds the
    /// node levels from it — skipping the `O(n³)` occupancy scan.
    codes: Vec<u64>,
    /// Monotonic counter bumped on every topology change (a brick appearing or
    /// disappearing). A [`SchoolBBuffer`] records this at `from_sparse` time and
    /// asserts it is unchanged before an in-place [`patch_leaf`](crate::SchoolBBuffer::patch_leaf):
    /// a topology edit renumbers leaf indices, so a stale patch would corrupt the
    /// buffer silently — this turns that into a loud panic.
    topo_gen: u64,
}

/// What an edit ([`SparseTree::set_voxel`]) did to the structure — and therefore
/// how little a GPU adapter must re-upload to stay in sync.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edit {
    /// The voxel already had the requested state; nothing changed.
    Unchanged,
    /// Exactly one leaf brick changed *in place* (the given leaf index): relative
    /// to the immediately-prior tree state, the Morton order, node masks, and all
    /// indices are unchanged, so the adapter need only re-upload that one leaf's
    /// words and bounds. (Indices are *not* stable across an intervening
    /// [`Topology`](Self::Topology) edit — see [`SchoolBBuffer::patch_leaf`].)
    ///
    /// [`SchoolBBuffer::patch_leaf`]: crate::SchoolBBuffer::patch_leaf
    Leaf(u32),
    /// A brick appeared or disappeared: the leaf array and node levels were
    /// rebuilt (no scan). Leaf indices have shifted; the adapter must
    /// re-serialize and re-upload the structure.
    Topology,
}

/// The voxel coordinates inside a solid sphere of `radius` voxels centred on
/// `center` (Euclidean membership: `dx² + dy² + dz² ≤ radius²`). `radius = 0`
/// yields just the centre voxel.
///
/// Coordinates that would fall below the grid origin are omitted; the upper
/// bound is left to [`SparseTree::set_voxel`], which treats out-of-bounds as
/// [`Edit::Unchanged`]. This is pure geometry shared by the viewer's edit brush
/// and the edit benchmarks, so both stamp the identical voxel set.
#[must_use]
pub fn brush_voxels(center: VoxelCoord, radius: u32) -> Vec<VoxelCoord> {
    let r = i64::from(radius);
    let r2 = r * r;
    let mut out = Vec::new();
    for dz in -r..=r {
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy + dz * dz > r2 {
                    continue;
                }
                let (x, y, z) = (
                    i64::from(center.x) + dx,
                    i64::from(center.y) + dy,
                    i64::from(center.z) + dz,
                );
                if let (Ok(x), Ok(y), Ok(z)) =
                    (u32::try_from(x), u32::try_from(y), u32::try_from(z))
                {
                    out.push(VoxelCoord::new(x, y, z));
                }
            }
        }
    }
    out
}

/// Groups a sorted list of child Morton codes into `4³` parent nodes.
///
/// Returns the parent nodes (each with its child mask and the base index of its
/// children in the input array) and the parents' own Morton codes, ready to be
/// grouped again one level up.
fn build_parents(child_codes: &[u64]) -> (Vec<GpuNode>, Vec<u64>) {
    let mut nodes = Vec::new();
    let mut parent_codes = Vec::new();
    let mut i = 0;
    while i < child_codes.len() {
        let parent = child_codes[i] >> 6;
        let child_base = u32::try_from(i).expect("stored child count exceeds u32::MAX");
        let mut mask = 0u64;
        while i < child_codes.len() && (child_codes[i] >> 6) == parent {
            mask |= 1u64 << (child_codes[i] & 63);
            i += 1;
        }
        nodes.push(GpuNode::new(mask, child_base));
        parent_codes.push(parent);
    }
    (nodes, parent_codes)
}

/// Builds the internal node levels `2..=k+1` bottom-up from the (ascending)
/// leaf-brick Morton codes by repeatedly OR-reducing `4³` groups. `nodes[L]`
/// holds level-`L` nodes; indices `0`/`1` are unused. Shared by [`SparseTree::build`]
/// (after the scan) and [`SparseTree::set_voxel`] (after a topology splice) — the
/// latter is why it is `O(bricks)` and scan-free.
fn build_levels(k: u32, leaf_codes: &[u64]) -> Vec<Vec<GpuNode>> {
    let mut nodes: Vec<Vec<GpuNode>> = vec![Vec::new(); (k + 2) as usize];
    if leaf_codes.is_empty() {
        return nodes;
    }
    let mut codes = leaf_codes.to_vec();
    for level in 2..=(k + 1) {
        let (parents, parent_codes) = build_parents(&codes);
        nodes[level as usize] = parents;
        codes = parent_codes;
    }
    nodes
}

/// Narrows a leaf-array index to the `u32` used by [`Edit::Leaf`] and the GPU
/// buffers (leaf counts are bounded well below `u32::MAX` by the build).
fn leaf_index(idx: usize) -> u32 {
    u32::try_from(idx).expect("stored leaf count exceeds u32::MAX")
}

/// Enumerates occupied bricks in the z-slab `[bz_lo, bz_hi)` — the parallel
/// unit of [`SparseTree::build`]'s scan.
fn enumerate_slab<F: OccupancyField>(
    field: &F,
    bpa: u32,
    bz_lo: u32,
    bz_hi: u32,
) -> Vec<(u64, LeafBrick)> {
    let mut out = Vec::new();
    for bz in bz_lo..bz_hi {
        for by in 0..bpa {
            for bx in 0..bpa {
                let mut leaf = LeafBrick::EMPTY;
                for lz in 0..8 {
                    for ly in 0..8 {
                        for lx in 0..8 {
                            let c = VoxelCoord::new(bx * 8 + lx, by * 8 + ly, bz * 8 + lz);
                            if field.is_occupied(c) {
                                leaf.set_local(lx, ly, lz);
                            }
                        }
                    }
                }
                if !leaf.is_empty() {
                    out.push((crate::morton::encode(bx, by, bz), leaf));
                }
            }
        }
    }
    out
}

impl SparseTree {
    /// Builds the sparse hierarchy from an occupancy field (`idea.md` §6.4
    /// steps 1–4).
    #[must_use]
    pub fn build<F: OccupancyField + Sync>(field: &F) -> Self {
        let resolution = field.resolution();
        let bpa = resolution.voxels_per_axis() / 8;

        // 1. Enumerate occupied bricks (Morton code + 512-bit leaf). This scan
        //    dominates build time at high resolution, so the z-slabs are split
        //    across threads (scoped, so `field` can be borrowed).
        let threads: u32 = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .ok()
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(1)
            .clamp(1, bpa.max(1));
        let bricks: Vec<(u64, LeafBrick)> = std::thread::scope(|scope| {
            let chunk = bpa.div_ceil(threads);
            let handles: Vec<_> = (0..threads)
                .map(|t| {
                    let lo = t * chunk;
                    let hi = ((t + 1) * chunk).min(bpa);
                    scope.spawn(move || enumerate_slab(field, bpa, lo, hi))
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("brick-enumeration thread panicked"))
                .collect()
        });

        // 2–4. Sort the occupied bricks and build the internal levels.
        Self::from_bricks(resolution, bricks)
    }

    /// Assembles the sparse hierarchy from already-enumerated occupied bricks —
    /// `(Morton code, leaf)` pairs in any order, where the code is
    /// [`morton::encode`](crate::morton::encode)`(bx, by, bz)` of the brick. This
    /// is steps 2–4 of [`build`](Self::build) for callers that enumerated the
    /// occupancy themselves (e.g. a GPU generator that evaluated the field in
    /// parallel and read back only the non-empty bricks), skipping the per-voxel
    /// CPU scan entirely.
    #[must_use]
    pub fn from_bricks(resolution: Resolution, mut bricks: Vec<(u64, LeafBrick)>) -> Self {
        let k = resolution.internal_levels();
        // Sort by Morton code (codes are unique, so the order is total).
        bricks.sort_unstable_by_key(|(code, _)| *code);
        let leaves: Vec<LeafBrick> = bricks.iter().map(|(_, leaf)| *leaf).collect();
        let codes: Vec<u64> = bricks.into_iter().map(|(code, _)| code).collect();

        // Build internal levels 2..=k+1 bottom-up by OR-reducing 4³ groups.
        let nodes = build_levels(k, &codes);

        Self {
            resolution,
            nodes,
            leaves,
            codes,
            topo_gen: 0,
        }
    }

    /// The topology generation — bumped each time a brick appears or disappears.
    /// A [`SchoolBBuffer`](crate::SchoolBBuffer) uses it to detect a stale
    /// in-place patch (see [`patch_leaf`](crate::SchoolBBuffer::patch_leaf)).
    #[must_use]
    pub fn topology_generation(&self) -> u64 {
        self.topo_gen
    }

    /// Sets or clears voxel `c`, updating the structure incrementally, and
    /// reports what changed (see [`Edit`]). An out-of-bounds or no-op edit
    /// returns [`Edit::Unchanged`].
    ///
    /// In-place edits (a brick that stays non-empty) are `O(1)`; topology edits
    /// (a brick appearing/disappearing) splice the sorted leaf/code arrays and
    /// rebuild the node levels in `O(bricks)` — skipping the `O(n³)` occupancy
    /// scan that dominates a full [`build`](Self::build).
    pub fn set_voxel(&mut self, c: VoxelCoord, occupied: bool) -> Edit {
        if !c.in_bounds(self.resolution) {
            return Edit::Unchanged;
        }
        let code = crate::morton::encode(c.x >> 3, c.y >> 3, c.z >> 3);
        let (lx, ly, lz) = (c.x & 7, c.y & 7, c.z & 7);

        match self.codes.binary_search(&code) {
            Ok(idx) => {
                let leaf = &mut self.leaves[idx];
                if leaf.get_local(lx, ly, lz) == occupied {
                    return Edit::Unchanged; // already in the requested state
                }
                if occupied {
                    leaf.set_local(lx, ly, lz);
                    Edit::Leaf(leaf_index(idx))
                } else {
                    leaf.clear_local(lx, ly, lz);
                    if leaf.is_empty() {
                        // Last voxel removed → the brick disappears (topology).
                        self.leaves.remove(idx);
                        self.codes.remove(idx);
                        self.rebuild_levels();
                        Edit::Topology
                    } else {
                        Edit::Leaf(leaf_index(idx))
                    }
                }
            }
            Err(insert_at) => {
                if !occupied {
                    return Edit::Unchanged; // clearing a voxel in an empty brick
                }
                // A new brick appears (topology).
                let mut leaf = LeafBrick::EMPTY;
                leaf.set_local(lx, ly, lz);
                self.leaves.insert(insert_at, leaf);
                self.codes.insert(insert_at, code);
                self.rebuild_levels();
                Edit::Topology
            }
        }
    }

    /// Rebuilds the internal node levels from the current `codes` (scan-free)
    /// and bumps the topology generation (leaf indices have changed).
    fn rebuild_levels(&mut self) {
        self.nodes = build_levels(self.resolution.internal_levels(), &self.codes);
        self.topo_gen = self.topo_gen.wrapping_add(1);
    }

    /// The grid resolution.
    #[must_use]
    pub fn resolution(&self) -> Resolution {
        self.resolution
    }

    /// The coarsest level index (`COARSE = k + 1`).
    #[must_use]
    pub fn coarse_level(&self) -> u32 {
        self.resolution.internal_levels() + 1
    }

    /// Total stored nodes across all internal levels.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.iter().map(Vec::len).sum()
    }

    /// Number of stored (non-empty) leaf bricks.
    #[must_use]
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    /// Stored `4³` nodes at internal level `L` (`0` for the voxel/leaf levels).
    #[must_use]
    pub fn nodes_at_level(&self, level: u32) -> usize {
        self.nodes.get(level as usize).map_or(0, Vec::len)
    }

    /// Total occupied voxels — the finest-level count `N(0)`.
    #[must_use]
    pub fn occupied_voxels(&self) -> u64 {
        self.leaves
            .iter()
            .map(|l| u64::from(l.count_occupied()))
            .sum()
    }

    /// Internal nodes at level `L` (empty for the voxel/leaf levels). Used by
    /// the School-B re-serialization.
    pub(crate) fn level_nodes(&self, level: u32) -> &[GpuNode] {
        match self.nodes.get(level as usize) {
            Some(v) => v,
            None => &[],
        }
    }

    /// The Morton-sorted leaf array, shared by both layouts.
    pub(crate) fn leaves_slice(&self) -> &[LeafBrick] {
        &self.leaves
    }

    /// Point query: whether voxel `c` is occupied, by descending the tree and
    /// testing the leaf bit. The independent check on the build + `popcount`
    /// addressing.
    #[must_use]
    pub fn is_occupied(&self, c: VoxelCoord) -> bool {
        if self.leaves.is_empty() || !c.in_bounds(self.resolution) {
            return false;
        }
        let (bx, by, bz) = (c.x >> 3, c.y >> 3, c.z >> 3);
        let mut level = self.coarse_level();
        let mut idx = 0usize;
        while level >= 2 {
            let node = self.nodes[level as usize][idx];
            let shift = 2 * (level - 2);
            let bit = node::child_bit((bx >> shift) & 3, (by >> shift) & 3, (bz >> shift) & 3);
            if !node.has_child(bit) {
                return false;
            }
            idx = node.child_slot(bit) as usize;
            level -= 1;
        }
        self.leaves[idx].get_local(c.x & 7, c.y & 7, c.z & 7)
    }

    /// Ray traversal (N-level hierarchical Amanatides–Woo). Returns the first
    /// occupied voxel, identical to the Tier-A oracle on the same field.
    ///
    /// Delegates to the layout-agnostic [`crate::layout::traverse`] over this
    /// School-A layout; the School-B buffer runs the exact same traversal.
    #[must_use]
    pub fn traverse(&self, ray: &Ray) -> Option<Hit> {
        crate::layout::traverse(self, ray)
    }

    /// Like [`traverse`](Self::traverse) but also returns the per-ray
    /// [`TraversalStats`] used by the §10 descent-frequency measurement.
    #[must_use]
    pub fn traverse_counted(&self, ray: &Ray) -> (Option<Hit>, TraversalStats) {
        crate::layout::traverse_counted(self, ray)
    }
}

impl NodeLayout for SparseTree {
    fn resolution(&self) -> Resolution {
        self.resolution
    }

    fn root(&self) -> Cell {
        if self.leaves.is_empty() {
            Cell::Empty
        } else if self.resolution.internal_levels() == 0 {
            Cell::Leaf(0)
        } else {
            Cell::Node(0)
        }
    }

    fn child(&self, node: u32, level: u32, child_bit: u32) -> Cell {
        let n = self.nodes[level as usize][node as usize];
        if !n.has_child(child_bit) {
            return Cell::Empty;
        }
        let slot = n.child_slot(child_bit);
        if level == 2 {
            Cell::Leaf(slot)
        } else {
            Cell::Node(slot)
        }
    }

    fn leaf(&self, idx: u32) -> &LeafBrick {
        &self.leaves[idx as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BitGrid;
    use crate::fixtures::{Checkerboard, Empty, OctantFractal, SingleVoxel, Solid};
    use crate::oracle;
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
    fn brush_voxels_is_a_clamped_sphere() {
        // radius 0 is the single centre voxel.
        let c = VoxelCoord::new(10, 10, 10);
        assert_eq!(brush_voxels(c, 0), vec![c]);

        // Every returned voxel lies within the Euclidean radius, and a sample of
        // in-range voxels is present (membership is exactly dx²+dy²+dz² ≤ r²).
        let r = 3u32;
        let got = brush_voxels(c, r);
        let r2 = i64::from(r) * i64::from(r);
        for v in &got {
            let (dx, dy, dz) = (
                i64::from(v.x) - 10,
                i64::from(v.y) - 10,
                i64::from(v.z) - 10,
            );
            assert!(dx * dx + dy * dy + dz * dz <= r2, "{v:?} outside radius");
        }
        assert!(got.contains(&VoxelCoord::new(13, 10, 10))); // on the axis, |d|=r
        assert!(!got.contains(&VoxelCoord::new(13, 13, 10))); // corner, outside

        // Voxels below the origin are dropped; the rest survive for set_voxel to
        // range-check (it treats out-of-bounds as Edit::Unchanged).
        let edge = brush_voxels(VoxelCoord::new(0, 0, 0), 2);
        assert!(edge.iter().all(|v| v.x <= 2 && v.y <= 2 && v.z <= 2));
        assert!(edge.contains(&VoxelCoord::new(0, 0, 0)));
    }

    #[test]
    fn point_query_matches_field_exhaustively() {
        // Every voxel in a 128³ grid descends to the correct occupancy.
        let field = OctantFractal::sierpinski_tetrahedron(res(128));
        let tree = SparseTree::build(&field);
        let n = field.resolution().voxels_per_axis();
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let c = VoxelCoord::new(x, y, z);
                    assert_eq!(tree.is_occupied(c), field.is_occupied(c), "voxel {c:?}");
                }
            }
        }
    }

    #[test]
    fn sparsity_drops_node_and_leaf_counts() {
        // A D=2 fractal in 512³ stores far fewer than the 64³ dense bricks.
        let field = OctantFractal::sierpinski_tetrahedron(res(512));
        let tree = SparseTree::build(&field);
        let dense_bricks = 64usize.pow(3);
        assert!(
            tree.leaf_count() < dense_bricks / 4,
            "expected sparse leaves, got {} of {dense_bricks}",
            tree.leaf_count()
        );
        assert_eq!(tree.coarse_level(), 4); // 512³: COARSE = L4
    }

    #[test]
    fn handles_single_brick_resolution() {
        // res 8 = k=0: no internal nodes, the root is the lone leaf.
        let field = SingleVoxel {
            resolution: res(8),
            voxel: VoxelCoord::new(2, 5, 1),
        };
        let tree = SparseTree::build(&field);
        assert_eq!(tree.node_count(), 0);
        assert_eq!(tree.leaf_count(), 1);
        assert!(tree.is_occupied(VoxelCoord::new(2, 5, 1)));
        assert!(!tree.is_occupied(VoxelCoord::new(2, 5, 2)));
    }

    #[test]
    fn empty_and_solid_edges() {
        let empty = SparseTree::build(&Empty {
            resolution: res(32),
        });
        assert_eq!(empty.leaf_count(), 0);
        let ray = Ray::new(DVec3::new(-1.0, 4.0, 4.0), DVec3::X);
        assert!(empty.traverse(&ray).is_none());

        let solid = SparseTree::build(&Solid {
            resolution: res(32),
        });
        let hit = solid.traverse(&ray).unwrap();
        assert_eq!(hit.voxel, VoxelCoord::new(0, 4, 4));
    }

    #[test]
    fn traverse_matches_oracle_on_random_rays() {
        let r = res(128);
        let nf = f64::from(r.voxels_per_axis());
        let checker = Checkerboard { resolution: r };
        let frac = OctantFractal::sierpinski_tetrahedron(r);
        let checker_tree = SparseTree::build(&checker);
        let frac_tree = SparseTree::build(&frac);

        let mut state = 0x1234_5678_9ABC_DEF0u64;
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

            for (oracle_hit, tree) in [
                (oracle::first_hit(&checker, &ray), &checker_tree),
                (oracle::first_hit(&frac, &ray), &frac_tree),
            ] {
                let tree_hit = tree.traverse(&ray);
                assert_eq!(
                    oracle_hit.is_some(),
                    tree_hit.is_some(),
                    "hit/miss, dir={dir:?}"
                );
                if let (Some(a), Some(b)) = (oracle_hit, tree_hit) {
                    assert!(
                        (a.t_enter - b.t_enter).abs() < 1e-6,
                        "t mismatch: oracle={} tree={} dir={dir:?}",
                        a.t_enter,
                        b.t_enter
                    );
                }
                compared += 1;
            }
        }
        assert!(compared > 1000);
    }

    /// The edit correctness gate: a tree mutated by a long sequence of random
    /// voxel toggles must be **byte-for-byte identical** to a fresh build of the
    /// same edited field — same leaves, same codes, same node levels — and agree
    /// with the reference field on every voxel. This is the incremental-edit
    /// analogue of the oracle differential.
    #[test]
    fn incremental_edits_match_fresh_build() {
        let r = res(32); // k = 1: exercises a real internal node level
        let n = r.voxels_per_axis();
        let base = OctantFractal::sierpinski_tetrahedron(r);
        let mut grid = BitGrid::from_field(&base); // mutable reference field
        let mut tree = SparseTree::build(&base);

        let mut state = 0xED17_0000_0000_0001u64;
        let rc = |s: &mut u64| u32::try_from(splitmix64(s) % u64::from(n)).unwrap();
        let (mut leaf_edits, mut topo_edits) = (0u32, 0u32);
        for _ in 0..4000 {
            let c = VoxelCoord::new(rc(&mut state), rc(&mut state), rc(&mut state));
            let occ = !grid.is_occupied(c); // always a real change
            if occ {
                grid.set(c);
            } else {
                grid.clear(c);
            }
            match tree.set_voxel(c, occ) {
                Edit::Leaf(_) => leaf_edits += 1,
                Edit::Topology => topo_edits += 1,
                Edit::Unchanged => panic!("a toggle must change something at {c:?}"),
            }
        }

        let fresh = SparseTree::build(&grid);
        assert_eq!(tree.codes, fresh.codes, "code arrays diverged");
        assert_eq!(tree.leaves, fresh.leaves, "leaf arrays diverged");
        assert_eq!(tree.nodes, fresh.nodes, "node levels diverged");
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let c = VoxelCoord::new(x, y, z);
                    assert_eq!(tree.is_occupied(c), grid.is_occupied(c), "voxel {c:?}");
                }
            }
        }
        assert!(
            leaf_edits > 0 && topo_edits > 0,
            "want both edit kinds exercised: leaf={leaf_edits} topo={topo_edits}"
        );
    }

    /// Pins the [`Edit`] classification an adapter relies on to decide how much
    /// to re-upload.
    #[test]
    fn edit_classification_is_correct() {
        let r = res(32);
        let mut tree = SparseTree::build(&Empty { resolution: r });
        let v = VoxelCoord::new;

        // First voxel in an empty region → a brick appears (topology).
        assert_eq!(tree.set_voxel(v(0, 0, 0), true), Edit::Topology);
        // Another voxel in the same (now-occupied) brick → in-place.
        assert!(matches!(tree.set_voxel(v(1, 0, 0), true), Edit::Leaf(_)));
        // Re-setting an already-set voxel → no change.
        assert_eq!(tree.set_voxel(v(1, 0, 0), true), Edit::Unchanged);
        // Clearing one of two voxels (brick stays non-empty) → in-place.
        assert!(matches!(tree.set_voxel(v(1, 0, 0), false), Edit::Leaf(_)));
        // Clearing the last voxel → the brick disappears (topology).
        assert_eq!(tree.set_voxel(v(0, 0, 0), false), Edit::Topology);
        // Clearing a voxel in an already-empty brick → no change.
        assert_eq!(tree.set_voxel(v(0, 0, 0), false), Edit::Unchanged);
        // Out of bounds → no change.
        assert_eq!(tree.set_voxel(v(32, 0, 0), true), Edit::Unchanged);
        assert_eq!(tree.leaf_count(), 0);
    }

    /// Edits at `k = 0` (a single `8³` brick, no internal nodes): the brick is
    /// the root, so add/remove toggles the root leaf directly.
    #[test]
    fn edits_handle_single_brick_resolution() {
        let r = res(8);
        let mut tree = SparseTree::build(&Empty { resolution: r });
        let v = VoxelCoord::new;

        assert_eq!(tree.set_voxel(v(2, 3, 4), true), Edit::Topology); // root leaf appears
        assert_eq!(tree.leaf_count(), 1);
        assert_eq!(tree.node_count(), 0);
        assert!(tree.is_occupied(v(2, 3, 4)));
        assert!(matches!(tree.set_voxel(v(5, 6, 7), true), Edit::Leaf(_)));
        assert!(matches!(tree.set_voxel(v(2, 3, 4), false), Edit::Leaf(_)));
        assert_eq!(tree.set_voxel(v(5, 6, 7), false), Edit::Topology); // root leaf removed
        assert_eq!(tree.leaf_count(), 0);
        assert!(!tree.is_occupied(v(5, 6, 7)));
        // Matches a fresh build of the now-empty field.
        let empty = SparseTree::build(&Empty { resolution: r });
        assert_eq!(tree.leaves, empty.leaves);
        assert_eq!(tree.codes, empty.codes);
    }

    /// Building up from empty and tearing back down to empty both reproduce a
    /// fresh build at each end.
    #[test]
    fn edits_from_empty_and_back_to_empty() {
        let r = res(32);
        let target = OctantFractal::sierpinski_tetrahedron(r);
        let n = r.voxels_per_axis();

        // Build the target field up voxel by voxel from empty.
        let mut tree = SparseTree::build(&Empty { resolution: r });
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let c = VoxelCoord::new(x, y, z);
                    if target.is_occupied(c) {
                        tree.set_voxel(c, true);
                    }
                }
            }
        }
        let fresh = SparseTree::build(&target);
        assert_eq!(tree.codes, fresh.codes);
        assert_eq!(tree.leaves, fresh.leaves);
        assert_eq!(tree.nodes, fresh.nodes);

        // Tear it all back down to empty.
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    tree.set_voxel(VoxelCoord::new(x, y, z), false);
                }
            }
        }
        assert_eq!(tree.leaf_count(), 0);
        assert_eq!(tree.node_count(), 0);
    }
}
