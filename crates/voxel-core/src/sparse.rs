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
        let child_base =
            u32::try_from(i).expect("child index exceeds u32 (resolution far beyond 2048³)");
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
        let k = resolution.internal_levels();
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
        let mut bricks: Vec<(u64, LeafBrick)> = std::thread::scope(|scope| {
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

        // 2. Sort by Morton code (codes are unique, so order is total).
        bricks.sort_unstable_by_key(|(code, _)| *code);
        let leaves: Vec<LeafBrick> = bricks.iter().map(|(_, leaf)| *leaf).collect();
        let mut codes: Vec<u64> = bricks.into_iter().map(|(code, _)| code).collect();

        // 3–4. Build internal levels 2..=k+1 bottom-up by OR-reducing 4³ groups.
        let mut nodes: Vec<Vec<GpuNode>> = vec![Vec::new(); (k + 2) as usize];
        for level in 2..=(k + 1) {
            let (parents, parent_codes) = build_parents(&codes);
            nodes[level as usize] = parents;
            codes = parent_codes;
        }

        Self {
            resolution,
            nodes,
            leaves,
        }
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
}
