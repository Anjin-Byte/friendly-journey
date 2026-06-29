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
use crate::palette::{LEAF_VOXELS, P_CAP};
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
    /// Per-leaf material grid, index-parallel with `leaves`/`codes`: one global
    /// material id per voxel in intra-brick Morton order (`0` = the reserved
    /// default / MISSING sentinel). Spliced in lockstep with `leaves` on every
    /// topology edit; the GPU's packed per-leaf palette is *derived* from this at
    /// upload time (so the palette is always minimal — no CPU palette/GC). Stays
    /// all-`0` until [`set_material`](SparseTree::set_material) colours voxels.
    // Boxed so a topology splice (`insert`/`remove`) shifts 8-byte pointers, not
    // 1 KiB grids — keeping it the same order as the parallel `leaves` shift.
    #[allow(clippy::vec_box)]
    materials: Vec<Box<[u16; LEAF_VOXELS]>>,
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
    /// A leaf's per-voxel material changed in place (no occupancy topology
    /// change): only that leaf's material slot need re-upload. `spilled` means
    /// the leaf now has more than `P_CAP` distinct occupied materials, so it no
    /// longer fits the inline palette and must ride the full reupload instead of
    /// an O(1) slot patch (treated as topology-class — the generation is bumped).
    Material {
        /// The affected leaf index (index-parallel with the GPU `leaf_mat` slot).
        leaf: u32,
        /// The leaf exceeded `P_CAP` distinct materials and now spills.
        spilled: bool,
    },
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

/// Whether a leaf has more than `P_CAP` distinct materials among its **occupied**
/// voxels — the spill condition. Only occupied voxels matter (the GPU palette is
/// built from them); empty voxels' material is irrelevant. Early-exits once the
/// cap is exceeded, so it is `O(512)` worst case with a tiny (≤17) linear scan.
fn leaf_spills(leaf: &LeafBrick, materials: &[u16; LEAF_VOXELS]) -> bool {
    let mut seen: Vec<u16> = Vec::with_capacity(P_CAP as usize + 1);
    for z in 0..8u32 {
        for y in 0..8u32 {
            for x in 0..8u32 {
                if leaf.get_local(x, y, z) {
                    let gid = materials[crate::morton::encode_brick(x, y, z) as usize];
                    if !seen.contains(&gid) {
                        seen.push(gid);
                        if seen.len() > P_CAP as usize {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
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
    ///
    /// # Examples
    /// ```
    /// use voxel_core::{Resolution, SparseTree, VoxelCoord};
    /// use voxel_core::fixtures::Solid;
    ///
    /// let tree = SparseTree::build(&Solid { resolution: Resolution::new(8).unwrap() });
    /// assert_eq!(tree.leaf_count(), 1); // an 8³ solid is a single full brick
    /// assert!(tree.is_occupied(VoxelCoord::new(0, 0, 0)));
    /// ```
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

        // Materials default to the all-0 sentinel; a builder with material data
        // (the voxelizer's owner→material) fills them after construction.
        let materials = (0..leaves.len())
            .map(|_| Box::new([0u16; LEAF_VOXELS]))
            .collect();

        Self {
            resolution,
            nodes,
            leaves,
            codes,
            materials,
            topo_gen: 0,
        }
    }

    /// Assembles the sparse hierarchy **and its per-voxel materials** from a stream
    /// of `(coord, global_id)` pairs — one per occupied voxel, in any order — where
    /// `global_id` is a renderer global material id (`0` = the magenta MISSING
    /// sentinel). Voxels are binned into fixed `8³` leaves by `coord >> 3`; a
    /// repeated coord keeps the **last** `global_id` written.
    ///
    /// This is the GPU-free core of the sparse mesh-material path
    /// (`docs/materials/09-sparse-material-bridge.md`): the voxelizer's GPU compact
    /// pass yields per-occupied-voxel `(coord, global_id)`, and this turns them into
    /// a renderer-ready tree without the `O(n³)` scan of [`build`](Self::build) or a
    /// dense `n³` owner grid — so it scales to `2048³`. Accumulation is **per
    /// brick**, so host memory tracks the brick count, not the voxel count.
    ///
    /// Coords MUST be in `[0, resolution)`; the caller filters out-of-range voxels
    /// (an out-of-grid coord would otherwise plant a phantom brick). Out-of-range
    /// coords panic in debug and are skipped in release.
    #[must_use]
    pub fn from_voxels(
        resolution: Resolution,
        voxels: impl IntoIterator<Item = (VoxelCoord, u16)>,
    ) -> Self {
        // Accumulate per BRICK (occupancy bits + the 512-voxel material grid) so
        // host memory scales with bricks, not the ~tens of millions of voxels.
        let mut by_brick: std::collections::HashMap<u64, (LeafBrick, Box<[u16; LEAF_VOXELS]>)> =
            std::collections::HashMap::new();
        for (c, gid) in voxels {
            if !c.in_bounds(resolution) {
                debug_assert!(false, "from_voxels: {c:?} out of [0,n); caller must filter");
                continue;
            }
            let code = crate::morton::encode(c.x >> 3, c.y >> 3, c.z >> 3);
            let (leaf, mats) = by_brick
                .entry(code)
                .or_insert_with(|| (LeafBrick::EMPTY, Box::new([0u16; LEAF_VOXELS])));
            let (lx, ly, lz) = (c.x & 7, c.y & 7, c.z & 7);
            leaf.set_local(lx, ly, lz);
            mats[crate::morton::encode_brick(lx, ly, lz) as usize] = gid;
        }

        // Sort by Morton code into the parallel leaves / codes / materials arrays
        // (unique codes by construction — one entry per distinct brick).
        let mut entries: Vec<(u64, LeafBrick, Box<[u16; LEAF_VOXELS]>)> =
            by_brick.into_iter().map(|(c, (l, m))| (c, l, m)).collect();
        entries.sort_unstable_by_key(|(c, _, _)| *c);

        let codes: Vec<u64> = entries.iter().map(|(c, _, _)| *c).collect();
        let leaves: Vec<LeafBrick> = entries.iter().map(|(_, l, _)| *l).collect();
        let materials: Vec<Box<[u16; LEAF_VOXELS]>> =
            entries.into_iter().map(|(_, _, m)| m).collect();
        let nodes = build_levels(resolution.internal_levels(), &codes);

        Self {
            resolution,
            nodes,
            leaves,
            codes,
            materials,
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
                if self.leaves[idx].get_local(lx, ly, lz) == occupied {
                    return Edit::Unchanged; // already in the requested state
                }
                if occupied {
                    self.leaves[idx].set_local(lx, ly, lz);
                    // STALE-BITS FIX: a newly-set voxel must read the leaf's
                    // default material (0), not whatever the previous tenant of
                    // this Morton slot left behind — keeps "every occupied voxel
                    // has a defined material" true after every in-place SET.
                    let m = crate::morton::encode_brick(lx, ly, lz) as usize;
                    self.materials[idx][m] = 0;
                    Edit::Leaf(leaf_index(idx))
                } else {
                    self.leaves[idx].clear_local(lx, ly, lz);
                    if self.leaves[idx].is_empty() {
                        // Last voxel removed → the brick disappears (topology).
                        self.leaves.remove(idx);
                        self.codes.remove(idx);
                        self.materials.remove(idx); // splice in lockstep
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
                // A fresh all-default material grid, spliced at the same index so
                // `materials` stays parallel to `leaves`/`codes`.
                self.materials
                    .insert(insert_at, Box::new([0u16; LEAF_VOXELS]));
                self.rebuild_levels();
                Edit::Topology
            }
        }
    }

    /// Assigns `global_id` (an index into the global material/colour table) to
    /// the voxel at `coord`. The voxel **must be occupied**; colouring an empty
    /// or out-of-bounds voxel is a no-op ([`Edit::Unchanged`]), as is recolouring
    /// to the same id. Returns [`Edit::Material`]; `spilled` is set when the leaf
    /// now exceeds `P_CAP` distinct occupied materials — a topology-class event
    /// that bumps the generation so the adapter re-uploads the whole structure.
    ///
    /// The occupancy bitmask is never touched, so a material edit cannot change
    /// the structure's topology or regress traversal.
    pub fn set_material(&mut self, coord: VoxelCoord, global_id: u16) -> Edit {
        if !coord.in_bounds(self.resolution) {
            return Edit::Unchanged;
        }
        let code = crate::morton::encode(coord.x >> 3, coord.y >> 3, coord.z >> 3);
        let (lx, ly, lz) = (coord.x & 7, coord.y & 7, coord.z & 7);
        let Ok(idx) = self.codes.binary_search(&code) else {
            return Edit::Unchanged; // no brick here
        };
        if !self.leaves[idx].get_local(lx, ly, lz) {
            return Edit::Unchanged; // colouring an empty voxel
        }
        let m = crate::morton::encode_brick(lx, ly, lz) as usize;
        if self.materials[idx][m] == global_id {
            return Edit::Unchanged; // already this colour
        }
        let was_spilled = leaf_spills(&self.leaves[idx], &self.materials[idx]);
        self.materials[idx][m] = global_id;
        let now_spilled = leaf_spills(&self.leaves[idx], &self.materials[idx]);
        if was_spilled != now_spilled {
            // Crossing the spill boundary changes the GPU layout for this leaf —
            // a topology-class event; bump the generation so the adapter does a
            // full reupload rather than an O(1) slot patch.
            self.topo_gen = self.topo_gen.wrapping_add(1);
        }
        Edit::Material {
            leaf: leaf_index(idx),
            spilled: now_spilled,
        }
    }

    /// The per-voxel material grid of leaf `idx` (intra-brick Morton order) — the
    /// source the GPU upload derives the packed per-leaf palette from. `0` is the
    /// default / MISSING sentinel.
    #[must_use]
    pub fn leaf_materials(&self, idx: usize) -> &[u16; LEAF_VOXELS] {
        &self.materials[idx]
    }

    /// The material global id at `coord` (`0` if empty / out of bounds).
    #[must_use]
    pub fn material_at(&self, coord: VoxelCoord) -> u16 {
        if !coord.in_bounds(self.resolution) {
            return 0;
        }
        let code = crate::morton::encode(coord.x >> 3, coord.y >> 3, coord.z >> 3);
        match self.codes.binary_search(&code) {
            Ok(idx) => {
                let m = crate::morton::encode_brick(coord.x & 7, coord.y & 7, coord.z & 7) as usize;
                self.materials[idx][m]
            }
            Err(_) => 0,
        }
    }

    /// The leaf slot (index into `leaves`, and into the School-B `leaf_mat` /
    /// `leaf_bounds` buffers) whose brick contains `coord`, or `None` if no brick
    /// is stored there. Maps a world voxel to its material slot.
    #[must_use]
    pub fn leaf_slot_of(&self, coord: VoxelCoord) -> Option<u32> {
        if !coord.in_bounds(self.resolution) {
            return None;
        }
        let code = crate::morton::encode(coord.x >> 3, coord.y >> 3, coord.z >> 3);
        self.codes.binary_search(&code).ok().map(leaf_index)
    }

    /// The voxel-space origin (min corner) of leaf `idx`'s 8³ brick —
    /// `decode(code) · 8`. Lets a post-build assembler map a leaf's local
    /// `(x, y, z)` back to a world voxel (the inverse of [`leaf_slot_of`]).
    ///
    /// [`leaf_slot_of`]: Self::leaf_slot_of
    #[must_use]
    pub fn leaf_origin(&self, idx: usize) -> VoxelCoord {
        let brick = crate::morton::decode(self.codes[idx]);
        VoxelCoord::new(brick.x * 8, brick.y * 8, brick.z * 8)
    }

    /// Bulk-assigns a material to every **occupied** voxel via
    /// `f(world_coord) -> global_id`, writing the per-leaf side-array directly (no
    /// per-voxel binary search). Unoccupied voxels keep the default global-0. This
    /// is the one-pass colouring path a builder uses after construction — the
    /// voxelizer's `owner_id → material_id` resolution feeds it. Occupancy is
    /// untouched, so it cannot change topology (no generation bump).
    pub fn fill_materials(&mut self, f: impl Fn(VoxelCoord) -> u16) {
        for idx in 0..self.codes.len() {
            let origin = crate::morton::decode(self.codes[idx]); // brick coords
            let (ox, oy, oz) = (origin.x * 8, origin.y * 8, origin.z * 8);
            let leaf = self.leaves[idx]; // Copy — ends the immutable borrow of leaves
            let mat = &mut self.materials[idx];
            for z in 0..8u32 {
                for y in 0..8u32 {
                    for x in 0..8u32 {
                        if leaf.get_local(x, y, z) {
                            let m = crate::morton::encode_brick(x, y, z) as usize;
                            mat[m] = f(VoxelCoord::new(ox + x, oy + y, oz + z));
                        }
                    }
                }
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
                Edit::Material { .. } => unreachable!("set_voxel never changes materials"),
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

    // ---- milestone 3: per-voxel material edit path ----

    #[test]
    fn set_material_colours_occupied_voxels() {
        let r = res(32);
        let mut tree = SparseTree::build(&Empty { resolution: r });
        let v = VoxelCoord::new;
        // Two voxels in the same brick.
        tree.set_voxel(v(0, 0, 0), true);
        tree.set_voxel(v(1, 0, 0), true);
        assert_eq!(tree.material_at(v(0, 0, 0)), 0); // default sentinel
        assert_eq!(
            tree.set_material(v(0, 0, 0), 7),
            Edit::Material {
                leaf: 0,
                spilled: false
            }
        );
        tree.set_material(v(1, 0, 0), 9);
        assert_eq!(tree.material_at(v(0, 0, 0)), 7);
        assert_eq!(tree.material_at(v(1, 0, 0)), 9);
        // Recolouring to the same id is a no-op.
        assert_eq!(tree.set_material(v(0, 0, 0), 7), Edit::Unchanged);
    }

    #[test]
    fn set_material_on_empty_or_oob_is_noop() {
        let r = res(32);
        let mut tree = SparseTree::build(&Empty { resolution: r });
        let v = VoxelCoord::new;
        tree.set_voxel(v(0, 0, 0), true);
        // Unoccupied voxel in an existing brick → no-op.
        assert_eq!(tree.set_material(v(1, 0, 0), 5), Edit::Unchanged);
        assert_eq!(tree.material_at(v(1, 0, 0)), 0);
        // Voxel in a brick that does not exist → no-op.
        assert_eq!(tree.set_material(v(16, 0, 0), 5), Edit::Unchanged);
        // Out of bounds → no-op.
        assert_eq!(tree.set_material(v(32, 0, 0), 5), Edit::Unchanged);
    }

    #[test]
    fn in_place_set_resets_stale_material() {
        // The stale-bits fix: a re-set voxel must read the default material,
        // not the colour it carried before being cleared.
        let r = res(32);
        let mut tree = SparseTree::build(&Empty { resolution: r });
        let v = VoxelCoord::new;
        tree.set_voxel(v(0, 0, 0), true);
        tree.set_voxel(v(1, 0, 0), true); // keep the brick non-empty
        tree.set_material(v(0, 0, 0), 9);
        assert_eq!(tree.material_at(v(0, 0, 0)), 9);
        // Clear then re-set voxel 0 in place (brick stays non-empty via voxel 1).
        assert!(matches!(tree.set_voxel(v(0, 0, 0), false), Edit::Leaf(_)));
        assert!(matches!(tree.set_voxel(v(0, 0, 0), true), Edit::Leaf(_)));
        assert_eq!(
            tree.material_at(v(0, 0, 0)),
            0,
            "stale material survived a re-set"
        );
    }

    #[test]
    fn materials_stay_index_parallel_across_topology() {
        let r = res(32);
        let mut tree = SparseTree::build(&Empty { resolution: r });
        let v = VoxelCoord::new;
        // Occupy a brick at local (1,0,0) and colour it.
        tree.set_voxel(v(8, 0, 0), true);
        tree.set_material(v(8, 0, 0), 5);
        assert_eq!(tree.materials.len(), tree.leaves.len());
        // Insert a brick at (0,0,0) — a LOWER Morton code → shifts the first
        // brick's index 0→1. The side-array must splice in lockstep.
        assert_eq!(tree.set_voxel(v(0, 0, 0), true), Edit::Topology);
        assert_eq!(tree.materials.len(), tree.leaves.len());
        assert_eq!(tree.leaf_count(), 2);
        // The colour followed its brick across the renumber; the new brick is 0.
        assert_eq!(tree.material_at(v(8, 0, 0)), 5);
        assert_eq!(tree.material_at(v(0, 0, 0)), 0);
    }

    #[test]
    fn removed_then_readded_leaf_starts_fresh() {
        let r = res(32);
        let mut tree = SparseTree::build(&Empty { resolution: r });
        let v = VoxelCoord::new;
        tree.set_voxel(v(0, 0, 0), true);
        tree.set_material(v(0, 0, 0), 9);
        // Clear the only voxel → the brick disappears (topology).
        assert_eq!(tree.set_voxel(v(0, 0, 0), false), Edit::Topology);
        assert_eq!(tree.materials.len(), 0);
        // Re-create the brick → a fresh material grid, NOT the old palette.
        assert_eq!(tree.set_voxel(v(0, 0, 0), true), Edit::Topology);
        assert_eq!(tree.material_at(v(0, 0, 0)), 0, "old material resurrected");
    }

    #[test]
    fn leaf_over_cap_spills_and_bumps_generation() {
        let r = res(32);
        let mut tree = SparseTree::build(&Empty { resolution: r });
        let v = VoxelCoord::new;
        // 17 occupied voxels in one 8³ brick. Uncolored occupied voxels carry
        // the default material 0 (the magenta sentinel), which is itself a
        // palette entry — so distinct = {0} ∪ {colours so far}. With 17 occupied
        // and 15 colours applied, two voxels remain at 0 ⇒ {0,1..15} = 16 entries
        // (inline). Colouring the 16th distinct ⇒ {0,1..16} = 17 ⇒ spill.
        let coord = |i: u32| v(i % 8, i / 8, 0); // all within the (0,0,0) brick
        for i in 0..17u32 {
            tree.set_voxel(coord(i), true);
        }
        for i in 0..15u32 {
            assert_eq!(
                tree.set_material(coord(i), u16::try_from(i + 1).unwrap()),
                Edit::Material {
                    leaf: 0,
                    spilled: false
                }
            );
        }
        let gen_before = tree.topology_generation();
        // The 16th distinct colour (with a 0 still present) makes 17 ⇒ spill,
        // a topology-class event that bumps the generation.
        assert_eq!(
            tree.set_material(coord(15), 16),
            Edit::Material {
                leaf: 0,
                spilled: true
            }
        );
        assert_eq!(tree.topology_generation(), gen_before + 1);
    }

    #[test]
    fn from_voxels_matches_incremental_build_and_carries_materials() {
        let r = res(32);
        // Voxels across two bricks (brick (0,0,0): the first three; brick (1,0,0):
        // the last two), each with a global material id.
        let pts: Vec<(VoxelCoord, u16)> = vec![
            (VoxelCoord::new(0, 0, 0), 3),
            (VoxelCoord::new(1, 0, 0), 3),
            (VoxelCoord::new(7, 7, 7), 5),
            (VoxelCoord::new(8, 0, 0), 7),
            (VoxelCoord::new(9, 2, 3), 7),
        ];
        let tree = SparseTree::from_voxels(r, pts.iter().copied());

        // Occupancy is bit-identical to building the same voxels incrementally.
        let mut inc = SparseTree::build(&Empty { resolution: r });
        for (c, _) in &pts {
            inc.set_voxel(*c, true);
        }
        assert_eq!(
            tree.leaves, inc.leaves,
            "occupancy diverged from incremental"
        );
        assert_eq!(tree.codes, inc.codes, "brick codes diverged");
        assert_eq!(tree.leaf_count(), 2);

        // Materials are carried per voxel; unoccupied reads global-0.
        for (c, gid) in &pts {
            assert_eq!(tree.material_at(*c), *gid, "material at {c:?}");
        }
        assert_eq!(tree.material_at(VoxelCoord::new(2, 0, 0)), 0);
    }

    #[test]
    fn from_voxels_duplicate_coord_keeps_last() {
        let r = res(32);
        // A repeated coord (chunk-boundary case) keeps the last global id written.
        let pts = [
            (VoxelCoord::new(4, 4, 4), 2u16),
            (VoxelCoord::new(4, 4, 4), 9u16),
        ];
        let tree = SparseTree::from_voxels(r, pts.iter().copied());
        assert_eq!(tree.leaf_count(), 1);
        assert_eq!(tree.material_at(VoxelCoord::new(4, 4, 4)), 9);
    }
}
