# Sparse MIP Voxel Structure — Design Document

**Status:** Draft v1
**Goal:** Minimize ray-traversal time over a binary occupancy grid on GPU, for scale-invariant (fractal-ish, low-to-moderate box-counting dimension) occupancy.
**Core idea:** A shallow MIP pyramid over binary occupancy, stored sparsely (empty subtrees elided), with bitmask leaves, traversed by a stackless Hierarchical DDA (HDDA).

---

## 1. Problem statement and assumptions

We are marching rays through a 3D binary occupancy grid (Amanatides & Woo style) and want to minimize traversal time by skipping empty space hierarchically.

**Assumptions driving the design (validate before committing — see §10):**

- **Occupancy is scale-invariant**, with box-counting dimension `D` in the sparse-to-moderate range (roughly `1 ≤ D ≤ 2.6`). This is what justifies a *sparse* structure over a dense one.
- **Target is GPU** (SIMT, 32-thread warps). Layout and scheduling decisions follow from warp behavior.
- **Field is static or slowly changing** (rebuilt infrequently relative to how often it is traced). If per-frame dynamic, this is an open question — see §12.
- **Primary workload is reasonably coherent rays** (e.g. camera primaries). Incoherent secondary rays are handled as an escalation, not the default (§7).

---

## 2. Key decisions at a glance

| Axis | Decision | Driver |
|---|---|---|
| Storage | Sparse (elide empty subtrees) | Low `D` ⇒ base grid is mostly empty; dense wastes `O(N³)` |
| Hierarchy | MIP pyramid, **uniform `4³` branching**, ~4–5 levels at 512³–2048³ | Lands in the optimal 3–6 band; `4³` chosen for level count and is required for any cross-level Morton address computation (applies to both School A and B) |
| Leaf encoding | **Bitmask** (1 bit/voxel), `8³` brick = 64 B | O(1) test; bitmask also serves as skip signal + `popcount`-rank index |
| Buffer layout | Serialized, pointerless, integer offsets; **roll-your-own** (not NanoVDB) | School B + uniform branching is incompatible with NanoVDB's `32³/16³/8³` |
| Intra-level order | **Morton (Z-order)** of occupied cells | Spatial neighbors → memory neighbors → warp coalescing |
| Cross-level order | **School B — shared Z-curve, subtree = contiguous interval (PROVISIONAL — pending §10 cache-residency + descent-frequency measurement)** | Descents stay local: coherent warp descends together, converged + coalesced — but only pays off if descent jumps are real misses; measurement decides |
| Traversal | **Stackless HDDA**, uniform-step loop | Keeps warp lanes on the same instruction; avoids divergence |
| Scheduling | **Coherence batching only**; no wavefront re-sort initially | A&W per-step work is cheap; wavefront overhead unlikely to be repaid |

---

## 3. The MIP is the information; sparsity is the layout

A MIP level of a binary occupancy grid **is** a bitmask: each coarse bit is the OR-reduction of its children. The reduction bottoms out at the individual **voxel** (the terminal, finest entity); the `8³` leaf brick is a *container* level, not the terminal — a set leaf bit means `≥1` of its 512 descendant voxels is set, not a surface hit.

```
mip[L+1][c] = OR over all children of c at mip[L]
```

(Conceptually 2³ per-axis subdivision; in this design the internal levels use `4³` branching = 64 children per node, and the leaf is an `8³` bitmask — see §4.)

A **clear** coarse bit ⇒ the entire subtree below is empty ⇒ skip it in one DDA step. A **set** bit ⇒ descend.

"Sparse bricks" are **not an alternative to the MIP** — they *are* the MIP, with the all-zero subtrees not stored. Same information, fewer cells allocated, at the cost of one indirection (coarse cell → packed slot of its stored child block).

Each level is stored as its own sparse set of occupied cells. The sets are **strictly nested upward** — the chain runs voxel → `8³` leaf brick → internal `4³` nodes → root: a cell exists at level `L` iff at least one descendant (ultimately a set voxel) exists below it. Every stored fine brick therefore has a stored parent chain to the root, which is what makes descent always well-defined.

Total memory ≈ `Σ_L N_occupied(L)`, dominated by the finest level; coarse levels are nearly free.

---

## 4. Level structure (concrete — `4³` internal, `8³` leaf)

**Committed structure:** uniform `4³` branching on internal levels, `8³` bitmask leaf, ~4–5 levels at 512³–2048³.

- **Leaf — `8³` bitmask brick.** 512 bits = **64 bytes** = one cache line / two NVIDIA sectors. Stored only where occupied. Internal `8³` Morton order; terminates its subtree *interval* in the buffer (§6.4, Option B). Note this is a **layout** fact, distinct from the traversal terminal: the leaf brick is the finest *stored* node, but traversal descends one level further into its 512 voxel bits — an occupied brick is **not** a hit (the terminal traversal level is the voxel, §7).
- **Internal levels — `4³` nodes.** Each node has a **64-bit child bitmask** (one bit per `4³` child) = the OR-reduction "does this child's subtree contain anything." This is the skip signal *and* the `popcount`-rank index. Stored sparsely, laid out on the shared Z-curve (§6.4).

Coordinate split (all shifts/masks; `res = 8 · 4^k` where `k` = number of internal levels). NB: this `k`-based count enumerates **storage** levels (k internal + 1 leaf brick); the traversal L-index (§7) additionally numbers the individual voxel as L=0, so the traversal levels run L=0 (voxel) … L=k+1 (coarsest internal). Do not conflate the two counts.

```
leaf_coord    = voxel_coord & 7              // low 3 bits/axis → position in 8³ leaf
internal_path = voxel_coord >> 3             // remaining bits, 2 bits/axis per 4³ level
// level L child index (L=0 finest/voxel, L=1 leaf brick, L≥2 internal): internal_path holds the 4³ path bits above the 8³ leaf. For internal level L (≥2), select its 2-bit per-axis field with a shift measured from the finest internal level: (internal_path >> (2*(L-2))) & 3  per axis  // larger L = coarser = higher bits; no L_max term
```

Valid resolutions are `8 × 4^k` per axis — only powers-of-4 multiples of 8 are representable without padding:

| k (internal levels) | Storage levels (k internal + leaf brick) | Grid resolution |
|---|---|---|
| 2 | 3 | 128³ |
| 3 | 4 | **512³** |
| 4 | 5 | **2048³** |

Storage levels = `k + 1` (k internal `4³` nodes + the `8³` leaf brick); the coarsest internal level is `COARSE = L_(k+1)` in the traversal index. The individual **voxel** sits one scale below the leaf brick as the terminal traversal level (L=0, §7) and is **not** counted as a storage level — the "optimal 3–6 band" refers to these storage/node levels.

512³ and 2048³ are the natural targets in the 4–5 level band. 1024³ is **not** representable with this scheme (`1024 / 8 = 128` is not a power of 4); grids sized 1024³ would need to be either padded to 2048³ or cropped to 512³.

**The coarsest level's job is fetch-avoidance, tune it for that.** The most efficient memory access is the one you never issue: a clear bit at the top level skips an entire subtree with a single 64-bit bitmask read and no touch of anything below. Because scale-invariant occupancy has a heavy-tailed gap distribution (lacunarity, §10) — a few enormous empty regions dominate skippable volume — most avoided-fetch benefit comes from the *top* level taking large strides cheaply. Keep the coarsest level cache-resident (§10); that is where the bandwidth savings concentrate, not at the leaves.

---

## 5. The bitmask does triple duty

The per-node child bitmask is the structural workhorse. One bitmask serves three jobs:

1. **Occupancy test** — index the bit, read it. O(1), branch-light: `(mask >> bit) & 1`.
2. **Skip signal** — a clear bit means "subtree empty, skip the whole extent."
3. **Index into elided storage** — the packed slot of a child among its stored siblings is the population count of set bits below it:

```
child_slot = subtree_base + popcount(mask & ((1ULL << bit) - 1))  // same base offset called `subtree_base` in the School B layout (§6.4); 64-bit shift; 1<<bit overflows for bit >= 32
```

This is how we store *only* occupied children yet address them in O(1) with no per-child index array — a single hardware `popcount`. This is the reason bitmask leaves beat CSR (which would turn the per-step test into an `O(log k)` search with poor locality and warp divergence).

**Alternative approach (not part of this design) — hash-keyed variant:** if cells are keyed by 64-bit Morton code in a hash map, only 63 bits are used (3×21). The spare high bit can carry the cell's `filled`/occupancy flag *inline with the key*, so the coarse skip test reads occupancy from the key itself with no separate lookup. This is a property of a hash-map-based structure, not of the offset-array structure described in §6. It is noted here for completeness; the primary design uses packed offset arrays with `popcount`-rank indexing, not a hash map.

---

## 6. Memory layout

### 6.1 Brick size ↔ hardware

| Platform | Cache line | Transaction granule | Leaf brick |
|---|---|---|---|
| NVIDIA / Apple | 128 B | 32 B sector | **`8³` = 64 B = 2 sectors** |
| AMD / CPU | 64 B | — | `8³` = 64 B = 1 line |

`8³` is the default: one brick's occupancy is one small, aligned transaction.

### 6.2 Ordering

- **Within a level:** sort occupied cells by **Morton code** of their coordinate. Spatially-adjacent cells → adjacent offsets → coherent warps coalesce.
- **Within a brick:** store the 512 bits in **Morton order** of intra-brick coordinates, so fine DDA walks contiguous bits.
- **Across levels (School B, provisional — §6.4):** one shared Morton curve; each subtree is a contiguous interval in a single buffer. Descend via `popcount`-rank into the child sub-interval. (School A alternative: independent per-level arrays, simpler bookkeeping — choose based on §10 measurement.)

**Morton encode method — platform-dependent, this matters on the hot path.** A 64-bit code holds 3×21 bits with one bit to spare (usable as an inline `filled`/occupancy flag on the key — see §5). Three implementations exist (ref: Baert 2013, §13):
- **For-loop** (bit-by-bit): ~10× slower, avoid on any hot path.
- **Magic bits** (`splitBy3` mask cascade): ALU-only (shifts + ANDs), no memory traffic. **Use this on the GPU** — it runs identically across all 32 warp lanes in lockstep with no serialization.
- **LUT** (256-entry tables): fastest on *CPU* in Baert's benchmark, **but often worse on GPU** — table lookups touch cache/memory and can serialize a warp. Do **not** copy the "LUT is fastest" CPU conclusion onto the GPU traversal path.
- **`pdep`/`pext`** (BMI2 on CPU; equivalent bit-deposit on recent NVIDIA): collapses `splitBy3` into ~one instruction — the best option for **CPU-side build-time** encoding.

Rule of thumb: **magic-bits for GPU traversal-time encoding; `pdep` (or LUT) for CPU build-time encoding.**

### 6.3 Serialization

One contiguous buffer, child references as **integer offsets** (not pointers). Single upload, read-only, coalescable. **Roll your own** (see §9) — NanoVDB's `32³/16³/8³` non-uniform branching is incompatible with the `4³` internal / `8³` leaf structure regardless of School A or B. NanoVDB remains useful as a reference and for I/O interop, not as the in-memory traversable format.

**Buffer direction depends on layout choice:**
- **School A:** root → internal → leaves (coarse to fine). Each level is a self-contained array; descent follows an offset into the next level's array.
- **School B:** leaves → internal → root (fine to coarse, post-order). Each subtree is a contiguous interval with the parent at the interval's end. See §6.4 for the DFS emission pass that produces this ordering.

### 6.4 Cross-level shared Z-curve — School B (PROVISIONAL)

**Provisional direction (not locked — see gate below):** one shared Morton curve across all internal levels, each coarse cell + its whole subtree occupying a **contiguous interval** (linear-octree layout). Descent stays local *and* sideways-within-a-cell stays local; you pay a long jump only when the ray exits the cell — which coincides with a hierarchy ascent. A coherent warp descending together touches one contiguous interval → control-converged *and* memory-coalesced at once.

**Gate: this decision requires measurement before committing.** School B's advantage is real only when descent jumps are actual cache misses — if the working set fits in L2, School A's simpler per-level layout may be sufficient with no meaningful miss penalty. Required data (from §10):
- Per-level footprint vs. GPU L2 size → do descent jumps cross a cache boundary?
- Measured descent frequency per ray on representative data → how often is the miss penalty paid?

Lock School B when both numbers confirm that (a) descent jumps miss L2 and (b) they occur frequently enough to justify the offset/interval bookkeeping complexity. If the working set is cache-resident, prefer School A.

**Branching: uniform `4³` on internal levels.** The shared curve requires every level to decompose the same way (a coarse cell's Morton code is the prefix of its descendants'). `4³` (2 bits/axis/level, Morton advances 6 bits/level) lands a 512³–2048³ grid in ~4–5 levels — the optimal band — with fewer descend transitions than `2³` (which would need ~10 levels). This rules out NanoVDB's `32³/16³/8³` (non-uniform → codes don't nest) — hence roll-your-own (§9).

**Leaf reconciliation (Option B).** Keep the cache-line `8³` bitmask leaf (§6.1). Run the shared `4³` Z-curve over the **internal** levels only; each leaf is the base case — a contiguous 64 B block terminating its subtree's interval, with its own internal `8³` Morton order. The prefix-nesting that School B needs applies to the internal hierarchy where cross-level interleaving happens; the leaf doesn't break it.

**Post-order convention (load-bearing — fix before coding).** The School B buffer uses post-order DFS: children precede their parent in the linear sequence. A node's children sit at *lower* addresses within its subtree interval and the parent sits at the interval's *end*. Descent moves toward the children's sub-interval; ascent moves out to the enclosing one. Get this backwards and every descend index is off by a subtree.

**Important:** standard integer sort of Morton codes does *not* produce post-order. A parent's truncated code is numerically smaller than any child's extended code, so plain integer sort is pre-order (parent before children). Post-order requires either a reverse sort, a DFS emission pass, or a custom comparator that ranks longer (finer) codes first. The "sort is the hierarchy" property applies to the *spatial* ordering within a level — it does not automatically produce the cross-level interleaving that School B requires.

**Build pipeline (School B):**
1. Voxelize → list of occupied fine-cell coordinates.
2. Compute Morton codes (`pdep`/LUT, CPU build-time — §6.2).
3. **Sort by Morton code** within the fine level → Z-order spatial layout.
4. Scan sorted list, OR-reduce each `4³` group into its parent, emit parent bitmasks, recurse up `log₄(res/8)` = `k` internal levels (the leaf consumes the factor of 8 = 3 bits/axis; only the `4³` internal levels are recursed). This produces per-level sorted arrays.
5. **DFS emission pass:** walk the tree recursively to emit the final buffer in post-order — each node's children sub-interval written before the node itself. This pass is `O(N)` over the already-built per-level arrays; the `O(N log N)` cost is dominated by the sort in step 3.
6. Record per-node subtree-base offsets during emission.

One sort + `log₄` linear scans + one linear DFS pass. No pointer-chasing construction.

**Descent indexing.** At a `4³` node the child bitmask is 64 bits; child `i`'s slot:
```
child_slot = subtree_base + popcount(mask64 & ((1ULL << child_bit) - 1))  // 64-bit shift; 1<<child_bit overflows for child_bit >= 32
```
`subtree_base` (start of this node's children sub-interval) is known from the post-order layout — no separate pointer.

**Costs accepted if School B is confirmed:** fiddlier offset/interval bookkeeping; convention bugs (post-order direction, subtree-base) are real, not hypothetical; NanoVDB no longer a drop-in for the structure (its traversal patterns and I/O tooling can still serve as reference). The §10 cache-residency measurement is the gate — if descent jumps don't miss L2, these costs are not justified and School A should be preferred.

---

## 7. Traversal: stackless HDDA

Run A&W at the coarse level; descend on occupied cells.

### 7.1 Per-axis setup (per level)

```
step.x    = sign(D.x)
// Guard BEFORE dividing — D.x == 0 means the ray is axis-aligned and never crosses an x-boundary
if D.x == 0:
    tMax.x   = +inf
    tDelta.x = +inf
else:
    tMax.x    = (next boundary in step dir - O.x) / D.x
    // L=0 is the VOXEL (terminal, 1 base voxel); L=1 is the 8³ LEAF BRICK (8 voxels/axis, ×8 step);
    // each internal level L≥2 multiplies extent by 4 (×4 step). cellSize(L) in base voxels:
    //     L == 0 -> 1                      (the voxel)
    //     L >= 1 -> 8 * 4^(L-1) = 2^(2L+1) (L=1 brick=8, L=2=32, L=3=128, ...)
    uint align = (L == 0u) ? 0u : (2u * L + 1u);              // 0,3,5,7,9,... ; voxel->brick jumps 0->3 (×8), then +2/level
    cellSize.x = baseVoxelSize * float(1u << align)           // 1,8,32,128,512,... base voxels; integer shift, cast to float
    tDelta.x  = cellSize.x / abs(D.x)
```

**Level-size convention:** L=0 is the **VOXEL** (terminal, cell extent = 1 base voxel; a set voxel bit = surface = HIT). L=1 is the `8³` **LEAF BRICK** (extent 8 base voxels/axis): voxel→brick is a `×8` step (3 bits). Each internal level `L≥2` adds 2 bits per axis (`4³` branching, `×4` step). So cell extent is 1 at L=0 and `8 × 4^(L-1) = 2^(2L+1)` base voxels at `L≥1` (L=1→8, L=2→32, L=3→128, …). The earlier `voxelSize << L` and `8 × 4^L` were both wrong: the former used a binary shift (`×2^L`) instead of `×4^L`, and the latter omitted the voxel level (off-by-one), giving 8 at the voxel and 64 at the leaf instead of 1 and 8.

### 7.2 Loop (uniform-step, stackless)

```
level = COARSE                            # = L_(k+1), the coarsest internal level
loop:
    occ = test_occupied(current cell, level)
    #   level == 0 (VOXEL)    -> read the voxel bit: brick512[ morton8(coord & 7) ]
    #   level == 1 (BRICK)    -> brick is present in the sparse set (its 512-bit mask is non-empty)
    #   level >= 2 (INTERNAL) -> (childMask64 >> childBit(coord, level)) & 1
    if occ == 0:                          # empty subtree / clear voxel bit — skip
        advance DDA at this level (smallest tMax axis), step one cell
        if exited grid: return MISS
        # stackless ascent: derive parent extent from base-voxel coords, no stored stack.
        # Guard: can only ascend while not already at the coarsest level.
        while level < COARSE:
            # parent is at level+1; under L=0=voxel its cellSize is 2^(2*(level+1)+1) = 2^(2*level+3)
            # base voxels/axis (L=0 voxel -> parent L=1 brick = 2^3 = 8; L=1 brick -> L=2 = 2^5 = 32; ...)
            parent_bits = 2 * level + 3           # = 2*(level+1)+1; at L=0 this is 3 (the voxel->brick ×8 step)
            parent_origin = (current_coord >> parent_bits) << parent_bits
            if current_coord inside [parent_origin, parent_origin + (1 << parent_bits)):
                break                    # still inside parent — stay at this level
            level = level + 1            # exited parent — ascend (coarser = larger L)
            RECOMPUTE tMax for new level from current position   # no stored parent tMax — stackless; see 7.3
            UPDATE   tDelta for new level                        # tDelta = cellSize(level)/abs(D); see 7.1
    else:                                # occupied: >=1 child/voxel set
        if level == 0:                   # VOXEL (L=0) is the terminal level: a set voxel bit
            return HIT                   # set voxel bit = surface = hit
        else:
            descend: level = level - 1   # finer = smaller L; brick(L=1)->voxel(L=0) is just
                                         # another descend, NOT a nested inner DDA (same loop body)
            RECOMPUTE child tMax from the actual entry face      # see 7.3
            UPDATE   tDelta for new level                        # tDelta = cellSize(level)/abs(D); see 7.1
```

Encode level as **data**, not control flow, so all warp lanes run the same instruction each iteration (only the *level value* differs). This is what avoids control divergence. Ascent is amortized cheap — a ray ascends at most as many times as it descends over its full path.

**Stackless ascent:** parent extent is derived from the current base-voxel coordinate by masking off the low `parent_bits(level)` bits — no per-level coordinate stack is stored. Under L=0=voxel, `parent_bits(level) = 2*level + 3` (the parent of the voxel L=0 is the `8³` brick at 3 bits = the `×8` step; the parent of a brick/internal at L≥1 is at `2*(level+1)+1 = 2*level+3` bits = a `×4` step). The check is pure integer arithmetic in registers. Because the design is stackless there is no stored parent `tMax` to resume, so on ascent `tMax` must be recomputed for the coarser level from the current position (§7.3), exactly as on descent — only `tDelta` is a simple per-level scale.

### 7.3 The descend recompute (main bug source)

On entry to an occupied cell the ray enters through a **face**, not at a child voxel boundary. Child `tMax` must be recomputed with the general ray-vs-plane form from the actual entry point — **do not inherit the parent's `tMax`**, or children misalign. `tDelta` simply scales by the level's voxel size.

**Ascent is symmetric (also stackless).** Because no parent state is stored, ascending to a coarser level cannot resume a saved `tMax` — the coarser level's `tMax` must be recomputed from the current ray position to the next coarser boundary (same ray-vs-plane form). Skipping this and reusing the finer level's `tMax` makes the next coarse-level DDA step fire at a stale finer boundary. `tDelta` likewise rescales to the coarser cell size.

### 7.4 Per-ray state → registers, SoA

Keep `tMax`, current coords, and level in registers. Any spilled/shared arrays are **Structure-of-Arrays** (`tMaxX[]`, `tMaxY[]`, …) so a warp's 32 threads touching "their" element coalesce. Keep state lean to preserve occupancy (resident warps hiding latency).

---

## 8. Ray scheduling

Mitigations in increasing cost; **build only the first two**:

1. **Coherence batching (build).** Group rays by origin + direction so a warp's 32 rays tend to skip/descend together → minimal divergence, coalesced brick fetches. Camera primaries are naturally coherent.
2. **Stackless uniform traversal (build).** Already in §7 — same instruction per lane per step.
3. **Wavefront / persistent-thread re-sort (DO NOT build initially).** Converts a divergence problem into a bandwidth problem (full per-ray state I/O + compaction per pass). For A&W's cheap per-step work + coherent primaries + modest state, the overhead is unlikely to be repaid. Reach for it **only** if profiling at §11 step 5 shows warp execution efficiency in the teens on a specific (incoherent) ray type, and apply it only to that ray type.

---

## 9. Build vs. adopt — DECIDED: roll your own

**Decision: roll your own structure.** If School B (§6.4) is confirmed by measurement, its shared cross-level Z-curve with uniform `4³` branching is incompatible with NanoVDB's `32³/16³/8³` non-uniform branching — codes don't nest into clean contiguous subtree intervals, so NanoVDB cannot serve as the in-memory traversable structure. Even if School A is chosen instead, NanoVDB's uniform branching still doesn't match `4³`, so the structure remains custom either way. NanoVDB remains useful as a **reference** (HDDA traversal patterns, the `popcount`-rank child-indexing trick) and for **I/O / interop tooling** (round-tripping to OpenVDB assets).

**What you reuse from the ecosystem anyway:**
- HDDA traversal structure (Museth 2014) — the per-level DDA loop pattern.
- The bitmask + `popcount`-rank addressing (VDB, ESVO).
- Morton primitives (Baert 2013) — magic-bits on GPU, `pdep`/LUT on CPU.

**What's genuinely yours to implement:**
- **Either school:** per-level Morton sort, the `popcount`-rank child indexing, and the stackless HDDA loop.
- **School B only:** the post-order DFS emission pass, subtree-base offset bookkeeping, and the shared-curve interval buffer.
- **School A only:** per-level offset tables and the coarse-to-fine level-array descent indexing.

---

## 10. Validation before committing (do this first)

Three measurements decide whether this design is even the right one:

1. **Measure `D`** — regress `log₄ N(L)` vs `L` over the MIP pyramid (slope = `−D`, since each level is a `4×` linear coarsening; equivalently regress `log N` vs `log(cell size)`). Convention: L=0 is the finest level (the individual **voxel**; the `8³` leaf brick is L=1), L increases toward the coarsest — so N(L) decreases as L increases, giving a negative slope equal to `−D`. Linearity confirms scale-invariance; a kink locates a real feature scale.
   - `D` low-to-moderate (≈ 1–2.6): **proceed with this design.**
   - `D` near 3 (near-solid): **abandon sparsity** — use a dense base + dense MIP, drop the brick-pool indirection. (Overhead depends on the MIP branching: a standard `2³` octree MIP is `8/7× ≈ 1.14×`; a dense `4³` MIP matching this design's branching is `64/63× ≈ 1.016×`. State which fallback you mean — do not quote `8/7×` against a `4³` structure.)
2. **Measure update rate** — how often does occupancy change vs. how often is it traced?
   - Static / slow: serialized sparse with sort-based build (this design).
   - Per-frame dynamic: prefer a cheaper-to-rebuild structure (dense MIP, or an in-place-updatable hash grid); the sort + DFS emission pass of the custom build may not fit the frame budget.
3. **Measure per-level footprint and cache residency** — for each level, compute `N_occupied(L) × bytes_per_cell` at your target resolution, and place each level against the cache hierarchy (L1 ~32–48 KB, L2 ~512 KB–2 MB, L3 tens of MB; on GPU, L2 is shared and the relevant level). This is the measurement the original draft omitted, and it drives three downstream calls:
   - **Whether to drop to a lower resolution** — if the coarsest level's footprint spills out of L2 at your target resolution, the options are: (a) drop to the next lower valid resolution (e.g. 2048³ → 512³, accepting coarser voxels), or (b) accept the cache miss on the top level and rely on lacunarity to limit how often it fires. A top-level skip that hits cache is nearly free; one that misses to DRAM is not. Adding an extra level of hierarchy beyond what `res = 8 × 4^k` gives is not an option without breaking the uniform Morton address scheme.
   - **The School A vs B call (§6.4)** — if the working set spills to DRAM, descend jumps become real misses and School B comes on the table earlier. Evaluate residency and descent frequency together, not separately.
   - **Coarsest-level tuning (§4)** — the top level should be sized so it is cheap-to-test *and* cache-resident, since that is where fetch-avoidance savings concentrate.

Optional: **lacunarity `Λ(L)`** (variance-to-mean² of per-cell occupancy mass per level) to set the constants in the depth optimum and predict per-ray cost variance.

Net: `D` decides sparse-vs-dense, update rate decides rebuild strategy (sort-based full rebuild for static/slow fields vs. an in-place-updatable structure for per-frame dynamic), and per-level cache residency decides level count, School A-vs-B, and coarse-level tuning. The roll-your-own decision (§9) is independent of update rate — NanoVDB's branching is incompatible with `4³` either way. Take all three measurements before writing structure code.

---

## 11. Build order (incremental, each step diff-tested against the previous)

1. **Dense-base single-level A&W.** Correctness oracle. Every later step is diffed against this.
2. **Add one coarse skip level** (dense coarse grid + `8³` bitmask leaves). Validates the skip mechanic and the descend recompute (§7.3) in isolation, before sparsity/layout complexity. Measure speedup.
3. **Make leaves sparse + Morton-order them.** Introduce the `popcount`-rank child indexing. Measure memory drop and coalescing. Still diff against the oracle.
4. **Complete §10 measurements; then build the buffer layout chosen.**
   - **If School B confirmed:** build the full `4³` shared-Z-curve buffer — sort-based pipeline (§6.4): voxelize → Morton codes → sort → OR-reduce up → emit post-order interval buffer. Get the post-order convention right (§6.4) before writing any traversal code.
   - **If School A chosen:** build independent per-level sparse arrays with per-level Morton order and a per-level offset table. Simpler bookkeeping, no post-order interval constraint.
   Either way, diff every cell against the oracle before moving to step 5.
5. **Stackless HDDA over the buffer + coherence-batch rays.** Measure warp execution efficiency.
6. **Stop and profile.** Only then decide if any incoherent ray type needs wavefront (§8, mitigation 3). (If School B was confirmed in §10, no layout migration remains. If School A was chosen instead, the buffer format is simpler and no migration is needed either.)

---

## 12. Open questions / to fill in

- Target resolution (sets exact level count: 512³→4 storage levels, 2048³→5 storage levels — see §4 resolution table; the voxel terminal is the finest traversal level below these; and whether the top level stays cache-resident, §10).
- Static vs per-frame dynamic. **Note:** structure is now roll-your-own with a sort-based build (§6.4); if per-frame dynamic, confirm the sort+scan rebuild fits the frame budget, else consider an updatable variant.
- Measured `D` (confirms sparse vs dense — still the gating call, §10).
- Ray mix: fraction coherent (primary) vs incoherent (secondary/GI) — sets whether §8, mitigation 3 (wavefront) is ever needed. (Also: School B's payoff is largest when rays dive/surface a lot, so a high dive/surface rate would be evidence in favor of School B — one input to the §6.4 gate, not a confirmation that School B is already chosen; the choice remains PROVISIONAL pending §10.)
- Anisotropy: is occupancy isotropic, or does it have a grain? If anisotropic, `D` is direction-dependent and level/leaf choices may want directional tuning.

---

## 13. References

- Amanatides & Woo (1987), *A Fast Voxel Traversal Algorithm for Ray Tracing.* http://www.cse.yorku.ca/~amana/research/grid.pdf
- Fujimoto, Tanaka & Iwata (1986), *ARTS: Accelerated Ray-Tracing System.* https://doi.org/10.1109/MCG.1986.276715
- Museth (2013), *VDB: High-Resolution Sparse Volumes with Dynamic Topology.* https://www.museth.org/Ken/Publications_files/Museth_TOG13.pdf
- Museth (2014), *Hierarchical Digital Differential Analyzer for Efficient Ray-Marching in OpenVDB.* https://doi.org/10.1145/2614106.2614136
- Museth (2021), *NanoVDB: A GPU-Friendly and Portable VDB Data Structure.* https://doi.org/10.1145/3450623.3464653
- Laine & Karras (2010), *Efficient Sparse Voxel Octrees.* https://research.nvidia.com/sites/default/files/pubs/2010-02_Efficient-Sparse-Voxel/laine2010i3d_paper.pdf
- Crassin, Neyret, Lefebvre & Eisemann (2009), *GigaVoxels.* https://doi.org/10.1145/1507149.1507152
- Aila & Laine (2009), *Understanding the Efficiency of Ray Traversal on GPUs.* http://www.tml.tkk.fi/~timo/publications/aila2009hpg_paper.pdf
- Laine, Karras & Aila (2013), *Megakernels Considered Harmful: Wavefront Path Tracing on GPUs.* https://research.nvidia.com/publication/2013-07_megakernels-considered-harmful-wavefront-path-tracing-gpus
- Lefebvre & Hoppe (2006), *Perfect Spatial Hashing.* https://doi.org/10.1145/1141911.1141926
- Baert (2013), *Morton Encoding/Decoding Through Bit Interleaving: Implementations.* https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/ — practical for-loop / magic-bits / LUT implementations and benchmarks for the Morton primitive (§6.2).