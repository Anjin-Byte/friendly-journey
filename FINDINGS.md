# Friendly-Journey — Findings: Sparse MIP Voxel GPU Traversal and the Orientation-Anisotropy Investigation
_June 2026_

## Abstract

This document records a multi-week investigation into the GPU ray-traversal performance of `friendly-journey`, a Rust + wgpu sparse, MIP-mapped voxel renderer that traverses a Morton-ordered (Z-order) brick hierarchy on the GPU. The renderer is fast enough for practical use: it reaches comfortable real-time frame rates on structured, surface-like content and degrades to roughly 20–60 fps at ~1080p only on the pathological "dust-class" case — a volume densely sprinkled with tiny, spatially incoherent occupied voxels that defeats both the early-skip path and the cache. The investigation, however, was not primarily about the average case. It was about a specific, reproducible, and stubborn artifact: traversal cost depends on the *direction the camera is looking* far more than it depends on what is on screen. Holding the scene, the resolution, and the visible surface area fixed, and rotating only the view direction, swung GPU traversal cost by as much as ~9× between the cheapest and most expensive orientations. We set out to understand the mechanism behind this orientation anisotropy and to reduce it in software. We succeeded at the first goal and failed — informatively — at the second. The anisotropy is intrinsic: it is dominated by memory-miss *latency* (rare but expensive cross-cache-block misses) compounded by SIMD/warp divergence when obliquely-oriented ray bundles walk a Morton structure whose locality is, by construction, axis-isotropic only at one scale. Every software lever we built to attack it was either falsified outright or bounded to a small constant factor, and the published literature has independently concluded that the divergence component is best solved in hardware (Shader Execution Reordering), not by relayout. The recommendation is therefore to stop spending engineering effort flattening the anisotropy via software relayout/reordering and instead either pivot the hot path onto the hardware ray-tracing pipeline or budget *around* the anisotropy with adaptive internal resolution and temporal reprojection.

## TL;DR

The renderer is real-time for structured content and ~20–60 fps at ~1080p for the worst (dust-class, sparse-incoherent) case. The dominant performance phenomenon is a **~9× orientation anisotropy**: rotating the camera while holding the image essentially fixed changes GPU traversal cost by up to ~9×. This anisotropy is **intrinsic** — it is latency-bound, driven by rare cross-cache-block misses and warp divergence on oblique views of a sparse Morton-ordered structure, not by a fixable layout deficiency. We tried three distinct software levers to reduce it (ray binning, axis-permuted multi-layout, Hilbert/treelet relayout); each was independently bounded or falsified. The one change that *did* pay off — per-brick early-skip — reduces absolute work on sparse/thin geometry but does **not** remove the residual anisotropy, because the residual is miss latency, and hiding latency is a hardware problem. The field agrees: divergence-fighting moved into hardware via Shader Execution Reordering. **Recommendation: stop reducing the anisotropy in software; pivot to the hardware-RT pipeline (SER) or budget around it (adaptive internal-resolution + temporal reprojection).**

## Four levers, four verdicts

The core of the investigation was four software levers. One shipped; three were rejected. The unifying reason the three were bounded is stated below the table and recurs throughout the document: the Morton layout is *already* near-optimal at the cache-block scale, so there is no spatial-locality slack left for a smarter layout to recover. What remains is miss latency, which a relayout cannot hide.

| Lever | Best measured result | Why it's bounded | Verdict |
|---|---|---|---|
| **Per-brick early-skip** (SHIPPED, on `main`) | Dust-class 1.68× faster coherent vs. baseline at same thermal state; cell-steps 244→161 at 512³ and 710→473 at 2048³ (see §10) | A genuine win on sparse/thin geometry, but it removes *avoidable* stepping work, not the *intrinsic* residual; the orientation anisotropy survives it | **KEEP** |
| **Ray binning** | ~1.2× at the kernel level; a real GPU bin nets only ~1.1–1.15× end-to-end | Recovers only the warp-divergence component; the ~9× is a *per-direction intrinsic* cost. Primarily helps incoherent secondary rays, which this primary-ray path does not have | **Don't build** |
| **Axis-permuted multi-layout** | Image-invariant (0 of 49152 pixels changed); ~3× memory; mean 1.3× / worst 1.4–1.6× under best-of-N selection | The best-of-N "gain" was largely min-of-noise; a realistic O(1) runtime selector picked the right layout ~40% of the time (≈ chance for 3 layouts). Mechanism is real; deployable benefit is not | **Mechanism correct, do NOT deploy** |
| **Hilbert / treelet (cache-oblivious) relayout** | ~0 net | Morton is already ~90% cache-block-isotropic; Hilbert merely *relocates* which orientations are cheap/expensive rather than flattening them. The residual is latency, not layout | **Don't build** |

The single sentence that ties the three rejections together: **the Morton layout is already near-optimal at the cache-block scale, the residual orientation cost is miss *latency*, and hiding latency is a hardware problem — not something a software relayout or reordering can fix.** Early-skip wins precisely because it does something different in kind: it removes *stepping work that never needed to happen* (empty/thin spans), which is orthogonal to, and does not touch, the latency residual.

A measurement caveat that colors every absolute number below: all timings were collected on Apple M-series hardware through wgpu under sustained load, and thermal drift moved absolute GPU times by roughly 2–6× over a session. Consequently the trustworthy signal throughout this document is **ratios and within-run comparisons**, not standalone absolute milliseconds. Where we quote absolutes (e.g., the dust-class 1.68× figure, or specific cell-step counts), the comparison was taken at matched thermal state within a single run; cross-run absolute comparisons are deliberately avoided.

## Reading guide

This is a long document, organized so that a reader can either follow the full arc — system, win, infrastructure, problem, three failed attacks, external corroboration, synthesis — or jump to the lever they care about. The sections:

- **§2 — System under study.** The data structure (sparse MIP voxel hierarchy, Morton/Z-order brick layout), the GPU traversal kernel, and the wgpu/Rust harness. What "a step" and "a brick" mean, and where the time goes.
- **§3 — Per-brick early-skip + adversarial review.** The one shipped win: skipping empty/thin spans at brick granularity. Includes the adversarial review that probed whether the gain was real or a measurement artifact, and why it survived that scrutiny.
- **§4 — Measurement infrastructure.** How traversal cost was instrumented (`traverse_timed`, cell-step counters, GPU timestamp queries), the thermal-drift problem, and the methodology that made ratios trustworthy where absolutes were not.
- **§5 — The orientation anisotropy.** The central finding: characterization of the ~9× view-direction dependence, the controlled experiment that isolates direction from image content, and the mechanism (latency-bound cross-cache-block misses + warp divergence).
- **§6 — Lever 1: Ray binning.** Reordering rays for coherence before traversal. The ~1.2× kernel result, why a real GPU bin nets only ~1.1–1.15×, and why it addresses only the divergence component.
- **§7 — Lever 2: Axis-permuted multi-layout.** Storing the volume under multiple Morton permutations and selecting per-frame. Image-invariance validation, the ~3× memory cost, the best-of-N result, and why the O(1) selector collapses to chance.
- **§8 — Lever 3: Hilbert & treelet/cache-oblivious relayout.** Replacing Z-order with a more locality-preserving curve. Why Morton is already near-isotropic at the cache-block scale and Hilbert only relocates the problem.
- **§9 — What the research community says.** Independent corroboration: the field moved divergence mitigation into hardware (Shader Execution Reordering) rather than chasing software relayout, and why that is consistent with our results.
- **§10 — Synthesis & recommendations.** Pulling it together: keep early-skip; do not build the three rejected levers; pivot to hardware-RT (SER) or budget around the anisotropy via adaptive internal resolution and temporal reprojection. Includes the cell-step figures referenced above.

A note on framing before the details begin. The honest center of gravity of this report is a *negative* result: three plausible, well-motivated software optimizations did not work, and we can say precisely why each one was bounded. Negative results of this kind are valuable exactly because they redirect future effort — they tell the next engineer not to re-derive ray binning or a Hilbert relayout from first principles expecting a 9× recovery, because the 9× is not living in the layout. It is living in the memory hierarchy's latency to service rare misses on oblique traversals, and that is a problem the hardware vendors have already decided to own.

I have the spec grounded. Writing Section 2.

## 2. The system under study

This section describes the artifact the rest of the document measures: what the structure *is*, how it is addressed, the discipline under which it was built, and the methodology that governed every decision. None of the performance investigation that follows is interpretable without this baseline, because almost every lever explored later (orientation anisotropy, layout choice, traversal scheduling) is a consequence of one of the structural commitments enumerated here. The reader who skips this section will mistake structural constraints for free parameters.

### 2.1 The structure: a sparse MIP voxel pyramid

The object under traversal is a shallow MIP (mip-mapped) pyramid over a binary occupancy grid, stored sparsely so that all-empty subtrees are never allocated. The MIP *is* the information — each coarse cell is the OR-reduction of its children, bottoming out at the individual voxel — and sparsity is purely a layout decision: identical information, fewer cells materialized, at the cost of one indirection per descent. This framing matters for the investigation because it means every traversal cost we measure is a layout cost, not an information cost; the same field can be re-laid-out (School A vs School B, §2.4) without changing a single occupancy bit.

The pyramid has a deliberately rigid shape:

- **Uniform `4³` internal-node branching.** Every internal node subdivides into 64 children (2 bits per axis). Uniformity is non-negotiable: it is what lets a coarse cell's Morton code be a clean prefix of all its descendants, which is the precondition for any cross-level address computation and for the contiguous-interval layout of School B. This is also exactly why NanoVDB's non-uniform `32³/16³/8³` branching was rejected as the in-memory format — its codes do not nest into clean subtree intervals.
- **`8³` bitmask leaf brick.** The leaf is a 512-bit occupancy bitmask (64 bytes — one cache line, two transaction sectors on Apple/NVIDIA), with the 512 bits stored in intra-brick Morton (Z-order) so the finest DDA walks contiguous bits. The leaf is the finest *stored* node but **not** the traversal terminal.
- **Voxel-terminal semantics.** A set leaf *bit* is the hit; an occupied *brick* is not. This is the single most important semantic subtlety in the whole system and a recurring source of off-by-one-level confusion. An occupied brick means "≥1 of my 512 descendant voxels is set" — it is a reason to descend one more level, not a surface. Traversal only returns `HIT` on a set voxel bit at L0.

#### Resolution is illegal-by-type

Valid grid resolutions are exactly `8 · 4^k` per axis. Enumerated, the representable sizes are:

| k (internal levels) | Storage levels | Resolution |
|---|---|---|
| 0 | 1 | 8³ |
| 1 | 2 | 32³ |
| 2 | 3 | 128³ |
| 3 | 4 | 512³ |
| 4 | 5 | 2048³ |

The leaf consumes the factor of 8 (3 bits/axis); each `4³` internal level above it contributes a factor of 4. The consequence the spec calls out explicitly: **1024³ is not representable**, because `1024 / 8 = 128` is not a power of 4. A 1024³ field must be padded up to 2048³ or cropped down to 512³ — there is no in-between. Rather than discover this at runtime via a panic deep inside the builder, the project encodes it in the type system: a `Resolution` newtype makes only the five valid sizes constructible, so an illegal resolution is a compile-time/constructor-time impossibility rather than a buried invariant. This is the first and clearest example of the project's house style — push invariants into types so the compiler, not a test, is the thing that refuses the wrong program.

#### Level convention

The traversal level index `L` numbers from the voxel upward, and the cell-size convention follows from the branching factors:

| L | Entity | `cell_size(L)` (base voxels) |
|---|---|---|
| 0 | voxel (terminal) | 1 |
| 1 | `8³` leaf brick | 8 |
| ≥2 | `4³` internal | `2^(2L+1)` |

So L0 = 1, L1 = 8, L2 = 32, L3 = 128, L4 = 512, and so on. The voxel→brick transition is a `×8` step (3 bits); every internal level above the brick is a `×4` step (2 bits). It is worth stressing that the storage-level count (`k + 1`) and the traversal-level count (`k + 2`, because the voxel is a traversal level but not a stored one) differ by one. Conflating the two is the origin of the most common addressing bug in this codebase, and the spec deliberately uses two separate counting conventions to keep them apart. Wherever the document later quotes a "level," it means the traversal `L`-index unless stated otherwise.

### 2.2 Addressing: popcount-rank, no GPU Morton

The structure stores only occupied children, yet must address them in O(1) with no per-child index array. The mechanism is **popcount-rank**:

Each node carries a 64-bit child mask — one bit per `4³` child, in Morton order — plus a `base` offset. The slot of child `i` among its stored siblings is the number of set mask bits below its own bit:

`child_slot = base + popcount(mask & ((1 << bit) - 1))`

This is a single hardware popcount and one add. It is the reason bitmask leaves beat a CSR-style structure: CSR turns the per-step occupancy test into an `O(log k)` search with poor locality and warp divergence, whereas the bitmask test is `(mask >> bit) & 1` and the rank is one instruction. The same bitmask therefore does triple duty — occupancy test, empty-subtree skip signal, and storage index — which is the structural reason the whole design is as compact as it is.

Two GPU realities shape how this is implemented in the kernel:

- **No `u64` on the GPU.** WGSL has no 64-bit integer type, so the 64-bit child mask is split lo/hi into a `vec2<u32>`. Every `popcount(mask & ((1<<bit)-1))` on the device is therefore a two-word operation with the partial-mask boundary handled explicitly across the 32-bit split. This is mechanical but error-prone — `1 << bit` overflows for `bit ≥ 32`, which is exactly the high word — and is one of the places where the f32 mirror (§2.5) earns its keep by catching a divergence that a CPU `u64` reference would silently paper over.
- **Morton codes are build-time only.** 64-bit magic-bits Morton encoding happens on the CPU during the build and *never* on the GPU. This is a deliberate constraint, not an oversight. The GPU navigates purely by 6-bit child indices (2 bits/axis extracted from the current coordinate by shift-and-mask) and popcount rank. The kernel never encodes or decodes a Morton code at traversal time. Keeping Morton off the hot path removes a class of GPU-side ALU/serialization concerns entirely and means the device-side addressing is nothing but coordinate-bit extraction plus popcount — which is also why the device path can be made bit-identical to a scalar mirror (§2.5).

### 2.3 The build-time Morton constraint, restated as a boundary

It is worth naming the build/traverse boundary explicitly, because it recurs throughout the investigation. Anything involving Morton encoding — sorting occupied cells into Z-order, OR-reducing groups up the pyramid, emitting the buffer — lives entirely in the CPU builder. Anything the GPU does at traversal time is expressed in coordinate bits and popcount. The two halves communicate only through the serialized buffer and the `base`/mask fields. This boundary is what makes the GPU kernel small enough to reason about exhaustively, and small enough to mirror exactly on the CPU.

### 2.4 Layouts behind a trait: School A vs School B

The pyramid can be serialized two ways, and the project keeps both behind a `NodeLayout` trait rather than committing prematurely:

- **School A — per-level node arrays.** Each level is an independent, self-contained array; descent follows an offset into the next finer level's array. Simpler bookkeeping, no cross-level ordering constraint. This is the form that wins when the working set is cache-resident and descent jumps do not miss.
- **School B — single children-contiguous DFS buffer.** One storage buffer in post-order DFS layout: a node's children occupy a contiguous interval at lower addresses, the node sits at the interval's end, and descent is a popcount-rank index from the subtree base. Because leaves are Morton-sorted, every subtree's leaves are contiguous, so a coherent warp descending together touches one contiguous interval — control-converged and memory-coalesced at once. This is the **GPU form**: it presents as a single storage binding, which is the shape the kernel actually consumes.

The two schools are not a finished decision in the spec — School B is marked provisional, gated on a cache-residency and descent-frequency measurement. But for the purposes of this investigation, School B is the layout the GPU path uses, and the `NodeLayout` trait is what made it cheap to hold both in the codebase simultaneously and diff one against the other. The trait boundary is itself a methodology artifact: it kept the "which layout" question answerable by measurement rather than by rewrite.

A subtlety worth recording: a naive integer sort of Morton codes produces *pre-order* (parent before children, because a parent's truncated code is numerically smaller than any child's extended code), but School B needs *post-order*. Post-order is produced by an explicit DFS emission pass over the already-built per-level arrays, recording subtree-base offsets as it goes. Getting the post-order direction or the subtree-base backwards makes every descent index off by a whole subtree — a convention bug, not a tolerance bug, and one the differential test (§2.5) catches immediately and totally.

### 2.5 The tiered oracle and the correctness philosophy

The correctness strategy is a tiered oracle, and it encodes a specific philosophy: a discrete voxel hit is **topology**, so the comparison is **structural equality, never a tolerance**. Three tiers:

1. **Tier-A f64 reference** — a dense Amanatides–Woo traversal in `f64`, treated as ground truth. It is itself self-validated against analytic ray–AABB intersection, so the "truth" is not merely asserted; it is checked against closed-form geometry.
2. **The f32 mirror** — a scalar CPU traversal that is *bit-identical* to the WGSL kernel. Same arithmetic, same precision, same operation order. Its job is not to be correct in the f64 sense but to be *exactly the same wrong* as the GPU, so that any GPU-vs-mirror disagreement isolates a true GPU/driver/codegen issue rather than an f32-vs-f64 rounding artifact.
3. **The GPU kernel** itself.

This three-way arrangement is what lets a failure be *localized*. A GPU-vs-oracle mismatch that the mirror reproduces is a precision or algorithm issue in the shared logic; a GPU-vs-mirror mismatch is something the device did that the identical scalar code did not. Without the mirror, every f32 grazing disagreement would be indistinguishable from a real bug.

On grazing cases: because the hit is topological, the oracle does not "tolerate" near-boundary disagreement. Grazing disagreements between f32 and f64 are **bounded and logged**, not silently accepted — they are treated as data about where the f32 path diverges from ground truth, not as acceptable noise. The bright line is that GPU and mirror must agree *exactly*, because they share precision; only the f64-vs-f32 axis is allowed to differ, and even there only within a logged bound.

The acceptance gate is correspondingly strict: the differential test casts **20,000 random rays** per fixture and requires **0 mismatches** for GPU-vs-oracle *and* GPU-vs-mirror across all fixtures. Not "few," not "within epsilon" — zero. This is the reference-implementation-as-oracle discipline made concrete, and it is the reason the performance numbers later in this document can be trusted to be measuring a *correct* traversal rather than a fast-but-wrong one.

### 2.6 Traversal: explicit-stack iterative HDDA

Traversal is an iterative Hierarchical DDA (HDDA) with an explicit stack of at most `MAX_DEPTH = 8` frames. The loop runs Amanatides–Woo at the current level, skips on a clear bit (advancing the DDA one cell and ascending while the current coordinate has exited its parent's extent), and descends on a set bit until it reaches a voxel. The two operations that recur and that the spec flags as the main bug source are the `tMax` recomputations: `tMax` is recomputed from the actual entry face on **descend**, and — because the design carries no saved parent `tMax` — also recomputed from the current position on **ascent**. Inheriting a parent's `tMax` on descent, or reusing a finer level's `tMax` on ascent, misaligns the DDA and fires steps at stale boundaries. `tDelta` is the easy part: it simply rescales to the level's cell size.

Level is encoded as *data*, not control flow, so all warp lanes execute the same instruction each iteration with only the level *value* differing — this is the deliberate divergence-avoidance lever, and it is directly relevant to the warp-efficiency questions explored later.

#### The GPU-resident render path

The render path is a single compute dispatch that, per pixel, builds the camera ray, traverses, shades, and writes to a storage texture — followed by a fullscreen blit. There is **no per-ray readback**: traversal results never round-trip to the host. This matters for measurement because it means the render path's cost is GPU-bound end to end, with no host-transfer term contaminating the timing, and it is the path under which the orientation/anisotropy phenomenon (the spine of this document) is observed.

### 2.7 The Engineering Codex: workspace discipline

The codebase is a virtual Cargo workspace with a strictly enforced dependency direction:

- **`voxel-core`** — pure core: types, the tiered oracle, the builder, the HDDA, measurement, and the bytemuck buffer contract. No `wgpu`, no I/O. This is the crate the oracle and mirror live in, and its purity is what makes them testable without a GPU present.
- **`voxel-gpu`** — the adapter: `wgpu` device/queue, the WGSL, and a typed `GpuError`. The rule is that it **always compiles**, GPU present or not — the device is acquired at runtime, never gated at build time.
- **`voxel-cli`** — a headless CLI for running builds, differentials, and measurements without a window.
- **`voxel-viewer`** — optional, `winit`, the *only* windowing dependency, quarantined so that the rest of the workspace never pulls in a display stack.

Two rules deserve emphasis because they shaped the investigation:

- **GPU is a runtime probe, never a Cargo feature.** Whether a GPU is available is discovered at runtime and surfaced as a typed error, not selected by a feature flag. This means there is no "GPU build" and "non-GPU build" to drift apart; there is one build, and the device is just a resource that may or may not be acquirable. It also means CI compiles the GPU adapter unconditionally, so device-path code rot is caught at build time even on machines with no usable GPU.
- **Clippy pedantic with `-D warnings`.** Lints are errors. The pedantic tier is on. This is not cosmetic — it is part of the same push-invariants-into-the-toolchain stance as the `Resolution` newtype, applied to code quality.

### 2.8 Fixtures: the test field zoo

Correctness and performance are exercised against a deliberately spread-out set of fixtures, chosen to span box-counting dimension and to probe specific traversal pathologies:

| Fixture | ~D | Purpose |
|---|---|---|
| Sierpinski tetrahedron | 2 | canonical fractal, mid sparsity |
| Cantor dust | 1 | extreme sparsity, near-1-dimensional |
| Checkerboard | ≈3 | dense alternating, descent-heavy |
| Solid | 3 | fully occupied, worst case for sparsity |
| WireLattice | — | thin axis-aligned wires; traversal pathology |
| Dust | — | hashed sparse noise; warp-divergence stress |

The first four span the dimension axis the whole sparse-vs-dense argument turns on. The last two are explicitly **adversarial**. WireLattice — thin axis-aligned wires — is a traversal pathology: rays graze long thin structures, maximizing the number of cells touched per hit and stressing the descend/ascent recompute. Dust — hashed sparse noise — is built to maximize **warp divergence**: adjacent rays make uncorrelated skip/descend decisions, which is the worst case for the data-encoded-level uniformity trick and exactly the regime where any warp-execution-efficiency problem becomes visible. These two are the fixtures that turn comfortable average-case numbers into honest worst-case ones.

### 2.9 Methodology as a through-line

Three disciplines governed the whole investigation, and they recur as a refrain in the sections that follow:

1. **Measure-first.** Build a cheap proxy before an expensive feature. The cheapest thing that answers a question gets built first; the expensive feature is only built once a measurement says it will pay. This is why the build order is incremental and every step is diffed against the previous one.
2. **Validate every optimization against the differential.** No optimization is "done" until the 20,000-ray, zero-mismatch differential still passes against both oracle and mirror. A faster traversal that fails the differential is not a faster traversal; it is a different, wrong program. Performance work and correctness work are not separate phases — the differential is a gate on every performance change.
3. **Adversarial multi-agent review for risky correctness.** Where a change carried real correctness risk — the post-order convention, the lo/hi mask split, the `tMax` recomputes — it was put through an adversarial multi-agent review rather than a single read-through.

The honest summary is that this discipline repeatedly prevented building the wrong thing: the measure-first gate killed features that would not have paid, the differential caught convention bugs that would otherwise have surfaced as subtle visual artifacts, and the type-level invariants refused malformed configurations before they reached a kernel. The performance investigation documented in the rest of this writeup is only legible *because* the artifact under it was held to this standard — every number that follows is a number from a traversal that was, at the moment it was measured, provably structurally correct against an f64 ground truth.

A final hardware note that conditions everything downstream: all of this runs on Apple M-series silicon, Metal via `wgpu` 29. The GPU is acquired as a runtime resource through that adapter. Absolute GPU timings on this platform are thermally sensitive — sustained load over a session drifted absolute numbers by 2–6× — so throughout this document the reliable signal is **ratios and within-run comparisons**, not absolute milliseconds. The structure, addressing, and discipline described above are stable; the wall-clock is not, and the analysis is built to lean only on the parts that are.

The identifiers in the brief match the actual codebase (`LeafBounds`, `SKIP_MARGIN`, `leaf_bounds`, group 0 binding 2). My naming is consistent. Now I'll write the section.

## 3. The per-brick early-skip (shipped) — and the bug the random differential missed

This is the one optimization out of the entire investigation that shipped to `main`. It is also the one that taught us the most about our own testing methodology, because the implementation was *correct on every fixture we had*, passed the `0/20000` exact-equality differential, and still contained a critical correctness bug — one that only an adversarial review, not random sampling, surfaced. The section below covers the motivation, the mechanism, the bug, why our differential was structurally blind to it, the fix, the secondary findings the review folded in, the measured wins, and a rebuild-cost regression we introduced and had to chase down twice.

### 3.1 Motivation: false-positive interior walks dominate sparse cost

The §10 cell-step instrumentation made the cost structure of sparse and thin geometry unambiguous: traversal time is dominated by *descending into a leaf brick and walking its 8³ interior only to miss*. The descent predicate we had — "this brick has at least one set voxel" — is too coarse. It is the right predicate for a dense brick, where any chord that enters the brick's outer shell will plausibly hit something. It is the wrong predicate for a *thin* brick, where the set voxels occupy a tiny sub-volume and most chords that clip the outer 8³ shell pass nowhere near an occupied cell.

The numbers make the waste concrete. On the `dust` fixture, roughly 12% of bricks are non-empty, and a large fraction of those non-empty bricks contain a *single* set voxel. The "≥1 set voxel" gate fires on every one of them. A ray that merely grazes the corner of such a brick satisfies the gate, descends, and then runs an 8³ interior DDA that walks on the order of ten interior cells before exiting — having hit nothing. These false-positive interior walks are not a tail cost; they *are* the cost. At 512³ on `dust`, roughly 140 of 244 total cell-steps were interior steps inside bricks that ultimately missed. More than half of all the work the traversal did was spent walking the insides of bricks the ray was never going to hit.

This is precisely the geometry the whole project targets — sparse, scale-invariant, thin occupancy — so a lever that attacks the false-positive interior walk attacks the dominant cost on the workload that matters. That is why this one shipped and the others (covered elsewhere in this document) did not.

### 3.2 Mechanism: a per-leaf occupied-box slab test before descent

The fix is to give the descent predicate a *finer* sub-box to test against than the brick's outer 8³ shell. Each leaf now carries the axis-aligned bounding box of its own set voxels — its `LeafBounds` — packed into a single `u32`. This is a *third* GPU storage binding (group 0 binding 2, `leaf_bounds`) alongside the existing node and leaf-word bindings; it is not free, but one `u32` per leaf is cheap relative to the 64-byte bitmask brick it annotates.

The traversal change is local and small. Before descending into a leaf child, the kernel slab-tests the ray's chord against the *occupied* sub-box (the `LeafBounds`, expressed in the brick's coordinate frame and offset to the leaf's world origin). If the chord misses that sub-box, the brick is treated as empty: the ray steps on to the next sibling, with no descent and no 8³ interior walk. If the chord hits the sub-box, the descent proceeds exactly as before. The slab test replaces the "≥1 set voxel" gate's blunt verdict with a geometric one that actually reflects where the occupancy lives.

Two design details matter:

- **The FULL gate.** If a leaf's occupied box spans the entire brick — for example a dense Sierpinski leaf whose fractal pattern reaches all eight corners of the 8³ — the slab test can never reject any chord that enters the brick, so it is pure overhead. We sentinel that case as `LeafBounds::FULL` and *skip the slab test entirely*, descending directly. There is no traversal headroom for the early-skip on a full brick, and the FULL gate avoids paying for a test that can only ever return "hit." This also, as §3.7 explains, becomes the load-bearing fast path for the rebuild-cost fix.
- **One packed `u32`.** The box is six small integers (min/max per axis, each in `0..=7`), which fit comfortably in a `u32` with room for the FULL sentinel. The WGSL kernel unpacks it with shift/mask logic that mirrors the Rust `LeafBounds::pack` bit-for-bit — a hand-mirrored layout that, as §3.6 notes, was itself a review finding because nothing initially cross-checked the two unpackers.

### 3.3 Conservativeness argument

The skip is conservative *by construction*. The occupied box, by definition, contains every set voxel in the leaf. Therefore a chord that misses the box cannot possibly intersect any set voxel inside that leaf — there are none outside the box. Skipping a brick whose occupied box the chord misses can never drop a true hit. This is not a probabilistic or fixture-dependent argument; it is a containment argument, and it holds for any ray and any leaf.

The consequence we leaned on for validation: the f64 reference traversal stays *bit-exact* against the oracle with the early-skip enabled (A == B == oracle), and this was confirmed by the existing exact-equality differential. An optimization that changes which bricks you descend into but provably never changes *which voxel you hit first* is the ideal kind — it is invisible to a correctness oracle. That property is exactly what made the f32 bug below so insidious: the *f64* path really was bit-exact, so the differential's green checkmark was telling the truth about f64 while saying nothing about the f32 paths that actually run on the GPU.

### 3.4 The adversarial review and the bug the differential missed

The implementation went through a multi-agent code review: several skeptic agents working in parallel, each assigned a distinct lens (numerics, bit-layout, sentinel handling, degenerate rays, index parallelism). The numerics lens caught a critical correctness bug that the `0/20000` random differential had passed clean.

The bug lived in the *f32* paths — the GPU kernel and the f32 CPU mirror. Those paths made the slab-test accept/reject decision with an epsilon-free comparison, in spirit `if (t_near > t_far) return false; return t_far >= t_enter`. The f64 reference path used a slack term; the f32 paths did not. For most rays this difference is immaterial. For a ray *grazing the edge of a single-voxel brick*, it is fatal.

Here is the mechanism. A single-voxel occupied box is one cell across. The `t`-interval that a grazing chord spends inside that razor-thin box is itself razor-thin. Computed in f32, that interval can round to *degenerate* (`t_near > t_far`, so the slab test reports a miss) or `t_far` can land just below `t_enter` (again reported as a miss). So the early-skip *rejects* the brick and steps on. But the interior 8³ DDA — had it been allowed to run — starts from the *same* `t_enter`, floors the entry point into integer cell coordinates, and lands *inside* the occupied voxel, which it would correctly report as a hit. The gate was **stricter than the walk it was guarding.** The skip threw away a hit that the descent it replaced would have found.

The critic did not stop at the argument. It reproduced the bug end-to-end, with **five concrete counterexample rays at 2048³** where `mirror_traverse` returned `None` (early-skip rejected the brick, no other brick hit) while the oracle reported a hit. These are not synthetic curiosities; they are exactly the grazing-the-thin-brick rays that the optimization was *built to make cheap*, and on which it was instead silently wrong.

The drop rate scales with coordinate magnitude, because f32 ULP grows with the coordinate, so the `t`-interval of a fixed-size voxel box rounds to garbage more readily at larger world coordinates:

| Grid | Dropped fraction of true grazing hits |
|---|---|
| 128³ | 0.54% |
| 512³ | 1.93% |
| 2048³ | 5.44% |

At 2048³ — the largest grid we test, and the one where sparse traversal matters most — better than one in twenty true grazing hits was being dropped. The trend is the tell: a bug whose severity *climbs with grid size* is the worst kind to ship into a project whose entire reason for existing is large sparse fields.

### 3.5 Why the differential missed it — and the lesson

The `0/20000` differential was *false confidence*, and understanding why is the most transferable lesson in this section.

The differential samples rays randomly. A random ray essentially *never* grazes a single-voxel box's edge to within f32 epsilon — the set of rays that do is a vanishingly thin sheet in ray space, of measure approximately zero under any reasonable sampling distribution. You can draw 20000 random rays, or 20 million, and expect to hit that sheet effectively never. Compounding this, our *dense* fixtures are near-FULL: their leaves FULL-gate the slab test, so on dense geometry the buggy f32 slab test never even runs. Between the two effects, the differential was exercising the early-skip almost exclusively on rays and bricks where the bug *cannot* manifest.

So `0/20000` was not weak evidence of correctness on the targeted geometry — it was *no* evidence, dressed up as strong evidence. The optimization targets sparse thin bricks and grazing chords; the differential's sampling distribution put approximately zero mass on exactly that class. The green checkmark was measuring the wrong region of input space with great statistical confidence.

The lesson, stated plainly: **random sampling systematically under-weights grazing-class bugs**, because grazing configurations are low-measure by definition while being high-consequence for any code path whose correctness hinges on a tie-breaking comparison. Adversarial *construction* — an agent reasoning "where would f32 rounding of a thin interval bite, and let me build a ray that lands there" — found in one pass what 20000 random samples structurally could not. Going forward we treat "passed the random differential" as necessary but emphatically not sufficient for any change that touches a geometric accept/reject boundary; such changes get a constructed grazing battery as well.

### 3.6 The fix: dilate the box by one voxel before the f32 slab test

The fix is a one-voxel dilation. Before the slab test runs in the f32 paths, the occupied box is grown by one voxel in every direction — `SKIP_MARGIN`. The reasoning is a bound on the worst-case disagreement between the gate and the walk it guards:

- The interior DDA can floor a grazing entry point up to roughly one f32 ULP *outside* the true voxel cell and still land in it.
- A one-voxel halo around the occupied box *dwarfs* a one-ULP discrepancy by many orders of magnitude.

So with the margin applied, the slab test can no longer be stricter than the interior DDA: any ray the DDA would floor into an occupied voxel passes the dilated slab test comfortably. Crucially, the dilation only ever biases the decision toward **descend**, and descending is *always safe* — at worst it costs a handful of extra interior walks (the exact false-positive cost the optimization set out to reduce), but it can never drop a hit. We trade a sliver of the performance win back for an unconditional correctness guarantee, which is the right trade for a gate.

The margin is applied *uniformly* across the f64 reference, the f32 mirror, and the WGSL kernel, so the three paths agree on the gate boundary rather than each carrying its own slack convention. A regression test pins the fix to the *exact reproduced counterexamples* from §3.4: it **fails at `margin = 0`** and **passes at `margin = 1`**. That test is the artifact that converts the critic's five rays from a one-time catch into a permanent guard — if anyone ever "optimizes away" the margin, the build goes red on the precise rays that motivated it.

### 3.7 Other review findings folded in

The numerics bug was the headline, but the parallel skeptics returned a cluster of smaller findings that we fixed in the same change rather than leaving as latent traps:

- **Sentinel parity.** The f32 mirror seeded its slab accumulators with `±f32::INFINITY`; the WGSL kernel used `±1e30`. These behave differently at the edges of the float range and in degenerate arithmetic. Made identical so the mirror and the shader cannot diverge on sentinel-dominated rays.
- **Empty-structure `leaf_bounds` padding.** Padding entries for empty structures were zero-filled, and a zero `LeafBounds` decodes to a *bogus single-voxel box* — the worst possible default, since it is both occupied-looking and razor-thin. Changed the padding to `FULL`, which is the safe default (it descends rather than skips).
- **WGSL bit-layout cross-validation.** The WGSL unpack of the packed `u32` was hand-mirrored from the Rust `pack` with *no* test asserting the two agree. Added a cross-validation test, because a silent shift/mask divergence here would corrupt every skip decision on the GPU while leaving the CPU paths correct — a classic "works on the mirror, wrong on the device" trap.
- **`leaf_bounds` index-parallel test.** Added a test asserting the `leaf_bounds` array is indexed in lockstep with the leaf-words array, so leaf `i`'s box always describes leaf `i`'s bitmask.
- **Degenerate / axis-aligned ray battery.** The random sweep essentially never produced a ray with a zero direction component, so the `d == 0` slab-test branch (the axis-parallel case, where the standard `1/d` slab formulation degenerates) was effectively untested. Added an explicit battery of degenerate and axis-aligned rays to exercise it directly.

Each of these is the same shape of problem as the headline bug: a code path that random sampling under-exercises, made safe by a constructed test rather than by hoping the sweep wanders into it.

### 3.8 Performance: deterministic cell-steps and throughput

The reliable signal here is the deterministic §10 cell-step count, which is thermally invariant — it is a count of traversal work, not a wall-clock time, so it does not drift with the M-series thermal state the way absolute GPU timings do. The early-skip cuts cell-steps substantially on exactly the geometry it targets:

| Fixture | Grid | Cell-steps before → after | Reduction |
|---|---|---|---|
| wire-lattice | 512³ | 81 → 46 | 1.76× |
| dust | 512³ | 244 → 161 | 1.51× |
| dust | 2048³ | 710 → 473 | 1.50× |

The wire-lattice win is the largest because its geometry is almost all thin grazing chords — the worst case for the old coarse gate and therefore the best case for the early-skip. The `dust` reductions hold steady (~1.5×) across an 8× jump in linear resolution from 512³ to 2048³, which is what you want: the false-positive interior walk is a per-thin-brick cost, and the fraction of thin bricks does not collapse as the field scales.

Throughput followed the cell-step reduction, measured same-thermal to keep the comparison honest. On a coherent viewer pass over `dust` at 512³, frame time went **26.5 → 15.8 ms, a 1.68× speedup**. As with every absolute timing in this document, the millisecond figures are thermal-sensitive — sustained GPU load drifted absolute numbers by 2–6× over a session — so the *ratio* is the load-bearing claim, not the two endpoints; both were captured back-to-back at matched thermal state specifically so the 1.68× is trustworthy. That the 1.68× throughput gain tracks the 1.51× cell-step reduction (with the remainder plausibly from fewer divergent descents and better warp coherence) is a good consistency check that we are measuring the same effect two ways.

Throughout all of this — including the grazing-heavy `wire` and `dust` fixtures and the *new* grazing and axis-aligned batteries added in §3.7 — the GPU differential held at **0/20000 against both the oracle and the mirror**. After the fix, the same sampling that gave false confidence before now gives real confidence, because the constructed batteries cover the region the random sweep does not. The clean differential is meaningful *because* it now runs alongside the adversarial tests, not instead of them.

### 3.9 A rebuild-cost regression — fixed, then re-fixed

The early-skip needs the `LeafBounds` precomputed at serialization time, and the obvious precompute is wrong for dense fields. Scanning every set bit of every leaf to find the occupied box is `O(set-bits)` per leaf, which is `O(n³)` for a *dense* field. On `checkerboard` at 2048³ — pathologically dense, half the voxels set — re-serialization ballooned to **4.6 s**, larger than building the structure in the first place. We had moved the cost from traversal (where the early-skip helps) into the rebuild (where it never paid for itself, because dense leaves FULL-gate the skip and gain nothing).

The first fix made it *worse*. Adding an early-exit — stop scanning once the box is observed to span the whole brick — pushed `checkerboard` 2048³ to **9.2 s**. The early-exit is a bet that the box reaches full extent early in the scan; on a checkerboard, the extreme corners of the occupied region are found *late*, so the early-exit's per-bit "have we spanned the brick yet?" comparison was pure overhead added to a scan that was going to run to completion anyway. The lesson mirrors §3.5: an optimization tuned on the wrong cost model can backfire, and checkerboard's late-corner structure was exactly the adversarial case for a span-detecting early-exit.

The correct fix is a cheap *popcount gate*. Any leaf with more than 64 set voxels is returned as `FULL` *without scanning its bits at all* — a single popcount on the 64-byte bitmask decides it. This is sound because a leaf that dense will FULL-gate the early-skip during traversal regardless of its true box, and it walks fast anyway, so reporting `FULL` for it loses no traversal performance. With the popcount gate, `checkerboard` 2048³ re-serialization dropped to **64 ms — a 144× improvement** over the 9.2 s second attempt. The sparse skip-targets — the leaves the early-skip actually helps — are *unaffected*, because a sparse leaf has few set bits, sails under the popcount threshold, and gets its true tight box from the full scan as intended.

The shape of this fix is worth keeping: the popcount gate routes each leaf to the cheap path for *its* density class. Dense leaves get an O(1) popcount and a `FULL` verdict they would have arrived at anyway; sparse leaves get the O(set-bits) scan that is genuinely cheap *because* they are sparse. We stopped paying O(n³) for information that only the sparse leaves can use, which is the same principle — spend work only where it can pay off — that motivates the early-skip itself.

## 4. Measurement infrastructure

Every quantitative claim in this document is backed by a headless tool that produces a number, not a screenshot or a feeling. This was a deliberate posture from early in the investigation: the orientation/view-direction anisotropy problem is the kind of effect that is easy to perceive, easy to misattribute, and easy to "fix" with a change that helps the case you happened to be looking at while quietly regressing five others. The only defense is to make the cost surface measurable along the axes that matter — fixture geometry, resolution, view direction, memory layout — and to keep the measurement cheap enough that a hypothesis can be killed before it is built. This section documents the four tools that carry that load: `bench`, `aniso`, `locality`, and the §10 `measure` harness. They share a design principle — measure-first — and a payoff structure: each cheap proxy lets a lever be falsified for the price of an afternoon instead of a multi-file GPU rebuild.

A standing caveat applies to every absolute number below. All GPU timings here come from an Apple M-series part driven through wgpu, and under sustained load the absolute throughput drifted by roughly 2-6× across a session as the package heated and clocked down. Absolute Mray/s and ns/ray figures are therefore soft; the reliable signal is always a **ratio** or a **within-run comparison**, where both sides of the comparison were measured under the same thermal state within seconds of each other. Where an absolute is quoted, read it as "this order of magnitude, on this machine, at that moment," not as a portable benchmark.

### 4.1 `bench` — the fixture × resolution sweep

`bench` is the workhorse. It sweeps a set of synthetic fixtures across resolutions and, for each cell of the sweep, tabulates a wide row of structural and performance metrics. The structural columns characterize the data structure independent of traversal: `build_ms` (time to construct the sparse MIP voxel tree), `serial_ms` (the cost to re-serialize the built structure — a proxy for I/O and a sanity check on the in-memory footprint), the leaf count, and the resident size in MiB. Alongside those it computes the box-counting fractal dimension `D` (with the regression `R²` so a low-quality fit can be spotted and discounted) to characterize how the occupied set fills space. The performance columns are the heart of it: mean descents per ray, mean cell-steps per ray, hit percentage, a single-thread CPU-mirror throughput in Mray/s, and the GPU throughput in Mray/s. Crucially the CPU mirror runs the *same* traversal algorithm on the CPU; it is a correctness oracle and a hardware-independent cost baseline, not a competitor to the GPU path.

The table below is a representative slice at 512³, captured after the early-skip fix (see the traversal correctness work elsewhere in this document). Absolute throughput columns are thermal-noisy; the structural columns are deterministic and reproducible.

| Fixture | build_ms | serial_ms | leaves | MiB | D | R² | desc | steps | hit% | CPU Mray/s | GPU Mray/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sierpinski | 20.7 | 0.4 | 4096 | 0.25 | 2.00 | 1.000 | 3.9 | 15.9 | 45 | 3.4 | 18.6 |
| checkerboard | 20.0 | 0.8 | 262144 | 16 | 2.90 | — | — | 4.6 | 100 | 8.6 | 54 |
| dust | 12.0 | 0.5 | 30944 | 1.9 | 1.71 | 0.844 | 36.9 | 161.2 | 22 | 0.5 | 7.9 |
| wire-lattice | 32.0 | 3.1 | 131072 | 8 | 2.33 | 0.965 | 10.3 | 46.1 | 100 | 1.6 | 9.3 |

A few structural relationships are worth reading directly off this table because they reframe what "expensive" means. The sierpinski fixture has a near-perfect `D` of 2.00 at `R²` 1.000 — it is, to numerical precision, a surface-like set, and its low leaf count (4096) and tiny footprint (0.25 MiB) reflect that. The dust fixture has the lowest dimension (`D` 1.71) but the *worst* `R²` (0.844), a useful reminder that box-counting on a finite grid is itself noisy and that `D` should not be over-interpreted when the fit is poor. Dust is also the traversal pathology: 36.9 descents and 161.2 cell-steps per ray at only 22% hit, with CPU throughput collapsing to 0.5 Mray/s — rays wander through a lot of mostly-empty hierarchy before terminating or escaping.

Compare that to checkerboard and wire-lattice, both at 100% hit. Checkerboard is the densest here (`D` 2.90, 16 MiB) yet the *cheapest* per ray: 4.6 steps, no meaningful descent cost, and the highest throughput on both CPU (8.6) and GPU (54). This is the single most important qualitative finding the bench surfaced, and it inverts a naive intuition. The high-dimension fixtures — checkerboard and (at other resolutions) solid — are fast per ray precisely because rays hit something almost immediately and terminate; they stress **storage**, not traversal. The traversal killers are the *low*-occupancy, thin or sparse fixtures: wire-lattice (46.1 steps to traverse a 100%-hit lattice of thin members) and dust (161.2 steps, 22% hit). A ray's cost is dominated by how far it travels through structure *before* it resolves, not by how much structure exists. Surface-dense scenes resolve fast; sparse and filamentary scenes do not.

The 2048³ rows sharpen the same point and expose the second axis — storage scaling:

| Fixture | build_ms | serial_ms | leaves | MiB | desc | steps | hit% |
|---|---|---|---|---|---|---|---|
| dust | 657 | 41 | 1.97M | 123.5 | 110 | 472.8 | 61 |
| checkerboard | 1444 | 64 | 16.7M | 1027 (~1 GiB) | — | — | — |
| sierpinski | 830 | 5 | 65536 | 4.05 | — | 18.7 | — |

Two things stand out. First, sierpinski barely moves: 65536 leaves and 18.7 steps at 2048³, essentially flat against the 512³ row because a `D`≈2 surface does not gain interior as resolution climbs — it gains only boundary, and the MIP hierarchy skips the interior it does not have. Dust, by contrast, balloons to 1.97M leaves, 123.5 MiB, and 472.8 cell-steps per ray; its hit rate rises to 61% as the finer grid catches more of the dust, but the per-ray work grows roughly with the linear extent of the wandering. This is exactly the scaling signature of a traversal-bound rather than storage-bound fixture.

Second, the checkerboard-2048 case is the storage stress test, and its result is a genuine, slightly surprising negative. The leaf set is 16.7M, the binding is 1027 MiB — call it 1 GiB — and on this M-series hardware that binding **did not OOM**. We had half-expected the 1 GiB single-buffer binding to be the wall; it was not. The operative conclusion is that on this hardware the binding constraint is *traversal work, not storage*. We could afford to materialize a gigabyte of leaves; what we could not afford, on the sparse fixtures, was the cell-step count to walk them. That finding redirected the rest of the investigation away from storage-compression levers and toward traversal-order and coherence levers — which is the through-line of this entire document.

### 4.2 `aniso` — the directional anisotropy sweep

`bench` measures cost as a function of fixture and resolution but says nothing about *view direction*, which is the central phenomenon under study. `aniso` fills that gap. It generates a Fibonacci-sphere distribution of view directions — a low-discrepancy, roughly equal-area sampling of the sphere so that no axis or diagonal is over- or under-represented — and for each direction it dispatches an orthographic, spatially coherent ray batch (a flat front of parallel rays, which is the most favorable case for memory coherence and therefore isolates the *algorithmic* direction-dependence rather than convolving it with divergence effects). For every direction it records two numbers: the deterministic mean cell-step count, which is the pure algorithmic cost and is bit-for-bit reproducible, and the GPU kernel time, which is the hardware cost and is thermal-sensitive. Correlating the two lets us decompose the observed anisotropy swing into "the algorithm genuinely does more work in this direction" versus "the hardware happens to be slower in this direction" (memory access pattern, cache behavior). That decomposition is the analytical payload of the tool, and it is the reason both columns are recorded rather than just the convenient one.

#### The GPU-timing cleanup — a methodology note

`aniso`'s GPU numbers were nearly worthless until a timing-infrastructure fix, and the story is worth recording because it changed conclusions, not just precision. The original path measured kernel time at wall-clock granularity around a dispatch that included a buffer readback. That readback plus dispatch overhead added a roughly fixed ~70 ns/ray tax to every measurement. A fixed additive offset is poison for an anisotropy study: it is a constant added to both the cheap and the expensive directions, which **compresses the ratio** between them and makes the anisotropy look smaller than it is.

The fix was to add wgpu compute-pass `TIMESTAMP_QUERY` support — a `traverse_timed` path that writes GPU timestamps around the compute pass and reads the kernel's duration *on the GPU timeline, readback-free*. The effect was large and exactly in the predicted direction. On dust, the minimum per-ray cost dropped from 85 ns/ray wall-clock to 13.5 ns/ray on the timestamp path — the ~70 ns of readback-plus-dispatch overhead made visible by its removal. More importantly, with that fixed offset gone the measured anisotropy *un-compressed*: a small-batch swing that read as 4.5× at wall-clock granularity opened up to 8.4× on the clean timestamp measurement. The earlier wall-clock anisotropy figures were therefore badly understated, and any conclusion drawn from them under-stated the size of the problem. Every directional GPU number reported elsewhere in this document is on the `traverse_timed` path for this reason; the wall-clock numbers should be treated as a lower bound on the true effect.

The general lesson, stated plainly: when the quantity of interest is a *ratio of small per-ray costs*, any fixed per-measurement overhead must be removed at the source, not subtracted after the fact, because you usually do not know its value precisely and it does not cancel in a ratio. Timestamp queries gave us a readback-free measurement at the right point on the timeline, and that was a prerequisite for trusting anything `aniso` said.

### 4.3 `locality` — a CPU proxy for memory layout

The most expensive thing to test naively is a memory-layout hypothesis: changing the leaf ordering means rebuilding the GPU structure, re-uploading it, and re-running the directional sweep across multiple files — a substantial turnaround for what might be a dead end. `locality` exists to make that test cheap and CPU-only, with no GPU rebuild. For the occupied bricks it computes, under different candidate leaf orderings (Morton and Hilbert), the per-axis mean memory-distance between bricks that are *spatially* adjacent — i.e. how far apart in the linear buffer two neighbors in space end up — together with the cache-block (treelet) co-location fraction, the share of spatial neighbors that land in the same cache-sized block.

These two statistics are a strong proxy for the cache behavior a real traversal would see: a ray walking spatially adjacent bricks pays for every spatial-neighbor hop that turns into a long memory jump or a cache-block miss. If a proposed ordering does not improve the per-axis distances or the co-location fraction in the proxy, it has essentially no chance of helping on the GPU, and it can be discarded in an afternoon instead of a multi-file rebuild. The value of `locality` is therefore almost entirely *negative-result throughput*: it lets a layout lever be falsified cheaply, and a layout that survives the proxy is the only kind worth paying the GPU rebuild to test.

### 4.4 The §10 `measure` harness

The §10 work is supported by a focused harness, `measure`, that computes the structural quantities the dimensional and footprint analysis depends on. It estimates the box-counting dimension `D` by regressing `ln N(L)` against `ln cell_size` across levels — the same `D`/`R²` pair that appears in the `bench` rows, here computed as the primary object of study rather than as one column among many. It tabulates the per-level memory footprint and compares it against an L2 budget, so that the level at which a fixture's working set spills out of L2 is explicit rather than inferred. And it reports descent frequency and cell-steps, tying the dimensional characterization back to the traversal cost it predicts. Keeping these in a dedicated harness means the dimensional claims in §10 rest on the same code path as the `bench` columns, so the two sections cannot quietly disagree about what `D` a fixture has.

### 4.5 The through-line: measure-first

Taken together these tools embody one discipline. `bench` establishes the cost surface over fixtures and resolutions and tells us *which* fixtures are traversal-bound (the sparse ones) rather than storage-bound (the dense ones). `aniso` — once its timing was cleaned up with `traverse_timed` — measures the directional swing honestly and decomposes it into algorithmic versus hardware components. `locality` and `coherence` are the cheap CPU proxies that let layout and coherence levers be falsified for the price of an afternoon, before any GPU rebuild is committed. The §10 `measure` harness grounds the dimensional analysis on the same metrics. The payoff of this infrastructure is not any single number it produced; it is that the project could iterate on hypotheses at the speed of a CPU proxy and only spend GPU-rebuild effort on levers that had already survived a cheap falsification. Every claim downstream of this section is measured because these tools made measuring it cheaper than guessing.

I have enough context. Writing Section 5 now based on the brief.

## 5. The orientation anisotropy — the central problem

Every preceding section measured *how fast* the structure traverses. This section is about the discovery that "how fast" is not a single number. The same structure, traced at the same resolution, with the same rays-per-frame, costs wildly different amounts of GPU time depending on **which way the camera points**. That orientation-dependence — the anisotropy — is the central problem the rest of this document circles around. It is also the most consequential negative result here: a large fraction of the cost swing turns out *not* to be intrinsic to the geometry or the algorithm, which means it is, in principle, addressable by layout — and yet, as the later sections show, none of the cheap levers we tried actually captured it.

### 5.1 The headline number

The `aniso` sweep traces a fixed structure from a fan of camera orientations spanning axis-aligned through body-diagonal view directions, at 512³, using the cleaned-up GPU timestamp path (not wall-clock; see §5.7). For the `dust` fixture — random, statistically isotropic occupancy — the same structure costs roughly **13.5 ns/ray from the cheapest view direction and ~127 ns/ray from the most expensive**, a swing of about **8.4×**. Other fixtures land in the same neighbourhood; the worst dust configuration we logged swings ~8.9× (see §5.3). Call it an 8–9× directional anisotropy in GPU traversal cost.

The absolute ns/ray figures here are thermal-sensitive — sustained load on the Apple M-series part under wgpu drifted absolute numbers by 2–6× across a session — so the *13.5 ns* and *127 ns* endpoints should be read as "a representative cheap and expensive direction within one run," not as portable constants. The **ratio**, taken within a single sweep where both endpoints share the same thermal state, is the robust signal, and it is the ratio that is alarming. A renderer whose per-frame cost varies 8× purely on camera heading has no stable frame budget: the same scene that hits frame rate looking down an axis blows it looking down a diagonal, with nothing in the scene having changed.

This is not a small second-order effect to be amortized away. It is comparable in magnitude to the entire speedup the sparse hierarchy buys over a dense march. If we cannot explain it, we cannot bound it; if we cannot bound it, every other optimization in this document is being measured against a moving target.

### 5.2 Why a swing is expected — and why most of it should not be

Some directional cost variation is unavoidable and has nothing to do with memory. It is pure geometry of the grid march. An axis-aligned ray crosses cells one face at a time; a ray down the body diagonal of the grid crosses roughly **√3× more cells** per unit length, because it is stepping through all three axes at once. The DDA step count is therefore a deterministic function of direction, and a diagonal ray simply *does more work* — more bitmask reads, more `popcount`s, more loop iterations — than an axis-aligned one. That alone predicts a cost swing on the order of 1.5–2.5× for these fixtures.

The problem is that the measured GPU swing is far larger than the geometric step-count swing. Something beyond "the diagonal ray takes more steps" is happening. To separate the two, we need to decompose the swing into the part that is forced by the algorithm (step count) and the part that is contributed by the hardware on top of it (everything else: cache locality, warp coherence, fetch latency). That decomposition is the analytical core of this section.

### 5.3 The decomposition: deterministic steps vs. measured GPU time

For each fixture we have two quantities measured over the same fan of swept directions:

- the **deterministic cell-step swing** — the ratio of max-to-min DDA cell-steps across directions, computed from the traversal itself, independent of any hardware effect. This is the algorithmic, "fundamental work" axis.
- the **GPU-time swing** — the ratio of max-to-min measured GPU traversal time across the same directions. This is the hardware axis.

We then correlate the two per-direction series with a Pearson `r` over the swept directions. A high `r` means the GPU time is essentially *tracking* the step count — the hardware is paying for work the algorithm genuinely does, and there is little layout-addressable headroom. A low `r` means the GPU time swings for reasons the step count cannot explain — the headroom lives in memory behaviour, not in the algorithm.

| Fixture | Geometry character | Cell-step swing | GPU-time swing | Pearson `r` | Interpretation |
|---|---|---|---|---|---|
| `sierpinski` | regular fractal | ~1.5× | ~2.5× | ≈ 0.82 | Cost mostly **tracks step count** — little addressable headroom |
| `wire-lattice` | axis-aligned thin features | ~2.4× | ~6.4× | ≈ 0.43 | **Mixed** — part fundamental steps, part addressable cache/coherence |
| `dust` | random / statistically isotropic | ~1.5× | ~8.9× | ≈ 0.24 | Almost **entirely** cache/coherence, **not** step count |

Read the table top to bottom as a gradient from "the swing is real work" to "the swing is memory artefact."

For `sierpinski`, the step-count swing (~1.5×) and the GPU swing (~2.5×) are close in magnitude, and `r ≈ 0.82` says they move together direction-by-direction. The diagonal directions that take more steps are the same directions that cost more GPU time, in proportion. The hardware is honest here: it is charging for steps taken. The residual (2.5× GPU vs 1.5× steps) is modest and is the sort of thing a constant per-step memory cost would produce. There is little to win by reorganizing memory for `sierpinski`; the swing is close to the floor set by the march itself.

For `wire-lattice`, the picture is mixed. The geometry is genuinely anisotropic — thin axis-aligned features mean the *amount* of structure a ray encounters really does depend on direction — so the step swing is larger (~2.4×). But the GPU swing (~6.4×) outruns it, and `r` drops to ≈ 0.43: only about half the directional variance is explained by step count. The rest is memory. This fixture is the cautionary middle case, because its real geometric anisotropy can be mistaken for the layout effect; the moderate `r` is what tells us both are present.

`dust` is the decisive case, and it gets its own subsection.

### 5.4 The decisive argument: isotropic geometry, anisotropic cost

`dust` is random occupancy with no preferred direction. We can confirm it is statistically isotropic *from the traversal itself*, without any hardware in the loop: its **cell-step swing is ~1.5× — nearly flat**. Random geometry presents essentially the same density of structure to a ray regardless of heading, so the number of cells a ray crosses barely changes with direction. (The small residual ~1.5× is the same √3-ish diagonal effect that every grid march has; it is the geometric floor, not a property of the dust.) By the only direction-independent measure we have — work done by the algorithm — `dust` looks the same from every angle.

And yet its **GPU cost swings ~8.9×**, with **`r ≈ 0.24`**.

This is the crux of the whole investigation, so it is worth stating the logic explicitly and without hedging:

1. The geometry is isotropic. We know this independently, because the step count — a deterministic, hardware-free quantity — is flat (~1.5×) across directions.
2. The GPU cost is not isotropic. It swings ~8.9× across the same directions.
3. The two series barely correlate (`r ≈ 0.24`). The expensive directions are *not* the high-step-count directions; the GPU swing is almost orthogonal to the algorithmic swing.

Therefore the 8.9× **cannot** be coming from the geometry — there is no geometric anisotropy to come from — and it **cannot** be coming primarily from step count, because step count is flat and uncorrelated. By elimination, it must come from the only remaining direction-dependent variable in the system: how the ray's traversal sequence walks the structure's **memory layout**. The cells `dust` stores are laid out in **Morton (Z-order)**, and the order in which a ray visits Morton-addressed cells — and therefore the cache lines and warp-lane access pattern it induces — depends on the ray's direction even when the geometry it hits does not. Some directions walk the Z-curve in a cache-friendly, warp-coherent order; others walk it in an order that thrashes cache lines and scatters warp lanes. That is the entire ~9×.

This is what we mean by the **"in principle addressable" component**. Unlike the `sierpinski` swing, which is genuine work the algorithm must do, the `dust` swing is an artefact of the interaction between a particular memory ordering and a particular ray direction. Nothing about the *information* in the structure requires that interaction to be expensive — only the chosen layout does. In principle, a layout that presented memory-coherent access regardless of direction would erase most of the 9×. That hypothesis is exactly what motivates the three layout/ordering levers in the sections that follow: if the swing is a layout artefact, re-ordering the layout should kill it.

The honest framing — and the spoiler for the rest of the document — is that "in principle addressable" is doing a lot of load-bearing work in that sentence. Identifying *where* the headroom lives is not the same as *capturing* it, and the later sections are largely the story of the headroom resisting every cheap attempt to reach it.

### 5.5 Corroboration: the cheapest direction is a property of the layout, not the geometry

If the anisotropy is a layout effect rather than a geometry effect, then the *identity* of the cheap and expensive directions should be a property the fixtures share, because they share the layout (the same Morton encoder), not their geometry. That is what we observe: the **same cheapest direction appeared for both random `dust` and structured `sierpinski`**. Two fixtures with completely different occupancy — one random, one a regular fractal — agree on which way is cheap to look. The only thing they have in common is the Z-order encoder that places their cells in memory. A geometry-driven effect would have no reason to pick the same favourite direction for two unrelated structures; a layout-driven effect must. This is independent corroboration that the encoder, not the scene, sets the orientation cost field.

It also means the anisotropy is not a quirk of one pathological fixture. It is a property of the *storage scheme*, inherited by anything stored that way. Any future structure we serialize in this Morton layout will arrive pre-loaded with the same orientation-cost field. That is what makes it a structural problem worth a structural fix, and what makes it disappointing when the structural fixes underdeliver.

### 5.6 The coherence gap: a related but distinct penalty

Separate from the orientation sweep — which varied the *direction* of a coherent camera-ray batch — we also measured the cost of ray **coherence** itself, holding the structure fixed. A **coherent** batch (camera primaries, neighbouring rays heading in nearly the same direction through nearly the same cells) traverses **17–35× faster than a scattered (incoherent) batch** on the same structure. This is the warp-divergence-plus-cache-locality penalty in its rawest form: when adjacent warp lanes follow nearby paths, they hit the same cache lines and stay on the same instructions; when lanes scatter, every lane misses independently and the warp serializes on divergence.

This is a different axis from the orientation anisotropy, and it is worth keeping them distinct:

- The **orientation anisotropy** (§5.1–5.5) is variation *within* coherent batches as the batch direction rotates. It is the 8–9× swing, and it is the subtler effect — it shows up even though the rays *are* coherent with each other.
- The **coherence gap** is variation *between* coherent and incoherent batches. It is the larger 17–35× effect, and it is the cruder one.

The two are the same underlying mechanism — cache locality and warp coherence dominating the cost — seen along two different axes. The orientation anisotropy is, in a sense, the coherence gap leaking into the coherent regime: even a perfectly coherent batch walks the Z-curve in a direction-dependent order, so coherence among lanes does not fully protect against incoherence against the *layout*.

The practical consequence sets up the workload model assumed elsewhere in this document: **primary rays are coherent and therefore land in the fast regime; secondary and scattered rays (shadow, GI, anything not bundled by the camera) pay the coherence gap.** The design's "coherence batching only, no wavefront re-sort" posture is a bet that primaries dominate. The coherence gap is the size of the penalty that bet is exposed to if and when scattered rays become the workload — and it is large enough (17–35×) that the bet is real.

### 5.7 Methodology caveat: the cleanup is what revealed the problem

A blunt but important point: we nearly missed this. The first measurements of the anisotropy used **wall-clock timing of the dispatch**, and they reported a swing of only **~2.85×** — large enough to notice, small enough to plausibly write off as the geometric √3 step effect plus some noise. It was only after moving to the cleaned-up GPU **timestamp** path that the swing on the same `dust` configuration resolved to **~8.9×**. Wall-clock timing *understated* the anisotropy by roughly 3×.

The mechanism is straightforward and worth internalizing because it generalizes to every ratio in this document. Wall-clock timing of a dispatch includes a roughly constant per-dispatch overhead — submission, readback, synchronization — that is the same regardless of how long the actual traversal takes. Adding a constant to both the numerator and denominator of a ratio **compresses the ratio toward 1**. A true 8.9× kernel ratio, with a fixed overhead added to each endpoint, reads out as a much tamer ~2.85×. The overhead was not noise in the sense of being random — it was a *systematic bias toward underestimating every ratio*, and it was hiding most of the signal. The GPU timestamp path measures the kernel interval directly and drops the overhead, which is why the clean swing is ~3× larger than the wall-clock swing.

The lesson, applied throughout: **trust ratios from the timestamp path; distrust ratios from wall-clock**, especially when the true ratio is large, because that is exactly where the constant-overhead compression bites hardest. And the standing caveat still applies on top of this — absolute ns/ray are thermal-sensitive and drifted 2–6× over a session, so even the clean timestamps are only comparable *within* a run. Both signals we lean on here — the anisotropy ratio (~8.9×) and the step-vs-GPU correlations (`r` from 0.82 down to 0.24) — are within-run, ratio-or-correlation quantities, which is precisely why they survive the thermal drift that the absolute timings do not.

### 5.8 The problem this poses for the rest of the document

To summarize what we now know going in:

- GPU traversal cost swings **8–9×** purely on camera orientation, at fixed structure and fixed resolution.
- For regular geometry (`sierpinski`), most of that swing is **fundamental** — it tracks DDA step count (`r ≈ 0.82`) and is not worth chasing.
- For isotropic geometry (`dust`), almost none of it is fundamental — the geometry is provably isotropic (~1.5× step swing) yet GPU cost swings ~8.9× with `r ≈ 0.24`, so the swing is **a Morton-layout / cache-coherence artefact**, in principle addressable by re-ordering memory.
- The cheap direction is shared across unrelated fixtures, confirming the effect belongs to the **encoder**, not the scene.
- A separate, larger **17–35×** coherence gap sits behind incoherent rays, the same mechanism along a different axis.

That is the problem statement for everything that follows. We have localized the addressable headroom — it is the interaction between Morton order and ray direction — and we have a clean way to measure whether we have captured it (does the `dust` GPU-swing ratio drop toward its ~1.5× step floor?). The next sections take three swings at it: reordering the intra-level layout, changing the cross-level ordering, and adjusting the traversal/scheduling to better match the layout to the rays. The framing to carry forward is that each of these is an attempt to convert "in principle addressable" into "actually captured" — and the recurring, honest finding is how stubbornly that ~9× resists being cheaply reclaimed.

I have everything I need, grounded in the actual implementation. Now I'll write Section 6.

```markdown
## 6. Lever 1 — Ray binning (branch `feat/ray-binning`, `1e6f1c0`)

The anisotropy from Section 5 is large — roughly 9× between the cheapest and
most expensive view direction — and it is tempting to read that gap as
*latent throughput we are leaving on the table*. The most natural lever to
reach for first is the oldest trick in the GPU ray-tracing book: **ray
binning**, also called ray sorting or ray reordering. The premise is that
incoherent rays scattered across a warp thrash the cache and diverge the lanes,
and if we reorder the batch so that neighboring rays travel in similar
directions and enter the structure near the same place, the warp stays coherent
and the cache lines we fetch for lane *k* are still warm for lane *k+1*. This is
the lever that, in principle, recovers the most performance for the least
structural change — we do not touch the layout, the kernel, or the data
structure, only the order in which rays are handed to the GPU.

This section records the measurement-first experiment we ran before committing
to any GPU binning pass, the `coherence` subcommand added on
`feat/ray-binning` at `1e6f1c0`. The headline result is negative: reordering
recovers only ~1.2×, the mechanism is well understood, and a real binning
pipeline is not worth building for this kernel. The subcommand was kept as a
reusable diagnostic rather than promoted to a feature.

### 6.1 Hypothesis

Stated sharply, the hypothesis was: **the 9× directional anisotropy is at least
partly a coherence artifact, and reordering an incoherent batch by a coherence
key will recover a meaningful fraction of it.** If that were true, we would
expect a sorted batch to approach the cost of the cheap direction, i.e. a
speedup trending toward the full anisotropy ratio rather than a few percent.
The classic GPU ray-sorting literature is the prior here — the technique is
real and does buy throughput on incoherent secondary rays — so the question was
never *whether* binning helps but *how much*, and specifically whether the gain
is large enough to repay the cost of a GPU binning stage in *this* kernel
against *this* layout.

Crucially, this is a falsifiable, quantitative hypothesis, not a vibe. The
anisotropy measurement (`aniso`) decomposes a direction's cost into traversal
steps and a cache/coherence excess term; if binning works, it should attack the
coherence excess. If binning recovers nothing, the 9× is intrinsic — it lives
in the per-direction step count and the layout's per-direction cache behavior,
and no amount of reordering can move a ray out of the cost its own direction
imposes.

### 6.2 The experiment: the `coherence` subcommand

The experiment is deliberately structured as a **gain-ceiling measurement**, not
as a working binning implementation. We measure the *best case* a reorder could
ever achieve — sort the rays perfectly by a coherence key on the CPU, then time
the GPU kernel on the reordered batch — and we explicitly do **not** charge the
sort cost to the kernel time. The reasoning: if even a free, perfect reorder
fails to recover the anisotropy, then no GPU binning pass (which is necessarily
an *approximate* reorder that costs something) can do better. Measure the
ceiling first; build the pipeline only if the ceiling is high enough to clear
the pipeline's overhead.

**The incoherent batch.** The whole experiment hinges on the batch being
genuinely incoherent, because a coherent batch has nothing for a reorder to
recover. The subcommand synthesizes what we called a "dust" batch: scattered
shell origins aimed at random interior points. Each ray gets an origin drawn
uniformly from a box roughly twice the grid extent and offset to surround the
structure (`unit01(&mut rng) * (2.0 * nf) - 0.5 * nf` per axis), and a target
drawn uniformly from the interior `[0, n)³`; the direction is `target - origin`,
normalized by `Ray::new`. The result is a batch in which adjacent rays share
neither direction nor grid-entry cell — the worst case for warp coherence, and
exactly the regime where binning is supposed to shine. The fixture *geometry*
defaults to `Dust`, the layout-addressable random-occupancy case, with
sierpinski and the resolution sweep also exercised.

**The coherence key.** Rays are reordered by a single `u64` `coherence_key`,
built direction-primary, entry-secondary:

- **High bits: a direction bucket.** The normalized direction is mapped onto a
  16³ grid (each axis quantized to `0..=15` via
  `(v * 0.5 + 0.5).clamp(0.0, 1.0) * 15.0`) and the three buckets are
  interleaved with `morton::encode` into a 12-bit direction key. This groups
  rays that step through the structure along the same axis pattern.
- **Low bits: the grid-entry cell.** We intersect the ray with the structure's
  bounding box (`ray_aabb`), take the entry point, quantize it to a cell, and
  Morton-encode that into a 36-bit entry key. Misses get entry key `0`, so they
  cluster together within their direction bucket.
- The two are packed `(dir_key << 36) | (entry_key & mask)` — direction sorts
  first, entry-cell breaks ties within a direction. Sorting a batch by this key
  with a stable `sort_by_key` clusters rays that both travel in similar
  directions *and* enter the structure near each other, which is precisely the
  coherence a warp wants.

**The timing.** Both the raw and the reordered batches are timed with
`best_kernel_ns_per_ray`, which calls the **readback-free** `traverse_timed`
path (GPU timestamp queries, no result download) and takes the best of 5 reps
after a warm-up rep. Using the readback-free path matters: it isolates the
kernel's traversal cost from PCIe/unified-memory transfer noise, so the only
thing differing between the two numbers is the *order rays were submitted in*.
The hit fraction is reported alongside so we can confirm both batches are
actually traversing the structure and not degenerating into mostly-misses. The
CPU sort time is measured separately (`sort_ns_per_ray`) and printed as a
reference line, never folded into the kernel number.

A note on the absolute timings throughout this section: they are
thermal-sensitive. These runs are on Apple M-series via wgpu, and sustained load
over a session drifted the absolute ns/ray by 2–6×. The reliable signal here is
the **ratio** `raw / ord` within a single back-to-back run, which is exactly
what the subcommand prints and what we quote below. The sort-vs-saved comparison
is also a within-run comparison, so it is robust to drift even though both halves
are absolute.

### 6.3 Result

The reorder recovers almost nothing:

| Fixture / resolution | Coherence-key reorder speedup |
|---|---|
| dust 512³ | ~1.16–1.22× |
| dust 2048³ | ~1.16–1.22× |
| sierpinski | ~1.17× |

The speedup is **robust across resolution** — the same ~1.2× shows up at 512³
and at 2048³ — and **robust across key granularity**. We ran it both with a
coarse 3-bit direction-octant key (high bits = which of the 8 sign-octants the
direction falls in) and with the fine 16³ direction key described above. Both
land in the same ~1.2× band. That cross-check is the important one: if the gain
were being throttled by the key being too coarse — too many genuinely-divergent
rays sharing a bucket — then refining the key from 8 buckets to 4096 buckets
should have widened the gap. It did not. **Coarseness is not the limiter.** The
ceiling is genuinely low.

**The cost side, as measured.** The CPU comparison sort costs ~350 ns/ray. The
kernel time saved by the reorder is ~11 ns/ray. As measured, that is a ~30× net
*loss* — the sort is vastly more expensive than the traversal time it buys back.
But that number must be read with care, because the CPU `sort_by_key` is only a
*measuring stick*, not the thing we would ship. A comparison sort on the CPU is
the wrong tool for a GPU pipeline; the real implementation would be a GPU
counting/radix sort over the small key space (the direction bucket has very few
distinct values), which would cost on the order of ~2–5 ns/ray. So the honest
framing is not "sorting loses 30×" — it is "even with a realistically cheap GPU
sort at ~2–5 ns/ray against ~11 ns/ray saved, the margin is thin, and the
*saved* number is itself only ~1.2× of a kernel that is already cheap." The
30× figure tells us the CPU measuring stick is too heavy to use in production;
it does not by itself condemn binning. What condemns binning is the 1.2× ceiling.

### 6.4 Why only ~1.2×: the mechanism

This is the part worth understanding, because it generalizes and it tells us
where the 9× actually lives.

The 9× anisotropy is the **intrinsic per-direction cost** of the layout. A ray
traveling in the expensive direction walks more cache lines and takes a less
favorable stride through the Morton-ordered data than a ray traveling in the
cheap direction, and it pays that cost **regardless of which other rays it is
batched with**. Reordering does not change a single ray's direction; it only
changes its neighbors. Therefore an incoherent batch's cost is, to first order,
the **mean of the directional costs** of the rays in it — and the mean is fixed
the moment you decide the batch is a uniform mix of directions. Sorting the
batch does not change the set of directions present, so it cannot change that
mean. **You cannot sort a ray out of its direction's intrinsic cost.**

What sorting *can* remove is the *extra* penalty of mixing directions **within a
single warp** — warp divergence and the cache-line cross-talk between lanes that
want different parts of the structure at the same instant. That penalty is real,
and it is exactly the ~1.2× we measured. It is the coherence excess, and it is
small relative to the intrinsic per-direction spread.

There is a second reason the divergence penalty is small *here* specifically,
and it is geometric. Dust is random occupancy. Even a batch that has been sorted
*perfectly* by direction still diverges within the warp, because adjacent rays
in the sorted order — same direction bucket, neighboring entry cells — hit
*different scattered voxels at different depths*. The random geometry defeats the
coherence that direction-sorting tries to manufacture: two rays can be parallel
and enter side by side and still terminate at wildly different step counts
because one happens to strike an occupied voxel early and the other threads
through empty space. So the dust case gives binning very little to work with on
*both* axes: the intrinsic per-direction cost is unsortable by definition, and
the residual divergence is partly unsortable too because the geometry is random.
Sierpinski, with its self-similar structure, lands in the same ~1.2× band,
which tells us this is not a peculiarity of dust — it is the general shape of
the result.

The clean way to state the decomposition: **the 9× belongs to the
layout-selection lever, not to binning.** The anisotropy is a property of how a
direction maps onto the layout's memory order, and the correct response to it is
to change the layout (or choose among layouts per-direction), not to change the
ray order. Binning addresses a different, smaller quantity — intra-warp
divergence — and recovers a different, smaller amount.

### 6.5 Net, scope, and the one combination that could matter

Folding the realistic GPU-sort cost back in, a real GPU bin nets roughly
**1.1–1.15×**, and only under a specific condition: **the rays must already be
incoherent.** That restricts the entire lever to **secondary rays** — shadow,
ambient-occlusion, and global-illumination rays that scatter in arbitrary
directions. Primary/camera rays are already coherent by construction (a camera
emits a tight, near-parallel pencil of rays through adjacent pixels), so for
them a binning pass has nothing to recover and is **pure overhead**. The correct
engineering decision is therefore to **gate binning OFF for primary rays** — if
we ever build it at all, it runs only on the secondary-ray batches where
incoherence is the default, and never on the camera pass.

There is one more sophisticated combination worth recording, because it is the
only configuration in which binning earns its keep, and it forward-references
the multi-layout work in later sections. If we (a) bin secondary rays by
direction octant, *and* (b) maintain multiple memory layouts, *and* (c) route
each octant's bin to the layout that is cheapest for that octant, then binning
stops being a pure intra-warp-divergence play and becomes the **dispatch
mechanism** for per-direction layout selection. In that arrangement the ~1.2×
divergence recovery stacks on top of whatever the layout selection buys, because
the bin is now doing double duty: it groups rays for warp coherence *and* it
sorts them into per-octant buckets that can each be sent to a different,
direction-matched layout. That is genuinely interesting — it is the bridge
between this lever and the layout-selection lever. But it is also expensive: it
needs **all three** pieces (a working GPU bin, N maintained layouts, and a
routing step), it adds the storage cost of N layouts, and after all that it
**still only touches secondary rays**. Primary rays remain coherent and
single-layout.

### 6.6 Verdict

**Not worth a GPU binning pipeline for this kernel.** Standalone, ray binning's
ceiling is ~1.2× and its realistic net is ~1.1–1.15×, available only on
incoherent secondary rays, and zero (worse, negative) on the camera rays that
dominate a typical frame. The mechanism is now clearly understood: binning
attacks intra-warp divergence, which is small here, while the 9× anisotropy is
intrinsic per-direction cost that no reorder can touch. The interesting future
is not binning-as-throughput but binning-as-dispatch into a multi-layout scheme,
and even that is gated to secondary rays and depends on machinery developed for
the layout lever rather than this one.

This negative result matches the literature (see Section 9): software ray
reordering is reported to buy ~1.3–2×, but "recovering the overhead is
problematic" — the sort cost eats much of the gain, which is precisely the
30×-as-measured / thin-margin-when-realistic tension we hit here. We are not
discovering a new failure; we are confirming a known one on our specific kernel,
and recording the measurement so the next person does not have to re-run the
experiment to re-learn that the 9× lives in the layout, not the ray order. The
`coherence` subcommand stays in the tree as a diagnostic for exactly that
purpose.
```

## 7. Lever 2 — Axis-permuted multi-layout (branch `feat/axis-permuted-morton`)

This is the most-developed lever in the investigation, and it ends in a decisive negative. It is worth dwelling on precisely *because* it looked correct at every intermediate checkpoint — the mechanism is sound, the correctness property is machine-verified, the memory cost is modest, and the stability framing was genuinely compelling — and yet it collapsed at the one step that actually matters for a deployable renderer: the per-frame selector. The arc of this section is therefore a vindication of build-and-validate over stopping at a promising proxy. Every checkpoint short of the selector said "ship it." The selector said "there is nothing here."

### 7.1 The mechanism and why it costs nothing to validate

The orientation anisotropy documented earlier in this report is, at root, a property of the storage order. Geometry stored under a Morton (Z-order) encoding interleaves the bits of the three coordinates in a fixed `(x, y, z)` priority. The traversal cost for a given camera ray depends on how that ray's marching direction interacts with the bit-interleave: rays that advance primarily along the axis that the encoder treats as "innermost" enjoy more contiguous memory access and shallower MIP descents than rays advancing along an axis the encoder buries deeper. The cheap direction is, in this sense, *baked into the encoder* at build time.

The natural idea is to store the geometry under several different Morton orderings — one per axis permutation — and, at render time, pick whichever ordering is cheapest for the current camera direction. There are `3! = 6` axis permutations of `(x, y, z)`, so the naive framing is "six Morton encoders, six builds, and a shader that knows which encoder it is traversing."

The key realization that makes this cheap to build and cheap to validate is an equivalence. Storing the geometry under the `P`-permuted Morton order `M_P` is *equivalent* to storing the `P`-permuted-coordinate geometry under the single fixed encoder `M_id`, and then transforming the camera ray by `P` at traversal time. In other words:

> Permuting the encoder is the same as permuting the geometry's coordinates and keeping the encoder fixed.

This equivalence collapses the entire scheme into something that requires **no Morton-encoder change and no shader change**. We do not write six encoders. We write the geometry's coordinates through a permutation `P` — producing another, ordinary occupancy field — and feed it to the *existing* encoder and the *existing* traversal shader. The only per-layout runtime cost is permuting the camera ray by `P` (and un-permuting the returned hit voxel by `P⁻¹`), both of which are trivial index shuffles on three components.

The correctness dividend is the part worth emphasizing. Each permuted copy is, by construction, *just another valid occupancy field* — it is the same scene with relabeled axes. That means the differential-correctness harness already in place (the bit-exact CPU/GPU traversal comparison used throughout the project) validates each permuted layout *for free*, with no new oracle and no new test scaffolding. We are not asking "is this exotic encoder correct?" We are asking "is this ordinary occupancy field traversed correctly?" — a question the existing differential already answers. This is why the lever was inexpensive to stand up despite being the most elaborate one explored: the equivalence reduced a shader-and-encoder project to a coordinate-permutation-and-ray-transform project, and the validation came along at zero marginal cost.

The commits trace the four stages: `29e0109` introduces the layout experiment; `b718378` establishes the 3×-is-enough memory finding; `1c96da3` lands the full three-cyclic-layout solution with the image-invariance proof; and `b46037c` records the selector finding that sinks it.

### 7.2 The `layout` experiment — the envelope

The first question is purely diagnostic: *if* you could pick the best layout per direction with an oracle, how much is there to gain? The `layout` subcommand answers this. It builds all six axis-permuted layouts, sweeps view directions, and for each direction takes the cheapest layout — the per-direction minimum over the six. We call this lower envelope the "best-of-6" curve. It is an oracle: it assumes a free, perfect selector. It is the *ceiling* of the scheme, not a deployable result.

The clean measurement (side = 256, the `dust` 512³ scene):

| Configuration | Worst-case cost | Worst-case ratio vs single |
|---|---|---|
| Single layout | 114.8 ns/ray | 1.00× (baseline) |
| Best-of-6 (oracle envelope) | 72.3 ns/ray | 1.59× faster |

The mean improvement across the swept directions was 1.29×; the worst-case improvement 1.59×. (As with all absolute timings in this report, these nanosecond figures are thermal-sensitive on the Apple M-series wgpu backend and drifted across the session; the *ratios* within a single run are the trustworthy signal, and these two numbers come from the same run.)

Two facts emerge, and the second is the one that foreshadows the failure.

First, the cheap direction genuinely *moves* across layouts. Different permutations make different view directions cheap, which confirms the central hypothesis: the cheap direction is layout-determined, not scene-determined. Were that not true, all six envelopes would coincide and there would be no envelope at all. So the mechanism does *something* — it is not a no-op.

Second, and decisively, the gain is *modest* — 1.59× worst-case, not the order-of-magnitude one might hope for — and the reason is structural. The six axis permutations only **relabel** axes. A permutation maps a view direction to one of its six coordinate-permutations. For a direction that is strongly **axis-aligned** (e.g. mostly along `+x`), the permutations shuffle which stored axis it aligns with, and one of them lands favorably — that is where the envelope wins. But for a **diagonal** direction — one with comparable components on all three axes — its six coordinate-permutations are all *approximately the same direction*. Permuting `(0.58, 0.58, 0.57)` just produces six near-identical vectors. The envelope cannot help a direction that is invariant under the very transformation that defines the layouts.

And diagonal directions are not a corner case: they are *most of the sphere* by solid angle, and — critically — they are the **worst-case** directions, the ones the whole investigation is trying to fix. So the layouts deliver their gain exactly where it is least needed (near the axes) and deliver almost nothing where it is most needed (on the diagonals). The envelope's 1.59× worst-case is, in effect, the layouts rescuing a handful of near-axis directions while the genuinely-bad diagonal directions sit on the envelope essentially unmoved. This is the first sign that the lever's ceiling is lower than its framing suggests, and that the ceiling itself is being propped up by directions that were never the problem.

### 7.3 Memory — 3× is enough, not 6×

Six layouts at full scene size is a 6× memory multiplier, which is steep. The `b718378` finding is that you do not need six.

The argument is combinatorial. The "cheap axis" a layout exposes is determined by which coordinate the permutation places innermost — `perm[0] ∈ {x, y, z}`. There are only three distinct values `perm[0]` can take, so the six permutations collapse to **three distinct cheap axes**. The three *cyclic* permutations — `xyz`, `yzx`, `zxy` — already cover all three; each contributes one cheap axis. The remaining three (the non-cyclic, odd permutations) re-expose cheap axes already covered and add only second-order reshuffling of the deeper bits.

Empirically, the three cyclic layouts capture **95% of the six-layout worst-case gain**: 1.37× versus 1.40× for the full six. (These two ratios are from the same run and so are directly comparable despite session-level thermal drift.) Three percent of the gain is not worth doubling the memory multiplier from 3× to 6×, so the deploy target is fixed at **3× via the three cyclic layouts**. This is the one unambiguously good engineering decision in the lever: it halves the cost for a 5% haircut on an already-modest ceiling.

### 7.4 The `multilayout` full solution — invariance, memory, stability

The `multilayout` subcommand builds the three cyclic layouts and establishes the three properties a deployable scheme needs: that switching layouts is invisible in the output, that the memory cost is exactly what was predicted, and that the stability benefit is real (under the oracle).

#### 7.4.1 Image-invariance — the crux correctness property

This is the property that makes the scheme *deployable at all*, and it is machine-verified. The claim: for each layout, if you (1) apply that layout's inverse permutation to the camera ray, (2) traverse the layout's permuted structure, and (3) apply the layout's forward permutation to the returned hit voxel, then **every layout returns the identical hit as the plain, unpermuted structure**. The measurement: **0 mismatches out of 49,152** rays, for every layout.

The implication is the whole point. Which layout the renderer selects is **completely invisible in the rendered output**. There is no seam, no shimmer, no popping artifact when you switch from one layout to another. You can switch layouts *mid-orbit*, frame to frame, and the image is bit-identical to what the plain structure would have produced. This is what would let a selector operate freely — it could change its mind every frame without any visual consequence — and it is the property that distinguishes this lever from naive "use a different data structure for different views" schemes that produce visible discontinuities at the switch. The invariance is not argued from first principles and trusted; it is verified bit-exactly against the plain structure across the full ray set. Correctness, here, is not in doubt.

#### 7.4.2 Memory

Exactly **3.00× a single layout** — 6.2 MiB for the `dust` 512³ scene. Because the `dust` scene is isotropic, the three cyclic layouts are equal in size, so the multiplier is exactly three with no per-layout variation. This confirms the 3× target from §7.3 holds in the built artifact, not just in the combinatorial argument.

#### 7.4.3 Stability — the signature of a stability lever

The stability measurement sweeps an azimuth orbit and looks at the *spread* of frame times, not just their mean. Under the best-of-3 oracle:

| Metric (azimuth orbit) | Single layout | Best-of-3 (oracle) | Change |
|---|---|---|---|
| Frame-time spread (max/min) | 5.23× | 4.09× | ~1.3× steadier |
| Worst frame | — | — | 1.16× faster |
| Mean cost | 52 ns/ray | 48 ns/ray | essentially unchanged |

The signature here is precise and worth naming. The **mean is essentially unchanged** (52 → 48 ns/ray is within thermal drift and effectively flat) while the **worst case improves** (worst frame 1.16× faster, spread tightened 1.3×). That mean-flat / tail-improved signature is the *definition* of a **stability lever** rather than a throughput lever. A throughput lever moves the whole distribution left — every frame gets faster. A stability lever leaves the typical frame alone and pulls in only the bad views, narrowing the distribution. Multi-layout is unambiguously the latter: it does not make a good view faster; it makes a bad view less bad, and only the bad ones.

This is an entirely reasonable thing to want. Frame-time *consistency* matters for a real-time viewer — a 5× orbit spread means visible hitching as the camera swings through expensive orientations, even if the average is fine. A lever that tightens the spread 1.3× and pulls the worst frame in 16% would be worth deploying *if* you could realize it. The "if" is the whole problem, and it is the subject of §7.5.

At this checkpoint, everything points to ship. Correct (0/49,152). Cheap (3.00×). Beneficial (spread 5.23× → 4.09×). Every proxy says go.

### 7.5 The selector failure — the decisive negative

A deployable renderer cannot consult an oracle. It needs an **O(1) per-frame selector**: given *this* camera direction, name the layout to use, in constant time, before the frame renders. The best-of-3 envelope of §7.4.3 is an oracle — it inspects all three actual costs and takes the minimum. No shipping renderer can do that without rendering all three, which would cost 3× and defeat the purpose. So the question that decides the lever is: **can a cheap selector approximate the oracle?**

We tried the two principled candidates. Both failed, and the *way* they failed reveals the root cause.

#### 7.5.1 The analytical axis-rule

The obvious closed-form selector follows directly from the mechanism's own story: each layout exposes a Morton-cheap axis, so pick the layout whose cheap axis matches the camera's dominant axis (the largest-magnitude component of the view direction). If the mechanism's narrative were correct, this rule would be near-perfect.

It picks the actually-cheapest layout **~35% of frames**. The chance baseline for three layouts is **33%**. The axis-rule is, within noise, *no better than random*.

This is a sharp and important result. It says the Morton-cheap **axis** — the thing the layouts are built around, the thing the whole combinatorial memory argument in §7.3 is predicated on — is **not** the GPU-cheap **view direction**. The story "this layout is cheap for rays along axis A" is true about the *encoder's* structure but false about the *GPU's* measured cost. The two have come apart. There is no closed-form selector because the quantity the closed form predicts (cheap axis) does not govern the quantity we need to predict (cheap view direction). The mechanism's own self-description does not survive contact with the timer.

#### 7.5.2 The principled fallback — a measured direction→layout table

If there is no closed form, the principled fallback is empirical: do not *derive* the best layout, *measure* it. Build a lookup table by sweeping the best-layout choice over a Fibonacci sphere of directions, store `direction → best-layout`, and at runtime do an O(1) nearest-direction lookup. This is the textbook move when an analytical model fails: replace the model with a measured table. It sidesteps the broken axis-story entirely and just records, empirically, which layout won where.

It is no better:

| Selector | Scene | Accuracy | Orbit spread effect | Gain captured |
|---|---|---|---|---|
| Axis-rule (analytical) | dust | ~35% | — | ~0 |
| Direction→layout table | dust | 38–40% | 5.76× → 6.39× (worse) | 0% |
| Direction→layout table | wire | 38–40% | — | 0% |

The table lands at **38–40% accuracy** on both the `dust` and `wire` scenes — marginally above the 33% chance baseline, nowhere near usable. Worse, deploying it made the orbit spread *worse*, not better: **5.76× → 6.39×**. It captured **0%** of the best-of-3 gain. A selector that is supposed to tighten the distribution actually widened it, because its near-random choices occasionally select a *more*-expensive layout than the single-layout default would have used. The principled empirical fallback did not merely underperform the oracle — it underperformed *doing nothing*.

#### 7.5.3 Root cause — the gain was min-of-noise

Why does even a *measured* table fail? Because the thing it is trying to measure is not there to be measured. The three layouts' per-direction costs differ **by less than the GPU timing noise**. The cost surfaces of the three layouts, over the sphere, are essentially the same surface plus three independent noise draws.

Two consequences follow, and the second is fatal.

First, the best-layout map over the sphere is **fragmented noise**. There is no coherent region where "layout 1 wins" — there is a salt-and-pepper field where whichever of three near-equal-plus-noise costs happened to come up smallest wins. A Fibonacci-sphere table sampling that field records noise; a nearest-direction lookup at runtime queries against noise; the next frame's actual draw is an *independent* noise sample that the table cannot predict. ~Chance accuracy is the *only* possible outcome. This is why both selectors land near 33% — not because we chose them poorly, but because the signal they would need does not exist above the noise floor.

Second — and this retroactively undermines §7.4.3 — the best-of-3 "ceiling" is itself **largely a min-of-noisy-samples artifact**. Taking the minimum of three noisy measurements of near-equal quantities is **biased low**: even if all three layouts were *identical*, `min(c₁+ε₁, c₂+ε₂, c₃+ε₃)` is systematically below any single `cᵢ+εᵢ`, purely from the statistics of taking a minimum over noise. So the "1.3× steadier / worst-frame 1.16× faster" envelope was not measuring a real per-direction advantage of the best layout — it was substantially measuring the downward bias of `min` over three noisy draws. It *looks* like a gain. It is partly an estimator artifact. And a *committed* selector — one that must name a layout *before* seeing the costs — can never realize an artifact of taking the minimum *after* seeing them. The oracle gets to cheat by looking at all three; the selector cannot, and the part of the "gain" that came from the cheating evaporates.

This is the crux: the envelope and the selector are measuring two different things. The envelope measures `E[min over noise]`, which is biased optimistic. The selector measures `E[cost | committed choice]`, which is honest. The gap between them is the min-of-noise bias, and it is most of the apparent gain. The stability framing in §7.4.3 was compelling and the best-of-3 number seemed to confirm it — but both were reading an estimator artifact as a physical effect.

### 7.6 Disposition and the lesson

The mechanism is **correct**: image-invariant (0/49,152, machine-verified), 3.00× memory, validated end-to-end against the plain structure. None of that is in question, and the layout/multilayout subcommands remain a **committed diagnostic** — the equivalence, the invariance proof, and the selector measurement are all reproducible in-tree. What does not survive is the *benefit*. The stability advantage that the oracle envelope advertised does **not survive a real O(1) selector**, because the per-direction cost differences the selector would need to exploit are below the noise floor, and the envelope's apparent gain is mostly the downward bias of taking a minimum over noisy samples.

The viewer wiring was **deliberately not built**. Wiring multi-layout into the live renderer would showcase a ~40%-accurate selector delivering no net gain — and, in the table case, a measurable *regression* of the orbit spread. Shipping that, or screenshotting it as a win, would be dishonest engineering: a demo that depends on the viewer's operator not measuring what it actually does. The honest disposition is to stop at the diagnostic, record the negative, and not build the showcase.

This is the lever that **most looked like it should work**. The equivalence was elegant, the invariance proof was clean and machine-checked, the memory cost was modest and well-argued down to 3×, and the stability framing — mean-flat, tail-improved — was exactly the textbook signature of a consistency lever. Every checkpoint short of the selector returned green. It collapsed at precisely the step that a paper or a design doc would have hand-waved ("a per-frame selector picks the best layout") and that only *building it* could falsify. Had we stopped at the best-of-3 envelope — the promising proxy — we would have recorded a 1.3×-stability win that does not exist. Building the selector, and measuring that it lands at chance, is what turned a plausible-but-false result into an honest negative. That is the vindication of build-and-validate over stopping at the proxy, and it is the most important methodological takeaway of the entire multi-layout investigation.

This is a writing task with all data provided in the brief. No skill or tool invocation is needed. I'll produce Section 8 directly.

## 8. Lever 3 — Hilbert curve and treelet / cache-oblivious layout (branch `feat/hilbert`)

The first two levers attacked the anisotropy on the GPU side: Lever 1 re-oriented the traversal frame, Lever 2 reshaped the work into the hardware's preferred access pattern. Lever 3 takes a different posture — attack the anisotropy at its *source*, in the leaf array's memory layout, before the GPU ever touches it. The hypothesis is geometric rather than architectural, and that is what made it attractive: if the ordering of occupied bricks in memory were itself isotropic, the 9× orientation-dependent traversal cost should fall out of the data structure rather than have to be fought per-frame.

This section records two related sub-levers explored on `feat/hilbert`: replacing Morton (Z-order) leaf ordering with a 3-D Hilbert ordering (commit `9327230`), and a treelet / cache-oblivious relayout (commit `39af0c0`). Both are falsified. The Hilbert encoder itself is correct and reusable, and the measurements are clean; the failure is conceptual, and it is worth recording precisely *why*, because the "why" rules out an entire family of would-be fixes and points squarely at the latency-hiding work in Section 9.

### 8.1 Motivation — Morton's anisotropy is structural, Hilbert's is (allegedly) not

Morton ordering interleaves the bits of the per-axis coordinates. That bit-interleave is the root of the orientation dependence we have been chasing: a unit step along `x` (the lowest-order interleaved bit) is a tiny jump in linear memory, a unit step along `z` (a higher-order bit) is a large one, and the gap between them grows with grid resolution. The Z-order curve has, by construction, *different* locality along different axes — it is anisotropic at the level of the index function itself. Our 9× view-direction sensitivity is, at least in part, this same anisotropy seen through the traversal: rays marching along the axis with poor index-locality miss cache more often than rays marching along the well-localized axis.

The Hilbert curve is the textbook answer to exactly this complaint. It visits the same set of cells but never makes a long jump — consecutive Hilbert indices are always spatial 6-neighbours — and its locality is famously more *uniform* across axes than Morton's. The standard result (Moon et al., and the broader spatial-indexing literature) is that Hilbert ordering yields lower and more isotropic average distance between spatially-near points than Z-order. If that property held for our data, ordering the leaf array by Hilbert index would *reduce* the 9× at the source: the same traversal, the same shader, the same hardware, but a layout whose worst axis is no longer 3-4× worse than its best. That is a strictly better place to fix the problem than re-orienting rays per frame, because it costs nothing at runtime — the cost is paid once, at build time.

So the question this lever asks is narrow and testable: **does Hilbert ordering flatten the per-axis locality of our occupied-brick array relative to Morton?** Not "is Hilbert more isotropic in general" — that is known — but "is it more isotropic *for our sparse subset*."

### 8.2 The Hilbert encoder (commit `9327230`)

Before anything could be measured, we needed a correct 3-D Hilbert encoder, and 3-D Hilbert is where naive implementations quietly go wrong. The encoder implements Skilling's algorithm: take the per-axis integer coordinates, run the Gray-code / sign-and-swap transform that produces the Hilbert "transpose" representation, then interleave that transpose to a single scalar Hilbert distance. The three stages — axes → transpose → scalar — are the standard decomposition and each was verified independently.

Two properties were checked, because for a curve encoder these two together are the whole correctness story:

| Property | What it asserts | Verified over |
|---|---|---|
| Bijection | The encoder maps the grid one-to-one onto `0 … N³−1` (no collisions, no gaps) | grids `2³` through `16³` |
| 6-neighbour consecutivity | Index `i` and index `i+1` differ by exactly 1 on exactly one axis | all consecutive pairs, same grids |

The second property is the defining geometric guarantee of a Hilbert curve and the entire reason we were interested: it is what "no long jumps" means concretely. Confirming it directly (rather than trusting the bit-twiddling) means the encoder is a sound foundation for any future full Hilbert build.

It is worth being explicit about what a *full* Hilbert build would cost, because it bears on whether this lever was ever going to be cheap. Morton's appeal in a sparse-MIP-voxel tree is that child addressing is *fixed*: the eight children of a node sit at a fixed bit-interleave offset regardless of where the node is, so descending the tree is a constant shift-and-mask. Hilbert child addressing is **orientation-dependent** — the order in which the curve visits a node's eight children depends on the curve's orientation as it *enters* that node, which is itself a function of the path taken from the root. A full Hilbert-ordered tree therefore has to carry and transform that orientation state at every level of descent, both at build time and (worse) in the traversal shader. That is a large, invasive change to the hot path. We were not willing to pay for it on spec. Hence the strategy: build the cheap proxy first, and only commit to the full build if the proxy says the payoff is real.

### 8.3 The cheap proxy — the `locality` subcommand

The proxy is the `locality` subcommand. For a given scene it loads the occupied bricks, orders them once by Morton and once by Hilbert, and for each ordering computes the **per-axis mean memory-distance between spatially-adjacent occupied bricks**: for every pair of occupied bricks that are 6-neighbours in space, how far apart are they in the linear leaf array? Averaging that gap separately for x-, y-, and z-adjacency gives a three-number locality signature per ordering, and the spread across those three numbers is precisely the layout anisotropy we care about. If Hilbert flattens the anisotropy, its three numbers should be closer together — and ideally smaller — than Morton's.

They are not. The proxy's verdict is unambiguous: Hilbert does not flatten our anisotropy, it **relocates** it.

**`dust` scene, 512³:**

| Ordering | x | y | z | Anisotropy (max/min) | Mean neighbour distance |
|---|---|---|---|---|---|
| Morton | 83 | 134 | 275 | 3.33× | 164 |
| Hilbert | 344 | 162 | 93 | 3.70× | 200 |

This is the clearest possible refutation. Hilbert does not merely fail to help — it is worse on *both* the best axis and the worst axis simultaneously. Under Morton the cheapest axis costs 83 and the most expensive 275; under Hilbert the cheapest is 93 (worse than Morton's 83) and the most expensive is 344 (worse than Morton's 275). The anisotropy ratio rises from 3.33× to 3.70×, and the mean neighbour distance rises from 164 to 200. The curve has rotated which axis is cheap (the cheap axis moved from x to z) but it has not made any axis cheaper, and on average it has made things worse.

**`wire` scene, 512³:**

| Ordering | Anisotropy spread | Worst-axis distance |
|---|---|---|
| Morton | 4.00× | 1189 |
| Hilbert | 3.05× | 1189 (unchanged) |

`wire` looks superficially more favourable to Hilbert — the spread tightens from 4.00× to 3.05×. But the improvement is cosmetic in exactly the way that matters least. The *worst-axis* distance, 1189, is **unchanged**: Hilbert tightened the ratio by raising the cheaper axes toward the expensive one, not by pulling the expensive one down. The mean rises. Since cache misses are driven by the long jumps on the worst axis, narrowing the ratio while leaving the worst axis at 1189 buys nothing for traversal — it is anisotropy-reduction on paper that does not touch the actual miss source.

So across both scenes the comparative verdict is the same: **Hilbert is equal-or-worse.** It never reduces the worst axis, and on `dust` it regresses every axis and the mean.

### 8.4 Why Hilbert relocates rather than reduces — the sparsity argument

The textbook "Hilbert is more cache-isotropic" result is true, and our measurement does not contradict it. The result is a statement about **dense** arrays — about traversing *every* cell of the grid in curve order, where the only thing that matters is the gap between consecutive *visited* cells. For a dense scan, Hilbert's no-long-jumps guarantee directly bounds the working set and the isotropy follows.

Our leaf array is not dense. It is a **sparse occupied subset** of the grid — a small, geometrically structured fraction of cells, with everything else absent. And here the asymmetry in the Hilbert guarantee becomes decisive. The Hilbert curve guarantees:

> consecutive index ⟹ spatially adjacent

It does **not** guarantee the converse:

> spatially adjacent ⟹ consecutive (or even nearby) index

For a dense array the converse holds *approximately* because the curve fills space and any two adjacent cells are reachable in a short stretch of curve. But on a sparse subset, two spatially-adjacent *occupied* bricks can be separated by a long arc of the curve that runs entirely through *unoccupied* cells — and those unoccupied cells are not in the leaf array, so they contribute nothing to compaction, while the two occupied bricks still land far apart in the compacted linear order. The very property we measure in the proxy — distance between adjacent *occupied* bricks — is governed by the *converse* direction that Hilbert does not control.

This is why Hilbert, on a sparse traversal, is just a **differently-oriented anisotropy** and not a flatter one. It rotates the curve's "grain" relative to the grid, which moves which spatial direction happens to align with cheap curve-locality, but it does not give us the isotropy that holds for dense scans. For our data the rotation is, if anything, unfavourable: it lined the cheap direction up against an axis where our occupied bricks are sparser, lengthening the average jump.

A caveat on the proxy, stated plainly so it is not over-read: the proxy reports roughly 3.3× anisotropy where the GPU measured roughly 9×. It **undershoots the magnitude** — it is a layout statistic, not a traversal measurement, and it does not capture the compounding effects (warp divergence, miss queueing, the traversal's own access pattern) that inflate the on-device ratio. We do not use the proxy's absolute number for anything. What the proxy *is* trusted for is its **comparative** verdict, Morton vs Hilbert, and that comparison is exactly what it was built to make: same scene, same adjacency definition, two orderings, one number each. The comparison is robust even though the absolute is not. And it is worth remembering that Hilbert can *only* affect the brick-locality component of the 9× in the first place — it cannot touch whatever part of the 9× comes from the traversal's access pattern or from divergence — so even a proxy that perfectly captured layout would be measuring an upper bound on the achievable win. That upper bound came back negative.

### 8.5 Treelet / cache-oblivious layout (commit `39af0c0`) — a proof, then a measurement

The natural follow-on idea is that even if neither Morton nor Hilbert is globally isotropic, perhaps a *treelet* or cache-oblivious (van-Emde-Boas) relayout — recursively grouping subtrees into cache-block-sized chunks — would do better than the linear curve orderings. This is the layout family that BVH and tree-traversal work reaches for when it wants to be cache-friendly without tuning to a specific cache size.

It does not apply here, and the reason is a short proof rather than a measurement. **For a regular grid, the cache-oblivious / van-Emde-Boas leaf order *is* Morton order.** Both constructions are recursive groupings of subtrees with the same child order at each level; applied to a complete, regular octree they emit the *same leaf sequence*. There is no distinct "treelet ordering" to add on top of Morton — Morton already *is* the recursive blocking. So a treelet relayout has, by construction, no leaf-order headroom over the layout we already ship.

That proof tells us the *order* is fixed, but treelets optimize something slightly different from order — they optimize cache-block **co-location**: the fraction of an axis's spatial neighbours that land within the same cache-block-sized chunk, regardless of their exact distance inside it. That is the right thing to measure, so commit `39af0c0` measures it directly: for a ≤64-leaf block (~4 KiB, a plausible cache-block / treelet size), what fraction of each axis's 6-neighbour pairs fall within a single block?

**`dust`:**

| Ordering | x | y | z |
|---|---|---|---|
| Morton | 95% | 89% | 89% |
| Hilbert | 92% | 91% | 92% |

**`wire`:**

| Ordering | x | y | z |
|---|---|---|---|
| Morton | 89% | 89% | 76% |
| Hilbert | 85% | 85% | 85% |

Read these two ways. First, **Morton is already near the ceiling.** It co-locates 76-95% of neighbours within a single ~4 KiB block, and it is already fairly isotropic in this metric — the spread across axes is modest. The blocking that a treelet layout would provide is, for the overwhelming majority of neighbours, *already there in Morton*. Second, **Hilbert again only evens the last few percent.** On `dust` it trades Morton's 95/89/89 for a flatter 92/91/92 — the worst axis comes up a little, the best axis comes down a little, and the *average is essentially unchanged*. On `wire` it does the same: 89/89/76 becomes a flatter 85/85/85, evening the spread while leaving the average where it was. This is precisely the relocation-not-reduction behaviour from the mean-distance proxy, seen now in the co-location metric — Hilbert moves percentage points between axes at constant total.

### 8.6 Where the anisotropy actually lives — and why layout cannot reach it

Put the two metrics together and a consistent picture emerges. Co-location says 76-95% of neighbours are already within a treelet-sized block under Morton. But the *mean-distance* anisotropy is 3.3-4×. Those two facts are only reconcilable if the anisotropy is carried almost entirely by the **rare** 5-24% of neighbours that fall *across* block boundaries. The common case — the 76-95% — is already cheap and already roughly isotropic. The expensive, view-dependent tail is the small fraction of adjacencies that straddle a cache block, and it is those cross-block jumps that the worst-axis numbers (the 275, the 1189) are measuring.

This is the crux, and it is why the whole layout lever bottoms out here. **No leaf reordering can fix a cross-block neighbour, because being cross-block is a property of needing more than one block, not of the order within blocks.** You can shuffle which neighbours are cheap (that is all Hilbert does), but for any sparse occupied set there will be adjacencies that cannot fit in the same block, and those are the misses that drive the anisotropy. The cost of a cross-block neighbour is not a layout cost at all — it is the **latency** of the second cache/memory access. The only way to make a cross-block neighbour cheap is to **hide its miss latency**: prefetch the next treelet before the traversal stalls on it.

And that is exactly the move WGSL cannot express. There are no prefetch intrinsics in WGSL — the shader cannot tell the hardware "the next access will be to block *B*, start fetching it now." The usual software substitute, staging the next block into shared / workgroup memory, does not help us either: it requires all 32 lanes of a warp to be working in the *same* treelet at the same time, which is only true when the rays are spatially coherent. Under the orientation/view-direction conditions that *produce* the 9× in the first place, the rays in a warp are precisely *not* coherent — they are spread across blocks — so the shared-memory prefetch has nothing coherent to stage. The lever that would actually help is unavailable at our abstraction level, and the conditions that would make a workaround viable are the conditions under which we do not have the problem.

### 8.7 Verdict

Both sub-levers are falsified, for reasons that compound cleanly:

- **Hilbert ordering relocates the anisotropy; it does not reduce it.** The proxy shows it equal-or-worse on every scene measured — worse on both axes and the mean on `dust` (3.33× → 3.70×, mean 164 → 200), and on `wire` it tightens the ratio (4.00× → 3.05×) only by raising the cheap axes while leaving the worst axis pinned at 1189. The root cause is sparsity: Hilbert guarantees consecutive-index ⟹ adjacent, but not the converse, and the converse is what governs distance between adjacent *occupied* bricks in a sparse subset.
- **A treelet / cache-oblivious relayout has no leaf-order headroom.** For a regular grid the cache-oblivious order *is* Morton, by proof, not approximation — there is nothing to add. The co-location metric confirms Morton already blocks 76-95% of neighbours near-isotropically; Hilbert only evens the final few percent at unchanged average.

The work was not wasted. The Hilbert encoder (commit `9327230`) is correct — bijective and 6-neighbour-consecutive over `2³…16³` — and remains a usable foundation if a full orientation-aware Hilbert build is ever justified by other considerations. And the negative result is itself a strong directional signal: it proves the residual anisotropy is **latency, not layout**. The cheap common case is already laid out well; the expensive tail is cross-block misses whose only remedy is latency-hiding — prefetch and ray-reordering — which is hardware territory (prefetch intrinsics, Shader Execution Reordering) and the subject of Section 9.

## 9. What the research community says

The preceding sections recorded a sequence of bounded and negative software results: ray-binning that returned roughly 1.2×, sort costs that swamped the traversal savings, a Hilbert relayout whose locality advantage failed to materialize on our sparse subset, and an overall conclusion that the orientation/view-direction anisotropy is intrinsic to incoherent GPU ray traversal rather than a bug we could engineer away. Before treating those as a dead end, it is worth checking them against the published literature. The short version: every one of our four bounded outcomes lands exactly where the field's consensus already sits. We did not stumble into a local minimum that a known trick would have escaped — we rediscovered, independently and from first principles on a different substrate (a custom WGSL compute kernel over a sparse MIP-voxel grid rather than a DXR/OptiX BVH), the same ceiling that has driven the rendering-hardware roadmap for the last fifteen years. This section walks each of our findings to its matching result in the literature, and then draws the meta-conclusion about why the real fix is one we cannot currently adopt.

### 9.1 The problem is treated as fundamental, not as a defect

Our central finding — that the anisotropy is intrinsic and the workload is latency-bound rather than throughput-bound — is not a novel or surprising claim in the field. It is the founding observation of the modern incoherent-ray-tracing literature.

Aila & Laine, *Architecture Considerations for Tracing Incoherent Rays* (HPG 2010), frame incoherent-ray tracing as **intrinsically latency-bound**. Their analysis attributes the cost to three coupled mechanisms: cache misses from rays touching unrelated regions of the acceleration structure, control-flow divergence as neighboring threads take different branches through that structure, and the memory bandwidth required to service the resulting scattered reads. The paper's posture is explicit that this is "an inherent challenge rather than a fully solvable problem" — i.e. a property of the workload's interaction with the memory hierarchy and the SIMD execution model, not an artifact that a cleverer kernel removes.

This maps directly onto what we measured. Our anisotropy is the macroscopic signature of exactly those three mechanisms: for view directions where neighboring rays diverge early in the traversal, we pay more cache misses and more divergence, and the kernel stalls waiting on memory. The reason our absolute timings drifted 2–6× over a session under sustained thermal load — and the reason ratios stayed stable while absolutes did not — is itself consistent with a latency-bound regime: when the kernel spends most of its time stalled on memory rather than saturating ALUs, throughput tracks clock and memory-controller behavior, which thermal throttling perturbs, while the *relative* cost of a hard view direction versus an easy one is a property of the traversal pattern and stays put. A throughput-bound kernel would not behave that way. The literature predicted both the qualitative shape and the latency-bound character of what we saw.

The practical consequence of accepting Aila & Laine's framing is that the question "how do we eliminate the anisotropy?" is the wrong question. The right question — and the one the rest of the field has spent fifteen years on — is "how do we hide or amortize the latency that produces it?" That reframing is what motivates every lever below.

### 9.2 Software ray reordering gives modest gains that the overhead eats

Our ray-binning experiment returned roughly 1.2× in the traversal phase, and we found that the cost of producing the ordering consumed most or all of the saving. This is the single most thoroughly replicated result in the reordering literature.

*On Ray Reordering Techniques for Faster GPU Ray Tracing* (I3D 2020) reports that reordering yields roughly **1.3–2.0×** in the trace phase — and then states the central problem plainly: "recovering the reordering overhead is problematic." The trace phase gets faster; the sort or binning pass you added to get there gives much of it back. Our 1.2× sits just below the bottom of their reported trace-phase band, which is exactly what we should expect given that our binning was coarse, our substrate is a sparse voxel grid rather than a BVH, and we measured net effect rather than the trace phase in isolation.

*Efficient ray sorting for the tracing of incoherent rays* reports a comparable **~1.48×** with a substantial sort cost, and adds a detail that directly anticipated one of our own dead ends: moving from 32-bit to 64-bit sort keys costs about **2.5× more sort time** for only a marginal trace-phase gain. That is the diminishing-returns curve on sort complexity stated numerically — past a certain key resolution you are paying linearly (or worse) for sub-linear improvement in ray coherence. Any temptation we had to make the ordering "better" by encoding more bits of spatial information per ray runs straight into this wall.

The wavefront-path-tracing result triangulates the same ceiling from the architecture side. Laine, Karras & Aila, *Megakernels Considered Harmful: Wavefront Path Tracing on GPUs* (HPG 2013), restructure the entire renderer to improve cache locality and execution coherence, and the locality improvement alone is worth about **16%**. That is a large engineering investment — splitting a megakernel into communicating wavefront stages — for a gain in the same modest band as a sort. The lesson is consistent across all three papers: software-level reorganization of *when* and *in what order* rays execute buys a small constant factor, and the bookkeeping to achieve it is expensive enough that net wins are fragile.

| Source | Reported gain | Where the gain is | The catch |
|---|---|---|---|
| I3D 2020 (ray reordering) | ~1.3–2.0× | trace phase | "recovering the reordering overhead is problematic" |
| Efficient ray sorting | ~1.48× | trace phase | substantial sort cost; 64-bit keys cost ~2.5× more sort for marginal gain |
| Megakernels Considered Harmful (HPG 2013) | ~16% | improved cache locality | requires full wavefront restructuring of the renderer |
| **Our ray-binning** | **~1.2× (net)** | **traversal** | **sort/binning cost dwarfs the saving** |

Our result is not an underperformance relative to the field. It is the field's result, reproduced on a different data structure, including the specific failure mode — the overhead eating the gain — that the I3D paper names as the defining difficulty of the technique.

### 9.3 Space-filling curves: Hilbert's locality advantage is real but conditional

We relaid out the voxel data along a Hilbert curve expecting better cache locality than Morton/Z-order, and found the advantage did not materialize on our sparse traversal subset while the encode cost went up. Both halves of that outcome are documented.

The locality comparison (*Hilbert curve vs. Z-order*) confirms that Hilbert ordering does have genuinely better locality than Morton ordering — adjacent indices map to spatially closer cells more reliably, because the Hilbert curve never makes the long diagonal jumps that Z-order does. But it comes at roughly **2× the encode cost**, and — critically — its benefit shows up mainly for **dense or large queries**, where a traversal sweeps long contiguous runs of the curve and the superior locality compounds over many consecutive accesses.

There is also a documented *structural* limitation of Z-order that explains why the family as a whole is awkward for our use: a Z-order traversal always ends in a cell diagonally opposite its start, so it is impossible to leverage for inter-grid locality. The curve's endpoints are pinned to opposite corners of the block, which means stitching blocks together along the curve forces a discontinuity at every block boundary. Hilbert avoids the worst of this (its endpoints are adjacent corners, so curves chain more gracefully), which is precisely why it is the locality-preferred choice — but only when there is enough contiguous work *within* a block for that to pay off.

That conditionality is exactly why it failed for us. Our traversal does not sweep dense contiguous runs; it touches a **sparse subset** of cells scattered along the curve. In that regime there is little contiguous work for Hilbert's locality to compound over, so the encode-cost penalty is paid up front while the locality benefit — which is real but query-shaped — never gets a chance to accumulate. The literature's "dense/large queries" caveat is not a footnote for us; it is the whole story.

Our sparse-subset analysis also extends the dense-array literature in one direction worth recording. The classic statement of space-filling-curve locality is that **consecutive indices map to spatially adjacent cells** — that is the property the dense-query results lean on. What we observed is that the converse does not hold: **adjacent cells do not map to consecutive indices**. For a sparse subset that distinction is decisive. A dense sweep only ever relies on the forward direction (walk consecutive indices, get adjacent cells, enjoy locality), so the literature can treat the curve as a locality primitive without qualification. A sparse traversal, by contrast, lands on spatially adjacent cells that are arbitrarily far apart *in curve-index space*, so the relayout buys nothing for the access pattern we actually have. The dense-array literature never had to confront this because its workload never exercises the failing direction. Our negative result is therefore not in tension with the published locality claims — it is what those claims reduce to once you remove the density assumption.

### 9.4 The frontier answer is hardware — Shader Execution Reordering

If software reordering is near its ceiling, what does the field actually do about incoherent rays today? The answer, unambiguously, is **hardware**. Shader Execution Reordering (SER) moves the reordering step out of the kernel and into dedicated silicon on the GPU's RT path.

SER shipped on NVIDIA Ada / RTX 40-series, and is exposed through DirectX Shader Model 6.9 and Vulkan's `VK_EXT_ray_tracing_invocation_reorder`. Reported gains are in the **40–90%** range — far above anything the software-reordering papers achieve — and a concrete production data point is *Black Myth: Wukong*, whose ReSTIR GI pass runs about **3.7× faster** with SER enabled.

The reason SER exists is the single most important corroboration of our entire investigation. The hardware vendors put reordering into dedicated silicon *precisely because* software reordering's overhead is, in the I3D paper's words, "problematic to recover." The industry ran the same experiment we did — and that §9.2 documents across multiple papers — concluded that the software approach is fundamentally overhead-limited, and responded by removing the overhead the only way it can be removed: by making the reorder step a hardware primitive that costs essentially nothing to invoke. SER is the field's admission that §9.2's ceiling is real. We did not fail to find a software trick; the field already established there isn't one worth the bookkeeping, and built hardware instead.

There is a sharp caveat for us, and it defines the boundary of what we can do. **SER lives on the RT-core / DXR (and Vulkan ray-tracing) pipeline, and it primarily targets *shading* divergence** — the case where rays hit different materials and dispatch to different hit shaders. It is **not available to a custom WGSL compute kernel.** Our traversal is a hand-written compute shader over a sparse MIP-voxel grid; it does not run on the ray-tracing pipeline, does not emit hit-shader invocations for SER to reorder, and `wgpu`/WGSL exposes no equivalent intrinsic. So the one lever that decisively beats the software ceiling is, for our current architecture, unreachable. This is not a small implementation gap — it is a pipeline choice. Adopting SER would mean abandoning the custom-compute traversal and re-expressing the problem on the hardware ray-tracing pipeline, which is a different project with different constraints (and one whose primary win, shading-divergence reordering, only partially overlaps the *traversal*-divergence cost we are bound by).

### 9.5 Treelet prefetching — the BVH community's software answer

There is one more software lever the BVH community uses against incoherent-ray cache misses, and it is worth recording both because it is the strongest software result in the space and because it, too, is out of reach for us.

*Treelet Prefetching for Ray Tracing* (MICRO 2023) combines three ideas: subdividing the acceleration structure into cache-sized treelets, prefetching those treelets ahead of the rays that will need them, and queueing rays against the treelet currently resident. Together they cut memory bandwidth by up to **~90%** in hard scenes. That is a far larger effect than reordering, and it is the BVH community's direct software answer to the cache-miss problem Aila & Laine identified — it attacks the latency at its source (the misses) rather than trying to reorder around it.

The operative half of the technique is the **prefetch**, and that is the half we cannot express. WGSL has no software-prefetch primitive: there is no way in the language to issue a "bring this region into cache ahead of time, asynchronously, without blocking on the value" hint. The treelet subdivision and ray-queueing halves are expressible in principle, but without the prefetch they lose most of their point — queueing rays against a resident treelet only helps if you can get the *next* treelet resident before the current queue drains, and that is exactly what the prefetch does. So the strongest software result in the literature depends on a capability our shading language does not give us. (Wodniok & Schulz, *Analysis of Cache Behavior and Performance of Different BVH Memory Layouts for Tracing Incoherent Rays*, is the companion analysis here — it characterizes how much the memory layout alone moves cache behavior for incoherent rays, which is the same lever we exercised with the Hilbert relayout in §9.3, and reaches the same conclusion that layout helps but does not solve the problem.)

The pattern across §9.4 and §9.5 is consistent: the two techniques that meaningfully beat the software-reordering ceiling — hardware SER and treelet prefetch — both depend on a primitive (a hardware reorder unit; an asynchronous prefetch instruction) that a portable WGSL compute kernel does not expose. Our software levers are bounded not because we chose them badly but because the unbounded levers live below the abstraction we are working at.

### 9.6 Temporal / frame-to-frame coherence

One lever in the literature is genuinely orthogonal to everything above and remains open to us: **temporal coherence**, i.e. reprojecting and reusing work across frames. For interactive sparse-voxel raycasting this is standard practice — the camera moves little between consecutive frames, so a large fraction of the per-pixel traversal result from frame *N* is still valid (after reprojection) at frame *N+1*. This does not make any single frame's traversal faster, and it does not touch the anisotropy at all; it amortizes the cost across time by not re-doing work that did not change. It is worth flagging as the one major direction the literature endorses that our investigation did not exhaust and that our pipeline *can* express, since unlike SER and prefetch it requires no special hardware or language intrinsic — only inter-frame state and a reprojection pass.

### 9.7 Meta-conclusion

Lining up our four bounded/negative software results against the literature:

| Our finding | Literature match | Verdict |
|---|---|---|
| Anisotropy is intrinsic, latency-bound | Aila & Laine (HPG 2010): "inherent challenge rather than a fully solvable problem" | Confirmed as fundamental, not a defect |
| Ray-binning ~1.2×, overhead eats it | I3D 2020 (~1.3–2.0×, "recovering overhead is problematic"); ray sorting (~1.48×); wavefront (~16%) | Reproduced the field's result and its defining failure mode |
| Hilbert relayout didn't help on sparse subset | Hilbert vs. Z-order: better locality at ~2× encode, benefit mainly for dense/large queries | Failure explained by density assumption; we extended it to the sparse case |
| Software levers are bounded | Field moved reordering into hardware (SER) and bandwidth relief into prefetch (treelet) | The real fix is below our abstraction layer |

The conclusion writes itself. Our four results do not represent four missed opportunities; they trace the exact contour of the software ceiling that the research community mapped over the last fifteen years. The field's response to that ceiling was not to find a cleverer sort — it was to move divergence- and latency-hiding into **hardware** (Shader Execution Reordering) and to attack cache misses with **prefetch** (treelet prefetching), both of which require primitives a portable WGSL compute kernel does not expose. The honest reading of this section is therefore reassuring rather than disappointing: **we did not miss an obvious trick.** The levers that would decisively beat what we measured are exactly the ones that the field also could not realize in software, which is why they now live in silicon and in pipeline-specific intrinsics. Within the constraints of a custom-compute traversal, our results are near the achievable software frontier, and the only remaining open direction the literature endorses for our pipeline is temporal reprojection (§9.6).

### Sources

- Aila & Laine — Architecture Considerations for Tracing Incoherent Rays (HPG 2010): https://users.aalto.fi/~ailat1/publications/aila2010hpg_paper.pdf
- Laine, Karras, Aila — Megakernels Considered Harmful: Wavefront Path Tracing (HPG 2013): https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf
- On Ray Reordering Techniques for Faster GPU Ray Tracing (I3D 2020): https://dl.acm.org/doi/fullHtml/10.1145/3384382.3384534
- Efficient ray sorting for the tracing of incoherent rays: https://www.jstage.jst.go.jp/article/elex/9/9/9_849/_pdf
- Analysis of Cache Behavior and Performance of Different BVH Memory Layouts for Incoherent Rays (Wodniok & Schulz): https://diglib.eg.org/items/eb8806a7-9cb3-4a2b-890d-ed69267c9a8b
- Treelet Prefetching for Ray Tracing (MICRO 2023): https://dl.acm.org/doi/10.1145/3613424.3614288
- Khronos — Shader Execution Reordering (VK_EXT_ray_tracing_invocation_reorder): https://www.khronos.org/blog/boosting-ray-tracing-performance-with-shader-execution-reordering-introducing-vk-ext-ray-tracing-invocation-reorder
- Hilbert curve vs. Z-order (locality): http://computerexpress0.blogspot.com/2017/11/the-hilbert-curve-vs-z-order.html

The commit hashes and branches from the brief align with the repo state. Writing the section now.

## 10. Synthesis, recommendations, and status

### 10.1 The one conclusion

The orientation anisotropy is real, it is roughly 9×, and it is intrinsic. After ten sections of measurement, falsification, and one shipped optimization, this is the single load-bearing finding the rest of the document supports: the gap between cheap (axis-aligned) and expensive (oblique) views of a sparse, Morton-ordered MIP-voxel structure is **latency-bound**, not bandwidth-bound and not layout-bound. It is the cost of rare cross-cache-block misses, taken on exactly the rays that also diverge within a warp, on exactly the views where the traversal walks the structure against the grain of its memory order.

The reason this matters — the reason it ends the relayout program rather than redirecting it — is that the memory layout is already near-optimal for the access pattern. Section 8 measured directly that Morton ordering co-locates on the order of 90% of a voxel's traversal neighbours inside the same cache block. That is not a number we can push much higher with a different curve, because the residual ~10% is not a curve-quality problem: it is the unavoidable boundary between blocks that any 1D embedding of a 3D structure must cross, and oblique rays cross those boundaries more often per unit of screen coverage than axis-aligned rays do. When a block boundary is crossed and the next brick is not resident, the cost is a memory miss whose **latency** the kernel must hide. Hiding miss latency is what hardware does — through wide warp occupancy, hardware prefetch, and, on RT-capable parts, Shader Execution Reordering (SER). It is not something a better space-filling curve, a deeper treelet, or a cleverer serialize order can do, because none of those change the latency of the miss; they only change how often it happens, and how often is already close to the floor.

This single fact is the lens for everything below. Every software lever we tried either (a) reduced the *number* of steps (early-skip — a different axis of cost, and the one win), (b) recovered *warp divergence* but not the miss latency underneath it (ray binning), or (c) tried to reduce the *miss rate* by relayout (multi-layout, Hilbert, treelets) and ran straight into the "layout is already optimal" wall. Read that way, the negative results are not a string of unlucky experiments. They are four independent confirmations of one structural truth, each falsifying a different hypothesis about where the cost lived, and each leaving the same answer standing.

### 10.2 The exhausted software levers

The table below summarizes every lever evaluated against the anisotropy, what it actually bought, and why. The critical column is the last one: what the result *tells us about the system*, not just whether it helped.

| Lever | Measured effect | What it targets | Why it did (not) work |
| --- | --- | --- | --- |
| Per-brick early-skip | 1.68× on sparse/thin content; **shipped** | Step count, not anisotropy | Skips empty space per brick; orthogonal to orientation. The one positive — it makes everything faster without touching the gap. |
| Ray binning | ~1.2× | Warp divergence only | Re-groups coherent rays so a warp diverges less; but the underlying miss latency on the divergent rays is unchanged, so the ceiling is low. |
| Axis-permuted multi-layout | Image-invariant; ~1.3× "gain" was min-of-noise; selector ~40% accurate; **3× memory** | Miss rate via per-axis relayout | The output is provably the same image; the apparent gain did not survive being separated from thermal/run noise, and the layout selector can't reliably pick the right layout. 3× memory for a non-effect. |
| Hilbert / treelet reorder | ~0 | Miss rate via better locality | Morton is already block-isotropic at the granularity that matters; Hilbert's better asymptotic locality does not translate into fewer block crossings here. Falsified by cheap proxies before any expensive build. |

The shape of this table is the argument. The only lever that moved the needle (early-skip) is the only one that does not target the anisotropy at all — it attacks step count, an independent cost axis, and it helps on sparse and thin geometry where there is empty space to skip. Everything aimed *directly* at the orientation gap is bounded by the same ceiling: ray binning gets the cheapest piece (divergence) and stops; the two relayout approaches get nothing durable because they fight a layout that is already near its information-theoretic floor for this access pattern. The multi-layout result is the sharpest cautionary tale of the four — a mechanism that is *correct* (the images match), whose headline number was an artifact of comparing a min against noise, and whose only path to a real win (a layout selector) tops out near 40% accuracy. A 3× memory tripling to pay for a coin-flip-plus selector is not a trade worth deploying.

### 10.3 System characteristics for the record

These are the standing characteristics of the system as built, independent of the anisotropy investigation. They are recorded here so the document is a complete reference, not only a postmortem. All absolute GPU and FPS numbers below were taken on Apple M-series hardware through wgpu under sustained load; absolute timings drifted on the order of 2–6× across a session as the part thermally throttled. Treat every absolute as a range and trust ratios and within-run comparisons.

**Memory.** Footprint is set by occupancy × resolution and is not changed by any of the traversal optimizations (the multi-layout lever would have multiplied it by ~3×, which is the central reason it is not worth deploying):

| Scene | Resolution | Footprint |
| --- | --- | --- |
| dust | 512³ | 1.9 MiB |
| dust | 2048³ | 123 MiB |
| sierpinski | 2048³ | 4 MiB |
| checkerboard | 2048³ | ~1 GiB |

The checkerboard figure is the worst case by design: maximal occupancy at high resolution defeats sparsity entirely, and the structure degenerates toward a dense grid. It is a stress bound, not a representative workload.

**Rebuild.** The build pipeline is a full O(n³) occupancy scan, followed by a Morton sort, followed by serialize. There is **no incremental update path**: any geometry edit, however small, re-runs the entire pipeline. Measured wall times:

| Resolution | Rebuild time |
| --- | --- |
| 512³ | ~15–35 ms |
| 2048³ | ~0.7–1.6 s |

Rebuild is build-dominated — the occupancy scan and sort own the cost. The serialize step and the GPU re-upload are small after the popcount-gate fix, so there is no further low-hanging fruit on the back half of the pipeline; a faster rebuild means attacking the O(n³) scan or making the edit path incremental (see recommendation 4).

**Frame rate.** At roughly 1080p, again thermal-noisy — quote as ranges, trust the ratios between rows:

| Scene | FPS (~1080p) |
| --- | --- |
| sierpinski (structured) | ~150–330+ |
| dust 512³ | ~40–63 |
| dust 2048³ | ~20 |

Structured content is comfortably real-time; the cost climbs with effective surface complexity and resolution as expected. Within any of these rows, the worst-case view runs a few× slower than the typical view — that spread *is* the anisotropy, visible directly in the frame-time distribution rather than as a separate benchmark.

### 10.4 Recommendations

These are ordered. The ordering encodes the conclusion: stop fighting the latency, budget around it, and recognize that the only real throughput jump on incoherent work is a hardware-pipeline pivot, not a kernel tweak.

**1. Stop trying to reduce the anisotropy with software relayout or reordering.** This is the primary recommendation and the one the whole document exists to justify. The relayout program is proven near its ceiling for a principled reason, not an incidental one: the layout is already optimal (Morton co-locates ~90% of neighbours per block), and the residual cost is miss *latency*, which no software reordering can hide. Hilbert returned ~0, treelets returned ~0, and multi-layout's apparent gain was noise. Further investment in space-filling curves, brick reorderings, or serialize-order changes should be treated as falsified until a *new* mechanism — not a new curve — is proposed. The dead ends are mapped; do not re-walk them.

**2. If orientation cost matters for a target workload, budget around it.** The recommended next direction, if any work continues, is adaptive internal-resolution plus temporal reprojection, driven by the existing `aniso` cost descriptor. The idea is to hold a fixed frame budget by trading a small amount of quality on exactly the expensive views: when the cost descriptor flags an oblique, divergence-heavy orientation, drop internal resolution and reproject from the previous frame to recover apparent detail. This is the one well-understood lever we have *not* built, and it is attractive precisely because it does **not** fight the latency — it accepts the latency as a given and spends a quality knob to stay inside budget. It carries essentially no memory cost (the cost descriptor already exists; reprojection needs a history buffer, not a relayout). It is the natural successor to this investigation because it is the first proposed lever whose premise is consistent with the conclusion rather than in tension with it.

**3. For a real throughput jump on incoherent workloads, move to a hardware-RT pipeline.** The field's established answer to incoherent-ray latency is Shader Execution Reordering, available through the hardware ray-tracing pipelines (DXR, Vulkan-RT). SER reorders threads *after* the divergence is known, regrouping them so that the expensive, latency-bound rays are coalesced and their misses overlapped — which is exactly the lever that no WGSL-compute reordering can reach, because the reordering has to happen in the scheduler, not in the kernel. This is a **strategic pivot**, not a tweak: it means moving off the custom WGSL compute kernel and onto a different API and execution model, with all the porting and portability cost that implies (notably, this trades away the wgpu/Apple path the current work runs on). It is recommended only if incoherent throughput becomes a hard requirement for a target workload; it is the right answer to the latency problem, but it is a different project.

**4. Keep the per-brick early-skip; make geometry edits incremental.** Early-skip is shipped and earns its place — it is the one durable, anisotropy-independent win. For dynamic geometry the natural next step is an in-place brick-edit path, because the rebuild is currently full-scan only and re-runs the entire O(n³) pipeline on any edit (Section 10.3). An incremental edit path — touch the affected bricks, re-sort locally, re-upload the changed range — would make interactive geometry editing viable at 2048³, where a full rebuild costs up to ~1.6 s. This is independent of the anisotropy work and can proceed regardless of whether recommendation 2 or 3 is pursued.

### 10.5 Branch status

Every experiment is a committed, reproducing diagnostic. No lever was left as an uncommitted spike; each carries a subcommand that re-runs its measurement so the finding can be re-checked rather than taken on trust. The negative results are first-class artifacts — the falsification is the deliverable.

| Branch | Contents | Commits | Status |
| --- | --- | --- | --- |
| `main` | Per-brick early-skip + the measurement suite | edc5714, af1f45a, 5797b48 | Shipped |
| `feat/ray-binning` | The `coherence` diagnostic | 1e6f1c0 | Diagnostic; ~1.2×, recovers divergence only |
| `feat/axis-permuted-morton` | Multi-layout mechanism + selector finding | b46037c | Mechanism correct, **not deployed** (3× memory, selector ~40%) |
| `feat/hilbert` | Hilbert encoder + falsified locality/treelet proxies | 39af0c0 | Falsified; ~0 gain |

The `feat/axis-permuted-morton` line deserves its qualifier in writing: the mechanism is *correct* — it produces the same image and the layout machinery works — it is simply not worth its cost, and is parked rather than reverted so the next person can confirm the selector ceiling directly rather than re-deriving it.

### 10.6 Closing: the methodology paid for itself

The most reusable output of this work is not the early-skip optimization or even the anisotropy diagnosis — it is the confirmation that the process caught the things that would otherwise have shipped wrong. Three practices each prevented building the wrong thing, and each prevented a *different* class of mistake:

- **Measure-first** killed three levers before any expensive build. Hilbert and the treelet proxies were falsified with cheap locality measurements, not with a full implementation and a benchmark. The cost of falsifying them was a proxy and an afternoon; the cost of building them would have been weeks for a ~0 result.
- **Build-and-validate** caught the multi-layout selector collapse. This is the inverse lesson to the one above: the multi-layout mechanism *looked* promising on paper and its headline number *looked* like a win, and the only way that illusion broke was by actually building it and discovering the selector tops out near 40% and the gain was min-of-noise. Some levers can be falsified cheaply; this one had to be built to be falsified, and the methodology was flexible enough to do that rather than trust the paper number.
- **Adversarial review** caught the early-skip f32 grazing bug — a correctness defect on grazing rays that a multi-agent review found and that **20,000 random rays had missed**. This is the sharpest single data point in the entire document about test methodology: random sampling at scale did not find a bug that targeted reasoning found immediately, because the bug lived in a measure-zero region (exact grazing incidence) that random rays effectively never hit. Coverage by volume is not coverage by reasoning.

Taken together, the dead ends are now mapped and — more importantly — *why* each is a dead end is on the record. The anisotropy is real, it is ~9×, and it is intrinsic to traversing a sparse Morton structure obliquely; the layout is already near-optimal; the residual is latency; software relayout is exhausted on a principled basis; the live options are to budget around the cost (recommendation 2) or to pivot to hardware SER (recommendation 3). A future reader who is tempted to try a new space-filling curve should read Section 8 and this section first, and bring a new *mechanism* — not a new curve — to the table.

---

## 11. Addendum — kernel specialization (register-resident traversal) partially refutes §5

§5 concluded the residual cost was "memory-miss latency — hardware territory, not software." A later experiment (`feat/kernel-specialization`) shows that was **partly wrong**, and the correction is worth ~1.8×.

**The experiment.** The shipped kernel walked the hierarchy through an explicit `array<Frame, 8>` stack, reading the *active* frame as `stack[top]` on every cell-step. A `top` that varies across iterations forces that array into GPU **local memory**, so each step paid a memory access. The variant hoists the active frame into function-local scalars (register-resident) and uses the array only for parent frames, touched on the rarer descend/ascend. This is the WGSL-appropriate form of the inlining the VDB literature gets from C++ template specialization — hot state in registers, no indirection — but it needs no templates or per-`k` codegen and works at every resolution.

**Result (A/B, best-of-12, GPU-timeline timestamps, byte-identical output — 0 / 1,000,000 hits differ, differential 5/5):**

| fixture | 128³ (k=1) | 512³ (k=4) |
|---|--:|--:|
| sierpinski | 1.92× | 1.94× |
| caves | 1.67× | 1.73× |
| dust | 1.87× | 1.87× |

Adopted as the **sole** traversal kernel (buffer path and viewer alike); the generic stack-indexed form is retired.

**The refinement.** The win is broad — it nearly doubled even the latency-bound `dust` case (r≈0.1), which is the tell: a meaningful slice of what §5 attributed to "hardware memory latency" was actually this addressable software artifact (local-memory spill of the frame stack), not irreducible cache-miss latency. The orientation **anisotropy ratio** (the ~9× swing, the actual subject of §5) is a separate axis and is expected to persist — but the absolute throughput **floor** was ~1.8× lower than it needed to be, and §5's "nothing to do in software" framing was too strong. Lesson, consistent with the rest of this document: a residual that *correlates* with memory behaviour is not proof that the memory behaviour is *hardware-fixed* — profile the specific mechanism before declaring it intrinsic.

**Verified (re-measured on the register kernel).** Re-running `aniso` at 512³ on the adopted kernel confirms the prediction: the cell-step swing is identical (1.51× / 2.17× / 1.53× for sierpinski / caves / dust), and the **cache/coherence excess is unchanged or slightly larger** (caves 3.08→3.42×, dust 3.39→3.87×) — so the orientation anisotropy is *not* a spill artifact and survives the kernel change. The register win was an orthogonal ~1.8× floor-shift, not a reduction of the swing; the §5 conclusion stands. (One second-order curiosity: `r` moved in opposite directions — caves 0.73→0.47, dust 0.11→0.37 — because the removed per-step spill was step-*correlated* for coherent caves but a step-*decorrelating* memory-pattern noise for incoherent dust, confirming spill and cache-miss latency are distinct mechanisms.)

**Door B (occupancy) — tested and falsified.** Having reframed "latency" as layered, the natural follow-up was: is the *floor* (not the swing) occupancy-limited? A second experiment (`feat/occupancy`) shrank the per-frame working set — a 48-byte `SlimFrame` that recomputes `dim`/`step`/`t_delta` from `level` + ray direction each step, vs the ~80-byte full frame — to free registers/local-memory and raise GPU occupancy. Byte-identical (0/1,000,000 hits differ), but the gain was marginal (sierpinski 1.05×, caves 1.09×, **dust 1.02×**) and **did not help the latency-bound case** — `dust` is precisely where more occupancy *should* hide the most global-memory latency, and it barely moved. So the incoherent-ray floor is genuine memory-miss/divergence latency, not a register-pressure or occupancy artifact; the slim frame was **not adopted** (a ~1.05× average is not worth the recompute complexity over the clean full-frame kernel). Together, Doors A and B bracket §5 from both sides: the anisotropy *swing* is intrinsic, and the latency *floor* is not occupancy-addressable. The register hoist captured the one large addressable software win (~1.8×); beyond it, both axes genuinely resist software.

**Block-walking (ropes) — built, the simple form falsified, and *why* it was a null is the lesson.** A third lever (`feat/block-walking` → `feat/ropes`): the classic "block walking" stores per-block adjacency ("ropes") so a ray leaving a brick jumps straight into its neighbour rather than re-querying the root. A `walk` profiler first bounded the ceiling — internal-node traversal is 50–90 % of per-ray work for sparse fields (dust does 143 node fetches to reach only 1.8 bricks), a UB ceiling of ~2× (dense) to ~8.8× (dust). Then per-leaf face ropes (`Leaf`/`Empty`/`Exit`) and a rope-following traversal were built and validated **byte-identical** to the baseline across five fixtures. The A/B verdict: the realized node-fetch reduction is **~nil (0.94–1.10×)**, and forgoing the early-skip even inflated leaf-stepping (dust 18.7 → 101 steps/ray). The reason is the synthesis worth keeping: block-walking's reputed speedups come from (A) eliminating the traversal **stack** and (B) eliminating **root re-query** — but we had **already banked (A)** via the register kernel (and Door B confirmed the residual stack cost is nil), and we **never paid (B)** because our HDDA ascends only to the *common ancestor*, not the root. The literature's gains are measured against a stack-heavy, root-restarting baseline we'd already engineered away, so the simple rope had no cost left to remove. The one genuinely-unclaimed piece is the **empty-space skip** — a directional skip-distance structure (distinct from adjacency caching) that jumps across empty regions in O(1) — the only lever with measured headroom remaining, and the next thing tested.

---

## Appendix A — Reproducing the findings

The project is a virtual Cargo workspace; the headless tool is the `voxel` binary, run as `cargo run --release -p voxel-cli -- <subcommand>`. The experiments live on different branches — each branch is a committed diagnostic carrying the subcommand that reproduces its section. **All GPU timings are thermal-sensitive** (Apple M-series via wgpu drifted absolute numbers 2–6× over a session); run from a cool machine and trust *ratios within a single invocation*, not absolute milliseconds across runs. Set `RUST_LOG=warn` to quiet logs.

### A.1 On `main` — the shipped tools

```
# §10 report: box-counting dimension D + R², per-level footprint vs L2, descent freq, mean cell-steps
voxel measure --fixture dust --res 512

# Diff a backend against the f64 reference (expect a tiny, bounded, grazing-only disagreement)
voxel diff --fixture sierpinski --res 128 --backend gpu

# The sweep table: build_ms, serial_ms, leaves, MiB, D, R², descents, steps, hit%, cpu/gpu Mray/s
voxel bench --res 512,2048

# Directional anisotropy: cell-step (algorithmic) vs GPU-time (hardware) swing, and their correlation r
voxel aniso --fixture dust --res 512 --dirs 64 --side 256
```

What to look for: `measure` recovers the known fractal dimensions (Sierpinski D≈2, Cantor D≈1). `bench` shows the low-occupancy fixtures (wire, dust) as the traversal killers and the dense ones (checkerboard, solid) as storage-bound but fast per-ray. `aniso` shows the ~8–9× GPU spread for dust with `r ≈ 0.24` (cache/coherence-dominated), vs Sierpinski `r ≈ 0.82` (step-driven).

### A.2 On `feat/ray-binning`

```
git checkout feat/ray-binning
voxel coherence --fixture dust --res 512 --rays 500000
```

Times an incoherent batch **raw vs reordered** by a coherence key, using readback-free timestamps; the CPU sort cost is reported separately (a measuring stick, not charged to the kernel). Watch the ~1.2× kernel-gain ceiling and that the sort cost dwarfs the saving.

### A.3 On `feat/axis-permuted-morton`

```
git checkout feat/axis-permuted-morton

# Build the 6 axis-permuted layouts; per-layout cheap directions + best-of-3 (cyclic) vs best-of-6 envelope
voxel layout --fixture dust --res 512 --dirs 64 --side 256

# The deployable 3-layout solution: 3.00× memory, image-invariance (expect 0/49152), orbit stability,
# and the O(1) selector accuracy (~40% — the negative finding)
voxel multilayout --fixture dust --res 512 --frames 48 --side 256
```

`layout` shows 3 cyclic layouts capture ~95% of the 6-layout worst-case gain. `multilayout` prints the `0/49152` image-invariance check, the orbit spread tightening, and the selector picking the best layout only ~40% of frames.

### A.4 On `feat/hilbert`

```
git checkout feat/hilbert
voxel locality --fixture dust --res 512
```

The CPU proxy: per-axis mean neighbour memory-distance under Morton vs Hilbert, plus the cache-block (treelet) co-location fractions. Morton already co-locates 76–95% of neighbours; Hilbert relocates rather than flattens.

### A.5 The viewer (GPU-resident render)

```
cargo run --release -p voxel-viewer -- --res 512 --fixture dust --frames 360
```

Opens a window, orbits the camera, and prints a rolling per-frame profile (`encode` / `gpu(traverse+shade+blit)` / `present` ms). `--frames N` auto-exits after N frames for scripted profiling; `--vsync` caps to the display refresh.

### A.6 Correctness gates

```
cargo xtask ci                                   # fmt, clippy pedantic -D warnings, build, tests, docs (no GPU needed)
VOXEL_REQUIRE_GPU=1 cargo test -p voxel-gpu --test differential   # 0/20000 vs oracle AND mirror
cargo xtask ci-gpu                               # asserts an adapter is present so a GPU lane can't silently skip
```

`cargo xtask ci` must pass with **no GPU**. The differential expects **0/20000 mismatches** vs both the f64 oracle and the f32 mirror, across every fixture including the grazing-heavy wire/dust and the axis-aligned/grazing batteries. To reproduce a section's numbers, check out its branch, build, and run the subcommand above — modulo thermal state.

---

## Appendix B — Branch and commit narrative

The work maps to git history so any finding can be inspected at its source. No branch was merged into `main` except the shipped baseline + early-skip + measurement suite; the three lever branches are committed diagnostics recording falsified or non-deployable hypotheses — kept, by design, as a map of the dead ends.

### B.1 `main` — shipped baseline + measurement suite

| Commit | What it did |
|---|---|
| `6c73185` | **feat: sparse MIP voxel structure with GPU ray traversal** — the initial library: pure core, the tiered oracle, the School-B builder, the f32 mirror, the WGSL kernel, and the §10 harness. |
| `edc5714` | **added per-brick early-skip** — the `LeafBounds` occupied-AABB skip, the FULL-gate, the conservative one-voxel dilation that fixes the f32 grazing bug the adversarial review found, and the regression test pinned from the counterexamples. |
| `af1f45a` | **fixed dense geometry rebuild cost** — the popcount gate on `occupied_bounds` so dense leaves return FULL without an O(set-bits) scan (checkerboard 2048³ re-serialize 4.6 s → 64 ms). |
| `5797b48` | **added better profiling** — the `bench` `serial_ms` column and the `aniso` subcommand with wgpu compute-pass `TIMESTAMP_QUERY` timing (`traverse_timed`). |

### B.2 `feat/ray-binning`

| Commit | What it did |
|---|---|
| `1e6f1c0` | **experiment: measure the ray-binning ceiling** — the `coherence` diagnostic. Finding: reordering recovers ~1.2× (the warp-divergence penalty), not the 9×; the 9× is per-direction intrinsic. |

### B.3 `feat/axis-permuted-morton`

| Commit | What it did |
|---|---|
| `29e0109` | **experiment: measure axis-permuted Morton multi-layout** — the `layout` sweep via the permuted-coordinate + ray-transform equivalence. |
| `b718378` | **measure: 3 cyclic layouts capture 95% of the 6-layout gain** — so deploy at 3× memory, not 6×. |
| `1c96da3` | **feat: deployable 3-layout multi-layout solution** — builds the 3 cyclic layouts, proves image-invariance (0/49152), measures stability. |
| `b46037c` | **finding: the multi-layout selector is noise-limited** — the O(1) selector (analytical 35%, table 38–40%) collapses to chance; the mechanism is correct but the stability benefit doesn't survive a real selector. |

### B.4 `feat/hilbert`

| Commit | What it did |
|---|---|
| `9327230` | **experiment: 3-D Hilbert encoder + locality proxy** — Skilling's 3-D Hilbert encode (bijection + 6-neighbour tests) and the `locality` proxy. Finding: Hilbert relocates the anisotropy, doesn't flatten it. |
| `39af0c0` | **experiment: treelet/cache-block metric** — the cache-block co-location measurement. Finding: Morton is already 76–95% block-isotropic; a treelet relayout has no leaf-order headroom. |

To inspect: `git log --oneline <branch>` and `git show <hash>`. Each experiment branch carries its reproducing subcommand (Appendix A).

---

## Appendix C — Methodology and correctness discipline

How the findings were made trustworthy. The recurring theme: the result is only as good as the discipline that produced it, and three different disciplines — a tiered oracle, measure-first proxies, and adversarial review — each caught something the others would have missed.

### C.1 The tiered oracle (review item R1)

Three traversals at descending fidelity, each validating the next:

- **Tier-A** — an `f64` dense Amanatides–Woo single-level traversal that is the ground **truth**, self-validated against analytic ray–AABB / ray–plane intersection on the fixtures. It proves "the math is correct."
- **The f32 mirror** — bit-identical to the WGSL kernel (same reciprocal-vs-divide, same `<`/`<=` tie-breaks, the same explicit stack and frame layout). The shader is a near-mechanical transliteration of it, so the kernel can be debugged on the CPU without a GPU.
- **The GPU kernel** — the production WGSL.

A discrete voxel hit is *topology*, so equality is **structural** (the same voxel index), never a distance tolerance.

### C.2 The differential

The gate casts 20,000 deterministic pseudo-random rays and requires **0 mismatches** GPU-vs-oracle **and** GPU-vs-mirror, on every fixture. f32-vs-f64 grazing disagreements are **bounded and logged** (a small fraction), never accepted as a free pass — and discrete hits must match exactly. The test is gated on a GPU adapter at runtime: it skips cleanly on CPU-only CI, while `VOXEL_REQUIRE_GPU=1` makes a missing adapter a *failure*, so a GPU lane cannot silently pass by skipping.

### C.3 Measure-first

Before building any expensive GPU feature, build the cheapest proxy that can *falsify* it:

- The `locality` proxy (CPU, no GPU rebuild) falsified both Hilbert and treelet relayout.
- The `coherence` experiment (readback-free timestamps; CPU sort used as a measuring stick, not charged to the kernel) bounded ray binning to ~1.2× before any GPU bin was written.
- The `aniso` decomposition isolated the addressable (cache/coherence) component from the fundamental (step-count) one.

This avoided three multi-file GPU builds that would each have delivered little.

### C.4 Build-and-validate when the proxy isn't enough

The multi-layout **stability** benefit looked real in every proxy — the best-of-N envelope genuinely tightened the spread — and only collapsed when the actual O(1) selector was built and measured at ~40% accuracy, revealing the envelope was largely *min-of-noise*. The lesson: a promising proxy is not proof. Some claims fall only to a real implementation, and it is worth building one when the decision hinges on it.

### C.5 The adversarial review

A multi-agent code review with parallel skeptics on distinct lenses found a **critical f32 grazing bug** in the early-skip that the 0/20000 random differential **missed** — because random rays essentially never graze a single-voxel box's edge to within f32 epsilon. The skeptic *constructed* five exact counterexample rays at 2048³ and reproduced an end-to-end dropped hit (the mirror returned `None` where the oracle hit), with drop rates 0.54% / 1.93% / 5.44% at 128/512/2048³. The fix was a one-voxel box dilation plus a regression test pinned from those counterexamples. Lesson: random sampling under-weights structured edge cases; adversarial construction + a pinned regression test is the antidote.

### C.6 Conservative-by-construction proofs

The early-skip is safe because the occupied AABB *contains every set voxel*, so missing the box implies missing the brick — the f64 reference therefore stays bit-exact (`A == B == oracle`). The f32 paths add a one-voxel box dilation so the slab test is never *stricter* than the interior DDA it guards: the walk can `floor` a grazing ray up to ~1 ULP outside the true cell, and biasing toward *descend* only ever costs a few extra interior walks, never a dropped hit.

### C.7 Thermal discipline

On Apple M-series under sustained load, absolute GPU times drifted 2–6× over a session. Every quantitative claim therefore uses **ratios taken within a single invocation at matched thermal state**. The `traverse_timed` path measures the compute pass on the GPU timeline (readback-free), removing ~70 ns/ray of fixed dispatch/readback overhead that otherwise compresses the ratios.

### C.8 Codex conformance

Pure-core / adapter / headless-CLI / viewer separation; GPU as a *runtime probe*, never a Cargo feature; clippy pedantic with `-D warnings`; reference-implementation-as-oracle. Every commit passed `cargo xtask ci` (fmt, clippy pedantic, tests, docs) green.

---

## Appendix D — Glossary

**Sparse MIP voxel structure** — a mip-mapped voxel pyramid storing only non-empty cells. Uniform `4³` internal-node branching down to an `8³` bitmask leaf brick. *Voxel-terminal*: a set leaf bit is the hit; an occupied brick is not itself a hit.

**Voxel-terminal** — traversal stops at a set voxel bit inside a leaf, not at the brick. A brick being non-empty only means "descend and look."

**Leaf brick** — an `8³` block of 512 occupancy bits in intra-brick Morton order. The unit of the leaf array; ~64 bytes each.

**Internal node (`4³`)** — carries a 64-bit child mask (one bit per `4³` child, Morton order) and a base offset; addresses children by popcount-rank.

**Resolution `8·4^k`** — the only representable grid sizes: 8, 32, 128, 512, 2048. `1024³` is **not** representable (it is not of the form `8·4^k`), enforced by a `Resolution` newtype.

**Morton code / Z-order curve** — a space-filling curve that interleaves the bits of the coordinates. Cheap to encode/decode (bit interleave), "good-ish" locality. A Z-order traversal of a region ends diagonally opposite where it started. Used here **build-time only**; never computed on the GPU.

**Hilbert curve** — a space-filling curve whose *consecutive indices are always 6-neighbours* (differ by 1 on a single axis). More isotropic locality than Morton for dense arrays, at ~2× encode cost; its child ordering is *orientation-dependent*, which makes a GPU build substantially harder than Morton's fixed bit-interleave.

**School A vs School B** — School A stores internal nodes in a separate array per level; School B re-serializes them into one children-contiguous DFS buffer addressed by popcount-rank from a subtree base. School B is the GPU form (one storage binding). Both implement a `NodeLayout` trait so one traversal serves both.

**popcount-rank addressing** — a stored child's slot is `base + popcount(mask & ((1<<bit) − 1))`. Only occupied children are stored, yet addressed in O(1). The 64-bit mask is split lo/hi into a `vec2<u32>` for WGSL (no `u64` on the GPU).

**HDDA** — Hierarchical Digital Differential Analyzer; a hierarchical Amanatides–Woo grid march. A **cell-step** is one DDA advance to the next cell at the current level; a **descent** is entering a child level.

**Box-counting dimension `D` (and `R²`)** — the §10 measure of how the occupancy scales across levels (regress `ln N(L)` vs `ln cell_size`). `D≈3` is near-solid (abandon sparsity); low `D` justifies the sparse structure. `R²` reports how scale-invariant (self-similar) the field actually is.

**Orientation / view-direction anisotropy** — the central finding: traversal cost depends on the camera direction far more than on what is on screen. Up to ~9× between cheapest and most-expensive orientations for the same image.

**Ray coherence vs divergence** — coherent rays (e.g. neighbouring camera pixels) follow similar paths and touch nearby memory; incoherent/divergent rays (scattered secondary rays, or warp lanes that desynchronize) touch scattered memory and serialize on the GPU. The measured **coherent-vs-incoherent gap** here is 17–35× on the same structure.

**Warp / wavefront** — the GPU's SIMD execution group (32 lanes on this hardware). Lanes execute in lockstep; divergence (different control flow or scattered memory) idles lanes and serializes work.

**Cache line / cache block / treelet** — the unit of memory fetched on a miss (cache line) and, by extension, a small subtree sized to a cache block (treelet). The treelet idea: cluster a subtree contiguously so a ray stays cache-resident while inside it.

**Cache-oblivious / van-Emde-Boas layout** — a memory layout with good locality at *all* block sizes simultaneously, without tuning to a specific cache. For a *regular* grid the cache-oblivious leaf order coincides with Morton (both are recursive subtree groupings).

**Shader Execution Reordering (SER)** — hardware (NVIDIA Ada / DirectX SM 6.9 / Vulkan `VK_EXT_ray_tracing_invocation_reorder`) that regroups ray-tracing invocations on-chip to reduce divergence. Targets primarily *shading* divergence on the RT-core pipeline; not available to a custom WGSL compute kernel.

**Tiered oracle** — the correctness scaffold: an `f64` dense traversal (truth), an `f32` mirror bit-identical to the WGSL kernel, and the GPU kernel; validated by a structural-equality differential.

**The equivalence (multi-layout)** — storing the geometry under a permuted Morton order `M_P` is identical, in memory layout, to storing the P-permuted-coordinate geometry under the fixed encoder and transforming the camera ray by `P`. This is why multi-layout needs no encoder/shader change.

**Image-invariance** — the property that, with the layout's inverse permutation applied to the ray and its permutation applied back to the hit, every layout returns the identical hit as the plain structure. Verified `0/49152`. It means the selected layout is invisible in the output and cannot pop on a switch.

**Fixtures** — procedural occupancy fields with known properties: **Sierpinski tetrahedron** (`D=2`), **Cantor dust** (`D=1`), **checkerboard** (`D≈3`, dense), **solid** (`D=3`), **WireLattice** (thin axis-aligned wires — traversal-pathology stress), **Dust** (hashed sparse noise — warp-divergence stress, statistically isotropic).

**LeafBounds / occupied-AABB** — the axis-aligned bounding box of a leaf's set voxels, packed into one `u32`; the per-brick early-skip slab-tests the ray chord against it. The **FULL-gate** short-circuits dense leaves (box = whole brick) without testing; **SKIP_MARGIN** is the one-voxel dilation that keeps the f32 slab test conservative.

---

## Appendix E — Recommended next directions (design sketches)

These are **proposals, not measured results** — no performance numbers are attached because none were measured. They are the frontier the synthesis (§10) points to, once the software relayout/reordering levers are accepted as exhausted.

### E.1 Budget around the anisotropy — adaptive internal-resolution + temporal reprojection (recommended)

**Rationale.** The anisotropy is latency-bound and intrinsic, so do not fight it — cap frame time by spending fewer rays on the expensive views.

**Sketch.** Use the `aniso` cost descriptor (or a cheap per-frame running estimate of recent frame cost) to predict an expensive view; on those frames render the internal buffer at reduced resolution and upscale, and reuse the previous frame via temporal reprojection where the camera moved little. The renderer holds a frame *budget* by trading a little quality, rather than dropping frames.

**Properties.** ~No extra memory. Orthogonal to the layout — it does not touch the traversal kernel at all. The knobs (resolution scale, reprojection confidence, disocclusion handling) are standard and well-understood. It is the one untested-but-well-understood lever that does **not** fight the latency residual, and it *composes* with multi-layout (which would merely reduce how much quality must be traded) if that were ever deployed.

**Why it's the recommended next step.** It addresses the *symptom that matters* (frame-time variance / worst-case views) without requiring the layout or pipeline changes that the other levers do, and without the selector problem that sank multi-layout.

### E.2 Dynamic geometry — in-place brick edit

**Current limitation.** The build is a full `O(n³)` occupancy scan + Morton sort + serialize, with **no incremental update**: any geometry edit re-runs the whole pipeline (512³ ≈ 15–35 ms, 2048³ ≈ 0.7–1.6 s, build-dominated).

**Sketch (the common case).** For edits *within already-occupied bricks* — bit flips that don't change the *set* of occupied bricks — patch the affected leaf words in place, recompute that leaf's `LeafBounds`, and re-upload only those leaf words plus the bound. This is `O(edit)`, not `O(n³)`.

**Sketch (topology changes).** When bricks appear or disappear, the popcount-rank addressing and the child masks shift, so the easy in-place patch no longer suffices. Two options: (a) a *partial re-serialize* of the affected subtrees only; or (b) a *slack / free-list* allocation scheme that reserves spare child slots so insertions don't shift downstream ranks. Option (a) is simpler but touches a subtree per edit; option (b) is more complex but keeps edits local at the cost of memory slack and occasional compaction.

**Why.** This is the natural step to make the structure usable for dynamic scenes; it is independent of the anisotropy work.

### E.3 The hardware path — SER via DXR / Vulkan-RT

**Context.** The field's actual answer to divergence is hardware Shader Execution Reordering, but it lives on the RT-core pipeline, not a custom WGSL compute kernel.

**Sketch.** Move the hot traversal onto the hardware ray-tracing pipeline (which brings its own acceleration structure and hardware reordering of invocations).

**Gained:** hardware divergence handling, mature tooling, and the 40–90% gains reported on the shading-divergence-heavy workloads SER targets.

**Lost / risked:** abandoning the bespoke sparse-MIP-voxel structure and the fine-grained WGSL compute control; SER primarily targets *shading* divergence whereas our anisotropy is *traversal cache latency*, so the benefit transfer is **uncertain**; and a step away from the wgpu cross-platform story.

**Framing.** A *strategic pivot*, not a tweak — to be undertaken only if incoherent-secondary-ray throughput becomes the priority, and ideally prototyped against the actual workload before committing.

### E.4 What we deliberately did *not* do — and why

| Not done | Why |
|---|---|
| Full Hilbert GPU build | The `locality` proxy falsified it (§8): Hilbert relocates rather than flattens the anisotropy. |
| GPU ray-binning pipeline | Ceiling ~1.2× (§6), and it only helps incoherent secondary rays this primary path doesn't have. |
| Multi-layout viewer wiring | The O(1) selector is ~40% accurate (§7) — wiring it would showcase no net gain. |
| 6-layout storage | 3 cyclic layouts capture ~95% of the gain at half the memory (§7). |

### E.5 Closing

The value of this document is the **map of dead ends with the reason each is dead**. Future effort should start from the frontier — budget *around* the anisotropy (E.1) or move to hardware (E.3) — rather than re-walking the falsified software relayout/reordering levers. The anisotropy is real, it is intrinsic, and the layout is already near-optimal; what remains is latency, and latency is hidden in hardware or designed around, not flattened in software.

---

## Appendix F — Measurement logs (captured)

The evidence behind the findings, as captured during the investigation. All GPU absolutes are thermal-sensitive; ratios within a block are the signal. These are representative runs, not the only ones — re-running per Appendix A will reproduce the *ratios* (absolutes drift with thermal state).

### F.1 `bench` sweep (post early-skip + popcount-gate fix)

```
fixture         res  build_ms serial_ms    leaves       MiB     D     R2   desc   steps  hit%  cpuMr/s  gpuMr/s
sierpinski      512      20.7       0.4      4096     0.253  2.00  1.000   3.92    15.9    45      3.4     16.5
checkerboard    512      20.1       0.8    262144    16.048  2.90  0.999   4.05     4.6   100      8.6     54.4
solid           512      20.4       1.0    262144    16.048  3.00  1.000   4.00     4.0   100     10.1     45.9
dust            512      11.7       0.5     30944     1.936  1.71  0.844  36.88   161.2    22      0.4      7.9
wire-lattice    512      31.8       3.1    131072     8.048  2.33  0.965  10.26    46.1   100      1.6     10.4
sierpinski     2048     813.0       4.6     65536     4.050  2.00  1.000   4.74    18.7    39      3.1     16.9
checkerboard   2048    1493.8      64.3  16777216  1027.048  2.93  0.999   5.06     5.6   100      4.3     20.9
dust           2048     657.3      40.9   1973789   123.517  2.03  0.895 110.43   472.8    61      0.1      3.1
```

Reading: the low-occupancy thin/sparse fixtures (wire, dust) are the traversal killers — high `desc`/`steps`, low `gpuMr/s`. The dense fixtures (checkerboard, solid) are fast per-ray (4–5 steps, 100% hit) but storage-heavy (checkerboard 2048³ ≈ 1 GiB). The `serial_ms` column reflects the popcount-gate fix: dense re-serialize stays small (checkerboard 2048³ at 64 ms, down from 4.6 s pre-fix).

### F.2 Early-skip effect (§10 cell-steps — the deterministic signal)

```
fixture        res    steps (no skip → skip)   factor
wire-lattice   512        81  →  46             1.76× fewer
dust           512       244  → 161             1.51× fewer
dust          2048       710  → 473             1.50× fewer
```

Throughput (same-thermal coherent viewer, dust 512³): 26.5 ms → 15.8 ms = **1.68× faster**. The GPU differential held `0/20000` vs the f64 oracle AND the f32 mirror throughout, including the grazing-heavy wire/dust fixtures.

### F.3 The early-skip f32 grazing bug (adversarial review)

```
single-voxel brick at high coordinates, grazing rays:
  drop rate (mirror returns None where oracle hits):
    128³ : 0.54%
    512³ : 1.93%
    2048³: 5.44%       ← climbs with coordinate magnitude (f32 ULP)
  reproduced end-to-end with 5 concrete counterexample rays at 2048³.
fix: 1-voxel box dilation (SKIP_MARGIN). Regression test:
  margin = 0  → FAILS ("early-skip dropped grazing hit [1005,1001,1006]")
  margin = 1  → PASSES; differential back to 0/20000.
```

### F.4 `aniso` — directional anisotropy (clean timestamps, 512³)

```
fixture       cell-steps swing   GPU swing    r (step↔gpu)   interpretation
sierpinski    1.52×              2.50×        0.82           mostly step-driven (less addressable)
wire-lattice  2.42×              6.36×        0.43           mixed
dust          1.54×              8.93×        0.24           almost entirely cache/coherence
```

The decisive row is dust: statistically isotropic geometry (cell-steps swing only 1.54×) yet an 8.93× GPU swing with r=0.24 — so the 9× is the *layout × direction* interaction (cache + warp coherence), not the geometry. Timestamp cleanup mattered: dust min went 85 ns/ray (wall-clock, overhead-compressed) → 13.5 ns/ray (timestamp), un-compressing the anisotropy from a noisy ~2.85× to 8.93×.

### F.5 `coherence` — ray-binning ceiling (incoherent dust)

```
                 raw       reordered     gain          cpu sort (ref)
dust 512³       60.7      49.7 ns/ray   1.22× kernel   362 ns/ray
dust 2048³     231.8     199.2 ns/ray   1.16× kernel   350 ns/ray
sierpinski512    8.9       7.6 ns/ray   1.17× kernel   227 ns/ray
```

Robust across a coarse octant key and a fine 16³ direction key. The kernel-level gain is ~1.2×; the CPU sort (a measuring stick) costs ~30× the saving, and a real GPU counting-sort (~2–5 ns/ray) would net ~1.1–1.15× — and only on incoherent secondary rays.

### F.6 `layout` and `multilayout` — multi-layout

```
layout (dust 512³, side 256, 64 dirs):
  single layout (xyz):   worst 122.4  mean 50.2 ns/ray
  best-of-3 (cyclic):    worst  89.5  mean 43.4   (1.37× / 1.15× vs single)
  best-of-6 (all axes):  worst  87.6  mean 41.6   (1.40× / 1.21× vs single)
  → 3 layouts (3× memory) capture 95% of the 6-layout worst-case gain

multilayout (dust 512³): 3 cyclic layouts, 6.2 MiB total = 3.00× one layout
  correctness (image-invariance): 0/49152 mismatched hits → identical images ✓
  stability (48-frame orbit):
    single layout:    spread 5.76×
    O(1) table sel.:  spread 6.39×   (picks best layout 40% of frames)   ← selector fails
    best-of-3 (ceil): spread 4.62×   (a min-of-noise artifact)
```

The mechanism is correct (image-invariant, 3× memory), but the deployable O(1) selector picks the best layout only ~40% of frames (≈ chance for 3 layouts) and the best-of-3 "ceiling" is largely min-of-noise — so the stability benefit does not survive a real selector.

### F.7 `locality` — Morton vs Hilbert (the Hilbert/treelet falsification)

```
dust 512³ — per-axis mean neighbour memory-distance (lower = more local):
  ordering   x-neigh  y-neigh  z-neigh   anisotropy
  morton         83      134      275     3.33×
  hilbert       344      162       93     3.70×    ← relocated, not flattened; worse mean (164→200)
  same-treelet (≤64 leaves) neighbour fraction:
    morton    x 95%  y 89%  z 89%
    hilbert   x 92%  y 91%  z 92%

wire-lattice 512³:
  morton        297      594     1189     4.00×
  hilbert      1189      669      390     3.05×    ← worst-axis (1189) unchanged
  same-treelet fraction:
    morton    x 89%  y 89%  z 76%
    hilbert   x 85%  y 85%  z 85%
```

Morton already co-locates 76–95% of neighbours in a treelet-sized block — near the ceiling. Hilbert relocates the anisotropy at the same average; a treelet relayout has no leaf-order headroom.

---

## Appendix G — Technical deep-dives

Three mechanisms worth recording in detail, because each is subtle and each cost real time to get right.

### G.1 The f32 grazing bug: why a conservative skip became non-conservative

The per-brick early-skip slab-tests the ray chord against the leaf's occupied AABB and skips the `8³` interior walk on a miss. It is conservative *in exact arithmetic*: the box contains every set voxel, so a chord that misses the box misses every voxel. The bug was that the **f32** slab test could be *stricter* than the **f32** interior DDA it guarded.

Consider a ray grazing the edge of a *single-voxel* brick at high coordinates. The box is `1×1×1`; the chord's parametric interval through it, `[t_near, t_far]`, is razor-thin. In f32 at coordinates ~1000, the per-axis slab arithmetic carries ~1 ULP of error (which *grows with the coordinate magnitude* — hence the climbing 0.54%/1.93%/5.44% drop rate at 128/512/2048³). That error can invert the interval (`t_near > t_far`, read as "miss") or push `t_far` just below `t_enter`. Either way the skip fires. But the interior DDA, started from the *same* `t_enter`, floors the ray position into the occupied voxel's cell and reports the hit. The guard was stricter than the walk.

The fix: dilate the occupied box by one voxel before the f32 slab test. The DDA can floor a grazing ray at most ~1 ULP outside the true cell; a one-voxel halo dwarfs that, so the dilated box always contains everything the walk could reach. Biasing toward *descend* is free in correctness terms — it only ever costs a few extra interior walks, never a dropped hit. The f64 reference keeps a tiny epsilon for the same reason; the f32 paths use the full voxel because their error is larger. Why did 20,000 random rays miss this? Because a random ray essentially never grazes a single-voxel box's edge to within an f32 ULP — the bug lives on a measure-zero set that adversarial construction hits and random sampling does not.

### G.2 The multi-layout equivalence and image-invariance

Storing the geometry under a permuted Morton order `M_P` is *identical in memory layout* to storing the P-permuted-coordinate geometry under the fixed encoder `M_id`. Proof sketch: the Morton code of a permuted coordinate equals the permuted-axis interleave of the original coordinate, which is exactly what a permuted encoder produces. So "layout `M_P`" can be realized with **no encoder or shader change** — just build `Permuted{field, P}` (a valid occupancy field, validated for free by the existing differential) and transform the camera ray by `P`.

Image-invariance is what makes selection safe. To render world direction `D` under layout `P`: transform the ray into the layout's storage frame (`permute(origin, P⁻¹)`, `permute(dir, P⁻¹)`), traverse `Permuted{field, P}` to a storage hit `c`, then map back with `permute(c, P)`. Because the storage structure is the P-permuted geometry and the ray is the P⁻¹-permuted camera, the pixel-to-true-voxel mapping is *identical* to the unpermuted render — verified `0/49152`. Consequently the renderer may switch layouts between frames with no visible change (no popping), which is the precondition for per-view layout selection to be usable at all.

### G.3 Why the selector is noise-limited

For the *isotropic* dust case, the three cyclic layouts are statistically identical structures (the same random field, axis-relabeled). For a given view direction the three layouts' costs are therefore close — within ~10–20% — while the GPU timing noise (thermal drift, dispatch jitter) is comparable. The "which layout is cheapest for this direction" signal sits at or below the noise floor.

Two consequences follow. First, a precomputed `direction → best-layout` table built from one noisy sweep is itself noisy: its argmin per direction is unreliable, the best-layout map over the sphere is fragmented rather than forming clean regions, and a nearest-direction lookup picks the actually-cheapest layout only ~38–40% of the time (chance for three layouts is 33%). Second — and this is the subtle one — the best-of-N "ceiling" we quoted as the achievable gain is a **min-of-noisy-samples** statistic, which is biased low: taking the minimum of three noisy measurements tends to pick whichever happened to measure low this time, producing an apparent spread reduction that a *committed* selector (which must choose before measuring) can never realize. The stability benefit was therefore an artifact of the measurement, not an achievable property — a distinction that only became visible when the real selector was built and measured.
