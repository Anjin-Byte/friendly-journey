# GPU counter capture — confirming the bottleneck before building

This is the **measure-first gate** for the traversal-speedup question (FINDINGS
§11). Every layout/work-reducing lever we tried failed, and the one win (the
register kernel, ~1.8×) removed a real *memory* indirection — which triangulates
to *"latency-bound on the cold cross-block **leaf** miss"* (Aila & Laine,
HPG 2010). But that bound has **never been measured with a GPU counter** — it is
inferred from the anisotropy correlation and a stack of falsified experiments.
The shipped `TIMESTAMP_QUERY` only brackets a pass begin→end (total ns); it is
blind to *why* the kernel is slow.

One counter capture (~1 day) either **greenlights exactly one engineering
family** (narrow the cold leaf dependent-fetch chain) or **refutes the premise**
and saves a multi-week misdirected build. Do this before building anything
memory-aimed.

## 1. Run the capture target

The `capture` subcommand finds the worst-anisotropy orientation and loops the
traversal kernel there — a stable, isolated, repeating workload to profile.

```sh
# 512³ — fits L2; the baseline cold-miss case.
cargo run --release -p voxel-cli -- capture --fixture dust --res 512  --iters 2000

# 2048³ — 123 MiB, spills L2 / SLC, where cold leaf misses should be MAXIMAL.
cargo run --release -p voxel-cli -- capture --fixture dust --res 2048 --iters 2000
```

`dust` is the decisive case: statistically isotropic geometry (cell-step swing
~1.5×) but ~9× GPU swing, so its expensive orientation is *pure* cache/coherence
cost, not algorithmic. Bump `--iters` if your profiler needs a longer sample
window (the harness prints the elapsed wall-time). Capture under controlled,
cool thermal state — but note the counters below are **ratios/percentages**,
far more thermally robust than the absolute-ns path the project distrusts.

## 2. Capture the counters

**Easiest — write a trace document directly (recommended).** `voxel capture
--gputrace` drives `MTLCaptureManager` itself and writes an Xcode-openable
`.gputrace` (a few dispatches at the worst orientation) — no attaching, no
`xctrace`, no scheme. It needs `METAL_CAPTURE_ENABLED=1`, which the make target
sets:

```sh
make gputrace RES=2048           # writes dust-2048.gputrace and opens it in Xcode
make gputrace RES=512
```

Open it (the make target does this for you) and you land in Xcode's GPU debugger
with the full counter set **and** the shader profiler — the per-load view that
distinguishes `leaf_words` from `nodes`. This is the path to the §3/§4 reads.

The external routes below still work if you prefer them:

- **Xcode → Debug → Capture GPU Workload** — attach to the running `voxel`
  process (you may need `METAL_CAPTURE_ENABLED=1` in the environment). Good for
  the per-pipeline "Performance" / shader-cost view.
- **Instruments → Metal System Trace** (or the *Game Performance* template) —
  record a few seconds while the harness loops, then inspect the compute
  encoder. This samples occupancy, limiters, and bandwidth over time — the data
  we actually want.

**Inspect the `traverse` *compute* encoder.** The harness's per-iteration
result readback is a separate copy/blit command; Instruments attributes counters
per encoder, so the compute pass's numbers are clean — ignore the readback.

## 3. The four reads (and the binding map)

A single root→voxel descent reads from **three separate storage bindings**, with
very different temperatures:

| binding | data | size | predicted temperature |
|---|---|---|---|
| `nodes` @0 | `GpuNode` (mask_lo, mask_hi, child_base) | 12 B | **warm** — root + near-root read by *every* ray, ~L1-pinned |
| `leaf_bounds` @2 | packed occupied-AABB, the early-skip probe | 4 B/leaf | warm-ish |
| `leaf_words` @1 | the 8³ occupancy brick | **64 B** | **cold** — one scattered, ray-unique brick per hit; the asserted culprit |

Read, in priority order:

1. **buffer-read L2 / last-level-cache hit rate, attributed per binding** — the
   decisive read. Is the miss concentrated on **`leaf_words` @1** (the predicted
   cold-miss culprit) vs `nodes` @0 (predicted warm) vs `leaf_bounds` @2?
2. **Top/Bottom limiter % + ALU-active vs memory-active %** — is the kernel
   **memory-limited**, or ALU/instruction-limited?
3. **Occupancy %** — confirm (or refute) Door B's verdict that the floor is *not*
   occupancy-addressable. If occupancy is already high and it's still slow, more
   warps won't help.
4. **Shader-execution stall attribution** — are stalls **memory-wait**, or
   instruction/sync/barrier?

## 4. Go / no-go

| Capture says | Verdict | Next |
|---|---|---|
| `leaf_words` @1 has a **poor L2 hit rate** AND the kernel is **memory-limited** | Premise **confirmed** | Greenlight **cold-leaf working-set restructuring** (FINDINGS rank 2): fewer distinct cold 128 B lines per leaf descent, cold line fetched first, shorten the dependent-fetch chain. **Do not** fold `leaf_bounds` into the leaf record — that kills the early-skip cheap-probe (the shipped ~1.51× dust win). |
| Kernel is **ALU / instruction-bound**, or stalls are not memory-wait | Premise **refuted** | Retire the cold-leaf framing. Redirect to the ALU hotspot (e.g. the 6-bit popcount-rank, the per-step `tMax` recompute). |
| **Occupancy-limited** (low occupancy, memory idle) | Partial | Run the **workgroup-size sweep** (rank 4) — the render path is hard-coded at 64 / (8,8). Cheap knob; reads out of the same capture. |
| `nodes` @0 (not `leaf_words` @1) is the miss | Premise **refuted** differently | Node-stream locality matters after all — reconsider node packing (otherwise a predicted dead-end). |

Whatever it says, it converts the bound from inference to fact. **Record the
capture result in FINDINGS §11** — confirmed or refuted, it retires (or
validates) the premise the whole strategy now rests on.

## 5. Independent of this capture

**Temporal reprojection + adaptive internal-resolution** (FINDINGS rank 3) does
not fight the latency — it amortizes traversal across frames and drops
resolution on the oblique views the aniso cost descriptor already flags. It can
be built in parallel regardless of what the capture says.
