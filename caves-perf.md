# Caves fixture — performance data

A focused performance characterization of the `caves` fixture (domain-warped
ridged multifractal — see [`NoiseField::caves`](crates/voxel-core/src/fixtures.rs)),
extending the orientation-anisotropy story in [FINDINGS.md](FINDINGS.md) §5 to an
*organic, mostly-coherent* volume.

**Setup.** Apple M4 Pro (Metal via wgpu), 2026-06-15. GPU times use readback-free
compute-pass timestamps. As everywhere in this project, **absolute GPU times are
thermal-sensitive — trust ratios within a single invocation, not absolutes
across runs.**

---

## 1. What `caves` is

A 3-D noise field thresholded into occupancy: 5-octave ridged multifractal
gradient noise, domain-warped (amplitude 0.45) so the isosurface bends into
swirling, interconnected veins and caverns with overhangs. Frequency scales with
the grid (resolution-independent macro shape; higher resolution adds octave
detail). Deterministic via a seed.

Measured character (`measure`/`bench`):

- **Box-counting dimension D ≈ 2.76** (`R² > 0.99`) — a *new regime* for this
  project, distinct from every prior fixture (Sierpinski 2.0, dust ~1.7, wire
  2.3, checkerboard 2.9). A solid-with-tunnels mass: nearly surface-filling but
  perforated.
- **~24 % fill, 100 % hit** — camera rays almost always strike the mass.
- **Coherent despite irregular geometry** — neighbouring rays hit at similar
  depths (solid blobs, not scattered voxels), so GPU warps stay in lockstep.

---

## 2. Build / footprint / throughput

`bench`, one thermal run, with the other fixtures for context.

| fixture | res | build ms | serial ms | leaves | MiB | D | R² | mean desc | mean steps | hit % | GPU Mray/s |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **caves** | 128 | 58.6 | 0.1 | 3 712 | 0.23 | 2.71 | 0.996 | 4.23 | 15.3 | 100 | 19.8 |
| **caves** | 512 | 2 425 | 3.0 | 142 078 | 8.72 | 2.76 | 0.999 | 6.70 | 23.6 | 100 | **24.2** |
| perlin | 512 | 793 | 1.6 | 100 503 | 6.17 | 2.76 | 0.999 | 8.36 | 31.3 | 100 | 17.4 |
| dust | 512 | 13.6 | 0.6 | 30 944 | 1.94 | 1.71 | 0.844 | 36.88 | 161.2 | 22 | 9.9 |
| sierpinski | 512 | 17.1 | 0.4 | 4 096 | 0.25 | 2.00 | 1.000 | 3.92 | 15.9 | 45 | 19.2 |
| wire-lattice | 512 | 39.1 | 3.7 | 131 072 | 8.05 | 2.33 | 0.965 | 10.26 | 46.1 | 100 | 8.6 |
| checkerboard | 512 | 25.8 | 1.6 | 262 144 | 16.05 | 2.90 | 0.999 | 4.05 | 4.6 | 100 | 47.1 |

Reading:

- **Throughput.** At 512³, `caves` is the **fastest non-dense fixture** (24.2
  Mray/s) — ahead of `sierpinski` (19.2), `perlin` (17.4), `dust` (9.9), and
  `wire` (8.6), behind only the trivially-shallow dense fixtures (checkerboard
  47, solid 42). Its coherence (100 % hit, similar depths) keeps warps busy
  despite the irregular geometry. The per-brick early-skip helps: `caves`
  averages 23.6 cell-steps vs `dust`'s 161.
- **Build cost.** The one expense: 2.4 s at 512³ — domain warp triples the
  per-voxel noise cost vs `perlin` (0.8 s). This is a one-time build (the viewer
  pays it once at startup); traversal is unaffected. The build is the only
  reason to filter `caves` out of large-resolution sweeps.
- **Footprint.** 8.7 MiB at 512³ — between `dust` (1.9) and `wire`/`checkerboard`
  (8–16). Scales ~64× per resolution doubling-squared as expected for a D≈2.76
  field.

---

## 3. Orientation anisotropy

The project's central finding (FINDINGS §5) is that traversal cost depends on
camera direction far more than on what is on screen. `aniso` decomposes the swing
into an *algorithmic* part (cell-steps — the DDA work, layout-addressable) and a
*hardware* part (GPU ns/ray — steps + cache + warp coherence), and reports their
correlation `r`. `caves` vs the two reference fixtures at the spectrum's ends:

512³, 64 directions (Fibonacci sphere), 256² rays/direction.

| fixture | cell-step swing | **GPU swing** | r (step↔gpu) | cache/coherence excess | character |
|---|--:|--:|--:|--:|---|
| sierpinski | 1.51× | 2.33× | **0.90** | 1.54× | step-driven |
| **caves** | **2.17×** | **6.69×** | **0.73** | **3.08×** | step-driven **+** cache |
| dust | 1.53× | 5.17× | 0.11 | 3.39× | cache-dominated |

Per-fixture detail (min / max / mean):

```
caves       cell-steps  6.4 / 14.0 / 10.7   gpu ns  8.7 / 58.1 / 38.7   r=0.73
dust        cell-steps 32.9 / 50.3 / 43.7   gpu ns 21.7 /112.4 / 50.7   r=0.11
sierpinski  cell-steps  4.3 /  6.4 /  5.3   gpu ns  9.5 / 22.0 / 17.6   r=0.90
```

(Cell-step counts differ from the §2 `bench` column — `aniso` casts orthographic
ray batches per direction, `bench` a single camera; the within-`aniso` ratios are
the signal.) The cheapest direction is axis-aligned (`[0, -0.14, -0.99]`, i.e.
near −Z) for all three — axis-aligned views minimise DDA steps and maximise
coherence.

### Interpretation

- **`caves` shows a strong 6.69× orientation swing** — larger than `dust`'s
  5.17×, despite `caves` being far more GPU-efficient on average. Organic ≠
  isotropic in cost.
- **It sits at r = 0.73 — toward the *step-driven* end** (Sierpinski 0.90), not
  the cache-dominated end (dust 0.11). And it has the **largest algorithmic swing
  of the three (2.17× cell-steps)**: the warped tunnels genuinely cost more DDA
  steps from some angles than others.
- **Implication for the layout lever.** FINDINGS showed the axis-permuted
  multi-layout was noise for `dust` because dust's anisotropy was ~all cache
  latency (r≈0.1–0.24), which the layout cannot touch. For `caves`, a real
  *addressable* algorithmic fraction exists (r=0.73, 2.17× step swing) — so the
  layout lever would have **more purchase here than it did for dust**. But the
  **~3× cache/coherence excess remains** — the irreducible hardware part the
  whole investigation converged on. `caves` is a fixture where the lever is worth
  re-measuring; it is not a fixture where the lever escapes the latency floor.

---

## 4. 2048³ scaling

`bench` and `aniso` at the largest representable resolution (build took 155 s —
the domain-warp noise scan over 8.6 G voxels; `serial_ms` stays at 68 ms thanks
to the dense-rebuild popcount gate).

```
bench:  build 154 880 ms  serial 67.7 ms  leaves 5 585 909  342.7 MiB
        D 2.79  R²0.999  desc 9.55  steps 32.2  hit 100%  GPU 10.0 Mray/s
aniso:  cell-steps 8.8 / 19.2 / 14.8  (swing 2.18×)
        gpu ns/ray 8.1 / 82.2 / 26.3  (swing 10.14×)   r = 0.49   excess 4.65×
        cheapest [0,-0.14,-0.99] (8.1 ns);  priciest [-0.83,-0.36,+0.43] (82.2 ns)
```

Cross-resolution — the important result:

| res | leaves | MiB | GPU Mray/s | step swing | **GPU swing** | **r** | cache excess |
|---|--:|--:|--:|--:|--:|--:|--:|
| 512³ | 142 k | 8.7 | 24.2 | 2.17× | 6.69× | **0.73** | 3.08× |
| 2048³ | 5.59 M | 342.7 | 10.0 | 2.18× | **10.14×** | **0.49** | 4.65× |

### What scaling reveals

- **The anisotropy gets *worse* with resolution, not better.** The GPU swing
  grows 6.69× → **10.14×** and the cache/coherence excess grows 3.08× → 4.65× as
  the structure (8.7 MiB → 343 MiB) outgrows cache.
- **The algorithmic swing is invariant** (2.17× → 2.18×): cell-step variation is
  a property of the *geometry/direction*, independent of resolution — exactly as
  expected for a scale-stable noise field.
- **`r` falls (0.73 → 0.49):** so the *fraction* of the swing that is algorithmic
  (layout-addressable) shrinks as resolution rises — the growth is entirely in
  the cache/latency term. Throughput halves (24 → 10 Mray/s) for the same reason.
- **Conclusion.** Even `caves` — the most step-driven noise fixture, the one where
  the layout lever looked most promising at 512³ — **trends toward the
  `dust` (latency-bound) regime as it exceeds cache.** The layout lever's purchase
  *shrinks* with scale. This reinforces FINDINGS' central result: the residual is
  memory-miss latency, it is intrinsic, and it *worsens* with structure size —
  hardware territory, not software.

---

## 5. Reproduce

```
voxel bench  --res 128,512 --fixtures caves          # footprint / throughput
voxel measure --fixture caves --res 512              # D, per-level footprint, descent freq
voxel aniso  --fixture caves --res 512 --dirs 64 --side 256   # orientation swing + r
voxel diff   --fixture caves --res 128 --backend gpu # correctness (0/20000 vs oracle)
cargo run --release -p voxel-viewer -- --res 512 --fixture caves   # fly through it
```
