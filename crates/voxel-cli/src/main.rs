//! `voxel` — headless CLI for the sparse MIP voxel structure.
//!
//! Orchestration only (argument parsing, I/O, reporting); all domain logic
//! lives in [`voxel_core`], the GPU path in [`voxel_gpu`], and mesh voxelization
//! in [`voxelizer`]. Backend selection is a runtime `--backend cpu|gpu|auto`
//! flag, never a Cargo feature.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use glam::DVec3;
use voxel_core::fixtures::{Checkerboard, Dust, NoiseField, OctantFractal, Solid, WireLattice};
use voxel_core::{
    Edit, Ray, Resolution, SchoolBBuffer, SparseTree, VoxelCoord, brush_voxels, measure,
    mirror_traverse,
};
use voxel_gpu::{GpuContext, GpuError, GpuRenderer, GpuTraverser};
use voxelizer::loader::load_mesh;
use voxelizer::reference_cpu::voxelize_surface_cpu;
use voxelizer::{
    GpuVoxelizer, GpuVoxelizerConfig, MeshInput, TileSpec, VoxelGrid, VoxelOccupancy, VoxelizeOpts,
};

#[derive(Parser)]
#[command(name = "voxel", about = "Sparse MIP voxel structure — headless tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build the structure and run the §10 measurements (dimension D, per-level
    /// footprint, descent frequency).
    Measure(MeasureArgs),
    /// Cast rays through the structure and diff a backend against the f64
    /// reference traversal.
    Diff(DiffArgs),
    /// Sweep fixtures × resolutions and tabulate build, size, §10, and
    /// CPU/GPU traversal throughput — a body of performance data.
    Bench(BenchArgs),
    /// Sweep view directions and report how traversal cost varies with
    /// orientation — the algorithmic (cell-step) anisotropy vs the hardware
    /// (GPU-time) anisotropy, decomposing how much of the swing is addressable.
    Aniso(AnisoArgs),
    /// Time incremental voxel edits (in-place vs topology) against a full
    /// rebuild — the cost of dynamic geometry.
    Edit(EditArgs),
    /// Loop the traversal kernel at its worst orientation as a stable target for
    /// an external GPU counter capture (Xcode / Instruments) — the measure-first
    /// step that confirms whether the floor is the cold leaf-miss latency. See
    /// `CAPTURE.md`.
    Capture(CaptureArgs),
    /// Voxelize a mesh file (`.gltf`/`.glb`, `.obj`, `.stl`) into the sparse
    /// structure and report it; `--diff` cross-checks the GPU voxelizer against
    /// the CPU oracle.
    Voxelize(VoxelizeArgs),
}

#[derive(Args)]
struct VoxelizeArgs {
    /// Mesh file to voxelize. Format is chosen by extension (`.gltf`/`.glb`,
    /// `.obj`, `.stl`).
    #[arg(long)]
    input: PathBuf,
    /// Grid resolution per axis (must be `8·4^k`: 8, 32, 128, 512, 2048, …).
    #[arg(long, default_value_t = 128)]
    res: u32,
    /// Voxels of margin to leave around the mesh's bounding box when fitting it
    /// into the cubic grid.
    #[arg(long, default_value_t = 2.0)]
    padding: f32,
    /// Which voxelizer to run.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    backend: Backend,
    /// Run BOTH the CPU oracle and the GPU path and report their agreement (the
    /// Reference-as-Oracle conservative-superset check) instead of a plain build.
    #[arg(long)]
    diff: bool,
}

#[derive(Args)]
struct MeasureArgs {
    /// Occupancy fixture to build from.
    #[arg(long, value_enum, default_value_t = Fixture::Sierpinski)]
    fixture: Fixture,
    /// Grid resolution per axis (must be `8·4^k`: 8, 32, 128, 512, 2048, …).
    #[arg(long, default_value_t = 512)]
    res: u32,
    /// Number of camera rays for the descent-frequency measurement.
    #[arg(long, default_value_t = 4000)]
    rays: u64,
}

#[derive(Args)]
struct DiffArgs {
    /// Occupancy fixture to build from.
    #[arg(long, value_enum, default_value_t = Fixture::Sierpinski)]
    fixture: Fixture,
    /// Grid resolution per axis.
    #[arg(long, default_value_t = 128)]
    res: u32,
    /// Number of rays to cast.
    #[arg(long, default_value_t = 20_000)]
    rays: u32,
    /// Which traversal backend to diff against the f64 reference.
    #[arg(long, value_enum, default_value_t = Backend::Auto)]
    backend: Backend,
}

#[derive(Args)]
struct BenchArgs {
    /// Fixtures to sweep (comma-separated; default: all).
    #[arg(long, value_enum, value_delimiter = ',')]
    fixtures: Vec<Fixture>,
    /// Resolutions to sweep (comma-separated).
    #[arg(long, value_delimiter = ',', default_value = "128,512")]
    res: Vec<u32>,
    /// Rays per throughput measurement.
    #[arg(long, default_value_t = 200_000)]
    rays: u32,
}

#[derive(Args)]
struct AnisoArgs {
    /// Occupancy fixture to analyze.
    #[arg(long, value_enum, default_value_t = Fixture::Sierpinski)]
    fixture: Fixture,
    /// Grid resolution per axis (`8·4^k`).
    #[arg(long, default_value_t = 512)]
    res: u32,
    /// Number of view directions on the sphere (Fibonacci lattice).
    #[arg(long, default_value_t = 64)]
    dirs: usize,
    /// Orthographic batch is `side²` parallel rays per direction.
    #[arg(long, default_value_t = 128)]
    side: u32,
    /// Print every direction's row, not just the summary.
    #[arg(long)]
    verbose: bool,
}

#[derive(Args)]
struct EditArgs {
    /// Occupancy fixture to build, then edit.
    #[arg(long, value_enum, default_value_t = Fixture::Caves)]
    fixture: Fixture,
    /// Grid resolution per axis (`8·4^k`).
    #[arg(long, default_value_t = 512)]
    res: u32,
    /// Number of random voxel toggles to time for the per-class costs.
    #[arg(long, default_value_t = 20_000)]
    edits: u32,
    /// Brush radius (voxels) for the stamp benchmark — a `~(2r+1)³` sphere.
    #[arg(long, default_value_t = 4)]
    brush_radius: u32,
    /// Number of brush stamps to time.
    #[arg(long, default_value_t = 200)]
    stamps: u32,
    /// Also print a compact fixture×resolution sweep (CPU only).
    #[arg(long)]
    sweep: bool,
}

#[derive(Args)]
struct CaptureArgs {
    /// Fixture to capture (the cold-miss signal is strongest on sparse `dust`).
    #[arg(long, value_enum, default_value_t = Fixture::Dust)]
    fixture: Fixture,
    /// Grid resolution. Capture both 512 and 2048 — 2048³ (123 MiB) spills L2,
    /// where cold leaf misses should be maximal.
    #[arg(long, default_value_t = 512)]
    res: u32,
    /// Rays per axis in the orthographic capture batch (`side²` rays/dispatch).
    #[arg(long, default_value_t = 1024)]
    side: u32,
    /// Directions searched to find the worst (most expensive) orientation.
    #[arg(long, default_value_t = 32)]
    search_dirs: usize,
    /// Kernel dispatches to loop — the capture window. Bump it if your profiler
    /// needs a longer sample; the harness prints the elapsed wall-time.
    #[arg(long, default_value_t = 1000)]
    iters: u32,
    /// Instead of looping for an external profiler, write a `.gputrace` document
    /// (a few dispatches) you open directly in Xcode — full counters + shader
    /// profiler, no attaching. Needs `METAL_CAPTURE_ENABLED=1` (see `make
    /// gputrace`).
    #[arg(long)]
    gputrace: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum Fixture {
    /// Sierpinski tetrahedron, `D = 2`.
    Sierpinski,
    /// Cantor dust, `D = 1`.
    Cantor,
    /// 3-D checkerboard, `D ≈ 3`.
    Checkerboard,
    /// Solid, `D = 3`.
    Solid,
    /// Thin 3-D wireframe lattice — traversal-pathology stress (#2).
    WireLattice,
    /// Sparse hashed noise — warp-divergence stress (#4).
    Dust,
    /// Perlin fBm isosurface — smooth organic clouds/caves.
    Perlin,
    /// Domain-warped ridged multifractal — interconnected veins/caverns.
    Caves,
}

#[derive(Clone, Copy, ValueEnum)]
enum Backend {
    /// The `f32` CPU mirror of the GPU kernel.
    Cpu,
    /// The GPU (WGSL) kernel; errors if no adapter is present.
    Gpu,
    /// GPU if an adapter is present, else the CPU mirror.
    Auto,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    match Cli::parse().command {
        Command::Measure(args) => measure_cmd(&args),
        Command::Diff(args) => diff_cmd(&args),
        Command::Bench(args) => {
            bench_cmd(&args);
            Ok(())
        }
        Command::Aniso(args) => aniso_cmd(&args),
        Command::Edit(args) => {
            edit_cmd(&args);
            Ok(())
        }
        Command::Capture(args) => capture_cmd(&args),
        Command::Voxelize(args) => voxelize_cmd(&args),
    }
}

/// Tiny splitmix64 step, mapped to a coordinate in `0..n`.
fn rng_coord(state: &mut u64, n: u32) -> u32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    u32::try_from(z % u64::from(n)).unwrap_or(0)
}

/// Per-class incremental-edit costs.
///
/// Topology edits (~µs each) are timed per call; in-place leaf edits — tens of
/// nanoseconds, at or below the platform timer-resolution floor (~40 ns on
/// macOS) — are instead timed as one homogeneous batch (`time_leaf_batch`), so
/// the figure is the real edit cost and not clock quantization.
struct ClassCosts {
    /// In-place leaf edit cost (seconds), from the single-timer batch.
    leaf_s: f64,
    /// Number of leaf edits actually timed in that batch.
    leaf_batch: u64,
    /// Topology edits in the random mix: per-call time (valid, ≫ floor) + count.
    topo: (std::time::Duration, u64),
    /// Leaf edits seen in the random mix (for the blended-crossover weighting).
    leaf_seen: u64,
}

#[allow(clippy::cast_precision_loss)]
fn per_edit_s((dur, count): (std::time::Duration, u64)) -> f64 {
    if count == 0 {
        0.0
    } else {
        dur.as_secs_f64() / count as f64
    }
}

impl ClassCosts {
    fn mean_topo_s(&self) -> f64 {
        per_edit_s(self.topo)
    }
    /// Blended mean over the random edit mix — leaf cost (from the batch) and
    /// topology cost (per call), weighted by how often each class occurred. This
    /// is the figure the incremental-vs-rebuild crossover divides into.
    #[allow(clippy::cast_precision_loss)]
    fn mean_s(&self) -> f64 {
        let n = self.leaf_seen + self.topo.1;
        if n == 0 {
            0.0
        } else {
            (self.leaf_seen as f64 * self.leaf_s + self.topo.1 as f64 * self.mean_topo_s())
                / n as f64
        }
    }
}

/// Classifies `edits` random toggles (timing the topology ones per call, which
/// are ≫ the timer floor) and measures the in-place leaf cost as a homogeneous
/// batch. Operates on clones, leaving `tree` untouched.
fn time_class_costs(tree: &SparseTree, n: u32, edits: u32, state: &mut u64) -> ClassCosts {
    let mut mix = tree.clone();
    let mut topo = (std::time::Duration::ZERO, 0u64);
    let mut leaf_seen = 0u64;
    for _ in 0..edits {
        let c = VoxelCoord::new(
            rng_coord(state, n),
            rng_coord(state, n),
            rng_coord(state, n),
        );
        let occupied = !mix.is_occupied(c);
        let t = std::time::Instant::now();
        let kind = mix.set_voxel(c, occupied);
        let dt = t.elapsed();
        match kind {
            Edit::Leaf(_) => leaf_seen += 1,
            Edit::Topology => {
                topo.0 += dt;
                topo.1 += 1;
            }
            Edit::Unchanged => {}
        }
    }
    let (leaf_s, leaf_batch) = time_leaf_batch(tree.clone(), n, edits.max(50_000), state);
    ClassCosts {
        leaf_s,
        leaf_batch,
        topo,
        leaf_seen,
    }
}

/// Single-timer cost of an in-place leaf edit, sampled representatively across
/// existing bricks: toggle the in-brick neighbour of randomly-sampled occupied
/// voxels, which always leaves the brick non-empty (so every edit is
/// [`Edit::Leaf`]). Batch timing keeps the measurement above the per-call timer
/// floor a single tens-of-nanosecond edit would sink below. Returns
/// `(seconds per leaf edit, edits timed)`.
#[allow(clippy::cast_precision_loss)]
fn time_leaf_batch(mut tree: SparseTree, n: u32, batch: u32, state: &mut u64) -> (f64, u64) {
    // Untimed: sample occupied voxels to anchor the edits in real bricks.
    let mut anchors: Vec<VoxelCoord> = Vec::new();
    let mut tries = 0u32;
    while anchors.len() < 256 && tries < 100_000 {
        tries += 1;
        let c = VoxelCoord::new(
            rng_coord(state, n),
            rng_coord(state, n),
            rng_coord(state, n),
        );
        if tree.is_occupied(c) {
            anchors.push(c);
        }
    }
    if anchors.is_empty() {
        return (0.0, 0);
    }

    // Timed: toggle each anchor's in-brick neighbour. Flipping the low x bit stays
    // in the same brick (`x>>3` unchanged), which the untouched anchor keeps
    // non-empty, so the edit is always in-place (Edit::Leaf).
    let len = u32::try_from(anchors.len()).unwrap_or(1);
    let t = std::time::Instant::now();
    let mut count = 0u64;
    for i in 0..batch {
        let a = anchors[(i % len) as usize];
        let v = VoxelCoord::new(a.x ^ 1, a.y, a.z);
        let set = (i / len) % 2 == 0; // alternate so the neighbour really toggles
        if matches!(tree.set_voxel(v, set), Edit::Leaf(_)) {
            count += 1;
        }
    }
    let secs = t.elapsed().as_secs_f64();
    (if count > 0 { secs / count as f64 } else { 0.0 }, count)
}

/// Aggregate cost of a series of brush stamps — the per-click work the viewer
/// pays. `total` covers only the `set_voxel` loop (the GPU-sync bookkeeping is
/// excluded), so it is the pure structural edit cost.
struct BrushStats {
    stamps: u32,
    total: std::time::Duration,
    voxels_changed: u64,
    /// Distinct leaves touched, summed over in-place (non-topology) stamps only —
    /// a stamp that changes topology renumbers leaf indices, so its post-edit
    /// indices aren't a meaningful distinct-leaf count.
    leaves_touched: u64,
    /// In-place stamps contributing to `leaves_touched` (the divisor for it).
    inplace_stamps: u32,
    topo_stamps: u32,
}

/// Times `stamps` add-brushes of `radius` at random centres, mutating `tree`.
fn time_brush_stamps(
    mut tree: SparseTree,
    n: u32,
    radius: u32,
    stamps: u32,
    state: &mut u64,
) -> BrushStats {
    let mut s = BrushStats {
        stamps,
        total: std::time::Duration::ZERO,
        voxels_changed: 0,
        leaves_touched: 0,
        inplace_stamps: 0,
        topo_stamps: 0,
    };
    for _ in 0..stamps {
        let center = VoxelCoord::new(
            rng_coord(state, n),
            rng_coord(state, n),
            rng_coord(state, n),
        );
        let coords = brush_voxels(center, radius);
        let t = std::time::Instant::now();
        let mut changed = 0u64;
        let mut any_topo = false;
        let mut leaves: Vec<u32> = Vec::new();
        for c in coords {
            match tree.set_voxel(c, true) {
                Edit::Leaf(idx) => {
                    changed += 1;
                    leaves.push(idx);
                }
                Edit::Topology => {
                    changed += 1;
                    any_topo = true;
                }
                Edit::Unchanged => {}
            }
        }
        s.total += t.elapsed();
        s.voxels_changed += changed;
        if any_topo {
            s.topo_stamps += 1;
        } else {
            // Indices are stable for an in-place stamp, so the distinct-leaf
            // count is meaningful; topology stamps renumber and are excluded.
            leaves.sort_unstable();
            leaves.dedup();
            s.leaves_touched += leaves.len() as u64;
            s.inplace_stamps += 1;
        }
    }
    s
}

/// Times the GPU sync of an edit on whatever adapter is present: a single
/// in-place leaf patch (`update_leaf` — CPU staging cost; the transfer is async,
/// flushed by the next frame's submit) vs a full structure re-upload
/// (`reupload`, which rebuilds the School-B buffer and recreates the buffers).
/// Returns `(µs per leaf patch, ms per full re-upload)`.
fn time_gpu_patch(ctx: &GpuContext, tree: &SparseTree) -> Result<(f64, f64)> {
    let structure = SchoolBBuffer::from_sparse(tree);
    let mut renderer = GpuRenderer::new(ctx, &structure)?;

    let patches = 10_000u32;
    let t = std::time::Instant::now();
    for _ in 0..patches {
        renderer.update_leaf(&structure, 0); // re-stage leaf 0's words (O(1))
    }
    let leaf_us = t.elapsed().as_secs_f64() * 1e6 / f64::from(patches);

    let reups = 50u32;
    let t = std::time::Instant::now();
    for _ in 0..reups {
        let s = SchoolBBuffer::from_sparse(tree);
        renderer.reupload(&s)?;
    }
    let reup_ms = t.elapsed().as_secs_f64() * 1e3 / f64::from(reups);

    renderer.flush_and_wait()?; // drain staged writes before drop
    Ok((leaf_us, reup_ms))
}

/// The edit-performance suite: rebuild baseline, per-class single-edit costs,
/// the incremental-vs-rebuild crossover, brush-stamp cost, and the GPU sync cost
/// (leaf patch vs full re-upload). `--sweep` adds a compact fixture×resolution
/// table.
#[allow(clippy::cast_precision_loss)]
fn edit_cmd(args: &EditArgs) {
    let Ok(resolution) = Resolution::new(args.res) else {
        eprintln!("invalid resolution {} (must be 8·4^k)", args.res);
        return;
    };
    let n = resolution.voxels_per_axis();
    let mut state = 0x9E37_79B9_7F4A_7C15u64;

    // (1) Full rebuild baseline — the O(n³) scan any change costs today. The
    // tree stays pristine: every bench below works on a clone or a borrow.
    let t = std::time::Instant::now();
    let tree = build_tree(args.fixture, resolution);
    let full_s = t.elapsed().as_secs_f64();
    tracing::info!(
        nodes = tree.node_count(),
        leaves = tree.leaf_count(),
        "structure built"
    );

    // (2) Per-class single-edit costs (leaf via batch, topology per call).
    let cc = time_class_costs(&tree, n, args.edits, &mut state);
    let leaf_s = cc.leaf_s;
    let topo_s = cc.mean_topo_s();

    println!(
        "edit cost — {} {n}³  ({} random toggles)",
        fixture_name(args.fixture),
        args.edits
    );
    println!(
        "  full rebuild (scan):            {:>10.2} ms   ← today's cost for ANY change",
        full_s * 1e3
    );
    if cc.leaf_batch > 0 {
        println!(
            "  in-place edit (Edit::Leaf):     {:>10.3} µs/edit   (N={:>6})  → {:.0}× cheaper",
            leaf_s * 1e6,
            cc.leaf_batch,
            full_s / leaf_s.max(1e-12),
        );
    }
    if cc.topo.1 > 0 {
        println!(
            "  topology edit (Edit::Topology): {:>10.3} ms/edit   (N={:>6})  → {:.0}× cheaper",
            topo_s * 1e3,
            cc.topo.1,
            full_s / topo_s.max(1e-12),
        );
    }

    // (3) Incremental-vs-rebuild crossover.
    let mean = cc.mean_s();
    if mean > 0.0 {
        println!(
            "  crossover:                      ≈ {:>8.0} incremental edits = one full rebuild",
            full_s / mean
        );
    }

    // (4) Brush stamp — the per-click cost the viewer pays.
    let bs = time_brush_stamps(tree.clone(), n, args.brush_radius, args.stamps, &mut state);
    if bs.stamps > 0 {
        let per_stamp_ms = bs.total.as_secs_f64() * 1e3 / f64::from(bs.stamps);
        let vox = bs.voxels_changed as f64 / f64::from(bs.stamps);
        let leaves = if bs.inplace_stamps > 0 {
            bs.leaves_touched as f64 / f64::from(bs.inplace_stamps)
        } else {
            0.0
        };
        let per_vox_us = if bs.voxels_changed > 0 {
            bs.total.as_secs_f64() * 1e6 / bs.voxels_changed as f64
        } else {
            0.0
        };
        println!(
            "  brush stamp (r={}):             {:>10.3} ms/stamp  ({vox:.0} voxels, {per_vox_us:.3} µs/voxel; {leaves:.1} leaves/in-place stamp)",
            args.brush_radius, per_stamp_ms,
        );
        println!(
            "    {} of {} stamps changed topology (a full re-upload on that click)",
            bs.topo_stamps, bs.stamps
        );
    }

    // (5) GPU sync cost — the renderer's real bottleneck (runtime-gated).
    if tree.leaf_count() == 0 {
        println!("  GPU patch cost: empty structure (skipped)");
    } else if let Ok(ctx) = GpuContext::try_new() {
        match time_gpu_patch(&ctx, &tree) {
            Ok((patch_us, reupload_ms)) => {
                println!(
                    "  GPU leaf patch (update_leaf):   {patch_us:>10.3} µs/leaf  (CPU staging; transfer async)"
                );
                println!(
                    "  GPU full re-upload (reupload):  {:>10.3} ms        → {:.0}× the leaf patch",
                    reupload_ms,
                    reupload_ms * 1e3 / patch_us.max(1e-12),
                );
            }
            Err(e) => eprintln!("GPU patch benchmark failed: {e:?}"),
        }
    } else {
        println!("  GPU patch cost: no adapter (skipped)");
    }

    println!(
        "  (in-place leaf = O(1) edit + one-leaf GPU patch; topology rebuilds node levels\n   O(bricks), scan-free + a full re-upload. Incremental wins below the crossover count.)"
    );

    if args.sweep {
        edit_sweep(&mut state);
    }
}

/// A compact CPU-only fixture×resolution table of rebuild, per-class, and
/// crossover costs — the spread across geometry and scale at a glance.
#[allow(clippy::cast_precision_loss)]
fn edit_sweep(state: &mut u64) {
    const FIXTURES: [Fixture; 5] = [
        Fixture::Sierpinski,
        Fixture::Dust,
        Fixture::Caves,
        Fixture::Checkerboard,
        Fixture::WireLattice,
    ];
    println!("\nfixture×resolution sweep (CPU; 5000 toggles each):");
    println!(
        "  {:<13} {:>5}  {:>10}  {:>10}  {:>10}  {:>10}",
        "fixture", "res", "rebuild", "leaf", "topo", "crossover"
    );
    for res in [128u32, 512] {
        let Ok(resolution) = Resolution::new(res) else {
            continue;
        };
        for f in FIXTURES {
            let t = std::time::Instant::now();
            let tree = build_tree(f, resolution);
            let rebuild_s = t.elapsed().as_secs_f64();
            let cc = time_class_costs(&tree, resolution.voxels_per_axis(), 5000, state);
            let mean = cc.mean_s();
            let crossover = if mean > 0.0 { rebuild_s / mean } else { 0.0 };
            println!(
                "  {:<13} {res:>5}  {:>8.2}ms  {:>8.3}µs  {:>8.3}ms  {crossover:>10.0}",
                fixture_name(f),
                rebuild_s * 1e3,
                cc.leaf_s * 1e6,
                cc.mean_topo_s() * 1e3,
            );
        }
    }
}

/// Builds the sparse structure for `fixture` at `resolution`.
fn build_tree(fixture: Fixture, resolution: Resolution) -> SparseTree {
    match fixture {
        Fixture::Sierpinski => {
            SparseTree::build(&OctantFractal::sierpinski_tetrahedron(resolution))
        }
        Fixture::Cantor => SparseTree::build(&OctantFractal::cantor_dust(resolution)),
        Fixture::Checkerboard => SparseTree::build(&Checkerboard { resolution }),
        Fixture::Solid => SparseTree::build(&Solid { resolution }),
        Fixture::WireLattice => SparseTree::build(&WireLattice::new(resolution)),
        Fixture::Dust => SparseTree::build(&Dust::new(resolution)),
        Fixture::Perlin => SparseTree::build(&NoiseField::perlin(resolution)),
        Fixture::Caves => SparseTree::build(&NoiseField::caves(resolution)),
    }
}

fn measure_cmd(args: &MeasureArgs) -> Result<()> {
    let resolution = Resolution::new(args.res)?;
    tracing::info!(
        n = resolution.voxels_per_axis(),
        "building structure (dense brick enumeration; may take a moment at 2048³)"
    );
    let tree = build_tree(args.fixture, resolution);
    tracing::info!(
        nodes = tree.node_count(),
        leaves = tree.leaf_count(),
        "structure built"
    );
    print!("{}", measure::measure(&tree, args.rays));
    Ok(())
}

/// Deterministic splitmix64-derived camera rays aimed at the grid centre.
#[allow(clippy::cast_precision_loss)]
fn camera_rays(resolution: Resolution, count: u32) -> Vec<Ray> {
    let nf = f64::from(resolution.voxels_per_axis());
    let centre = DVec3::splat(nf * 0.5);
    let mut state = 0x0DDB_1A5E_5EED_F00Du64;
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut rays = Vec::with_capacity(count as usize);
    while rays.len() < count as usize {
        let origin = DVec3::new(
            next() * (2.0 * nf) - 0.5 * nf,
            next() * (2.0 * nf) - 0.5 * nf,
            next() * (2.0 * nf) - 0.5 * nf,
        );
        let jitter = DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * (nf * 0.25);
        let dir = (centre + jitter) - origin;
        if dir.length() > 1e-6 {
            rays.push(Ray::new(origin, dir));
        }
    }
    rays
}

#[allow(clippy::cast_precision_loss)]
fn diff_cmd(args: &DiffArgs) -> Result<()> {
    let resolution = Resolution::new(args.res)?;
    let tree = build_tree(args.fixture, resolution);
    let structure = SchoolBBuffer::from_sparse(&tree);
    let rays = camera_rays(resolution, args.rays);

    // f64 reference (School-A recursive traversal — proven equal to the oracle).
    let reference: Vec<Option<VoxelCoord>> = rays
        .iter()
        .map(|r| tree.traverse(r).map(|h| h.voxel))
        .collect();

    let (label, results) = run_backend(args.backend, &structure, &rays)?;

    let mut mismatches = 0u32;
    let mut hits = 0u32;
    for (got, want) in results.iter().zip(&reference) {
        if got != want {
            mismatches += 1;
        }
        if got.is_some() {
            hits += 1;
        }
    }
    let total = rays.len() as f64;
    println!(
        "diff [{label}] vs f64 reference: {} rays, {hits} hits, {mismatches} mismatches ({:.4}%)",
        rays.len(),
        f64::from(mismatches) / total * 100.0,
    );
    if mismatches == 0 {
        println!("  exact match.");
    } else {
        println!("  (mismatches are grazing rays: f32 vs f64 picks a different DDA step)");
    }
    Ok(())
}

/// Runs the selected backend, returning a label and the per-ray hits.
fn run_backend(
    backend: Backend,
    structure: &SchoolBBuffer,
    rays: &[Ray],
) -> Result<(&'static str, Vec<Option<VoxelCoord>>)> {
    let cpu = || {
        (
            "cpu-mirror",
            rays.iter().map(|r| mirror_traverse(structure, r)).collect(),
        )
    };

    match backend {
        Backend::Cpu => Ok(cpu()),
        Backend::Gpu => {
            let ctx = GpuContext::try_new().context("no GPU; use --backend cpu or auto")?;
            Ok(("gpu", gpu_traverse(&ctx, structure, rays)?))
        }
        Backend::Auto => match GpuContext::try_new() {
            Ok(ctx) => Ok(("gpu", gpu_traverse(&ctx, structure, rays)?)),
            Err(GpuError::NoAdapter) => {
                tracing::info!("no GPU adapter; falling back to the CPU mirror");
                Ok(cpu())
            }
            Err(e) => Err(e.into()),
        },
    }
}

fn gpu_traverse(
    ctx: &GpuContext,
    structure: &SchoolBBuffer,
    rays: &[Ray],
) -> Result<Vec<Option<VoxelCoord>>> {
    let traverser = GpuTraverser::new(ctx, structure)?;
    Ok(traverser.traverse(rays)?)
}

#[allow(clippy::cast_precision_loss)]
fn bench_cmd(args: &BenchArgs) {
    use clap::ValueEnum;

    let fixtures = if args.fixtures.is_empty() {
        Fixture::value_variants().to_vec()
    } else {
        args.fixtures.clone()
    };
    let gpu = GpuContext::try_new().ok();
    if gpu.is_none() {
        eprintln!("(no GPU adapter — GPU column shows '-')");
    }

    println!(
        "{:<13} {:>5} {:>9} {:>9} {:>9} {:>9} {:>5} {:>6} {:>6} {:>7} {:>5} {:>8} {:>8}",
        "fixture",
        "res",
        "build_ms",
        "serial_ms",
        "leaves",
        "MiB",
        "D",
        "R2",
        "desc",
        "steps",
        "hit%",
        "cpuMr/s",
        "gpuMr/s",
    );

    for &res in &args.res {
        let Ok(resolution) = Resolution::new(res) else {
            eprintln!("skip res {res}: not 8·4^k");
            continue;
        };
        for &fix in &fixtures {
            // `build_ms` (full n³ scan + Morton sort + OR-reduce) and `serial_ms`
            // (School-B re-serialize + leaf-bounds) are the per-edit rebuild cost:
            // any geometry change re-runs both — there is no incremental update.
            let t = std::time::Instant::now();
            let tree = build_tree(fix, resolution);
            let build_ms = t.elapsed().as_secs_f64() * 1000.0;
            let t = std::time::Instant::now();
            let structure = SchoolBBuffer::from_sparse(&tree);
            let serialize_ms = t.elapsed().as_secs_f64() * 1000.0;
            let report = measure::measure(&tree, 4000);

            let rays = camera_rays(resolution, args.rays);
            let nrays = rays.len() as f64;

            // CPU mirror throughput (single thread); black-box so it can't be
            // optimized away.
            let t = std::time::Instant::now();
            let mut hits = 0u64;
            for r in &rays {
                hits += u64::from(mirror_traverse(&structure, r).is_some());
            }
            std::hint::black_box(hits);
            let cpu_mrays = nrays / t.elapsed().as_secs_f64() / 1e6;

            let gpu_col = match &gpu {
                Some(ctx) => match bench_gpu(ctx, &structure, &rays) {
                    Ok(mr) => format!("{mr:8.1}"),
                    Err(_) => format!("{:>8}", "oom"),
                },
                None => format!("{:>8}", "-"),
            };

            let hit_pct =
                100.0 * report.descent.rays_hit as f64 / report.descent.rays_cast.max(1) as f64;
            let mib = report.total_bytes as f64 / (1u64 << 20) as f64;
            println!(
                "{:<13} {:>5} {:>9.1} {:>9.1} {:>9} {:>9.3} {:>5.2} {:>6.3} {:>6.2} {:>7.1} {:>5.0} {:>8.1} {}",
                fixture_name(fix),
                res,
                build_ms,
                serialize_ms,
                tree.leaf_count(),
                mib,
                report.dimension.dimension,
                report.dimension.r_squared,
                report.descent.mean_descents,
                report.descent.mean_cell_steps,
                hit_pct,
                cpu_mrays,
                gpu_col,
            );
        }
    }
}

/// GPU batch throughput (Mray/s, including readback) after a warm-up dispatch.
#[allow(clippy::cast_precision_loss)]
fn bench_gpu(ctx: &GpuContext, structure: &SchoolBBuffer, rays: &[Ray]) -> Result<f64> {
    let traverser = GpuTraverser::new(ctx, structure)?;
    let _ = traverser.traverse(&rays[..rays.len().min(2000)])?;
    let t = std::time::Instant::now();
    let _ = traverser.traverse(rays)?;
    Ok(rays.len() as f64 / t.elapsed().as_secs_f64() / 1e6)
}

fn fixture_name(f: Fixture) -> &'static str {
    match f {
        Fixture::Sierpinski => "sierpinski",
        Fixture::Cantor => "cantor",
        Fixture::Checkerboard => "checkerboard",
        Fixture::Solid => "solid",
        Fixture::WireLattice => "wire-lattice",
        Fixture::Dust => "dust",
        Fixture::Perlin => "perlin",
        Fixture::Caves => "caves",
    }
}

/// One view direction's algorithmic and hardware cost.
struct DirSample {
    dir: DVec3,
    steps: f64,
    hit: f64,
    gpu_ns: f64,
}

/// Per-direction cost sweep: cell-steps (algorithmic) and, if a GPU is present,
/// GPU ns/ray for a coherent orthographic batch (hardware).
fn aniso_collect(args: &AnisoArgs) -> Result<Vec<DirSample>> {
    let resolution = Resolution::new(args.res)?;
    let tree = build_tree(args.fixture, resolution);
    let structure = SchoolBBuffer::from_sparse(&tree);
    let dirs = measure::fibonacci_dirs(args.dirs.max(1));

    let gpu = GpuContext::try_new().ok();
    let traverser = if let Some(ctx) = &gpu {
        eprintln!(
            "GPU timer: {}",
            if ctx.supports_timestamps() {
                "compute-pass timestamps (readback-free kernel time)"
            } else {
                "wall-clock (no timestamp support; includes readback)"
            }
        );
        Some(GpuTraverser::new(ctx, &structure)?)
    } else {
        eprintln!("(no GPU adapter — hardware-anisotropy column omitted)");
        None
    };

    let mut samples = Vec::with_capacity(dirs.len());
    for &d in &dirs {
        let stat = measure::directional_steps(&tree, d, args.side);
        let gpu_ns = gpu_ns_per_ray(traverser.as_ref(), resolution, d, args.side)?;
        samples.push(DirSample {
            dir: stat.dir,
            steps: stat.mean_cell_steps,
            hit: stat.hit_frac,
            gpu_ns,
        });
    }
    Ok(samples)
}

/// Best-of-3 GPU kernel ns/ray for an orthographic batch along `dir`, or `NaN`
/// with no GPU. Uses compute-pass timestamps when available (readback-free —
/// the clean kernel time); otherwise falls back to wall-clock.
#[allow(clippy::cast_precision_loss)]
fn gpu_ns_per_ray(
    traverser: Option<&GpuTraverser>,
    resolution: Resolution,
    dir: DVec3,
    side: u32,
) -> Result<f64> {
    let Some(t) = traverser else {
        return Ok(f64::NAN);
    };
    let rays = measure::ortho_rays(resolution, dir, side);
    let _ = t.traverse_timed(&rays)?; // warm
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let (_, gpu_ns) = t.traverse_timed(&rays)?;
        let ns = if let Some(ns) = gpu_ns {
            ns
        } else {
            let t0 = std::time::Instant::now();
            let _ = t.traverse(&rays)?;
            t0.elapsed().as_secs_f64() * 1e9
        };
        best = best.min(ns);
    }
    Ok(best / rays.len() as f64)
}

#[allow(clippy::cast_precision_loss)]
fn aniso_cmd(args: &AnisoArgs) -> Result<()> {
    let samples = aniso_collect(args)?;
    let has_gpu = samples.first().is_some_and(|s| s.gpu_ns.is_finite());

    let step_vals: Vec<f64> = samples.iter().map(|s| s.steps).collect();
    let gpu_vals: Vec<f64> = samples.iter().map(|s| s.gpu_ns).collect();
    let (step_min, step_max, step_mean) = min_max_mean(&step_vals);
    let (gpu_min, gpu_max, gpu_mean) = min_max_mean(&gpu_vals);
    let step_ratio = step_max / step_min.max(1e-9);
    let gpu_ratio = gpu_max / gpu_min.max(1e-9);

    println!(
        "anisotropy — {} {}³  ({} directions, {}² rays/dir)",
        fixture_name(args.fixture),
        args.res,
        args.dirs.max(1),
        args.side,
    );
    println!(
        "  {:<11} {:>8} {:>8} {:>7} {:>8}",
        "metric", "min", "max", "ratio", "mean"
    );
    println!(
        "  {:<11} {:>8.1} {:>8.1} {:>6.2}× {:>8.1}   algorithmic (DDA + skip)",
        "cell-steps", step_min, step_max, step_ratio, step_mean,
    );
    if has_gpu {
        println!(
            "  {:<11} {:>8.1} {:>8.1} {:>6.2}× {:>8.1}   hardware (steps + cache + coherence)",
            "gpu ns/ray", gpu_min, gpu_max, gpu_ratio, gpu_mean,
        );
        println!(
            "  step↔gpu correlation r = {:.2}   → algorithmic swing {:.2}×, hardware swing {:.2}× \
             (cache/coherence excess ≈ {:.2}×)",
            pearson(&step_vals, &gpu_vals),
            step_ratio,
            gpu_ratio,
            gpu_ratio / step_ratio.max(1e-9),
        );
        report_extremes(&samples, has_gpu);
    } else {
        report_extremes(&samples, has_gpu);
    }

    if args.verbose {
        println!(
            "  {:>22} {:>9} {:>6} {:>10}",
            "direction", "steps", "hit%", "gpu_ns"
        );
        for s in &samples {
            println!(
                "  {:>22} {:>9.1} {:>6.0} {:>10.1}",
                fmt_dir(s.dir),
                s.steps,
                s.hit * 100.0,
                s.gpu_ns,
            );
        }
    }
    Ok(())
}

/// Loops the GPU traversal kernel at its worst-anisotropy orientation, giving an
/// external profiler (Xcode GPU capture / Instruments Metal System Trace) a
/// stable, isolated, repeating workload to read counters from. This is the
/// measure-first step: confirm the kernel is memory-latency-bound on the cold
/// `leaf_words` stream before committing engineering to that premise. The
/// counter checklist and go/no-go thresholds are in `CAPTURE.md`.
#[allow(clippy::cast_precision_loss)]
fn capture_cmd(args: &CaptureArgs) -> Result<()> {
    let resolution = Resolution::new(args.res)?;
    let tree = build_tree(args.fixture, resolution);
    let structure = SchoolBBuffer::from_sparse(&tree);
    let ctx = GpuContext::try_new()
        .context("the capture harness needs a GPU adapter (this is a GPU-counter target)")?;
    let traverser = GpuTraverser::new(&ctx, &structure)?;

    // Find the worst (most expensive) orientation — where cold misses peak. A
    // small search batch keeps the sweep quick; the capture loop uses --side.
    let search_side = args.side.min(256);
    eprintln!(
        "searching {} directions ({}² rays each) for the worst orientation…",
        args.search_dirs.max(1),
        search_side,
    );
    let mut worst = (DVec3::Z, 0.0f64);
    for d in measure::fibonacci_dirs(args.search_dirs.max(1)) {
        let ns = gpu_ns_per_ray(Some(&traverser), resolution, d, search_side)?;
        if ns.is_finite() && ns > worst.1 {
            worst = (d, ns);
        }
    }
    let (dir, search_ns) = worst;
    let rays = measure::ortho_rays(resolution, dir, args.side);

    println!(
        "capture target — {} {}³",
        fixture_name(args.fixture),
        args.res
    );
    println!(
        "  worst orientation {}  (~{:.1} ns/ray at search size; {} rays/dispatch)",
        fmt_dir(dir),
        search_ns,
        rays.len(),
    );
    let _ = traverser.traverse(&rays[..rays.len().min(2000)])?; // warm

    if args.gputrace {
        // Write a Xcode-openable trace document wrapping a few dispatches.
        let path = format!("{}-{}.gputrace", fixture_name(args.fixture), args.res);
        println!("  writing a GPU trace document ({path})…");
        voxel_gpu::capture_gputrace(std::path::Path::new(&path), || {
            for _ in 0..4 {
                traverser.traverse(&rays)?;
            }
            Ok(())
        })?;
        println!("  wrote {path} — open it in Xcode:  open {path}");
    } else {
        println!(
            "  looping {} kernel dispatches — record an Instruments Metal System\n  Trace now and read counters per CAPTURE.md",
            args.iters
        );
        let t = std::time::Instant::now();
        for _ in 0..args.iters {
            let _ = traverser.traverse(&rays)?;
        }
        let secs = t.elapsed().as_secs_f64();
        let total = rays.len() as f64 * f64::from(args.iters);
        println!(
            "  done: {} dispatches in {:.2}s  ({:.1} Mrays/s, {:.1} ns/ray wall)",
            args.iters,
            secs,
            total / secs / 1e6,
            secs / total * 1e9,
        );
    }
    Ok(())
}

/// Reports the cheapest and priciest directions (by GPU time if present, else by
/// cell-steps).
fn report_extremes(samples: &[DirSample], has_gpu: bool) {
    let key = |s: &DirSample| if has_gpu { s.gpu_ns } else { s.steps };
    let cheap = samples
        .iter()
        .min_by(|a, b| key(a).total_cmp(&key(b)))
        .expect("at least one direction");
    let pricey = samples
        .iter()
        .max_by(|a, b| key(a).total_cmp(&key(b)))
        .expect("at least one direction");
    println!(
        "  cheapest {} (steps {:.1}{}); priciest {} (steps {:.1}{})",
        fmt_dir(cheap.dir),
        cheap.steps,
        if has_gpu {
            format!(", {:.1} ns", cheap.gpu_ns)
        } else {
            String::new()
        },
        fmt_dir(pricey.dir),
        pricey.steps,
        if has_gpu {
            format!(", {:.1} ns", pricey.gpu_ns)
        } else {
            String::new()
        },
    );
}

#[allow(clippy::cast_precision_loss)]
fn min_max_mean(v: &[f64]) -> (f64, f64, f64) {
    let finite: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    (min, max, mean)
}

#[allow(clippy::cast_precision_loss)]
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (*a, *b))
        .collect();
    let n = pairs.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let mx = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for (a, b) in &pairs {
        cov += (a - mx) * (b - my);
        vx += (a - mx).powi(2);
        vy += (b - my).powi(2);
    }
    if vx == 0.0 || vy == 0.0 {
        return f64::NAN;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

fn fmt_dir(d: DVec3) -> String {
    format!("[{:+.2},{:+.2},{:+.2}]", d.x, d.y, d.z)
}

fn voxelize_cmd(args: &VoxelizeArgs) -> Result<()> {
    let resolution = Resolution::new(args.res)?;
    let mesh =
        load_mesh(&args.input).with_context(|| format!("loading mesh {}", args.input.display()))?;
    tracing::info!(
        triangles = mesh.triangles.len(),
        n = resolution.voxels_per_axis(),
        "mesh loaded"
    );

    // Fit the mesh into the cubic grid; tile for the GPU dispatch.
    let grid = VoxelGrid::fit_mesh(resolution, &mesh, args.padding);
    let tiles = TileSpec::new([4, 4, 4], grid.dims())?;
    // Occupancy-only: this command consumes only the bitset (→ SparseTree), so
    // skip the per-voxel owner/color buffers, which would be n³·4 bytes each and
    // exceed the storage limit at n ≥ 512.
    let opts = VoxelizeOpts {
        epsilon: 1e-4,
        store_owner: false,
        store_color: false,
    };

    if args.diff {
        return voxelize_diff(&mesh, &grid, &tiles, &opts);
    }

    let (label, occupancy) = run_voxelize_backend(args.backend, &mesh, &grid, &tiles, &opts)?;

    // Bridge to the renderer's structures (the whole point of the native output).
    let tree = occupancy.to_sparse_tree();
    let structure = SchoolBBuffer::from_sparse(&tree);

    let n = resolution.voxels_per_axis();
    println!("voxelize [{label}] {}", args.input.display());
    println!("  mesh:      {} triangles", mesh.triangles.len());
    println!("  grid:      {n}³  (voxel_size {:.5})", grid.voxel_size);
    println!("  occupied:  {} voxels", occupancy.count_occupied());
    println!(
        "  structure: {} sparse nodes, {} leaves; {} School-B nodes",
        tree.node_count(),
        tree.leaf_count(),
        structure.node_count()
    );
    Ok(())
}

/// Runs the selected voxelizer backend, returning a label and the occupancy.
fn run_voxelize_backend(
    backend: Backend,
    mesh: &MeshInput,
    grid: &VoxelGrid,
    tiles: &TileSpec,
    opts: &VoxelizeOpts,
) -> Result<(&'static str, VoxelOccupancy)> {
    let cpu = || {
        (
            "cpu-oracle",
            voxelize_surface_cpu(mesh, grid, tiles, opts).occupancy,
        )
    };

    match backend {
        Backend::Cpu => Ok(cpu()),
        Backend::Gpu => {
            let ctx = GpuContext::try_new().context("no GPU; use --backend cpu or auto")?;
            Ok(("gpu", voxelize_gpu(&ctx, mesh, grid, tiles, opts)?))
        }
        Backend::Auto => match GpuContext::try_new() {
            Ok(ctx) => Ok(("gpu", voxelize_gpu(&ctx, mesh, grid, tiles, opts)?)),
            Err(GpuError::NoAdapter) => {
                tracing::info!("no GPU adapter; using the CPU oracle");
                Ok(cpu())
            }
            Err(e) => Err(e.into()),
        },
    }
}

/// Voxelizes on the GPU, sharing the renderer's device via `from_device` (one GPU
/// for both the voxelizer and the renderer).
fn voxelize_gpu(
    ctx: &GpuContext,
    mesh: &MeshInput,
    grid: &VoxelGrid,
    tiles: &TileSpec,
    opts: &VoxelizeOpts,
) -> Result<VoxelOccupancy> {
    let vox = pollster::block_on(GpuVoxelizer::from_device(
        &ctx.device,
        &ctx.queue,
        GpuVoxelizerConfig::default(),
    ))?;
    let out = pollster::block_on(vox.voxelize_surface(mesh, grid, tiles, opts))?;
    Ok(out.occupancy)
}

/// Cross-checks the GPU voxelizer against the CPU SAT oracle on the same mesh: the
/// GPU must be a *conservative superset* (never under-mark) — Engineering Codex:
/// Reference Implementation as Oracle.
fn voxelize_diff(
    mesh: &MeshInput,
    grid: &VoxelGrid,
    tiles: &TileSpec,
    opts: &VoxelizeOpts,
) -> Result<()> {
    let ctx = GpuContext::try_new().context("--diff requires a GPU")?;
    let cpu = voxelize_surface_cpu(mesh, grid, tiles, opts).occupancy;
    let gpu = voxelize_gpu(&ctx, mesh, grid, tiles, opts)?;

    let mut under = 0u64; // CPU-marked but GPU-missed — must be 0
    let mut over = 0u64; // GPU over-marked — a small FP-tangent margin only
    for (c, g) in cpu.words().iter().zip(gpu.words()) {
        under += u64::from((c & !g).count_ones());
        over += u64::from((g & !c).count_ones());
    }

    println!(
        "voxelize diff (CPU oracle vs GPU): {} CPU-occupied, {} GPU-occupied",
        cpu.count_occupied(),
        gpu.count_occupied()
    );
    println!("  CPU∖GPU (under-marks, must be 0): {under}");
    println!("  GPU∖CPU (FP-tangent over-marks):  {over}");
    if under == 0 {
        println!("  ✓ GPU is a conservative superset of the CPU oracle (no under-marking).");
        Ok(())
    } else {
        anyhow::bail!("GPU under-marked {under} voxels — a real divergence, not a tangent effect")
    }
}
