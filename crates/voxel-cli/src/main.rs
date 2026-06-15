//! `voxel` — headless CLI for the sparse MIP voxel structure.
//!
//! Orchestration only (argument parsing, I/O, reporting); all domain logic
//! lives in [`voxel_core`] and the GPU path in [`voxel_gpu`]. Backend selection
//! is a runtime `--backend cpu|gpu|auto` flag, never a Cargo feature.

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use glam::DVec3;
use voxel_core::fixtures::{Checkerboard, Dust, OctantFractal, Solid, WireLattice};
use voxel_core::{
    Ray, Resolution, SchoolBBuffer, SparseTree, VoxelCoord, measure, mirror_traverse,
};
use voxel_gpu::{GpuContext, GpuError, GpuTraverser};

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
        "{:<13} {:>5} {:>9} {:>9} {:>9} {:>5} {:>6} {:>6} {:>7} {:>5} {:>8} {:>8}",
        "fixture",
        "res",
        "build_ms",
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
            let t = std::time::Instant::now();
            let tree = build_tree(fix, resolution);
            let build_ms = t.elapsed().as_secs_f64() * 1000.0;
            let structure = SchoolBBuffer::from_sparse(&tree);
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
                "{:<13} {:>5} {:>9.1} {:>9} {:>9.3} {:>5.2} {:>6.3} {:>6.2} {:>7.1} {:>5.0} {:>8.1} {}",
                fixture_name(fix),
                res,
                build_ms,
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
    }
}
