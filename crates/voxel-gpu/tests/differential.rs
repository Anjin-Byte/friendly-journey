//! GPU-vs-reference differential (`idea.md` §11.5).
//!
//! Runs the WGSL kernel on real hardware and checks it against the `f64` oracle
//! and the `f32` mirror. Gated at runtime: with no adapter the test *skips*
//! (passes), so CPU-only CI stays green; with `VOXEL_REQUIRE_GPU=1` a missing
//! adapter *fails*, so a GPU CI lane can't silently pass by skipping (review R2).

use glam::DVec3;
use voxel_core::fixtures::{Checkerboard, Dust, OctantFractal, WireLattice};
use voxel_core::{
    OccupancyField, Ray, Resolution, SchoolBBuffer, SparseTree, VoxelCoord, mirror_traverse, oracle,
};
use voxel_gpu::{GpuContext, GpuError, GpuTraverser};

fn require_gpu() -> bool {
    std::env::var_os("VOXEL_REQUIRE_GPU").is_some()
}

/// Returns a context, or `None` (after honoring `VOXEL_REQUIRE_GPU`) to skip.
fn context_or_skip() -> Option<GpuContext> {
    match GpuContext::try_new() {
        Ok(ctx) => Some(ctx),
        Err(GpuError::NoAdapter) if !require_gpu() => {
            eprintln!("skip: no GPU adapter present (set VOXEL_REQUIRE_GPU=1 to require one)");
            None
        }
        Err(e) => panic!("GPU unavailable: {e}"),
    }
}

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
fn gpu_axis_aligned_hits_are_exact() {
    let Some(ctx) = context_or_skip() else { return };
    let field = OctantFractal::sierpinski_tetrahedron(res(128));
    let structure = SchoolBBuffer::from_sparse(&SparseTree::build(&field));
    let traverser = GpuTraverser::new(&ctx, &structure).unwrap();

    // Axis-aligned rays through occupied columns: no grazing, exact hits.
    let mut rays = Vec::new();
    let mut expected = Vec::new();
    for (x, y) in [(0u32, 0u32), (3, 3), (12, 5), (40, 40)] {
        let ray = Ray::new(
            DVec3::new(f64::from(x) + 0.5, f64::from(y) + 0.5, -1.0),
            DVec3::Z,
        );
        rays.push(ray);
        expected.push(oracle::first_hit(&field, &ray).map(|h| h.voxel));
    }
    let gpu = traverser.traverse(&rays).unwrap();
    assert_eq!(
        gpu, expected,
        "GPU axis-aligned hits must match the oracle exactly"
    );
}

#[test]
fn gpu_matches_oracle_and_mirror_on_random_rays() {
    let Some(ctx) = context_or_skip() else { return };
    check_fixture(&ctx, &OctantFractal::sierpinski_tetrahedron(res(128)));
    check_fixture(
        &ctx,
        &Checkerboard {
            resolution: res(128),
        },
    );
}

#[test]
fn gpu_matches_on_pathological_fixtures() {
    // The traversal-pathology and warp-divergence stress fixtures: thin wires
    // and scattered dust maximize grazing, so this is where an f32 kernel bug
    // would surface first. The grazing-disagreement bound must still hold.
    let Some(ctx) = context_or_skip() else { return };
    check_fixture(&ctx, &WireLattice::new(res(128)));
    check_fixture(&ctx, &Dust::new(res(128)));
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn gpu_matches_mirror_on_grazing_and_axis_aligned_rays() {
    // The random battery above never aims a ray to graze a single-voxel brick's
    // occupied-box face or travel exactly axis-aligned — the geometry the
    // per-brick early-skip's f32 box test is most fragile on. Aim rays straight
    // at occupied voxels (jittered to graze edges) and exactly down the axes, and
    // assert the GPU agrees with the f32 mirror EXACTLY (they share the kernel,
    // so any disagreement is a real bug, not f32-vs-f64 grazing).
    let Some(ctx) = context_or_skip() else { return };
    let field = Dust::new(res(128));
    let structure = SchoolBBuffer::from_sparse(&SparseTree::build(&field));
    let traverser = GpuTraverser::new(&ctx, &structure).unwrap();
    let n = field.resolution().voxels_per_axis();

    // Collect occupied voxels to aim at.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut targets = Vec::new();
    while targets.len() < 40 {
        let x = u32::try_from(splitmix64(&mut state) % u64::from(n)).unwrap();
        let y = u32::try_from(splitmix64(&mut state) % u64::from(n)).unwrap();
        let z = u32::try_from(splitmix64(&mut state) % u64::from(n)).unwrap();
        if field.is_occupied(VoxelCoord::new(x, y, z)) {
            targets.push([x, y, z]);
        }
    }

    let axes = [
        DVec3::X,
        DVec3::Y,
        DVec3::Z,
        DVec3::new(1.0, 1.0, 0.0),
        DVec3::new(1.0, 0.0, 1.0),
        DVec3::new(0.0, 1.0, 1.0),
        DVec3::new(1.0, 1.0, 1.0),
    ];
    let mut rays = Vec::new();
    for t in &targets {
        let centre = DVec3::new(
            f64::from(t[0]) + 0.5,
            f64::from(t[1]) + 0.5,
            f64::from(t[2]) + 0.5,
        );
        for dir in axes {
            for _ in 0..16 {
                let aim = centre
                    + DVec3::new(
                        unit(&mut state) * 1.6 - 0.8,
                        unit(&mut state) * 1.6 - 0.8,
                        unit(&mut state) * 1.6 - 0.8,
                    );
                rays.push(Ray::new(aim - dir * 50.0, dir));
            }
        }
    }

    let gpu = traverser.traverse(&rays).unwrap();
    let mut mismatch = 0u32;
    for (ray, g) in rays.iter().zip(&gpu) {
        if *g != mirror_traverse(&structure, ray) {
            mismatch += 1;
        }
    }
    assert_eq!(
        mismatch,
        0,
        "GPU disagreed with the f32 mirror on {mismatch}/{} grazing/axis-aligned rays",
        rays.len()
    );
}

#[test]
fn traverse_timed_matches_traverse_and_times_the_kernel() {
    // The timed path must return identical hits to the plain path, and — where
    // the device supports timestamp queries — a positive, finite kernel time.
    let Some(ctx) = context_or_skip() else { return };
    let field = OctantFractal::sierpinski_tetrahedron(res(128));
    let structure = SchoolBBuffer::from_sparse(&SparseTree::build(&field));
    let traverser = GpuTraverser::new(&ctx, &structure).unwrap();

    // A full 128² front-face batch, so the kernel clearly spans several
    // timestamp ticks (a handful of rays can round to a 0-tick pass).
    let mut rays = Vec::new();
    for y in 0..128u32 {
        for x in 0..128u32 {
            rays.push(Ray::new(
                DVec3::new(f64::from(x) + 0.5, f64::from(y) + 0.5, -1.0),
                DVec3::Z,
            ));
        }
    }

    let plain = traverser.traverse(&rays).unwrap();
    let _ = traverser.traverse_timed(&rays).unwrap(); // warm: prime the query set
    let (timed, gpu_ns) = traverser.traverse_timed(&rays).unwrap();
    assert_eq!(plain, timed, "timed and untimed hits must agree");

    if ctx.supports_timestamps() {
        let ns = gpu_ns.expect("timestamps supported ⇒ Some");
        assert!(
            ns > 0.0 && ns.is_finite(),
            "kernel time should be positive: {ns}"
        );
    }
}

#[allow(clippy::cast_precision_loss)]
fn check_fixture<F: OccupancyField + Sync>(ctx: &GpuContext, field: &F) {
    let structure = SchoolBBuffer::from_sparse(&SparseTree::build(field));
    let traverser = GpuTraverser::new(ctx, &structure).unwrap();
    let nf = f64::from(field.resolution().voxels_per_axis());

    let mut state = 0x1357_9BDF_2468_ACE0u64;
    let mut rays = Vec::new();
    for _ in 0..20_000 {
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
        rays.push(Ray::new(origin, dir));
    }

    let gpu = traverser.traverse(&rays).unwrap();
    assert_eq!(gpu.len(), rays.len());

    let mut vs_oracle = 0u32;
    let mut vs_mirror = 0u32;
    for (ray, gpu_hit) in rays.iter().zip(&gpu) {
        let oracle_hit: Option<VoxelCoord> = oracle::first_hit(field, ray).map(|h| h.voxel);
        if *gpu_hit != oracle_hit {
            vs_oracle += 1;
        }
        if *gpu_hit != mirror_traverse(&structure, ray) {
            vs_mirror += 1;
        }
    }

    let n = rays.len() as f64;
    let oracle_rate = f64::from(vs_oracle) / n;
    let mirror_rate = f64::from(vs_mirror) / n;
    eprintln!(
        "GPU diff ({} rays): vs oracle {vs_oracle} ({:.3}%), vs mirror {vs_mirror} ({:.3}%)",
        rays.len(),
        oracle_rate * 100.0,
        mirror_rate * 100.0,
    );

    // The GPU is an f32 kernel; disagreements with the f64 oracle are grazing
    // rays only and must stay a small bounded fraction — a kernel bug blows
    // far past this.
    assert!(
        oracle_rate < 0.01,
        "GPU vs oracle mismatch {:.3}% > 1%",
        oracle_rate * 100.0
    );
    // GPU and the f32 mirror run identical arithmetic; only fma fusion on the
    // GPU may differ, so they agree even more tightly.
    assert!(
        mirror_rate < 0.005,
        "GPU vs mirror mismatch {:.3}% > 0.5%",
        mirror_rate * 100.0
    );
}
