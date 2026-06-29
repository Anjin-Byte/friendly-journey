//! Workspace build automation.
//!
//! Three gates, per the Engineering Codex *CI Quality Bar* and the GPU-on-CI
//! discipline from *Reference Path Cargo Wiring*:
//!
//! - `cargo xtask ci` — the always-green floor: format check; clippy with
//!   warnings-as-errors (workspace, plus a `voxelizer` **feature-matrix** lane so
//!   a non-default loader-feature combo can't slip the gate); test
//!   (`--all-targets`) and doctests; and doc. Passes with **no GPU** present (the
//!   GPU differential tests skip themselves when no adapter is found).
//! - `cargo xtask ci-gpu` — sets `VOXEL_REQUIRE_GPU=1` so the GPU differentials
//!   *fail* instead of skipping when no adapter is present. This closes the
//!   "a skipped test looks green" hole on GPU-equipped CI lanes.
//! - `cargo xtask ci-deps` — supply-chain audit (advisories / bans / licenses /
//!   sources) via `cargo-deny`. **Opt-in and tooling-gated** (needs the external
//!   `cargo-deny` binary), so it is deliberately *not* part of the always-green
//!   `ci` floor; justified by the untrusted `gltf`/`obj`/`stl`/`image` byte-parsers.

use std::process::Command;

use anyhow::{Context, Result, bail};

fn main() -> Result<()> {
    let task = std::env::args().nth(1);
    match task.as_deref() {
        Some("ci") => ci(GpuPolicy::Optional),
        Some("ci-gpu") => ci(GpuPolicy::Required),
        Some("ci-deps") => ci_deps(),
        Some(other) => bail!("unknown task {other:?}; usage: cargo xtask <ci|ci-gpu|ci-deps>"),
        None => bail!("missing task; usage: cargo xtask <ci|ci-gpu|ci-deps>"),
    }
}

/// Whether the GPU differentials must run (`Required`) or may skip when no
/// adapter is present (`Optional`).
#[derive(Clone, Copy)]
enum GpuPolicy {
    Optional,
    Required,
}

/// Run one `cargo <args>` step, failing the gate if it does not succeed. `envs`
/// sets extra environment variables for this step only.
fn run_step(label: &str, args: &[&str], envs: &[(&str, &str)]) -> Result<()> {
    eprintln!("\n== xtask: cargo {}", args.join(" "));
    let mut cmd = Command::new(cargo());
    cmd.args(args);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let status = cmd
        .status()
        .with_context(|| format!("failed to spawn `cargo {}`", args.join(" ")))?;
    if !status.success() {
        bail!("ci step '{label}' failed");
    }
    Ok(())
}

fn ci(gpu: GpuPolicy) -> Result<()> {
    run_step("fmt", &["fmt", "--all", "--", "--check"], &[])?;
    run_step(
        "clippy",
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
        &[],
    )?;
    // The default workspace clippy only sees the all-features-on set; the
    // voxelizer loader features (gltf/obj/stl) are otherwise unexercised, so a
    // break under a non-default combo (e.g. an example/test importing a cfg'd-out
    // symbol) would slip the gate. This lane closes that.
    feature_matrix()?;

    // GPU differentials honor VOXEL_REQUIRE_GPU; pass it through under ci-gpu.
    let test_env: &[(&str, &str)] = match gpu {
        GpuPolicy::Required => &[("VOXEL_REQUIRE_GPU", "1")],
        GpuPolicy::Optional => &[],
    };
    // `--all-targets` covers examples/benches but EXCLUDES doctests, so the
    // doctest rung is a separate explicit run (Documentation as Truth).
    run_step("test", &["test", "--workspace", "--all-targets"], test_env)?;
    run_step("doctest", &["test", "--workspace", "--doc"], test_env)?;

    run_step(
        "doc",
        &["doc", "--workspace", "--no-deps"],
        // Documentation as Truth: broken doc links fail the build.
        &[("RUSTDOCFLAGS", "-D warnings")],
    )?;

    eprintln!("\nxtask: all steps passed.");
    Ok(())
}

/// Clippy the `voxelizer` feature matrix: every loader-feature combination must
/// compile clean, not just the default all-on set.
fn feature_matrix() -> Result<()> {
    let combos: &[&[&str]] = &[
        &["--no-default-features"],
        &["--no-default-features", "--features", "gltf"],
        &["--no-default-features", "--features", "obj"],
        &["--no-default-features", "--features", "stl"],
        &["--all-features"],
    ];
    for combo in combos {
        let mut args = vec!["clippy", "-p", "voxelizer", "--all-targets"];
        args.extend_from_slice(combo);
        args.extend_from_slice(&["--", "-D", "warnings"]);
        run_step("features", &args, &[])?;
    }
    Ok(())
}

/// Supply-chain audit via `cargo-deny` (advisories, bans, licenses, sources).
/// Opt-in: not part of the always-green `ci` floor because it needs the external
/// `cargo-deny` binary and a fresh advisory database.
fn ci_deps() -> Result<()> {
    // Friendly preflight: a missing binary would otherwise surface as a cryptic
    // spawn error from `cargo deny`.
    let have_deny = Command::new(cargo())
        .args(["deny", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if !have_deny {
        bail!(
            "cargo-deny not found; install it with `cargo install --locked cargo-deny` \
             (this step is opt-in and not part of `cargo xtask ci`)"
        );
    }
    run_step("ci-deps", &["deny", "check"], &[])?;
    eprintln!("\nxtask: supply-chain audit passed.");
    Ok(())
}

/// The cargo binary that invoked us (so the right toolchain is reused).
fn cargo() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned())
}
