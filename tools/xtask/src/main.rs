//! Workspace build automation.
//!
//! Two gates, per the Engineering Codex *CI Quality Bar* and the GPU-on-CI
//! discipline from *Reference Path Cargo Wiring*:
//!
//! - `cargo xtask ci` — the always-green floor: format check, clippy with
//!   warnings-as-errors, build, test, and doc. Passes with **no GPU** present
//!   (the GPU differential test skips itself when no adapter is found).
//! - `cargo xtask ci-gpu` — sets `VOXEL_REQUIRE_GPU=1` so the GPU differential
//!   *fails* instead of skipping when no adapter is present. This closes the
//!   "a skipped test looks green" hole on GPU-equipped CI lanes.

use std::process::Command;

use anyhow::{Context, Result, bail};

fn main() -> Result<()> {
    let task = std::env::args().nth(1);
    match task.as_deref() {
        Some("ci") => ci(GpuPolicy::Optional),
        Some("ci-gpu") => ci(GpuPolicy::Required),
        Some(other) => bail!("unknown task {other:?}; usage: cargo xtask <ci|ci-gpu>"),
        None => bail!("missing task; usage: cargo xtask <ci|ci-gpu>"),
    }
}

/// Whether the GPU differential must run (`Required`) or may skip when no
/// adapter is present (`Optional`).
#[derive(Clone, Copy)]
enum GpuPolicy {
    Optional,
    Required,
}

fn ci(gpu: GpuPolicy) -> Result<()> {
    // (step label, cargo args, extra RUSTDOCFLAGS).
    let steps: &[(&str, &[&str])] = &[
        ("fmt", &["fmt", "--all", "--", "--check"]),
        (
            "clippy",
            &[
                "clippy",
                "--workspace",
                "--all-targets",
                "--",
                "-D",
                "warnings",
            ],
        ),
        ("test", &["test", "--workspace"]),
        ("doc", &["doc", "--workspace", "--no-deps"]),
    ];

    for (label, args) in steps {
        eprintln!("\n== xtask: cargo {}", args.join(" "));
        let mut cmd = Command::new(cargo());
        cmd.args(*args);
        if *label == "doc" {
            // Documentation as Truth: broken doc links fail the build.
            cmd.env("RUSTDOCFLAGS", "-D warnings");
        }
        if *label == "test" {
            if let GpuPolicy::Required = gpu {
                cmd.env("VOXEL_REQUIRE_GPU", "1");
            }
        }
        let status = cmd
            .status()
            .with_context(|| format!("failed to spawn `cargo {}`", args.join(" ")))?;
        if !status.success() {
            bail!("ci step '{label}' failed");
        }
    }

    eprintln!("\nxtask: all steps passed.");
    Ok(())
}

/// The cargo binary that invoked us (so the right toolchain is reused).
fn cargo() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned())
}
