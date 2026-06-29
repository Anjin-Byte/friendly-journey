//! End-to-end CLI contract tests for the `voxel` binary.
//!
//! `voxel-cli` is the workspace default-member, so its arg parsing and
//! exit-code behaviour are part of the headless interface. These spawn the built
//! binary (`CARGO_BIN_EXE_voxel`, the pattern proven in
//! `voxel-viewer/tests/exit_code.rs`) and assert on exit status. All cases are
//! deterministic and GPU-agnostic: they use tiny fixtures and the CPU backend,
//! so no GPU adapter is required.

use std::process::Command;

/// Run `voxel <args>` and return its exit status.
fn run(args: &[&str]) -> std::process::ExitStatus {
    Command::new(env!("CARGO_BIN_EXE_voxel"))
        .args(args)
        .output()
        .expect("spawn voxel binary")
        .status
}

#[test]
fn help_exits_zero() {
    assert!(run(&["--help"]).success(), "--help must exit 0");
}

#[test]
fn measure_runs_on_a_tiny_fixture() {
    // A pure-CPU measurement on an 8³ fixture — fast, deterministic, no GPU.
    let status = run(&[
        "measure",
        "--fixture",
        "sierpinski",
        "--res",
        "8",
        "--rays",
        "50",
    ]);
    assert!(status.success(), "measure must succeed, got {status:?}");
}

#[test]
fn diff_cpu_backend_runs_without_a_gpu() {
    // `--backend cpu` diffs the CPU mirror against the f64 reference: no adapter
    // needed, so this exercises the build → rays → traverse → diff path headlessly.
    let status = run(&[
        "diff",
        "--backend",
        "cpu",
        "--fixture",
        "sierpinski",
        "--res",
        "8",
        "--rays",
        "50",
    ]);
    assert!(
        status.success(),
        "diff --backend cpu must succeed, got {status:?}"
    );
}

#[test]
fn invalid_resolution_exits_nonzero() {
    // 7 is not a valid `8·4^k` resolution → `Resolution::new` errors → non-zero exit.
    let status = run(&["measure", "--fixture", "solid", "--res", "7"]);
    assert!(
        !status.success(),
        "an invalid --res must exit non-zero, got {status:?}"
    );
}

#[test]
fn unknown_subcommand_exits_nonzero() {
    assert!(
        !run(&["definitely-not-a-command"]).success(),
        "an unknown subcommand must exit non-zero (clap usage error)"
    );
}
