//! The viewer must exit **non-zero** when initialization fails.
//!
//! `resumed` exits the winit event loop cleanly even when `Viewer::new` fails,
//! so `run_app` returns `Ok` and — before the fix — `main` exited 0 regardless.
//! A scripted/`--frames` profiling run then could not distinguish a failed load
//! from a successful render. This pins the init-error → non-zero-exit contract.
//!
//! The viewer is a windowed binary, so this spawns it as a child process. In an
//! environment that cannot create an event loop / window (headless CI), the
//! child either fails fast (still non-zero, fine) or never reaches `resumed`; in
//! the latter case the deadline elapses and the test **skips** rather than
//! fails, mirroring the GPU-gated tests elsewhere in the workspace.

use std::process::{Command, ExitStatus};
use std::time::{Duration, Instant};

/// Spawn the viewer with arguments that force an init failure and return its
/// exit status. Returns `None` (→ skip) if the binary cannot be spawned or does
/// not exit within the deadline (no windowing environment).
fn run_until_exit(args: &[&str]) -> Option<ExitStatus> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_voxel-viewer"))
        .args(args)
        .spawn()
        .ok()?;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                eprintln!("skip: viewer did not exit within deadline (no event loop?)");
                return None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(_) => return None,
        }
    }
}

/// A `--mesh` pointing at a file that does not exist fails the load inside
/// `Viewer::new`; the process must exit non-zero. (`--frames 1` guarantees the
/// run terminates promptly even if the load were somehow to succeed.) Any other
/// init failure — bad `--res`, GPU device request — propagates through the same
/// `init_error` path, so this one case pins the whole contract.
#[test]
fn nonexistent_mesh_exits_nonzero() {
    let Some(status) = run_until_exit(&[
        "--mesh",
        "/nonexistent-voxelizer-hardening-test-mesh.glb",
        "--frames",
        "1",
    ]) else {
        return; // skipped: no windowing environment available
    };
    assert!(
        !status.success(),
        "a missing mesh must exit non-zero, got {status:?}"
    );
}
