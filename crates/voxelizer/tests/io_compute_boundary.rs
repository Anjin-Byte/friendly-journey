//! Enforces the `io ⊥ compute` invariant within the single `voxelizer` crate.
//!
//! The crate keeps its outside-world IO (`src/io/`) fenced off from the
//! format-agnostic compute modules. A single crate cannot express that to the
//! compiler — sibling `pub mod`s freely see each other's `pub(crate)` items — so
//! this source-text check is the substitute until the future `voxel-io`
//! extraction (after which the compiler enforces the edge for free and this test
//! should be DELETED).
//!
//! Three-way classification, the only two banned directions:
//!   - files under `src/io/`  MUST NOT name a COMPUTE path (`crate::bake` etc.);
//!   - COMPUTE files          MUST NOT name `crate::io`;
//!   - everything else (`lib.rs`, `core.rs`, `error.rs`, `appearance.rs`) is
//!     EXEMPT — it is the shared DTO/wiring layer. Do NOT add `core`/`error`/
//!     `appearance` to either ban list, and `lib.rs` legitimately re-exports
//!     `crate::io::…` upward, so it must never be classified as compute.
//!
//! Legitimate edges that must NOT flag: `io` to `core` / `error` / `appearance`
//! / `voxel_core`; compute to those three; and compute to compute (e.g. a
//! `gpu/*` file naming `crate::csr` / `crate::reference_cpu`).

use std::path::{Path, PathBuf};

/// Fully-qualified compute-module paths an `io` file must never name.
const COMPUTE_PATHS: &[&str] = &[
    "crate::bake",
    "crate::csr",
    "crate::gpu",
    "crate::reference_cpu",
    "crate::materials",
    "crate::truecolor",
];

/// A file is COMPUTE if its path contains one of these. `gpu` is a *directory*
/// module, so match `/gpu/` to cover every `gpu/*.rs` submodule.
fn is_compute(path: &str) -> bool {
    [
        "/bake.rs",
        "/csr.rs",
        "/gpu/",
        "/gpu.rs",
        "/reference_cpu.rs",
        "/materials.rs",
        "/truecolor.rs",
    ]
    .iter()
    .any(|m| path.contains(m))
}

/// The code portion of a line: everything before the first `//`. This drops
/// full-line doc/comments (`//`, `//!`) and trailing comments in one move, so a
/// compute path mentioned in prose (the module-map doc, this crate's own header
/// notes) never false-positives — only real code references count.
fn code_of(line: &str) -> &str {
    line.split("//").next().unwrap_or("")
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read_dir src") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn io_and_compute_do_not_import_each_other() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_rs(&src, &mut files);
    assert!(!files.is_empty(), "found no source files under {src:?}");

    let mut violations: Vec<String> = Vec::new();
    for file in &files {
        let path_str = file.to_string_lossy().replace('\\', "/");
        let in_io = path_str.contains("/io/");
        let in_compute = is_compute(&path_str);
        // Exempt layer (lib.rs / core.rs / error.rs / appearance.rs): neither.
        if !in_io && !in_compute {
            continue;
        }
        let text = std::fs::read_to_string(file).expect("read source file");
        for (i, line) in text.lines().enumerate() {
            let code = code_of(line);
            if in_io {
                for bad in COMPUTE_PATHS {
                    if code.contains(bad) {
                        violations.push(format!(
                            "{}:{} — io file names compute path `{bad}`",
                            path_str,
                            i + 1
                        ));
                    }
                }
            } else if code.contains("crate::io") {
                violations.push(format!(
                    "{}:{} — compute file names `crate::io`",
                    path_str,
                    i + 1
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "io ⊥ compute boundary violated:\n{}",
        violations.join("\n")
    );
}
