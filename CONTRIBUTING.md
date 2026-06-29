# Contributing

## The merge gate

All work is gated by `cargo xtask` (a workspace member, so the gate is itself
testable code, not opaque CI YAML). Three commands, in increasing strength:

| Command | What it checks | Where it runs |
| --- | --- | --- |
| `cargo xtask ci` | The always-green floor: `fmt --check`, `clippy --workspace --all-targets -D warnings`, a `voxelizer` **feature-matrix** lane (each `gltf`/`obj`/`stl` combo), `test --workspace --all-targets`, doctests, and `doc` with `RUSTDOCFLAGS=-D warnings`. GPU differential tests **skip** when no adapter is present. | Automatically on every PR / push to `main` ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). |
| `cargo xtask ci-gpu` | Everything `ci` does, **plus** `VOXEL_REQUIRE_GPU=1` so the CPU↔GPU differential tests **fail instead of skipping** when no adapter is found. | **Local, on a GPU host (Metal/Vulkan) — this is the merge gate.** Hosted CI runners have no reliable GPU, so it is not automated; run it yourself before merging. |
| `cargo xtask ci-deps` | Supply-chain: `cargo-deny check` (advisories / bans / licenses / sources) against [`deny.toml`](deny.toml). Needs `cargo install --locked cargo-deny`. | The `supply-chain` job in CI runs the equivalent check. |

**Before merging to `main`:** run `cargo xtask ci-gpu` locally on a GPU host and
confirm it is green. The hosted CI proves the floor on every PR, but the
bit-exact CPU↔GPU contracts are only *witnessed* (not skipped) on real hardware.

## Conventions

- The dependency graph runs strictly inward toward the pure, GPU/IO-free
  `voxel-core`; `unsafe_code = deny` workspace-wide (scoped, audited allows
  only). Mesh IO lives in `voxelizer::io` and must stay fenced off from the
  format-agnostic compute modules (enforced by
  `crates/voxelizer/tests/io_compute_boundary.rs`).
- Engineering principles (testing, error handling, type-driven design,
  workspace shape) follow the vault under `Engineering_Codex/`.
