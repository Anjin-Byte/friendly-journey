//! wgpu adapter: the optimized GPU traversal path.
//!
//! Wraps wgpu behind a narrow API and runs the stackless Hierarchical DDA
//! (`idea.md` §7) as a WGSL compute kernel over the School-B buffer built by
//! [`voxel_core`]. It is the *optimized* path in the reference/optimized pair;
//! the canonical answer lives in `voxel-core`, and the differential tests
//! cross-validate the two.
//!
//! GPU availability is discovered at **runtime** ([`GpuContext::try_new`]
//! returns [`GpuError::NoAdapter`] when absent), not via a Cargo feature — the
//! Engineering Codex forbids a "CPU vs GPU" feature toggle. The crate always
//! compiles; handling the absence of a GPU is the caller's job.

mod buffers;
mod context;
mod error;
mod render;
mod traverse;

pub use context::GpuContext;
pub use error::GpuError;
pub use render::{GpuCamera, GpuRenderer, OUTPUT_FORMAT};
pub use traverse::GpuTraverser;
