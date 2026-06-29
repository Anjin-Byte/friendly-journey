//! IO boundary — the crate's interface with the diverse, messy outside world.
//!
//! [`import`] reads external mesh formats into the crate's world-space
//! [`MeshInput`](crate::core::MeshInput) (glTF/GLB, OBJ, STL), behind one
//! [`load_mesh`] dispatcher, each format gated by its
//! `gltf`/`obj`/`stl` cargo feature. [`export`] is the deferred output side
//! (the mirror of import).
//!
//! This module is deliberately fenced off from the format-agnostic compute
//! modules (`reference_cpu` / `csr` / `gpu` / `bake` / `materials` / `truecolor`)
//! — the `io ⊥ compute` invariant, enforced by `tests/io_compute_boundary.rs` —
//! and is the pre-staging point for a future `voxel-io` crate extraction.

pub mod export;
pub mod import;

#[cfg(feature = "gltf")]
pub use import::{load_gltf_path, load_gltf_slice};
#[cfg(any(feature = "gltf", feature = "obj", feature = "stl"))]
pub use import::{load_mesh, rotation_degrees};
#[cfg(feature = "obj")]
pub use import::{load_obj_path, load_obj_slice};
#[cfg(feature = "stl")]
pub use import::{load_stl_path, load_stl_slice};
