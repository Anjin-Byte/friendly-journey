//! Surface / conservative mesh voxelizer.
//!
//! Turns a triangle-soup [`MeshInput`] into `voxel-core`-native occupancy
//! ([`VoxelOccupancy`], which implements [`OccupancyField`]) that feeds the
//! engine's [`SparseTree`] / [`SchoolBBuffer`] structures.
//!
//! A `wgpu` compute path ([`gpu`]) performs the voxelization on the GPU and is
//! validated against the CPU SAT oracle in [`reference_cpu`]: bit-exact on
//! tangent-free meshes, and at floating-point tangent voxels the GPU is a
//! *conservative superset* (it may over-mark a boundary voxel by one, never
//! under-marks).
//!
//! Module map:
//! - [`core`] — public types (grids, tiles, mesh input, outputs).
//! - [`csr`] — CPU tile / brick binning that maps grid partitions to candidate
//!   triangles (compressed sparse row).
//! - [`gpu`] — the `wgpu` compute pipeline ([`GpuVoxelizer`]).
//! - [`reference_cpu`] — the CPU SAT reference voxelizer used as a test oracle.
//! - [`loader`] — input adapters that read external mesh formats into
//!   [`MeshInput`] (glTF/GLB, OBJ, and STL), behind one [`load_mesh`]
//!   dispatcher. Gated behind the `gltf` / `obj` / `stl` cargo features.

// GPU index / dimension arithmetic converts freely between integer widths and
// `f32`/`f64` for workgroup, brick, and voxel counts; these conversions are
// intentional and bounded by device limits. Follows the workspace precedent in
// `voxel-viewer` / `voxel-gpu`.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

pub mod bake;
pub mod core;
pub mod csr;
pub mod error;
pub mod gpu;
pub mod loader;
pub mod materials;
pub mod reference_cpu;
pub mod truecolor;

pub use crate::core::{
    CompactVoxel, DispatchStats, MeshInput, SparseVoxelizationOutput, TileSpec, VoxelGrid,
    VoxelOccupancy, VoxelizationOutput, VoxelizeOpts,
};
pub use crate::error::{VoxelizeGpuError, VoxelizerError};
pub use crate::gpu::{GpuVoxelizer, GpuVoxelizerConfig};
#[cfg(feature = "gltf")]
pub use crate::loader::{load_gltf_path, load_gltf_slice};
#[cfg(any(feature = "gltf", feature = "obj", feature = "stl"))]
pub use crate::loader::{load_mesh, rotation_degrees};
#[cfg(feature = "obj")]
pub use crate::loader::{load_obj_path, load_obj_slice};
#[cfg(feature = "stl")]
pub use crate::loader::{load_stl_path, load_stl_slice};
pub use crate::materials::{apply_mesh_materials, material_table_for_sparse, tree_from_compact};
pub use crate::truecolor::{bake_leaf_colors, cull_mask_cutout};

// The `voxel-core` types that appear in this crate's public API, re-exported so
// callers (and the renderer bridge) need not depend on `voxel-core` directly.
pub use voxel_core::{
    MaterialTable, OccupancyField, Resolution, SchoolBBuffer, SparseTree, VoxelCoord,
};
