//! WGSL sources for the GPU voxelizer + sparse-brick compaction passes.
//!
//! Kept as separate `shaders/*.wgsl` files (loaded via `include_str!`) like the
//! rest of the workspace, rather than inlined Rust raw strings. The constant
//! names are unchanged, so [`super::pipelines`] consumes them exactly as before.

/// Surface voxelizer compute pass (dense grid + sparse brick paths).
pub(crate) const VOXELIZER_WGSL: &str = include_str!("../../shaders/voxelize.wgsl");

/// Sparse-brick → compacted-position compaction pass.
pub(crate) const COMPACT_WGSL: &str = include_str!("../../shaders/compact.wgsl");

/// Sparse-brick → compacted-voxel (global coord + material) compaction pass.
pub(crate) const COMPACT_VOXELS_WGSL: &str = include_str!("../../shaders/compact_voxels.wgsl");

/// Sparse-brick → compacted-attribute compaction pass.
pub(crate) const COMPACT_ATTRS_WGSL: &str = include_str!("../../shaders/compact_attrs.wgsl");
