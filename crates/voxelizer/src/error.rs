//! Typed errors at the voxelizer's boundaries (Engineering Codex: *Domain Errors
//! at Boundaries*). The CPU/validation surface and the GPU surface get distinct
//! enums; neither leaks raw `wgpu` types beyond the `#[from]` conversions the
//! caller can still match on.

use thiserror::Error;

/// Errors from the pure CPU validation boundary (grid/tile/mesh specs).
#[derive(Debug, Clone, PartialEq, Error)]
pub enum VoxelizerError {
    /// `voxel_size` was not finite or not strictly positive.
    #[error("voxel_size must be finite and > 0 (got {0})")]
    NonPositiveVoxelSize(f32),

    /// A grid dimension was zero (the grid must have at least one voxel per axis).
    #[error("grid dimensions must be >= 1")]
    ZeroGridDim,

    /// The grid origin was not finite.
    #[error("origin_world must be finite")]
    NonFiniteOrigin,

    /// The supplied `world_to_grid` transform was not finite.
    #[error("world_to_grid must be finite")]
    NonFiniteTransform,

    /// A tile dimension was zero (each tile must span at least one voxel per axis).
    #[error("tile_dims must be >= 1")]
    ZeroTileDim,

    /// The tile's voxel count exceeds the device's per-workgroup invocation limit.
    #[error("tile_dims product must be <= {limit} (got {got})")]
    TileTooLarge {
        /// The tile's voxel count.
        got: u32,
        /// The device's `max_compute_invocations_per_workgroup`.
        limit: u32,
    },

    /// The `material_ids` length did not match the triangle count.
    #[error("material_ids length ({ids}) must match triangles length ({tris})")]
    MaterialIdLenMismatch {
        /// Number of supplied material ids.
        ids: usize,
        /// Number of triangles.
        tris: usize,
    },

    /// A triangle contained a non-finite vertex.
    #[error("triangle contains a non-finite vertex")]
    NonFiniteVertex,

    /// A wrapped occupancy word buffer was shorter than `ceil(n³/32)`.
    #[error("occupancy word buffer too small: got {got} words, need {need}")]
    OccupancyBufferTooSmall {
        /// Number of words supplied.
        got: usize,
        /// Number of words required (`ceil(n³/32)`).
        need: usize,
    },

    /// The voxelize epsilon was not finite or was negative.
    #[error("voxelize epsilon must be finite and >= 0 (got {0})")]
    InvalidEpsilon(f32),

    /// `store_color` was requested without `store_owner`.
    #[error("store_color requires store_owner (color is hashed from the owning triangle)")]
    ColorRequiresOwner,

    /// `voxel_size` was so small its reciprocal is not finite (e.g. a deep
    /// subnormal), which would make the derived world→grid matrix non-finite.
    #[error("voxel_size {0} is too small: its reciprocal is not finite")]
    VoxelSizeTooSmall(f32),

    /// An input adapter (e.g. the glTF loader) failed to parse, read, or
    /// validate a source mesh. Carries the underlying cause as a `String` so
    /// this enum stays `Clone`/`PartialEq` (the upstream error types are not),
    /// mapped via `.to_string()` at the boundary.
    #[error("failed to load mesh: {0}")]
    MeshLoad(String),
}

/// Errors from the GPU boundary: adapter probing, dispatch limits, buffer
/// readback, and shader/pipeline validation.
#[derive(Debug, Error)]
pub enum VoxelizeGpuError {
    /// No compatible GPU adapter is present.
    #[error("no compatible GPU adapter found")]
    NoAdapter,

    /// The adapter would not grant a device with the requested limits.
    #[error("failed to request GPU device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),

    /// Mapping a buffer for readback failed.
    #[error("GPU buffer mapping failed: {0}")]
    BufferMap(#[from] wgpu::BufferAsyncError),

    /// The device was polled or a readback channel closed before completing.
    #[error("GPU device poll failed")]
    Poll,

    /// A dispatch's workgroup count exceeds the device's per-dimension limit.
    #[error("{label}: workgroups {workgroups} exceed max {limit}")]
    WorkgroupsExceeded {
        /// Label of the dispatch that overflowed.
        label: &'static str,
        /// Requested workgroup count.
        workgroups: u32,
        /// The device's `max_compute_workgroups_per_dimension`.
        limit: u32,
    },

    /// A storage buffer would exceed the adapter's `max_storage_buffer_binding_size`.
    #[error("{label}: buffer size {bytes} bytes exceeds max {limit} bytes")]
    StorageExceeded {
        /// Label of the buffer that overflowed.
        label: &'static str,
        /// Requested size in bytes.
        bytes: u64,
        /// The adapter's per-binding storage limit.
        limit: u64,
    },

    /// An operation required `store_owner = true` but it was not set.
    #[error("compact_surface_sparse requires store_owner = true")]
    OwnerRequired,

    /// `brick_dim` was zero, or its cube (`brick_dim³`, the per-brick voxel
    /// count) overflows `u32`. Caught up front so the compaction validators
    /// cannot panic on a multiply-overflow.
    #[error("invalid brick_dim {got}: zero or its cube overflows u32")]
    InvalidBrickDim {
        /// The offending brick dimension.
        got: u32,
    },

    /// A pipeline or compute pass failed shader/validation, or produced an
    /// internally inconsistent result (e.g. missing owner ids, empty dispatch).
    #[error("{0}")]
    PipelineValidation(String),
}
