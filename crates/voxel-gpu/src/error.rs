//! Typed errors at the GPU adapter boundary (Engineering Codex: *Domain Errors
//! at Boundaries*). These do not leak raw wgpu error types beyond `#[from]`
//! conversions the caller can still match on.

use thiserror::Error;

/// Anything that can go wrong setting up or running the GPU traversal.
#[derive(Debug, Error)]
pub enum GpuError {
    /// No compatible GPU adapter is present — the runtime gate that lets CPU-only
    /// CI skip the GPU path (review R2).
    #[error("no compatible GPU adapter found")]
    NoAdapter,

    /// The adapter would not grant a device with the requested limits.
    #[error("failed to request GPU device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),

    /// Mapping a buffer for readback failed.
    #[error("GPU buffer mapping failed: {0}")]
    BufferMap(#[from] wgpu::BufferAsyncError),

    /// A storage buffer would exceed the adapter's `max_storage_buffer_binding_size`.
    #[error("structure needs {needed} B but the adapter caps storage bindings at {limit} B")]
    BufferTooLarge {
        /// Bytes the structure needs in one binding.
        needed: u64,
        /// The adapter's per-binding limit.
        limit: u64,
    },

    /// The device was polled and reported an internal failure.
    #[error("GPU device poll failed")]
    Poll,
}
