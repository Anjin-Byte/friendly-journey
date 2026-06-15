//! The GPU context: a runtime-probed device and queue.
//!
//! `try_new` returns [`GpuError::NoAdapter`] when no GPU is present, which is
//! how the differential test skips on CPU-only CI and how `xtask ci-gpu`
//! detects a missing adapter (review R2). There is no Cargo feature gating the
//! GPU — `voxel-gpu` always compiles.

use crate::error::GpuError;

/// A ready GPU device and queue.
pub struct GpuContext {
    /// The wgpu device.
    pub device: wgpu::Device,
    /// The wgpu queue.
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Probes for a GPU adapter and requests a device, or returns
    /// [`GpuError::NoAdapter`] if none is available.
    ///
    /// Requests a raised `max_storage_buffer_binding_size` so larger structures
    /// fit; falls back to the adapter's reported maximum.
    pub fn try_new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|_| GpuError::NoAdapter)?;

        // Ask for as much storage-buffer headroom as the adapter allows.
        let adapter_limits = adapter.limits();
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            ..wgpu::Limits::default()
        };

        // Enable compute-pass timestamps when the adapter supports them, so the
        // traverser can time the kernel on the GPU timeline (readback-free). It
        // is optional — absent it, timed paths fall back to wall-clock.
        let mut features = wgpu::Features::empty();
        if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("voxel-gpu device"),
                required_features: features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            }))?;

        Ok(Self { device, queue })
    }

    /// Whether compute-pass timestamp queries are available on this device.
    #[must_use]
    pub fn supports_timestamps(&self) -> bool {
        self.device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
    }

    /// The adapter's per-binding storage-buffer size cap.
    #[must_use]
    pub fn max_storage_binding(&self) -> u64 {
        self.device.limits().max_storage_buffer_binding_size
    }
}
