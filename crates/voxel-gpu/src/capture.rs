//! Programmatic GPU-trace capture: wrap a workload so Metal writes a
//! `.gputrace` document to disk, openable in Xcode's GPU debugger (full counters
//! and shader profiler) with no attaching, no `xctrace`, no scheme. macOS-only;
//! see `CAPTURE.md`.

// Unsafe Quarantine: the Objective-C message sends into `MTLCaptureManager`.
#![allow(unsafe_code)]

use std::path::Path;

use crate::error::GpuError;

/// Runs `workload` while a Metal GPU trace is recorded to `path` (a `.gputrace`
/// bundle), openable in Xcode. Captures the system-default Metal device — the
/// same physical GPU `wgpu` uses on a single-GPU Apple-silicon Mac — so every
/// command buffer the workload submits lands in the trace.
///
/// Requires `METAL_CAPTURE_ENABLED=1` in the environment (programmatic capture
/// is gated; the `make gputrace` target sets it). Keep the workload to a few
/// dispatches — a trace document replays every captured command buffer.
#[cfg(target_os = "macos")]
pub fn capture_gputrace<R>(
    path: &Path,
    workload: impl FnOnce() -> Result<R, GpuError>,
) -> Result<R, GpuError> {
    use objc2::runtime::AnyObject;
    use objc2_foundation::{NSString, NSURL};
    use objc2_metal::{
        MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager,
        MTLCreateSystemDefaultDevice,
    };

    let device = MTLCreateSystemDefaultDevice()
        .ok_or_else(|| GpuError::Capture("no system-default Metal device".to_string()))?;
    // SAFETY: `sharedCaptureManager` returns the process-wide singleton.
    let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };

    let descriptor = MTLCaptureDescriptor::new();
    let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
    let capture_object: &AnyObject = (*device).as_ref();
    // SAFETY: setCaptureObject takes an `id`; the system-default device is a
    // valid, live capture target.
    unsafe { descriptor.setCaptureObject(Some(capture_object)) };
    descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
    descriptor.setOutputURL(Some(&url));

    manager
        .startCaptureWithDescriptor_error(&descriptor)
        .map_err(|e| {
            GpuError::Capture(format!(
                "startCapture failed (is METAL_CAPTURE_ENABLED=1 set?): {e:?}"
            ))
        })?;

    let result = workload();

    manager.stopCapture();
    result
}

/// Non-macOS stub: programmatic GPU-trace capture is a Metal feature.
#[cfg(not(target_os = "macos"))]
pub fn capture_gputrace<R>(
    _path: &Path,
    _workload: impl FnOnce() -> Result<R, GpuError>,
) -> Result<R, GpuError> {
    Err(GpuError::Capture(
        "programmatic .gputrace capture is macOS-only".to_string(),
    ))
}
