//! Output adapters (DEFERRED) — the mirror of [`super::import`]: lower a
//! voxelization result back to an external mesh format.
//!
//! Nothing is built here yet. The export source-of-truth is the voxel structure
//! (`voxel_core::SparseTree` / `SchoolBBuffer` / `MaterialTable`), read as an
//! output adapter parallel to import; a geometry-lowering pass (voxel-cubes, then
//! re-mesh) would feed a `MeshOutput` DTO that per-format writers serialize. The
//! DTO, geometry strategies, per-format writers, streaming, and any
//! source/sink traits are spec-only for now and intentionally unimplemented —
//! see `docs/materials/11` and the IO-boundary design notes for the deferral
//! rationale and the `voxel-io` extraction triggers.

// (intentionally empty: the output direction is a documented placeholder)
