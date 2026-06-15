//! Pure domain core for the sparse MIP voxel structure.
//!
//! This crate holds the deterministic, GPU-free reference implementation that
//! everything else is measured against: domain types, the dense
//! Amanatides-Woo oracle, the School-B buffer builder, the stackless HDDA
//! reference traversal, and the `bytemuck` buffer contract shared with the GPU
//! adapter. It is the correctness oracle the optimized `voxel-gpu` adapter is
//! diffed against (Engineering Codex: *Reference Implementation as Oracle*).
//!
//! Everything here is pure and deterministic — no device access, no I/O, no
//! windowing — so it runs anywhere, including in CI with no GPU and on
//! `wasm32`. The specification implemented is `idea.md` at the repository root;
//! section references in the source (e.g. "§7.2") point there.
//!
//! # Layout
//!
//! - [`Resolution`] / [`Level`] — the `8·4^k` grid sizes and the traversal
//!   level formulas, made illegal-by-type.
//! - [`VoxelCoord`] and [`morton`] — integer coordinates and the build-time
//!   Z-order codec.
//! - [`Ray`] / [`ray_aabb`] — the `f64` ray and an independent slab test.
//! - [`OccupancyField`] / [`BitGrid`] / [`fixtures`] — the binary input field.
//! - [`oracle`] — the Tier-A reference traversal (the correctness oracle).
//!
//! Later phases (`idea.md` §11) add the School-B builder, the `f32` mirror
//! traversal, the GPU buffer contract, and the §10 measurement harness.

pub mod fixtures;
pub mod layout;
pub mod measure;
pub mod mip;
pub mod mirror;
pub mod morton;
pub mod node;
pub mod oracle;
pub mod school_b;
pub mod sparse;

mod coord;
mod dda;
mod leaf;
mod level;
mod noise;
mod occupancy;
mod ray;
mod resolution;

pub use coord::VoxelCoord;
pub use layout::{Cell, NodeLayout, TraversalStats, traverse};
pub use leaf::{LeafBounds, LeafBrick};
pub use level::Level;
pub use mip::BrickGrid;
pub use mirror::mirror_traverse;
pub use node::GpuNode;
pub use occupancy::{BitGrid, OccupancyField};
pub use oracle::Hit;
pub use ray::{Ray, ray_aabb};
pub use resolution::{Resolution, ResolutionError};
pub use school_b::SchoolBBuffer;
pub use sparse::{Edit, SparseTree};
