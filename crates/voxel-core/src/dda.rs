//! A single-level Amanatides–Woo walker, shared across hierarchy levels.
//!
//! [`DdaWalker`] steps a ray through a uniform grid of `dim³` cells of edge
//! `cell_size`, anchored at a world `origin`. The hierarchical traversal (P2's
//! two-level march, and the general HDDA in P4) drives one walker per level:
//! the coarse level walks bricks, and on a hit it spawns an inner walker over
//! the `8³` voxels of that brick — the **descend recompute** of `idea.md` §7.3,
//! since the inner walker's `t_max` is computed afresh from the brick entry
//! point, never inherited.

use crate::ray::Ray;

/// Floors a cell-space coordinate to an index clamped into `[0, dim)`.
///
/// The single audited `f64 → u32` site in the walker: the value is clamped
/// non-negative and below `dim ≤ 2³¹` before the cast.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub(crate) fn floor_to_cell(v: f64, dim: u32) -> u32 {
    let max = f64::from(dim - 1);
    v.floor().clamp(0.0, max) as u32
}

/// Amanatides–Woo traversal state over a `dim³` grid of `cell_size` cells.
pub(crate) struct DdaWalker {
    cell: [u32; 3],
    /// Per-axis step direction (`-1`, `0`, or `+1`).
    step: [i8; 3],
    t_max: [f64; 3],
    t_delta: [f64; 3],
    dim: u32,
    /// Ray parameter at which the current cell was entered.
    t_entry: f64,
}

impl DdaWalker {
    /// Begins traversal at the cell containing the ray's position at `t_enter`.
    ///
    /// `origin` is the grid's lower world corner, `dim` the cells per axis, and
    /// `cell_size` the cell edge in world units. The caller is responsible for
    /// having clipped the ray so `t_enter` is where it enters the grid box.
    pub(crate) fn enter(
        ray: &Ray,
        origin: [f64; 3],
        dim: u32,
        cell_size: f64,
        t_enter: f64,
    ) -> Self {
        let o = ray.origin.to_array();
        let d = ray.dir.to_array();

        let mut cell = [0u32; 3];
        let mut step = [0i8; 3];
        let mut t_max = [f64::INFINITY; 3];
        let mut t_delta = [f64::INFINITY; 3];

        for a in 0..3 {
            let entry = o[a] + t_enter * d[a];
            let local = (entry - origin[a]) / cell_size;
            let idx = floor_to_cell(local, dim);
            cell[a] = idx;

            if d[a] > 0.0 {
                step[a] = 1;
                let next = origin[a] + (f64::from(idx) + 1.0) * cell_size;
                t_max[a] = t_enter + (next - entry) / d[a];
                t_delta[a] = cell_size / d[a];
            } else if d[a] < 0.0 {
                step[a] = -1;
                let next = origin[a] + f64::from(idx) * cell_size;
                t_max[a] = t_enter + (next - entry) / d[a];
                t_delta[a] = -cell_size / d[a];
            } // else: parallel to this axis; t_max/t_delta stay +inf.
        }

        Self {
            cell,
            step,
            t_max,
            t_delta,
            dim,
            t_entry: t_enter,
        }
    }

    /// The current cell index (each component in `[0, dim)`).
    pub(crate) fn cell(&self) -> [u32; 3] {
        self.cell
    }

    /// Ray parameter at which the current cell was entered.
    pub(crate) fn t_entry(&self) -> f64 {
        self.t_entry
    }

    /// Advances to the next cell along the ray. Returns `false` (and leaves the
    /// walker unchanged) when the ray exits the grid box.
    ///
    /// Ties break toward the lowest axis index — the fixed convention the `f32`
    /// mirror and GPU kernel copy.
    pub(crate) fn step(&mut self) -> bool {
        let mut axis = 0;
        if self.t_max[1] < self.t_max[axis] {
            axis = 1;
        }
        if self.t_max[2] < self.t_max[axis] {
            axis = 2;
        }
        if !self.t_max[axis].is_finite() {
            return false; // ray is parallel to every remaining axis
        }

        match self.step[axis] {
            1 => {
                if self.cell[axis] + 1 >= self.dim {
                    return false;
                }
                self.cell[axis] += 1;
            }
            -1 => {
                if self.cell[axis] == 0 {
                    return false;
                }
                self.cell[axis] -= 1;
            }
            _ => unreachable!("a finite-t_max axis always has a ±1 step"),
        }
        self.t_entry = self.t_max[axis];
        self.t_max[axis] += self.t_delta[axis];
        true
    }
}
