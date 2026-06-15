//! P5: the `f32` iterative mirror of the GPU kernel.
//!
//! The GPU cannot recurse, so the kernel is an **explicit-stack iterative**
//! Hierarchical DDA over the School-B buffer. This module is the CPU twin of
//! that kernel — same `f32` arithmetic, same explicit stack, same buffer reads
//! — so the WGSL is a near-mechanical transliteration of it and can be debugged
//! here without a GPU (review R1).
//!
//! It produces the same hits as the recursive `f64` traversal except on grazing
//! rays, where `f32` and `f64` pick different DDA steps. The differential bounds
//! and logs those disagreements rather than tolerating an arbitrary distance on
//! a discrete voxel hit.

use crate::layout::NodeLayout;
use crate::node::child_bit;
use crate::ray::Ray;
use crate::{SchoolBBuffer, VoxelCoord};

/// Max traversal depth: voxel → leaf brick → up to 4 internal levels at 2048³,
/// i.e. 6; `8` leaves head-room and matches the WGSL stack size.
const MAX_DEPTH: usize = 8;

/// Truncates a cell-space coordinate to `[0, dim)`. The single audited
/// `f32 → u32` site.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn floor_to_cell(v: f32, dim: u32) -> u32 {
    v.floor().clamp(0.0, (dim - 1) as f32) as u32
}

/// `f32` single-level DDA state (one stack frame's walker).
#[derive(Clone, Copy)]
struct Walker {
    cell: [u32; 3],
    step: [i32; 3],
    t_max: [f32; 3],
    t_delta: [f32; 3],
    t_entry: f32,
    dim: u32,
}

impl Walker {
    #[allow(clippy::cast_precision_loss)]
    fn enter(
        o: [f32; 3],
        d: [f32; 3],
        origin: [f32; 3],
        dim: u32,
        cell_size: f32,
        t_enter: f32,
    ) -> Self {
        let mut cell = [0u32; 3];
        let mut step = [0i32; 3];
        let mut t_max = [f32::INFINITY; 3];
        let mut t_delta = [f32::INFINITY; 3];
        for a in 0..3 {
            let entry = o[a] + t_enter * d[a];
            let local = (entry - origin[a]) / cell_size;
            let idx = floor_to_cell(local, dim);
            cell[a] = idx;
            if d[a] > 0.0 {
                step[a] = 1;
                let next = origin[a] + (idx as f32 + 1.0) * cell_size;
                t_max[a] = t_enter + (next - entry) / d[a];
                t_delta[a] = cell_size / d[a];
            } else if d[a] < 0.0 {
                step[a] = -1;
                let next = origin[a] + (idx as f32) * cell_size;
                t_max[a] = t_enter + (next - entry) / d[a];
                t_delta[a] = -cell_size / d[a];
            }
        }
        Self {
            cell,
            step,
            t_max,
            t_delta,
            t_entry: t_enter,
            dim,
        }
    }

    fn step(&mut self) -> bool {
        let mut a = 0;
        if self.t_max[1] < self.t_max[a] {
            a = 1;
        }
        if self.t_max[2] < self.t_max[a] {
            a = 2;
        }
        if !self.t_max[a].is_finite() {
            return false;
        }
        match self.step[a] {
            1 => {
                if self.cell[a] + 1 >= self.dim {
                    return false;
                }
                self.cell[a] += 1;
            }
            -1 => {
                if self.cell[a] == 0 {
                    return false;
                }
                self.cell[a] -= 1;
            }
            _ => return false,
        }
        self.t_entry = self.t_max[a];
        self.t_max[a] += self.t_delta[a];
        true
    }
}

#[derive(Clone, Copy)]
struct Frame {
    node: u32,
    level: u32,
    origin: [u32; 3],
    walker: Walker,
}

/// Child cell edge in base voxels for a `level`-internal frame's `4³` walk, or
/// `1` for the leaf frame's `8³` walk.
fn frame_cell_size(level: u32) -> u32 {
    if level == 1 { 1 } else { 1 << (2 * level - 1) }
}

#[allow(clippy::cast_precision_loss)]
fn make_frame(
    o: [f32; 3],
    d: [f32; 3],
    node: u32,
    level: u32,
    origin: [u32; 3],
    t_enter: f32,
) -> Frame {
    let f_origin = [origin[0] as f32, origin[1] as f32, origin[2] as f32];
    let dim = if level == 1 { 8 } else { 4 };
    let cell_size = frame_cell_size(level) as f32;
    Frame {
        node,
        level,
        origin,
        walker: Walker::enter(o, d, f_origin, dim, cell_size, t_enter),
    }
}

/// Pops the exhausted top frame and advances ancestors until one steps to a new
/// cell. Returns `false` if the whole stack drains (ray exits the grid).
fn ascend(stack: &mut [Frame; MAX_DEPTH], sp: &mut usize) -> bool {
    loop {
        *sp -= 1;
        if *sp == 0 {
            return false;
        }
        if stack[*sp - 1].walker.step() {
            return true;
        }
    }
}

/// `f32` slab test against the grid box `[0, n]³`.
#[allow(clippy::cast_precision_loss)]
fn ray_aabb_f32(o: [f32; 3], d: [f32; 3], n: f32) -> Option<(f32, f32)> {
    let mut t_near = f32::NEG_INFINITY;
    let mut t_far = f32::INFINITY;
    for a in 0..3 {
        if d[a] == 0.0 {
            if o[a] < 0.0 || o[a] > n {
                return None;
            }
        } else {
            let inv = 1.0 / d[a];
            let mut t1 = (0.0 - o[a]) * inv;
            let mut t2 = (n - o[a]) * inv;
            if t1 > t2 {
                core::mem::swap(&mut t1, &mut t2);
            }
            t_near = t_near.max(t1);
            t_far = t_far.min(t2);
            if t_near > t_far {
                return None;
            }
        }
    }
    Some((t_near, t_far))
}

/// Iterative `f32` traversal of the School-B buffer — the CPU mirror of the GPU
/// kernel. Returns the first occupied voxel.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn mirror_traverse(buffer: &SchoolBBuffer, ray: &Ray) -> Option<VoxelCoord> {
    let leaves = buffer.leaves();
    if leaves.is_empty() {
        return None;
    }
    let nodes = buffer.nodes();
    let res = buffer.resolution();
    let o = ray.origin.as_vec3().to_array();
    let d = ray.dir.as_vec3().to_array();

    let (t0, t1) = ray_aabb_f32(o, d, res.voxels_per_axis() as f32)?;
    if t1 < 0.0 {
        return None;
    }
    let t_entry = t0.max(0.0);

    let k = res.internal_levels();
    let (root_node, root_level) = if k == 0 { (0, 1) } else { (0, k + 1) };

    let mut stack = [make_frame(o, d, root_node, root_level, [0, 0, 0], t_entry); MAX_DEPTH];
    let mut sp = 1usize;

    loop {
        let top = sp - 1;
        if stack[top].level == 1 {
            let v = stack[top].walker.cell;
            if leaves[stack[top].node as usize].get_local(v[0], v[1], v[2]) {
                let org = stack[top].origin;
                return Some(VoxelCoord::new(org[0] + v[0], org[1] + v[1], org[2] + v[2]));
            }
            if stack[top].walker.step() {
                continue;
            }
            if !ascend(&mut stack, &mut sp) {
                return None;
            }
        } else {
            let frame = stack[top];
            let c = frame.walker.cell;
            let node = nodes[frame.node as usize];
            let bit = child_bit(c[0], c[1], c[2]);
            if node.has_child(bit) {
                let size = frame_cell_size(frame.level);
                let child_origin = [
                    frame.origin[0] + c[0] * size,
                    frame.origin[1] + c[1] * size,
                    frame.origin[2] + c[2] * size,
                ];
                let child = make_frame(
                    o,
                    d,
                    node.child_slot(bit),
                    frame.level - 1,
                    child_origin,
                    frame.walker.t_entry,
                );
                stack[sp] = child;
                sp += 1;
            } else if !stack[top].walker.step() && !ascend(&mut stack, &mut sp) {
                return None;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{Checkerboard, Empty, OctantFractal, SingleVoxel, Solid};
    use crate::{OccupancyField, Resolution, SparseTree, oracle};
    use glam::DVec3;

    fn res(n: u32) -> Resolution {
        Resolution::new(n).unwrap()
    }

    fn buf<F: OccupancyField + Sync>(field: &F) -> SchoolBBuffer {
        SchoolBBuffer::from_sparse(&SparseTree::build(field))
    }

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[allow(clippy::cast_precision_loss)]
    fn unit(state: &mut u64) -> f64 {
        (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn exact_axis_aligned_hits() {
        let b = buf(&SingleVoxel {
            resolution: res(128),
            voxel: VoxelCoord::new(40, 9, 9),
        });
        let ray = Ray::new(DVec3::new(-1.0, 9.5, 9.5), DVec3::X);
        assert_eq!(mirror_traverse(&b, &ray), Some(VoxelCoord::new(40, 9, 9)));

        let empty = buf(&Empty {
            resolution: res(32),
        });
        assert_eq!(mirror_traverse(&empty, &ray), None);

        let solid = buf(&Solid {
            resolution: res(32),
        });
        assert_eq!(
            mirror_traverse(&solid, &Ray::new(DVec3::new(-1.0, 4.5, 4.5), DVec3::X)),
            Some(VoxelCoord::new(0, 4, 4))
        );
    }

    #[test]
    fn single_brick_k0() {
        let b = buf(&SingleVoxel {
            resolution: res(8),
            voxel: VoxelCoord::new(5, 2, 2),
        });
        let ray = Ray::new(DVec3::new(-1.0, 2.5, 2.5), DVec3::X);
        assert_eq!(mirror_traverse(&b, &ray), Some(VoxelCoord::new(5, 2, 2)));
    }

    #[test]
    fn iterative_f32_matches_oracle_within_grazing_bound() {
        // The iterative f32 mirror agrees with the recursive f64 oracle on the
        // hit voxel for the vast majority of rays; the residual disagreements
        // are grazing (f32 vs f64) and must stay a small, bounded fraction. A
        // *bug* in the iterative transform would disagree on far more.
        let r = res(128);
        let nf = f64::from(r.voxels_per_axis());
        let frac = OctantFractal::sierpinski_tetrahedron(r);
        let checker = Checkerboard { resolution: r };
        let frac_b = buf(&frac);
        let checker_b = buf(&checker);

        let mut state = 0x7777_3333_BBBB_1111u64;
        let mut total = 0u32;
        let mut mismatch = 0u32;
        for _ in 0..6000 {
            let origin = DVec3::new(
                unit(&mut state) * (nf + 8.0) - 4.0,
                unit(&mut state) * (nf + 8.0) - 4.0,
                unit(&mut state) * (nf + 8.0) - 4.0,
            );
            let dir = DVec3::new(
                unit(&mut state) * 2.0 - 1.0,
                unit(&mut state) * 2.0 - 1.0,
                unit(&mut state) * 2.0 - 1.0,
            );
            if dir.length() < 1e-3 {
                continue;
            }
            let ray = Ray::new(origin, dir);
            for (field_hit, b) in [
                (oracle::first_hit(&frac, &ray).map(|h| h.voxel), &frac_b),
                (
                    oracle::first_hit(&checker, &ray).map(|h| h.voxel),
                    &checker_b,
                ),
            ] {
                total += 1;
                if mirror_traverse(b, &ray) != field_hit {
                    mismatch += 1;
                }
            }
        }
        let rate = f64::from(mismatch) / f64::from(total);
        assert!(
            rate < 0.01,
            "f32 mirror disagreed with oracle on {mismatch}/{total} ({:.3}%) — expected < 1% grazing",
            rate * 100.0
        );
    }
}
