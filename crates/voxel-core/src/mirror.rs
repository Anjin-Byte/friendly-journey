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

use crate::layout::{NodeLayout, SKIP_MARGIN};
use crate::leaf::LeafBounds;
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

/// Per-brick early-skip, `f32`, **bit-identical to WGSL `leaf_reaches`**: whether
/// `(o, d)` can reach leaf `slot`'s occupied sub-box (brick corner `origin`,
/// entered at `t_enter`). `false` ⇒ skip the `8³` walk. Conservative: the box —
/// dilated by [`SKIP_MARGIN`] so the `f32` slab test is never stricter than the
/// interior DDA — contains every set voxel, so a chord that misses it (the box
/// plus its halo) cannot reach any voxel the walk would `floor` into.
// `as f32`: u32 coords (≤ 2055) and the 1.0 margin are all exact in f32.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn leaf_reaches_f32(
    buffer: &SchoolBBuffer,
    slot: u32,
    o: [f32; 3],
    d: [f32; 3],
    origin: [u32; 3],
    t_enter: f32,
) -> bool {
    let b = LeafBounds::unpack(buffer.leaf_bounds_words()[slot as usize]);
    // Full-brick bound ⇒ the test can never skip; descend without it.
    if b == LeafBounds::FULL {
        return true;
    }
    let m = SKIP_MARGIN as f32;
    let lo = [
        (origin[0] + b.min[0]) as f32 - m,
        (origin[1] + b.min[1]) as f32 - m,
        (origin[2] + b.min[2]) as f32 - m,
    ];
    let hi = [
        (origin[0] + b.max[0] + 1) as f32 + m,
        (origin[1] + b.max[1] + 1) as f32 + m,
        (origin[2] + b.max[2] + 1) as f32 + m,
    ];
    // ±1e30, not ±inf, to stay bit-identical with WGSL `leaf_reaches` (`BIG`).
    let mut t_near = -1e30f32;
    let mut t_far = 1e30f32;
    for a in 0..3 {
        if d[a] == 0.0 {
            if o[a] < lo[a] || o[a] > hi[a] {
                return false;
            }
        } else {
            let inv = 1.0 / d[a];
            let mut t1 = (lo[a] - o[a]) * inv;
            let mut t2 = (hi[a] - o[a]) * inv;
            if t1 > t2 {
                core::mem::swap(&mut t1, &mut t2);
            }
            t_near = t_near.max(t1);
            t_far = t_far.min(t2);
        }
    }
    if t_near > t_far {
        return false;
    }
    t_far >= t_enter
}

/// Iterative `f32` traversal of the School-B buffer — the CPU mirror of the GPU
/// kernel. Returns the first occupied voxel.
#[must_use]
pub fn mirror_traverse(buffer: &SchoolBBuffer, ray: &Ray) -> Option<VoxelCoord> {
    mirror_with(buffer, ray, true)
}

/// The mirror traversal with the per-brick early-skip toggleable. The skip is a
/// pure optimization: `mirror_with(.., true)` must return the *same* voxel as
/// `mirror_with(.., false)` for every ray. Tests assert exactly that to prove
/// the skip is conservative without conflating it with baseline `f32` grazing.
#[allow(clippy::cast_precision_loss)]
fn mirror_with(buffer: &SchoolBBuffer, ray: &Ray, enable_skip: bool) -> Option<VoxelCoord> {
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
            let child_level = frame.level - 1;
            let size = frame_cell_size(frame.level);
            let child_origin = [
                frame.origin[0] + c[0] * size,
                frame.origin[1] + c[1] * size,
                frame.origin[2] + c[2] * size,
            ];
            let mut descend = node.has_child(bit);
            let slot = if descend { node.child_slot(bit) } else { 0 };
            // Per-brick early-skip: a leaf child whose occupied sub-box the chord
            // misses is treated as empty — step on instead of descending.
            if enable_skip && descend && child_level == 1 {
                descend = leaf_reaches_f32(buffer, slot, o, d, child_origin, frame.walker.t_entry);
            }
            if descend {
                stack[sp] = make_frame(o, d, slot, child_level, child_origin, frame.walker.t_entry);
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

    /// The five rays the adversarial review reproduced as dropped hits: each is
    /// `(occupied voxel, ray origin, ray dir)` at 2048³ where, before the box
    /// was dilated, the `f32` skip returned `None` while the oracle and the
    /// skip-off `f32` walk both hit the voxel.
    const GRAZING_CASES: [([u32; 3], [f64; 3], [f64; 3]); 5] = [
        (
            [1005, 1001, 1006],
            [
                982.984_239_308_813,
                982.401_992_822_304_1,
                984.798_651_116_279_4,
            ],
            [
                23.015_626_101_891_94,
                18.597_909_656_474_485,
                21.201_340_994_433_167,
            ],
        ),
        (
            [1000, 1002, 1003],
            [
                980.935_792_703_746_6,
                983.143_088_286_634_6,
                983.980_312_831_914_7,
            ],
            [
                20.064_276_842_389_29,
                19.856_810_088_007_137,
                19.019_775_035_212_888,
            ],
        ),
        (
            [1007, 1004, 1003],
            [
                984.994_431_263_870_6,
                983.936_235_976_935_9,
                982.417_227_211_502_8,
            ],
            [
                23.005_431_150_791_31,
                20.063_647_513_767_933,
                20.582_903_845_754_23,
            ],
        ),
        (
            [1007, 1002, 1000],
            [
                981.134_829_893_952_8,
                984.417_325_050_859_5,
                985.904_370_355_243_5,
            ],
            [
                26.865_040_272_210_194,
                17.582_846_420_236_48,
                15.095_767_354_692_498,
            ],
        ),
        (
            [1005, 1001, 1001],
            [
                983.183_967_708_553,
                984.670_597_813_287_4,
                985.603_651_590_570_6,
            ],
            [
                22.815_834_187_861_583,
                16.329_279_511_667_096,
                15.396_220_851_206_749,
            ],
        ),
    ];

    #[test]
    fn early_skip_keeps_high_coordinate_grazing_hits() {
        // Adversarial-review regression. The per-brick early-skip must be a PURE
        // optimization: skip-ON must return the same voxel as skip-OFF for every
        // ray. At 2048³ the f32 box interval for a *single-voxel* brick is razor
        // thin, so a grazing chord rounded degenerate and the skip dropped real
        // hits — until the box was dilated by SKIP_MARGIN. Each voxel must live
        // ALONE in its brick (else the merged occupied-box is no longer thin), so
        // we build a separate `SingleVoxel` per case. Comparing skip-on vs
        // skip-off isolates the skip from baseline f32-vs-f64 grazing entirely.
        let r = res(2048);
        // Three of the five reproduced cases — enough to pin the regression while
        // bounding the per-case 2048³ build.
        for (v, o, d) in GRAZING_CASES.iter().take(3).copied() {
            let voxel = VoxelCoord::new(v[0], v[1], v[2]);
            let b = buf(&SingleVoxel {
                resolution: r,
                voxel,
            });
            let ray = Ray::new(DVec3::from_array(o), DVec3::from_array(d));
            assert_eq!(
                mirror_with(&b, &ray, false),
                Some(voxel),
                "skip-off walk missed {v:?}"
            );
            assert_eq!(
                mirror_with(&b, &ray, true),
                Some(voxel),
                "early-skip dropped grazing hit {v:?}"
            );
        }
    }

    #[test]
    fn early_skip_matches_no_skip_on_axis_aligned_rays() {
        // The d==0 (axis-parallel) branch of the box slab test is never reached
        // by the random differential's continuous directions. Exercise it: with
        // exactly-axis-aligned and single-zero-component directions, jittered to
        // graze a voxel's faces/edges/corners, skip-on must equal skip-off.
        let r = res(512);
        let v = VoxelCoord::new(300, 301, 302);
        let b = buf(&SingleVoxel {
            resolution: r,
            voxel: v,
        });
        let centre = DVec3::new(300.5, 301.5, 302.5);
        let dirs = [
            DVec3::X,
            DVec3::Y,
            DVec3::Z,
            -DVec3::X,
            -DVec3::Y,
            -DVec3::Z,
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 1.0),
            DVec3::new(1.0, 0.0, 1.0),
        ];
        let mut state = 0xDEAD_BEEF_0042_0042u64;
        let mut checked = 0u32;
        for dir in dirs {
            for _ in 0..400 {
                let aim = centre
                    + DVec3::new(
                        unit(&mut state) * 1.6 - 0.8,
                        unit(&mut state) * 1.6 - 0.8,
                        unit(&mut state) * 1.6 - 0.8,
                    );
                let ray = Ray::new(aim - dir * 80.0, dir);
                assert_eq!(
                    mirror_with(&b, &ray, true),
                    mirror_with(&b, &ray, false),
                    "early-skip changed the hit for axis-aligned ray {ray:?}"
                );
                checked += 1;
            }
        }
        assert!(checked > 2000, "only checked {checked} axis-aligned rays");
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
