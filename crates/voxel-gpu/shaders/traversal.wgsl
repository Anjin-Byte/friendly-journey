// Shared HDDA traversal core over the School-B buffer — the exact algorithm of
// voxel-core's `mirror.rs`. This file is concatenated *ahead* of an entry-point
// module: `hdda.wgsl` (buffer path: rays in, hits out) or `render.wgsl` (camera
// path: traverse + shade straight to a storage texture). Both call
// `traverse_ray`, so the validated kernel and the viewer kernel never drift.

struct Node {
    mask_lo: u32,
    mask_hi: u32,
    child_base: u32,
}

@group(0) @binding(0) var<storage, read> nodes: array<Node>;
@group(0) @binding(1) var<storage, read> leaf_words: array<u32>;
// One packed LeafBounds per leaf (min.xyz in bits 0/3/6, max.xyz in 9/12/15).
@group(0) @binding(2) var<storage, read> leaf_bounds: array<u32>;

const BIG: f32 = 1e30;

fn cell_size_of(level: u32) -> u32 {
    if (level == 1u) { return 1u; }
    return 1u << (2u * level - 1u);
}

fn child_bit(c: vec3<u32>) -> u32 {
    return (c.x & 1u) | ((c.y & 1u) << 1u) | ((c.z & 1u) << 2u)
         | ((c.x & 2u) << 2u) | ((c.y & 2u) << 3u) | ((c.z & 2u) << 4u);
}

fn morton8(v: vec3<u32>) -> u32 {
    return (v.x & 1u) | ((v.y & 1u) << 1u) | ((v.z & 1u) << 2u)
         | ((v.x & 2u) << 2u) | ((v.y & 2u) << 3u) | ((v.z & 2u) << 4u)
         | ((v.x & 4u) << 4u) | ((v.y & 4u) << 5u) | ((v.z & 4u) << 6u);
}

fn has_child(n: Node, bit: u32) -> bool {
    if (bit < 32u) {
        return ((n.mask_lo >> bit) & 1u) == 1u;
    }
    return ((n.mask_hi >> (bit - 32u)) & 1u) == 1u;
}

fn child_slot(n: Node, bit: u32) -> u32 {
    var below: u32 = 0u;
    if (bit < 32u) {
        let m = (1u << bit) - 1u;
        below = countOneBits(n.mask_lo & m);
    } else {
        let m = (1u << (bit - 32u)) - 1u;
        below = countOneBits(n.mask_lo) + countOneBits(n.mask_hi & m);
    }
    return n.child_base + below;
}

fn leaf_bit(leaf_idx: u32, v: vec3<u32>) -> bool {
    let idx = morton8(v & vec3<u32>(7u));
    let word = leaf_words[leaf_idx * 16u + (idx >> 5u)];
    return ((word >> (idx & 31u)) & 1u) == 1u;
}

// Per-brick early-skip — the exact f32 algorithm of mirror.rs `leaf_reaches_f32`.
// Whether the ray can reach leaf `slot`'s occupied sub-box (brick corner
// `origin`, entered at `t_enter`). `false` ⇒ the 8³ walk is skipped. The box
// contains every set voxel, so a chord that misses it provably misses the brick.
fn leaf_reaches(slot: u32, o: vec3<f32>, d: vec3<f32>, origin: vec3<u32>, t_enter: f32) -> bool {
    let packed = leaf_bounds[slot];
    let mn = vec3<u32>(packed & 7u, (packed >> 3u) & 7u, (packed >> 6u) & 7u);
    let mx = vec3<u32>((packed >> 9u) & 7u, (packed >> 12u) & 7u, (packed >> 15u) & 7u);
    // Full-brick bound ⇒ the test can never skip; descend without it.
    if (all(mn == vec3<u32>(0u, 0u, 0u)) && all(mx == vec3<u32>(7u, 7u, 7u))) {
        return true;
    }
    // Dilate the occupied box by one voxel (SKIP_MARGIN) so this f32 slab test is
    // never stricter than the interior DDA — without it, grazing rays on
    // single-voxel bricks get wrongly skipped (the box interval rounds degenerate).
    let m = vec3<f32>(1.0, 1.0, 1.0);
    let lo = vec3<f32>(origin + mn) - m;
    let hi = vec3<f32>(origin + mx + vec3<u32>(1u)) + m;

    var t_near = -BIG;
    var t_far = BIG;
    if (d.x == 0.0) { if (o.x < lo.x || o.x > hi.x) { return false; } }
    else { let inv = 1.0 / d.x; var a = (lo.x - o.x) * inv; var b = (hi.x - o.x) * inv; if (a > b) { let t = a; a = b; b = t; } t_near = max(t_near, a); t_far = min(t_far, b); }
    if (d.y == 0.0) { if (o.y < lo.y || o.y > hi.y) { return false; } }
    else { let inv = 1.0 / d.y; var a = (lo.y - o.y) * inv; var b = (hi.y - o.y) * inv; if (a > b) { let t = a; a = b; b = t; } t_near = max(t_near, a); t_far = min(t_far, b); }
    if (d.z == 0.0) { if (o.z < lo.z || o.z > hi.z) { return false; } }
    else { let inv = 1.0 / d.z; var a = (lo.z - o.z) * inv; var b = (hi.z - o.z) * inv; if (a > b) { let t = a; a = b; b = t; } t_near = max(t_near, a); t_far = min(t_far, b); }

    if (t_near > t_far) { return false; }
    return t_far >= t_enter;
}

struct Frame {
    node: u32,
    level: u32,
    dim: u32,
    _pad: u32,
    origin: vec3<u32>,
    cell: vec3<u32>,
    step: vec3<i32>,
    t_max: vec3<f32>,
    t_delta: vec3<f32>,
    t_entry: f32,
}

struct Axis {
    cell: u32,
    step: i32,
    t_max: f32,
    t_delta: f32,
}

fn axis_init(o: f32, d: f32, origin: f32, dim: u32, cs: f32, t_enter: f32) -> Axis {
    let entry = o + t_enter * d;
    let local = (entry - origin) / cs;
    let idx = u32(clamp(floor(local), 0.0, f32(dim - 1u)));
    var ax: Axis;
    ax.cell = idx;
    ax.step = 0;
    ax.t_max = BIG;
    ax.t_delta = BIG;
    if (d > 0.0) {
        ax.step = 1;
        let next = origin + (f32(idx) + 1.0) * cs;
        ax.t_max = t_enter + (next - entry) / d;
        ax.t_delta = cs / d;
    } else if (d < 0.0) {
        ax.step = -1;
        let next = origin + f32(idx) * cs;
        ax.t_max = t_enter + (next - entry) / d;
        ax.t_delta = -cs / d;
    }
    return ax;
}

fn make_frame(o: vec3<f32>, d: vec3<f32>, node: u32, level: u32, origin: vec3<u32>, t_enter: f32) -> Frame {
    let dim = select(4u, 8u, level == 1u);
    let cs = f32(cell_size_of(level));
    let ax = axis_init(o.x, d.x, f32(origin.x), dim, cs, t_enter);
    let ay = axis_init(o.y, d.y, f32(origin.y), dim, cs, t_enter);
    let az = axis_init(o.z, d.z, f32(origin.z), dim, cs, t_enter);
    var f: Frame;
    f.node = node;
    f.level = level;
    f.dim = dim;
    f.origin = origin;
    f.t_entry = t_enter;
    f.cell = vec3<u32>(ax.cell, ay.cell, az.cell);
    f.step = vec3<i32>(ax.step, ay.step, az.step);
    f.t_max = vec3<f32>(ax.t_max, ay.t_max, az.t_max);
    f.t_delta = vec3<f32>(ax.t_delta, ay.t_delta, az.t_delta);
    return f;
}

fn walker_step(f: ptr<function, Frame>) -> bool {
    let tm = (*f).t_max;
    var a = 0u;
    var best = tm.x;
    if (tm.y < best) { a = 1u; best = tm.y; }
    if (tm.z < best) { a = 2u; best = tm.z; }

    if (a == 0u) {
        let s = (*f).step.x;
        if (s == 0) { return false; }
        if (s > 0) {
            if ((*f).cell.x + 1u >= (*f).dim) { return false; }
            (*f).cell.x = (*f).cell.x + 1u;
        } else {
            if ((*f).cell.x == 0u) { return false; }
            (*f).cell.x = (*f).cell.x - 1u;
        }
        (*f).t_entry = (*f).t_max.x;
        (*f).t_max.x = (*f).t_max.x + (*f).t_delta.x;
    } else if (a == 1u) {
        let s = (*f).step.y;
        if (s == 0) { return false; }
        if (s > 0) {
            if ((*f).cell.y + 1u >= (*f).dim) { return false; }
            (*f).cell.y = (*f).cell.y + 1u;
        } else {
            if ((*f).cell.y == 0u) { return false; }
            (*f).cell.y = (*f).cell.y - 1u;
        }
        (*f).t_entry = (*f).t_max.y;
        (*f).t_max.y = (*f).t_max.y + (*f).t_delta.y;
    } else {
        let s = (*f).step.z;
        if (s == 0) { return false; }
        if (s > 0) {
            if ((*f).cell.z + 1u >= (*f).dim) { return false; }
            (*f).cell.z = (*f).cell.z + 1u;
        } else {
            if ((*f).cell.z == 0u) { return false; }
            (*f).cell.z = (*f).cell.z - 1u;
        }
        (*f).t_entry = (*f).t_max.z;
        (*f).t_max.z = (*f).t_max.z + (*f).t_delta.z;
    }
    return true;
}

/// Marches one ray through the structure. Returns `(vx, vy, vz, 1)` for the
/// first occupied voxel, or `(0, 0, 0, 0)` for a miss.
///
/// The *active* DDA frame is kept in function-local (register-resident) scalars;
/// the `stack` array holds only its ancestors, touched on descend/ascend rather
/// than on every cell-step. Same algorithm as voxel-core's `mirror.rs` (only the
/// active frame's storage differs) — it traverses ~1.8× faster than indexing
/// `stack[top]` each step, which spilled the frame array to GPU local memory.
/// Validated byte-identical to that stack form by the differential.
fn traverse_ray(o: vec3<f32>, d: vec3<f32>, n: f32, k: u32) -> vec4<u32> {
    let miss = vec4<u32>(0u, 0u, 0u, 0u);

    // Grid-clip (f32 slab) against [0, n]³ — identical to traverse_ray.
    var t_near = -BIG;
    var t_far = BIG;
    var missed = false;
    if (d.x == 0.0) { if (o.x < 0.0 || o.x > n) { missed = true; } }
    else { let inv = 1.0 / d.x; var a = (0.0 - o.x) * inv; var b = (n - o.x) * inv; if (a > b) { let t = a; a = b; b = t; } t_near = max(t_near, a); t_far = min(t_far, b); }
    if (d.y == 0.0) { if (o.y < 0.0 || o.y > n) { missed = true; } }
    else { let inv = 1.0 / d.y; var a = (0.0 - o.y) * inv; var b = (n - o.y) * inv; if (a > b) { let t = a; a = b; b = t; } t_near = max(t_near, a); t_far = min(t_far, b); }
    if (d.z == 0.0) { if (o.z < 0.0 || o.z > n) { missed = true; } }
    else { let inv = 1.0 / d.z; var a = (0.0 - o.z) * inv; var b = (n - o.z) * inv; if (a > b) { let t = a; a = b; b = t; } t_near = max(t_near, a); t_far = min(t_far, b); }

    if (missed || t_near > t_far || t_far < 0.0) {
        return miss;
    }
    let t_entry = max(t_near, 0.0);

    var root_level = 1u;
    if (k > 0u) {
        root_level = k + 1u;
    }
    // The cur frame lives here (registers); `stack` holds only its ancestors.
    var cur = make_frame(o, d, 0u, root_level, vec3<u32>(0u, 0u, 0u), t_entry);
    var stack: array<Frame, 8>;
    var sp = 0u; // number of parent frames on the stack

    for (var iter = 0u; iter < 200000u; iter = iter + 1u) {
        if (cur.level == 1u) {
            let v = cur.cell;
            if (leaf_bit(cur.node, v)) {
                let org = cur.origin;
                return vec4<u32>(org.x + v.x, org.y + v.y, org.z + v.z, 1u);
            }
            if (walker_step(&cur)) { continue; }
            // Ascend: pop a parent into `cur` and step it.
            loop {
                if (sp == 0u) { return miss; }
                sp = sp - 1u;
                cur = stack[sp];
                if (walker_step(&cur)) { break; }
            }
        } else {
            let c = cur.cell;
            let bit = child_bit(c);
            let node = nodes[cur.node];
            let child_level = cur.level - 1u;
            let size = cell_size_of(cur.level);
            let child_origin = cur.origin + c * size;
            var descend = has_child(node, bit);
            var slot = 0u;
            if (descend) {
                slot = child_slot(node, bit);
                if (child_level == 1u) {
                    descend = leaf_reaches(slot, o, d, child_origin, cur.t_entry);
                }
            }
            if (descend) {
                stack[sp] = cur;
                sp = sp + 1u;
                cur = make_frame(o, d, slot, child_level, child_origin, cur.t_entry);
            } else if (!walker_step(&cur)) {
                loop {
                    if (sp == 0u) { return miss; }
                    sp = sp - 1u;
                    cur = stack[sp];
                    if (walker_step(&cur)) { break; }
                }
            }
        }
    }
    return miss;
}
