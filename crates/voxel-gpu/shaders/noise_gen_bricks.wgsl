// Compacted-leaf occupancy generator — concatenated after `noise_core.wgsl`
// (which provides `Params`@0 and `occupied(x, y, z)`).
//
// One invocation per 8³ brick: it evaluates the brick's 512 voxels into a
// register-resident 512-bit Morton leaf (the exact `LeafBrick` layout), and if
// the brick is non-empty, atomic-appends its packed coordinate + 16 leaf words to
// the output. So the host reads back ONLY the occupied bricks (≈⅓ at 2048³) and
// hands them straight to `SparseTree::from_bricks` — no dense 1 GiB readback and
// no CPU re-scan, unlike the dense bitset path.

@group(0) @binding(1) var<storage, read_write> counter: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> out_coords: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_leaves: array<u32>;

// Intra-brick Morton index of local voxel (x, y, z), each 0..8 — the 9-bit
// interleave `LeafBrick` / `traverse_ray`'s `leaf_bit` use.
fn morton_local(x: u32, y: u32, z: u32) -> u32 {
    return (x & 1u) | ((y & 1u) << 1u) | ((z & 1u) << 2u)
        | ((x & 2u) << 2u) | ((y & 2u) << 3u) | ((z & 2u) << 4u)
        | ((x & 4u) << 4u) | ((y & 4u) << 5u) | ((z & 4u) << 6u);
}

@compute @workgroup_size(64)
fn generate_bricks(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let brick_idx = gid.y * (nwg.x * 64u) + gid.x;
    let bpa = params.n / 8u; // bricks per axis
    let total_bricks = bpa * bpa * bpa;
    if (brick_idx >= total_bricks) {
        return;
    }
    let bx = brick_idx % bpa;
    let by = (brick_idx / bpa) % bpa;
    let bz = brick_idx / (bpa * bpa);

    // Build the brick's 512-bit Morton leaf in registers.
    var leaf: array<u32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        leaf[i] = 0u;
    }
    var any = false;
    for (var lz = 0u; lz < 8u; lz = lz + 1u) {
        for (var ly = 0u; ly < 8u; ly = ly + 1u) {
            for (var lx = 0u; lx < 8u; lx = lx + 1u) {
                if (occupied(bx * 8u + lx, by * 8u + ly, bz * 8u + lz)) {
                    let m = morton_local(lx, ly, lz);
                    leaf[m >> 5u] = leaf[m >> 5u] | (1u << (m & 31u));
                    any = true;
                }
            }
        }
    }
    if (!any) {
        return; // empty brick — not emitted (this is the compaction)
    }

    // Atomic append: claim a slot, write coord (10 bits/axis) + the 16 leaf words.
    let slot = atomicAdd(&counter, 1u);
    out_coords[slot] = bx | (by << 10u) | (bz << 20u);
    let base = slot * 16u;
    for (var i = 0u; i < 16u; i = i + 1u) {
        out_leaves[base + i] = leaf[i];
    }
}
