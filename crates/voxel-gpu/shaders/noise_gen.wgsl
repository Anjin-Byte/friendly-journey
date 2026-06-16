// Dense-bitset occupancy generator — concatenated after `noise_core.wgsl`
// (which provides `Params`@0 and `occupied(x, y, z)`).
//
// One invocation per output u32 word: it evaluates that word's 32 voxels (linear
// index `x + y·n + z·n²`, the BitGrid bit order) and packs them. No atomics —
// every word has a single writer. The host reads the dense bitset back into a
// `BitGrid` and runs the (now bit-read-only) CPU scan. Simple, but it reads back
// the whole grid (1 GiB at 2048³) and re-scans it; the brick path avoids both.

@group(0) @binding(1) var<storage, read_write> bits: array<u32>;

@compute @workgroup_size(256)
fn generate(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    // 2D dispatch (the word count exceeds the 65535 workgroups-per-dimension cap
    // at high resolution): flatten to a 1D word index. `nwg.x * 256` is the x row
    // stride in invocations.
    let word_idx = gid.y * (nwg.x * 256u) + gid.x;
    if (word_idx >= params.total_words) {
        return;
    }
    let n = params.n;
    var word = 0u;
    if (n >= 32u) {
        // 32 | n: a word's 32 voxels are one 32-wide x-run inside a single (y, z)
        // row, so the coords derive from the (u32) word index WITHOUT forming the
        // linear voxel index `x + y·n + z·n²` — which overflows u32 at 2048³.
        let k = n / 32u; // u32 words per row
        let x_base = (word_idx % k) * 32u;
        let y = (word_idx / k) % n;
        let z = word_idx / (n * k); // n·k = n²/32
        for (var j = 0u; j < 32u; j = j + 1u) {
            if (occupied(x_base + j, y, z)) {
                word = word | (1u << j);
            }
        }
    } else {
        // n < 32 (only the 8³ single-leaf grid): the full linear index fits u32.
        let nn = n * n;
        for (var j = 0u; j < 32u; j = j + 1u) {
            let i = word_idx * 32u + j;
            if (occupied(i % n, (i / n) % n, i / nn)) {
                word = word | (1u << j);
            }
        }
    }
    bits[word_idx] = word;
}
