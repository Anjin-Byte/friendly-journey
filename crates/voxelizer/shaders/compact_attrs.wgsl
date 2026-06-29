struct CompactAttrsParams {
  brick_dim: u32,
  brick_count: u32,
  max_entries: u32,
  _pad0: u32,
  grid_dims: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> occupancy: array<u32>;
@group(0) @binding(1) var<storage, read> brick_origins: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> owner_id: array<u32>;
@group(0) @binding(3) var<storage, read> color_rgba: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_indices: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_owner: array<u32>;
@group(0) @binding(6) var<storage, read_write> out_color: array<u32>;
@group(0) @binding(7) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> params: CompactAttrsParams;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let brick_index = wg_id.x;
  if (brick_index >= params.brick_count) {
    return;
  }
  let brick_dim = params.brick_dim;
  let brick_voxels = brick_dim * brick_dim * brick_dim;
  let words_per_brick = (brick_voxels + 31u) / 32u;
  let base_word = brick_index * words_per_brick;
  let origin = brick_origins[brick_index].xyz;

  var linear = lid.x;
  loop {
    if (linear >= brick_voxels) {
      break;
    }
    let word = base_word + (linear >> 5u);
    let bit = linear & 31u;
    let mask = 1u << bit;
    if ((occupancy[word] & mask) != 0u) {
      let idx = atomicAdd(&counter[0], 1u);
      if (idx < params.max_entries) {
        let vx = linear % brick_dim;
        let vy = (linear / brick_dim) % brick_dim;
        let vz = (linear / (brick_dim * brick_dim));
        let gx = origin.x + vx;
        let gy = origin.y + vy;
        let gz = origin.z + vz;
        let linear_index =
          gx + params.grid_dims.x * (gy + params.grid_dims.y * gz);
        let attr_index = brick_index * brick_voxels + linear;
        out_indices[idx] = linear_index;
        out_owner[idx] = owner_id[attr_index];
        out_color[idx] = color_rgba[attr_index];
      }
    }
    linear = linear + 64u;
  }
}
