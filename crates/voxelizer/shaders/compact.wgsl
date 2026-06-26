struct CompactParams {
  brick_dim: u32,
  brick_count: u32,
  max_positions: u32,
  _pad0: u32,
  origin_world: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> occupancy: array<u32>;
@group(0) @binding(1) var<storage, read> brick_origins: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> out_positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: CompactParams;
@group(0) @binding(5) var<storage, read_write> debug: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let brick_index = wg_id.x;
  if (brick_index >= params.brick_count) {
    return;
  }
  if (lid.x == 0u) {
    atomicAdd(&debug[0], 1u);
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
      atomicAdd(&debug[1], 1u);
      let idx = atomicAdd(&counter[0], 1u);
      if (idx < params.max_positions) {
        let vx = linear % brick_dim;
        let vy = (linear / brick_dim) % brick_dim;
        let vz = (linear / (brick_dim * brick_dim));
        let gx = f32(origin.x + vx) + 0.5;
        let gy = f32(origin.y + vy) + 0.5;
        let gz = f32(origin.z + vz) + 0.5;
        let world = params.origin_world.xyz + vec3<f32>(gx, gy, gz) * params.origin_world.w;
        out_positions[idx] = vec4<f32>(world, 1.0);
      }
    }
    linear = linear + 64u;
  }
}
