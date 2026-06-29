struct CompactVoxelsParams {
  brick_dim: u32,
  brick_count: u32,
  max_entries: u32,
  material_table_len: u32,
  g_origin: vec4<i32>,
};

struct CompactVoxelGpu {
  vx: i32,
  vy: i32,
  vz: i32,
  material: u32,
};

@group(0) @binding(0) var<storage, read> occupancy: array<u32>;
@group(0) @binding(1) var<storage, read> brick_origins: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> owner_id: array<u32>;
@group(0) @binding(3) var<storage, read> material_table: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_voxels: array<CompactVoxelGpu>;
@group(0) @binding(5) var<storage, read_write> counter: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: CompactVoxelsParams;

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

        // Resolve material from owner triangle → material table. The table maps
        // triangle → renderer GLOBAL id (u16, two per word); an unowned voxel or
        // an unresolved triangle stays material 0 = global-0 = magenta MISSING
        // (docs/materials/05 hole 1). No 0→1 clamp: a real 0 must survive so the
        // sentinel renders loud magenta instead of aliasing a real material.
        let attr_index = brick_index * brick_voxels + linear;
        let tri = owner_id[attr_index];
        var mat: u32 = 0u; // global-0 (MISSING) default
        if (tri != 0xFFFFFFFFu && params.material_table_len > 0u) {
          let word_idx = tri >> 1u;
          let shift = (tri & 1u) << 4u;
          if (word_idx < params.material_table_len) {
            let packed = material_table[word_idx];
            mat = (packed >> shift) & 0xFFFFu;
          }
        }

        let gx = params.g_origin.x + i32(origin.x + vx);
        let gy = params.g_origin.y + i32(origin.y + vy);
        let gz = params.g_origin.z + i32(origin.z + vz);

        out_voxels[idx] = CompactVoxelGpu(gx, gy, gz, mat);
      }
    }
    linear = linear + 64u;
  }
}
