struct Params {
  grid_dims: vec4<u32>,
  tile_dims: vec4<u32>,
  num_tiles_xyz: vec4<u32>,
  num_triangles: u32,
  num_tiles: u32,
  tile_voxels: u32,
  store_owner: u32,
  store_color: u32,
  debug: u32,
  dispatch_xy: vec2<u32>,
};

@group(0) @binding(0) var<storage, read> tris: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> tri_indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> occupancy: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> owner_id: array<u32>;
@group(0) @binding(8) var<storage, read_write> color_rgba: array<u32>;
@group(0) @binding(9) var<uniform> params: Params;
@group(0) @binding(10) var<storage, read> brick_origins: array<vec4<u32>>;
@group(0) @binding(11) var<storage, read_write> debug_counts: array<atomic<u32>>;

override WORKGROUP_SIZE: u32 = 64u;
override TILES_PER_WORKGROUP: u32 = 1u;
const TRI_STRIDE: u32 = 6u;
const MAX_ACTIVE_TRIS: u32 = 256u;
const MAX_TILES_PER_WORKGROUP: u32 = 4u;
var<workgroup> active_tris: array<u32, MAX_ACTIVE_TRIS * MAX_TILES_PER_WORKGROUP>;
var<workgroup> active_count: array<u32, MAX_TILES_PER_WORKGROUP>;
var<workgroup> active_overflow: array<u32, MAX_TILES_PER_WORKGROUP>;

fn hash_color(id: u32) -> u32 {
  var x = id * 1664525u + 1013904223u;
  let r = x & 255u;
  x = x * 1664525u + 1013904223u;
  let g = x & 255u;
  x = x * 1664525u + 1013904223u;
  let b = x & 255u;
  return r | (g << 8u) | (b << 16u) | (255u << 24u);
}

fn axis_test(axis: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, half: vec3<f32>) -> bool {
  let p0 = dot(v0, axis);
  let p1 = dot(v1, axis);
  let p2 = dot(v2, axis);
  let min_p = min(p0, min(p1, p2));
  let max_p = max(p0, max(p1, p2));
  let r = half.x * abs(axis.x) + half.y * abs(axis.y) + half.z * abs(axis.z);
  return !(min_p > r || max_p < -r);
}

fn plane_box_intersects(normal: vec3<f32>, d: f32, center: vec3<f32>, half: vec3<f32>) -> bool {
  let r = half.x * abs(normal.x) + half.y * abs(normal.y) + half.z * abs(normal.z);
  let s = dot(normal, center) + d;
  return abs(s) <= r;
}

fn triangle_box_overlap(center: vec3<f32>, half: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, normal: vec3<f32>, d: f32, tri_min: vec3<f32>, tri_max: vec3<f32>) -> bool {
  let v0 = a - center;
  let v1 = b - center;
  let v2 = c - center;
  let e0 = v1 - v0;
  let e1 = v2 - v1;
  let e2 = v0 - v2;

  // Fast AABB reject (triangle AABB vs box).
  let box_min = center - half;
  let box_max = center + half;
  if (tri_min.x > box_max.x || tri_max.x < box_min.x) {
    return false;
  }
  if (tri_min.y > box_max.y || tri_max.y < box_min.y) {
    return false;
  }
  if (tri_min.z > box_max.z || tri_max.z < box_min.z) {
    return false;
  }

  // Plane test before edge axes to early reject (precomputed plane).
  if (!plane_box_intersects(normal, d, center, half)) {
    return false;
  }

  let axes = array<vec3<f32>, 9>(
    vec3<f32>(0.0, -e0.z, e0.y),
    vec3<f32>(0.0, -e1.z, e1.y),
    vec3<f32>(0.0, -e2.z, e2.y),
    vec3<f32>(e0.z, 0.0, -e0.x),
    vec3<f32>(e1.z, 0.0, -e1.x),
    vec3<f32>(e2.z, 0.0, -e2.x),
    vec3<f32>(-e0.y, e0.x, 0.0),
    vec3<f32>(-e1.y, e1.x, 0.0),
    vec3<f32>(-e2.y, e2.x, 0.0)
  );

  for (var i = 0u; i < 9u; i = i + 1u) {
    if (!axis_test(axes[i], v0, v1, v2, half)) {
      return false;
    }
  }

  return true;
}

@compute @workgroup_size(WORKGROUP_SIZE, TILES_PER_WORKGROUP, 1)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tile_lane = lid.y;
  // Linearize the (possibly 3-D) workgroup id back to a flat workgroup index so a
  // dense grid with more tiles than the per-dimension dispatch limit still maps
  // correctly. dispatch_xy = the dispatch's x,y extents; for a 1-D dispatch
  // (sparse / small dense) wg_id.y = wg_id.z = 0, so this reduces to wg_id.x.
  let linear_wg = wg_id.x + wg_id.y * params.dispatch_xy.x
    + wg_id.z * params.dispatch_xy.x * params.dispatch_xy.y;
  let tile_index = linear_wg * TILES_PER_WORKGROUP + tile_lane;
  let valid_tile = tile_index < params.num_tiles;
  if (lid.x == 0u && valid_tile && params.debug != 0u) {
    atomicAdd(&debug_counts[0], 1u);
  }

  var tile_min = vec3<u32>(0u, 0u, 0u);
  if (valid_tile && params.num_tiles_xyz.x > 0u) {
    let tile_x = tile_index % params.num_tiles_xyz.x;
    let tile_y = (tile_index / params.num_tiles_xyz.x) % params.num_tiles_xyz.y;
    let tile_z = tile_index / (params.num_tiles_xyz.x * params.num_tiles_xyz.y);
    tile_min = vec3<u32>(
      tile_x * params.tile_dims.x,
      tile_y * params.tile_dims.y,
      tile_z * params.tile_dims.z
    );
  } else if (valid_tile) {
    tile_min = brick_origins[tile_index].xyz;
  }

  var offset = 0u;
  var end = 0u;
  if (valid_tile) {
    offset = tile_offsets[tile_index];
    end = tile_offsets[tile_index + 1u];
  }
  let has_tris = valid_tile && (offset != end);

  let tile_max = vec3<u32>(
    min(tile_min.x + params.tile_dims.x, params.grid_dims.x),
    min(tile_min.y + params.tile_dims.y, params.grid_dims.y),
    min(tile_min.z + params.tile_dims.z, params.grid_dims.z)
  );
  let tile_min_f = vec3<f32>(f32(tile_min.x), f32(tile_min.y), f32(tile_min.z));
  let tile_max_f = vec3<f32>(f32(tile_max.x), f32(tile_max.y), f32(tile_max.z));
  let tile_center = (tile_min_f + tile_max_f) * 0.5;
  let tile_half = (tile_max_f - tile_min_f) * 0.5;

  if (lid.x == 0u) {
    active_count[tile_lane] = 0u;
    active_overflow[tile_lane] = 0u;
    if (has_tris) {
      let base_index = tile_lane * MAX_ACTIVE_TRIS;
      for (var i = offset; i < end; i = i + 1u) {
        let tri = tri_indices[i];
        if (tri >= params.num_triangles) {
          continue;
        }
        let base = tri * TRI_STRIDE;
        let plane = tris[base + 5u];
        if (plane_box_intersects(plane.xyz, plane.w, tile_center, tile_half)) {
          if (active_count[tile_lane] < MAX_ACTIVE_TRIS) {
            active_tris[base_index + active_count[tile_lane]] = tri;
            active_count[tile_lane] = active_count[tile_lane] + 1u;
          } else {
            active_overflow[tile_lane] = 1u;
          }
        }
      }
    }
  }
  workgroupBarrier();
  let half = vec3<f32>(0.5, 0.5, 0.5);

  let tile_voxels = params.tile_voxels;
  if (has_tris) {
    var linear = lid.x;
    loop {
      if (linear >= tile_voxels) {
        break;
      }
      let vx = linear % params.tile_dims.x;
      let vy = (linear / params.tile_dims.x) % params.tile_dims.y;
      let vz = (linear / (params.tile_dims.x * params.tile_dims.y));
      let gx = tile_min.x + vx;
      let gy = tile_min.y + vy;
      let gz = tile_min.z + vz;

      if (gx < params.grid_dims.x && gy < params.grid_dims.y && gz < params.grid_dims.z) {
        let center = vec3<f32>(f32(gx) + 0.5, f32(gy) + 0.5, f32(gz) + 0.5);
        var hit = false;
        var best = 0xffffffffu;
        if (active_overflow[tile_lane] == 0u) {
          let base_index = tile_lane * MAX_ACTIVE_TRIS;
          for (var i = 0u; i < active_count[tile_lane]; i = i + 1u) {
            let tri = active_tris[base_index + i];
            let base = tri * TRI_STRIDE;
            let a = tris[base].xyz;
            let b = tris[base + 1u].xyz;
            let c = tris[base + 2u].xyz;
            let tri_min = tris[base + 3u].xyz;
            let tri_max = tris[base + 4u].xyz;
            let plane = tris[base + 5u];
            if (triangle_box_overlap(center, half, a, b, c, plane.xyz, plane.w, tri_min, tri_max)) {
              hit = true;
              if (tri < best) {
                best = tri;
              }
            }
          }
        } else {
          for (var i = offset; i < end; i = i + 1u) {
            let tri = tri_indices[i];
            if (tri >= params.num_triangles) {
              continue;
            }
            let base = tri * TRI_STRIDE;
            let a = tris[base].xyz;
            let b = tris[base + 1u].xyz;
            let c = tris[base + 2u].xyz;
            let tri_min = tris[base + 3u].xyz;
            let tri_max = tris[base + 4u].xyz;
            let plane = tris[base + 5u];
            if (triangle_box_overlap(center, half, a, b, c, plane.xyz, plane.w, tri_min, tri_max)) {
              hit = true;
              if (tri < best) {
                best = tri;
              }
            }
          }
        }

        if (params.debug != 0u) {
          atomicAdd(&debug_counts[1], 1u);
        }
        if (hit) {
          if (params.debug != 0u) {
            atomicAdd(&debug_counts[2], 1u);
          }
          if (params.num_tiles_xyz.x > 0u) {
            // Global X-major occupancy WORD index, computed WITHOUT forming the
            // full linear index in u32: at 2048³, gx + n·gy + n²·gz reaches 8.6e9
            // > u32::MAX, overflowing for gz > 1024 (the top half — silently
            // dropped). The grid is a power of two, so for n >= 32 (32-aligned)
            // the low 5 bits come from gx and the word index stays in range:
            // word = (gx>>5) + (n>>5)·(gy + n·gz), max = n³/32 < u32.
            let nx = params.grid_dims.x;
            let row = gy + params.grid_dims.y * gz;
            var word: u32;
            var bit: u32;
            if (nx >= 32u) {
              word = (gx >> 5u) + (nx >> 5u) * row;
              bit = gx & 31u;
            } else {
              let linear = gx + nx * row;
              word = linear >> 5u;
              bit = linear & 31u;
            }
            atomicOr(&occupancy[word], 1u << bit);
            // owner/color are per-voxel arrays, only allocated for n <= 512
            // (n³·4 bytes <= the storage limit), where the full index fits u32.
            if (params.store_owner == 1u) {
              owner_id[gx + nx * row] = best;
            }
            if (params.store_color == 1u) {
              color_rgba[gx + nx * row] = hash_color(best);
            }
          } else {
            let local_index = vx + params.tile_dims.x * (vy + params.tile_dims.y * vz);
            let word = (tile_index * ((params.tile_voxels + 31u) / 32u)) + (local_index >> 5u);
            let bit = local_index & 31u;
            atomicOr(&occupancy[word], 1u << bit);
            if (params.store_owner == 1u) {
              owner_id[tile_index * params.tile_voxels + local_index] = best;
            }
            if (params.store_color == 1u) {
              color_rgba[tile_index * params.tile_voxels + local_index] = hash_color(best);
            }
          }
        }
      }
      linear = linear + WORKGROUP_SIZE;
    }
  }
}
