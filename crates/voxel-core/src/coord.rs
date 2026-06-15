//! Integer voxel coordinates.

use crate::Resolution;

/// A base-voxel coordinate `(x, y, z)`, each component a voxel index along its
/// axis. Whether it is in bounds depends on a [`Resolution`]; see
/// [`VoxelCoord::in_bounds`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VoxelCoord {
    /// X index.
    pub x: u32,
    /// Y index.
    pub y: u32,
    /// Z index.
    pub z: u32,
}

impl VoxelCoord {
    /// Constructs a coordinate from its three components.
    #[must_use]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Constructs from an array, the layout used by the per-axis DDA loop.
    #[must_use]
    pub const fn from_array(a: [u32; 3]) -> Self {
        Self {
            x: a[0],
            y: a[1],
            z: a[2],
        }
    }

    /// The components as an array, for per-axis iteration.
    #[must_use]
    pub const fn to_array(self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }

    /// `true` iff every component is `< resolution.voxels_per_axis()`.
    #[must_use]
    pub fn in_bounds(self, resolution: Resolution) -> bool {
        let n = resolution.voxels_per_axis();
        self.x < n && self.y < n && self.z < n
    }
}
