//! Pure camera math for the viewer.
//!
//! Two cameras share one [`GpuCamera`] output:
//! - [`orbit_camera`] — a deterministic turntable used by the scripted
//!   `--frames` profiling path (and the default until the user takes control).
//! - [`FlyCamera`] — an interactive free-fly camera driven by [`Input`].
//!
//! Everything here is pure: positions and angles in, a [`GpuCamera`] uniform
//! out. No winit, no device access, no clock — the viewer's event loop reads the
//! clock and devices and feeds the results in as `dt` and an [`Input`] snapshot
//! (Engineering Codex: *Pure Core, Effectful Edges*). That keeps the camera math
//! unit-testable with zero setup.

use glam::Vec3;
use voxel_gpu::GpuCamera;

use crate::input::Input;

/// Vertical field of view, in degrees.
const FOV_Y_DEG: f32 = 60.0;
/// Pitch is clamped just shy of straight up/down to keep the basis well-defined.
const MAX_PITCH: f32 = 1.553_343; // ≈ 89° in radians
/// Radians of look rotation per pixel of mouse motion.
const LOOK_SENSITIVITY: f32 = 0.003;
/// Multiplicative speed change per scroll-wheel notch.
const SCROLL_SPEED_FACTOR: f32 = 1.15;
/// Hold-to-boost movement multiplier.
const BOOST_MULTIPLIER: f32 = 4.0;

/// Builds the orthonormal camera basis `(forward, right, up)` for a forward
/// direction, matching the renderer's convention (right-handed, world-up `+Y`).
fn basis(forward: Vec3) -> (Vec3, Vec3, Vec3) {
    let forward = forward.normalize();
    let right = forward.cross(Vec3::Y).normalize();
    let up = right.cross(forward);
    (forward, right, up)
}

/// Packs an eye/forward pair into the GPU camera uniform for a `w×h` viewport
/// over an `n³` grid with `k` internal levels.
fn pack(eye: Vec3, forward: Vec3, w: u32, h: u32, n: f32, k: u32) -> GpuCamera {
    let (forward, right, up) = basis(forward);
    GpuCamera {
        eye: eye.to_array(),
        tan: (FOV_Y_DEG.to_radians() * 0.5).tan(),
        forward: forward.to_array(),
        aspect: w as f32 / h as f32,
        right: right.to_array(),
        n,
        up: up.to_array(),
        pad: 0.0,
        dims: [w, h, k, 0],
    }
}

/// The orbiting turntable camera at `angle` radians around an `n³` grid.
///
/// Deterministic in `angle`, so the scripted `--frames` profiling run is
/// reproducible regardless of frame rate.
#[must_use]
pub(crate) fn orbit_camera(angle: f32, n: f32, w: u32, h: u32, k: u32) -> GpuCamera {
    let (eye, forward) = orbit_eye_forward(angle, n);
    pack(eye, forward, w, h, n, k)
}

/// The orbit camera's eye position and forward direction at `angle`. Exposed so
/// the free camera can be seeded from the current orbit pose without a jump.
#[must_use]
pub(crate) fn orbit_eye_forward(angle: f32, n: f32) -> (Vec3, Vec3) {
    let centre = Vec3::splat(n * 0.5);
    let radius = n * 1.6;
    let eye = centre + Vec3::new(angle.cos() * radius, n * 0.35, angle.sin() * radius);
    (eye, (centre - eye).normalize())
}

/// An interactive free-fly camera: world position plus yaw/pitch look angles and
/// a movement speed (world units per second).
#[derive(Clone, Copy, Debug)]
pub(crate) struct FlyCamera {
    /// World-space eye position.
    pub(crate) eye: Vec3,
    /// Yaw (radians) about world `+Y`; `0` looks toward `+Z`.
    pub(crate) yaw: f32,
    /// Pitch (radians); positive looks up, clamped to ±[`MAX_PITCH`].
    pub(crate) pitch: f32,
    /// Base movement speed in world units per second.
    pub(crate) speed: f32,
}

impl FlyCamera {
    /// Seeds a free camera from an eye position and forward direction (e.g. the
    /// current orbit pose), with a movement speed scaled to the grid size `n`.
    #[must_use]
    pub(crate) fn from_eye_forward(eye: Vec3, forward: Vec3, n: f32) -> Self {
        let forward = forward.normalize();
        Self {
            eye,
            yaw: forward.x.atan2(forward.z),
            pitch: forward.y.clamp(-1.0, 1.0).asin(),
            speed: (n * 0.6).max(1.0),
        }
    }

    /// The unit forward direction implied by the current yaw/pitch.
    #[must_use]
    pub(crate) fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(sy * cp, sp, cy * cp)
    }

    /// Advances the camera by `dt` seconds under the current [`Input`]: applies
    /// mouse-look, scroll-to-speed, and `WASDQE` movement. Pure — `dt` and
    /// `input` are supplied by the event loop.
    pub(crate) fn apply(&mut self, dt: f32, input: &Input) {
        // Look: yaw follows horizontal motion, pitch follows vertical (inverted
        // so dragging up looks up), clamped to keep the basis well-defined.
        self.yaw += input.look_dx * LOOK_SENSITIVITY;
        self.pitch = (self.pitch - input.look_dy * LOOK_SENSITIVITY).clamp(-MAX_PITCH, MAX_PITCH);

        // Speed: each scroll notch scales the base speed geometrically.
        if input.scroll != 0.0 {
            self.speed = (self.speed * SCROLL_SPEED_FACTOR.powf(input.scroll)).clamp(0.05, 1.0e6);
        }

        // Movement: along the look basis, with world-up for vertical.
        let (forward, right, _up) = basis(self.forward());
        let axis = |pos: bool, neg: bool| f32::from(pos) - f32::from(neg);
        let dir = forward * axis(input.forward, input.back)
            + right * axis(input.right, input.left)
            + Vec3::Y * axis(input.up, input.down);
        if dir.length_squared() > 0.0 {
            let boost = if input.boost { BOOST_MULTIPLIER } else { 1.0 };
            self.eye += dir.normalize() * self.speed * boost * dt;
        }
    }

    /// Packs the current pose into a [`GpuCamera`] for a `w×h` viewport over an
    /// `n³` grid with `k` internal levels.
    #[must_use]
    pub(crate) fn to_gpu(self, w: u32, h: u32, n: f32, k: u32) -> GpuCamera {
        pack(self.eye, self.forward(), w, h, n, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn forward_at_origin_angles_points_along_z() {
        let cam = FlyCamera {
            eye: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            speed: 1.0,
        };
        let f = cam.forward();
        assert!(
            approx(f.x, 0.0) && approx(f.y, 0.0) && approx(f.z, 1.0),
            "{f:?}"
        );
    }

    #[test]
    fn gpu_basis_is_orthonormal() {
        let cam = FlyCamera {
            eye: Vec3::new(10.0, 5.0, -3.0),
            yaw: 0.7,
            pitch: 0.3,
            speed: 1.0,
        };
        let g = cam.to_gpu(800, 600, 512.0, 3);
        let f = Vec3::from_array(g.forward);
        let r = Vec3::from_array(g.right);
        let u = Vec3::from_array(g.up);
        for v in [f, r, u] {
            assert!(approx(v.length(), 1.0), "not unit: {v:?}");
        }
        assert!(approx(f.dot(r), 0.0) && approx(f.dot(u), 0.0) && approx(r.dot(u), 0.0));
    }

    #[test]
    fn moving_forward_advances_along_forward() {
        let mut cam = FlyCamera {
            eye: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            speed: 10.0,
        };
        let f = cam.forward();
        let before = cam.eye;
        let input = Input {
            forward: true,
            ..Default::default()
        };
        cam.apply(0.5, &input);
        let moved = cam.eye - before;
        assert!(moved.dot(f) > 0.0, "should move along forward: {moved:?}");
        assert!(
            approx(moved.length(), 5.0),
            "10 u/s * 0.5 s = 5: {}",
            moved.length()
        );
    }

    #[test]
    fn pitch_clamps_to_avoid_gimbal() {
        let mut cam = FlyCamera {
            eye: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            speed: 1.0,
        };
        let input = Input {
            look_dy: -1.0e6, // slam the look way up
            ..Default::default()
        };
        cam.apply(0.016, &input);
        assert!(cam.pitch <= MAX_PITCH && cam.pitch >= -MAX_PITCH);
        assert!(approx(cam.pitch, MAX_PITCH));
    }

    #[test]
    fn from_eye_forward_round_trips_direction() {
        let eye = Vec3::new(1.0, 2.0, 3.0);
        let dir = Vec3::new(0.3, -0.6, 0.74).normalize();
        let cam = FlyCamera::from_eye_forward(eye, dir, 512.0);
        let f = cam.forward();
        assert!(
            approx(f.x, dir.x) && approx(f.y, dir.y) && approx(f.z, dir.z),
            "{f:?} vs {dir:?}"
        );
    }

    #[test]
    fn scroll_scales_speed() {
        let mut cam = FlyCamera {
            eye: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            speed: 100.0,
        };
        let input = Input {
            scroll: 1.0,
            ..Default::default()
        };
        cam.apply(0.016, &input);
        assert!(approx(cam.speed, 100.0 * SCROLL_SPEED_FACTOR));
    }
}
