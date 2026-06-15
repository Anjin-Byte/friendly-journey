//! The input snapshot bridging winit events and the pure [`FlyCamera`].
//!
//! The winit handlers (the effectful edge) set the held-key flags and accumulate
//! the per-frame mouse/scroll deltas here; [`FlyCamera::apply`] reads it as plain
//! data. Held flags persist across frames (they reflect key state); deltas are
//! accumulated within a frame and cleared by [`Input::end_frame`] once consumed.
//!
//! [`FlyCamera`]: crate::camera::FlyCamera
//! [`FlyCamera::apply`]: crate::camera::FlyCamera::apply

/// A frame's worth of camera input: persistent key state plus accumulated
/// pointer deltas.
//
// Six movement/modifier flags is the natural shape for keyboard state; a packed
// bitset would obscure it (Engineering Codex: Boring Data Layouts).
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Input {
    /// `W` — move along the forward direction.
    pub(crate) forward: bool,
    /// `S` — move opposite the forward direction.
    pub(crate) back: bool,
    /// `A` — strafe left.
    pub(crate) left: bool,
    /// `D` — strafe right.
    pub(crate) right: bool,
    /// `E` / Space — rise along world `+Y`.
    pub(crate) up: bool,
    /// `Q` / Ctrl — descend along world `−Y`.
    pub(crate) down: bool,
    /// Shift — apply the movement speed boost.
    pub(crate) boost: bool,
    /// Accumulated horizontal mouse motion (pixels) since the last frame.
    pub(crate) look_dx: f32,
    /// Accumulated vertical mouse motion (pixels) since the last frame.
    pub(crate) look_dy: f32,
    /// Accumulated scroll-wheel notches since the last frame.
    pub(crate) scroll: f32,
}

impl Input {
    /// Clears the per-frame deltas after they have been applied. Held key flags
    /// are intentionally preserved.
    pub(crate) fn end_frame(&mut self) {
        self.look_dx = 0.0;
        self.look_dy = 0.0;
        self.scroll = 0.0;
    }
}
