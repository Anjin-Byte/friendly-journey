//! The on-screen performance HUD.
//!
//! A small overlay drawn after the image blit: text (FPS, frame/kernel times,
//! scene info, camera pose) plus a frame-time sparkline. It is rendered as
//! instanced quads sampling an 8×8 bitmap-font atlas — see [`hud.wgsl`].
//!
//! The font data is the public-domain `font8x8` table (a pure, no-`std`,
//! no-churn data crate kept local to this UI binary; Engineering Codex: a leaf
//! data dependency, not a heavy adapter — *Shared Dependencies*). Layout is pure
//! and unit-tested; only [`Hud`] touches the device.
//!
//! [`hud.wgsl`]: ../../shaders/hud.wgsl

use bytemuck::{Pod, Zeroable};

/// Atlas geometry: 128 ASCII glyphs in a 16×8 grid of 8×8 cells.
const GLYPH_PX: u32 = 8;
const ATLAS_COLS: u32 = 16;
const ATLAS_ROWS: u32 = 8;
const ATLAS_W: u32 = ATLAS_COLS * GLYPH_PX; // 128
const ATLAS_H: u32 = ATLAS_ROWS * GLYPH_PX; // 64
/// Maximum quads per frame (text glyphs + panel + sparkline bars). Fixed so the
/// instance buffer and its bind group are built once.
const MAX_QUADS: usize = 8192;

/// One instanced HUD rectangle, matching `HudQuad` in `hud.wgsl`.
///
/// Must be exactly 64 bytes: that is the storage-array stride the WGSL side
/// reads (its `HudQuad` is laid out to 64 bytes via three scalar `u32` pads —
/// a `vec3<u32>` there would make it 80 and scatter every instance). The
/// `const` assertion below pins the Rust side of that contract.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct HudQuad {
    rect: [f32; 4],
    uv: [f32; 4],
    color: [f32; 4],
    kind: u32,
    _pad: [u32; 3],
}

const _: () = assert!(
    std::mem::size_of::<HudQuad>() == 64,
    "HudQuad must be 64 bytes to match the hud.wgsl std430 instance stride"
);

/// HUD uniform (viewport size), matching `HudUniform` in `hud.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HudUniform {
    screen: [f32; 2],
    _pad: [f32; 2],
}

/// The 8×8 bitmap rows for ASCII byte `c` (row 0 top; bit `1<<x` is column `x`
/// from the left). Non-printable / out-of-range bytes render blank.
fn glyph_rows(c: u8) -> [u8; 8] {
    font8x8::legacy::BASIC_LEGACY
        .get(c as usize)
        .copied()
        .unwrap_or([0; 8])
}

/// Rasterizes the atlas: an `ATLAS_W × ATLAS_H` R8 image, 255 where a glyph bit
/// is set. Pure, so the font bit-order is pinned by a unit test.
fn build_atlas() -> Vec<u8> {
    let mut px = vec![0u8; (ATLAS_W * ATLAS_H) as usize];
    for code in 0u32..(ATLAS_COLS * ATLAS_ROWS) {
        let rows = glyph_rows(code as u8);
        let (cx, cy) = (code % ATLAS_COLS, code / ATLAS_COLS);
        for (y, row) in rows.iter().enumerate() {
            for x in 0..GLYPH_PX {
                if (row >> x) & 1 == 1 {
                    let ax = cx * GLYPH_PX + x;
                    let ay = cy * GLYPH_PX + y as u32;
                    px[(ay * ATLAS_W + ax) as usize] = 255;
                }
            }
        }
    }
    px
}

/// Atlas UV rectangle `(u0, v0, u1, v1)` for ASCII byte `c`.
fn glyph_uv(c: u8) -> [f32; 4] {
    let (cx, cy) = (u32::from(c) % ATLAS_COLS, u32::from(c) / ATLAS_COLS);
    let (u0, v0) = (cx as f32 / ATLAS_COLS as f32, cy as f32 / ATLAS_ROWS as f32);
    [
        u0,
        v0,
        u0 + 1.0 / ATLAS_COLS as f32,
        v0 + 1.0 / ATLAS_ROWS as f32,
    ]
}

/// A buffer of HUD quads being assembled for one frame, plus its layout cursor.
pub(crate) struct HudBuilder {
    quads: Vec<HudQuad>,
    /// Pixels advanced per glyph (slightly tighter than the 8px cell).
    scale: f32,
}

impl HudBuilder {
    /// A builder rendering text at `scale` device pixels per font pixel.
    pub(crate) fn new(scale: f32) -> Self {
        Self {
            quads: Vec::with_capacity(256),
            scale,
        }
    }

    /// A solid filled rectangle (pixel-space, top-left origin).
    pub(crate) fn solid(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        self.quads.push(HudQuad {
            rect: [x, y, w, h],
            uv: [0.0; 4],
            color,
            kind: 1,
            _pad: [0; 3],
        });
    }

    /// Horizontal advance per character (glyph cell plus a little letter spacing).
    pub(crate) fn advance(&self) -> f32 {
        7.5 * self.scale
    }

    /// The pixel width of `s` at this scale (for sizing panels to content).
    pub(crate) fn text_width(&self, s: &str) -> f32 {
        s.len() as f32 * self.advance()
    }

    /// Lays out a line of text with its top-left at `(x, y)`, returning the x
    /// just past the last glyph (so callers can append). Non-printable bytes
    /// advance without drawing.
    pub(crate) fn text(&mut self, x: f32, y: f32, s: &str, color: [f32; 4]) -> f32 {
        let cell = GLYPH_PX as f32 * self.scale;
        let advance = self.advance();
        let mut cx = x;
        for &b in s.as_bytes() {
            if b != b' ' && (32..128).contains(&b) {
                self.quads.push(HudQuad {
                    rect: [cx, y, cell, cell],
                    uv: glyph_uv(b),
                    color,
                    kind: 0,
                    _pad: [0; 3],
                });
            }
            cx += advance;
        }
        cx
    }

    /// The baseline-to-baseline pixel pitch of a text line (glyph cell plus a
    /// generous inter-line gap so stacked lines read comfortably).
    pub(crate) fn line_height(&self) -> f32 {
        GLYPH_PX as f32 * self.scale + 6.0 * self.scale
    }

    /// Draws a frame-time sparkline filling `rect` (x, y, w, h), one bar per
    /// sample, scaled so `max_ms` reaches the top. Bars over `budget_ms` are
    /// tinted to flag spikes.
    #[allow(clippy::many_single_char_names)] // x/y/w/h are the natural rect names
    pub(crate) fn sparkline(
        &mut self,
        rect: [f32; 4],
        samples: &[f32],
        max_ms: f32,
        budget_ms: f32,
    ) {
        let [x, y, w, h] = rect;
        self.solid(x, y, w, h, [0.10, 0.12, 0.16, 0.85]);
        if samples.is_empty() || max_ms <= 0.0 {
            return;
        }
        let n = samples.len();
        let bar_w = (w / n as f32).max(1.0);
        for (i, &ms) in samples.iter().enumerate() {
            let frac = (ms / max_ms).clamp(0.0, 1.0);
            let bh = frac * h;
            let bx = x + i as f32 * bar_w;
            let color = if ms > budget_ms {
                [0.95, 0.45, 0.30, 0.95] // over budget — warm
            } else {
                [0.40, 0.85, 0.55, 0.90] // healthy — green
            };
            self.solid(bx, y + h - bh, bar_w.max(1.0) - 0.5, bh, color);
        }
    }

    /// The assembled quads, truncated to the per-frame capacity.
    fn quads(&self) -> &[HudQuad] {
        let n = self.quads.len().min(MAX_QUADS);
        &self.quads[..n]
    }
}

/// The GPU resources for drawing [`HudBuilder`] output over the surface.
pub(crate) struct Hud {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    instance_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    count: u32,
}

impl Hud {
    /// Builds the font atlas, the instance/uniform buffers, and the alpha-blended
    /// quad pipeline targeting `surface_format`.
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let (atlas_view, sampler) = upload_atlas(device, queue);

        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud instances"),
            size: (MAX_QUADS * std::mem::size_of::<HudQuad>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud uniform"),
            size: std::mem::size_of::<HudUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = bind_layout(device);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hud bind group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: instance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline = build_pipeline(device, &layout, surface_format);

        Self {
            pipeline,
            bind_group,
            instance_buf,
            uniform_buf,
            count: 0,
        }
    }

    /// Uploads this frame's quads and viewport size; call before [`draw`].
    ///
    /// [`draw`]: Self::draw
    pub(crate) fn prepare(&mut self, queue: &wgpu::Queue, w: u32, h: u32, builder: &HudBuilder) {
        let quads = builder.quads();
        self.count = u32::try_from(quads.len()).unwrap_or(0);
        queue.write_buffer(
            &self.uniform_buf,
            0,
            bytemuck::bytes_of(&HudUniform {
                screen: [w as f32, h as f32],
                _pad: [0.0; 2],
            }),
        );
        if !quads.is_empty() {
            queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(quads));
        }
    }

    /// Draws the prepared quads into an in-progress render pass over the surface.
    pub(crate) fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..6, 0..self.count);
    }
}

/// Rasterizes and uploads the font atlas (R8) once, returning its view and a
/// nearest-filter sampler (crisp scaled-up pixels).
fn upload_atlas(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::TextureView, wgpu::Sampler) {
    let extent = wgpu::Extent3d {
        width: ATLAS_W,
        height: ATLAS_H,
        depth_or_array_layers: 1,
    };
    let atlas = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hud atlas"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &atlas,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &build_atlas(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(ATLAS_W),
            rows_per_image: Some(ATLAS_H),
        },
        extent,
    );
    let view = atlas.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("hud sampler"),
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });
    (view, sampler)
}

/// The HUD bind-group layout: instance storage + viewport uniform (vertex), and
/// the font atlas + sampler (fragment).
fn bind_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let buffer = |binding, vis, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("hud layout"),
        entries: &[
            buffer(
                0,
                wgpu::ShaderStages::VERTEX,
                wgpu::BufferBindingType::Storage { read_only: true },
            ),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            buffer(
                3,
                wgpu::ShaderStages::VERTEX,
                wgpu::BufferBindingType::Uniform,
            ),
        ],
    })
}

/// The alpha-blended quad pipeline targeting `surface_format`.
fn build_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    surface_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("hud"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/hud.wgsl").into()),
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hud pl"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("hud pipeline"),
        layout: Some(&pl),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins the font bit-order so the atlas is never silently mirrored/flipped:
    /// the glyph 'L' must have a left vertical stroke and a bottom horizontal
    /// stroke, with the top-right corner empty.
    /// Ink mass `(left, right, top, bottom)` of a glyph, splitting the 8×8 cell
    /// into halves. Under the convention `build_atlas` uses (bit `1<<x` is
    /// column `x` from the left, row 0 is the top).
    fn ink_mass(c: u8) -> (u32, u32, u32, u32) {
        let rows = glyph_rows(c);
        let (mut left, mut right, mut top, mut bottom) = (0, 0, 0, 0);
        for (y, r) in rows.iter().enumerate() {
            for x in 0..8u32 {
                if (r >> x) & 1 == 1 {
                    if x < 4 {
                        left += 1;
                    } else {
                        right += 1;
                    }
                    if y < 4 {
                        top += 1;
                    } else {
                        bottom += 1;
                    }
                }
            }
        }
        (left, right, top, bottom)
    }

    /// Pins the font bit-order so the atlas is never silently mirrored or
    /// flipped: 'F' is top- and left-heavy; 'L' is bottom- and left-heavy.
    /// Together these fix both axes (a horizontal mirror flips left/right; a
    /// vertical flip swaps top/bottom).
    #[test]
    fn font_orientation_is_upright() {
        let (fl, fr, ft, fb) = ink_mass(b'F');
        assert!(fl > fr, "F left-heavy: {fl} vs {fr}");
        assert!(ft > fb, "F top-heavy: {ft} vs {fb}");
        let (ll, lr, lt, lb) = ink_mass(b'L');
        assert!(ll > lr, "L left-heavy: {ll} vs {lr}");
        assert!(lb > lt, "L bottom-heavy: {lb} vs {lt}");
    }

    #[test]
    fn space_glyph_is_blank() {
        assert_eq!(glyph_rows(b' '), [0; 8]);
    }

    #[test]
    fn text_emits_one_quad_per_visible_glyph() {
        let mut b = HudBuilder::new(2.0);
        b.text(0.0, 0.0, "A B", [1.0; 4]); // space is skipped
        assert_eq!(b.quads.len(), 2);
    }

    #[test]
    fn glyph_uv_is_within_unit_square() {
        for c in [b'A', b'z', b'0', b'~'] {
            let [u0, v0, u1, v1] = glyph_uv(c);
            assert!((0.0..=1.0).contains(&u0) && (0.0..=1.0).contains(&u1));
            assert!((0.0..=1.0).contains(&v0) && (0.0..=1.0).contains(&v1));
            assert!(u1 > u0 && v1 > v0);
        }
    }

    #[test]
    fn atlas_has_expected_size_and_ink() {
        let a = build_atlas();
        assert_eq!(a.len(), (ATLAS_W * ATLAS_H) as usize);
        assert!(a.contains(&255), "atlas should contain glyphs");
    }
}
