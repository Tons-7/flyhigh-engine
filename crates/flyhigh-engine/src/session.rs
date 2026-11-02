//! `Session`: the plans of one model on one engine for one frame size (ARCHITECTURE.md). Two frames in, one frame at phase
//! `t` out, on textures the session owns or on views the host passes. No timestamps, no clocks, no decision about
//! which interval or phase to generate: that is the Stream's and the Pacer's.
use crate::gen_tex::{TexMode, TexOptions};
use crate::gpu::readback::Readback;
use crate::options::Options;
use crate::{Engine, Gpu, Model, NetOpts, OptionsError, SessionExt, Step, TimedRun, TuneTable};
use anyhow::{Context, Result};
use serde::Serialize;
use std::cell::Cell;
use std::rc::Rc;

/// The network's working resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Internal {
    /// The frame itself up to 960 px on the long side, above that the 480-line rung (`default_internal`).
    Auto,
    /// The frame size.
    Frame,
    /// An explicit size (32 x 32 up to the frame size); the model is scale-specific, keep it near the 480-line rung.
    Size(u32, u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderSampling {
    /// The sampler the checkpoint was trained with.
    Model,
    Bilinear,
    Bicubic,
}

/// Pixels unchanged for `frames` source pairs (a HUD, burnt-in subtitles, a logo) are taken from the sources instead of
/// the network.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct StaticProtection {
    pub frames: u32,
    /// Largest per-channel change (0..255 levels) still counted as unchanged.
    pub tolerance: u32,
}

impl Default for StaticProtection {
    fn default() -> Self {
        StaticProtection { frames: crate::DEFAULT_STATIC_FRAMES, tolerance: crate::DEFAULT_STATIC_TOLERANCE }
    }
}

/// Fixed when a session is built; changing one means a new session (tens of milliseconds, plus the autotune at the
/// first build of a new size). The Stream takes the same options in `Stream::open`, the CPU convenience `api::Options`
/// mirrors them. `validate()` names the field and the reason when one is out of range.
#[derive(Clone, Debug, PartialEq, Options)]
pub struct SessionOptions {
    /// Width of the source frames and of the generated frames, at least 64.
    #[option(default = "(required)", values = ">= 64")]
    pub width: u32,
    /// Height of the source frames and of the generated frames, at least 64.
    #[option(default = "(required)", values = ">= 64")]
    pub height: u32,
    /// Resolution the network runs at; only the final render runs at the frame size. Automatic: the frame itself up to
    /// 960 px on the long side, above that the 480-line rung (1920 x 1080 runs at 856 x 480). The model is trained at
    /// that scale: raising it above about 540 lines makes the motion estimates worse, not better, and costs more. Lower
    /// it only to save GPU time.
    #[option(
        default = "automatic",
        values = "automatic, frame, w x h (32 x 32 up to the frame)",
        c = "internal_w, internal_h (0, 0 = automatic)"
    )]
    pub internal: Internal,
    /// Refinement passes of the fine flow. 2 costs about 20% more GPU time for a small gain on hard motion; 3 rarely
    /// helps.
    #[option(values = "1 to 3")]
    pub k: u32,
    /// Periodic-structure repair of the flow (fences, railings, window grids under a camera pan). 1 = safe: only
    /// switches where the motion clearly jumped by a period; measured neutral or better on games, anime and natural
    /// video; costs about 0.2 ms per source pair at the default network size on an RTX 5090, 0.6 ms on an RTX 4060 Ti
    /// with the network at 720p. 2 = aggressive: better on drawn content (+0.4 dB on the anime test clip), worse on
    /// natural video (about -0.8 dB). 0 = off.
    #[option(values = "0, 1, 2")]
    pub repair: u32,
    /// How the source frames are sampled when the in-between is rendered. The model manifest says what it was trained
    /// with (the shipped model: bicubic). Override only to compare.
    #[option(default = "the model's", values = "model, bilinear, bicubic", c = "render (0, 1, 2)")]
    pub render: RenderSampling,
    /// Keeps what did not move still: pixels unchanged for the last `frames` source frames that also stayed put while
    /// the picture around them moved, twice (a game's HUD, the crosshair, chat, burnt-in subtitles, a logo), are taken
    /// from the source frames instead of being warped with the motion behind them; `tolerance` is the largest
    /// per-channel change still counted as unchanged. On for games and screen capture with the defaults, frames 8 and
    /// tolerance 4, which also suit streamed (compressed) video: a looser tolerance or fewer frames hold parts of the
    /// moving picture as overlays. Nothing is protected for the first `frames` frames after a session starts or an
    /// overlay appears, nor in the first in-between after the camera starts moving.
    #[option(
        values = "off, or frames 1 to 255 with tolerance 0 to 255",
        c = "keep_static (0, 1), static_frames, static_tolerance"
    )]
    pub static_protection: Option<StaticProtection>,
    /// Renders the in-between straight from the source textures into the output (one RGBA texel per tap, no
    /// full-resolution intermediate). The same picture as the exact render (38.16 against 38.16 dB on the natural
    /// set), 30 to 40% cheaper per generated frame. Needs an internal size below the frame size; the texture path only.
    #[option(values = "on, off", c = "fast_render (0, 1)")]
    pub fast_render: bool,
    /// With fast render: the t-dependent part of the network runs once per source pair (at t = 0.5) and the other
    /// phases scale its motion linearly, so each further generated frame is only a render (0.95 to 0.21 ms at 1080p on
    /// the portable kernels). Phase quality on 60 fps video unchanged (t = 0.25 / 0.5 / 0.75: 40.58 / 39.63 / 40.83 dB
    /// against 40.54 / 39.64 / 40.83 exact); motion that bends within one source interval is followed less closely.
    #[option(values = "on, off", c = "linear_t (0, 1)")]
    pub linear_t: bool,
    /// Keep the repair stage's maps after the pair: the Stream's cadence re-timing reads them. Raises `repair` to 1.
    #[option(values = "on, off", c = "keep_maps (0, 1)")]
    pub keep_maps: bool,
    /// The host paints a mask of pixels never to interpolate (`write_mask`: one byte per pixel, 0 interpolates,
    /// anything else protects): a HUD, a crosshair, subtitles whose place the host knows. Protected pixels take the
    /// source frames' value like statically protected ones, and the render keeps the motion around them off them.
    /// Works with or without `static_protection`; a rewritten mask applies from the next pair.
    #[option(values = "on, off", c = "protection_mask (0, 1)")]
    pub protection_mask: bool,
    /// Kernel choices from an earlier autotune on this device at this size (the text of a `TuneTable`); without one
    /// the engine tunes at the first build, or uses the fixed routing when the autotune is off.
    #[option(default = "none", values = "an autotune table", c = "tune (text, NULL = none)")]
    pub tune: Option<TuneTable>,
}

impl SessionOptions {
    pub fn new(width: u32, height: u32) -> SessionOptions {
        SessionOptions {
            width,
            height,
            internal: Internal::Auto,
            k: 1,
            repair: crate::DEFAULT_REPAIR,
            render: RenderSampling::Model,
            static_protection: None,
            fast_render: false,
            linear_t: false,
            keep_maps: false,
            protection_mask: false,
            tune: None,
        }
    }

    pub fn internal_size(&self) -> (u32, u32) {
        match self.internal {
            Internal::Auto => crate::default_internal(self.width, self.height),
            Internal::Frame => (self.width, self.height),
            Internal::Size(w, h) => (w.min(self.width), h.min(self.height)),
        }
    }

    pub fn validate(&self) -> Result<(), OptionsError> {
        let err = |field, reason: String| Err(OptionsError { field, reason });
        if self.width < 64 || self.height < 64 {
            return err("width", format!("frame size {}x{} is below 64x64", self.width, self.height));
        }
        if !(1..=3).contains(&self.k) {
            return err("k", format!("{} is not 1..=3", self.k));
        }
        if self.repair > 2 {
            return err("repair", format!("{} is not 0..=2", self.repair));
        }
        let (iw, ih) = self.internal_size();
        if iw < 32 || ih < 32 {
            return err("internal", format!("internal size {iw}x{ih} is below 32x32"));
        }
        if let Some(s) = self.static_protection {
            if !(1..=255).contains(&s.frames) {
                return err("static_protection", format!("frames {} is not 1..=255", s.frames));
            }
            if s.tolerance > 255 {
                return err("static_protection", format!("tolerance {} is above 255", s.tolerance));
            }
        }
        if self.linear_t && !self.fast_render {
            return err("linear_t", "needs fast_render".into());
        }
        Ok(())
    }

    pub(crate) fn net_opts(&self, model: &Model) -> NetOpts {
        let s = self.static_protection.unwrap_or_default();
        NetOpts {
            repair: if self.keep_maps { self.repair.max(1) } else { self.repair },
            keep_maps: self.keep_maps,
            cubic: match self.render {
                RenderSampling::Model => model.blob().render_cubic,
                RenderSampling::Bilinear => false,
                RenderSampling::Bicubic => true,
            },
            keep_static: self.static_protection.is_some(),
            static_frames: s.frames,
            static_tolerance: s.tolerance,
            fast_render: self.fast_render,
            protection_mask: self.protection_mask,
        }
    }

    /// The same options in the generator's shape (mode and cadence are the Stream's).
    pub fn tex_options(&self) -> TexOptions {
        let (iw, ih) = self.internal_size();
        let s = self.static_protection.unwrap_or_default();
        TexOptions {
            w: self.width,
            h: self.height,
            k: self.k,
            iw,
            ih,
            mode: TexMode::Realtime,
            repair: self.repair,
            cadence: false,
            cubic: match self.render {
                RenderSampling::Model => None,
                RenderSampling::Bilinear => Some(false),
                RenderSampling::Bicubic => Some(true),
            },
            keep_static: self.static_protection.is_some(),
            static_frames: s.frames,
            static_tolerance: s.tolerance,
            fast_render: self.fast_render,
            linear_t: self.linear_t,
            protection_mask: self.protection_mask,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SessionReport {
    pub width: u32,
    pub height: u32,
    pub internal_width: u32,
    pub internal_height: u32,
    pub k: u32,
    pub repair: u32,
    pub render: &'static str,
    pub fast_render: bool,
    pub linear_t: bool,
    pub static_protection: Option<StaticProtection>,
    pub protection_mask: bool,
    pub inputs: usize,
    pub outputs: usize,
    pub pair_steps: usize,
    pub mid_steps: usize,
}

pub struct Session {
    gpu: Rc<Gpu>,
    ext: SessionExt,
    opts: SessionOptions,
    internal: (u32, u32),
    /// The slots of the last pair (own slots only; `pair_views` pairs are the caller's to remember).
    pair: Cell<Option<(usize, usize)>>,
    /// linear_t: the t = 0.5 maps of the current pair are computed.
    half_done: Cell<bool>,
}

impl Session {
    pub fn new(engine: &Engine, model: &Model, opts: &SessionOptions) -> Result<Session> {
        opts.validate()?;
        let gpu = engine.shared();
        let (iw, ih) = opts.internal_size();
        let net = opts.net_opts(model);
        let ext = SessionExt::new(
            &gpu,
            model.blob(),
            model.ops(),
            opts.width,
            opts.height,
            opts.k,
            iw,
            ih,
            opts.tune.as_ref(),
            &net,
        );
        Ok(Session {
            gpu,
            ext,
            opts: opts.clone(),
            internal: (iw, ih),
            pair: Cell::new(None),
            half_done: Cell::new(false),
        })
    }

    pub fn options(&self) -> &SessionOptions {
        &self.opts
    }

    pub fn internal_size(&self) -> (u32, u32) {
        self.internal
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The plans underneath (the cadence and the generator use them).
    pub fn ext(&self) -> &SessionExt {
        &self.ext
    }

    /// Adds an input slot (an rgba8 texture the host writes into); returns its index.
    pub fn add_input(&mut self) -> usize {
        self.ext.add_input(&self.gpu)
    }

    /// A texture the host owns (its own, or one imported with `interop`) as an input slot: the frame size, rgba8 or
    /// bgra8, sampleable. Returns its index.
    pub fn add_input_texture(&mut self, tex: wgpu::Texture) -> Result<usize> {
        let (w, h) = (self.opts.width, self.opts.height);
        anyhow::ensure!(
            tex.width() == w && tex.height() == h,
            "input texture is {}x{}, the session takes {w}x{h}",
            tex.width(),
            tex.height()
        );
        anyhow::ensure!(
            matches!(tex.format(), wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm),
            "input texture format {:?}: rgba8unorm or bgra8unorm",
            tex.format()
        );
        anyhow::ensure!(tex.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING), "input texture is not sampleable");
        let v = tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.ext.ins.push((tex, v));
        Ok(self.ext.ins.len() - 1)
    }

    /// Adds an output slot (an rgba8 storage texture the host presents from); returns its index.
    pub fn add_output(&mut self) -> usize {
        self.ext.add_output(&self.gpu)
    }

    pub fn inputs(&self) -> usize {
        self.ext.ins.len()
    }

    pub fn outputs(&self) -> usize {
        self.ext.outs.len()
    }

    pub fn input_texture(&self, slot: usize) -> &wgpu::Texture {
        &self.ext.ins[slot].0
    }

    pub fn input_view(&self, slot: usize) -> &wgpu::TextureView {
        &self.ext.ins[slot].1
    }

    pub fn output_texture(&self, slot: usize) -> &wgpu::Texture {
        &self.ext.outs[slot].0
    }

    pub fn output_view(&self, slot: usize) -> &wgpu::TextureView {
        &self.ext.outs[slot].1
    }

    fn check_in(&self, slot: usize) -> Result<()> {
        anyhow::ensure!(slot < self.ext.ins.len(), "bad input slot {slot} (the session has {})", self.ext.ins.len());
        Ok(())
    }

    fn check_out(&self, slot: usize) -> Result<()> {
        anyhow::ensure!(slot < self.ext.outs.len(), "bad output slot {slot} (the session has {})", self.ext.outs.len());
        Ok(())
    }

    /// The protection mask (`protection_mask`): one byte per pixel, width x height, 0 interpolates, anything else is
    /// taken from the source frames. Rewrite it when the overlay changes; it applies from the next pair.
    pub fn write_mask(&self, mask: &[u8]) -> Result<()> {
        anyhow::ensure!(self.opts.protection_mask, "the session was built without protection_mask");
        self.ext.write_mask(&self.gpu, mask)
    }

    /// CPU hosts: a frame (RGBA8, width x height x 4 bytes) into an input slot.
    pub fn write_rgba(&self, slot: usize, rgba: &[u8]) -> Result<()> {
        self.check_in(slot)?;
        let (w, h) = (self.opts.width, self.opts.height);
        anyhow::ensure!(
            rgba.len() == (w * h * 4) as usize,
            "frame is {} bytes, {}x{}x4 = {} expected",
            rgba.len(),
            w,
            h,
            w * h * 4
        );
        self.gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.ext.ins[slot].0,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        Ok(())
    }

    /// CPU hosts: an output slot back as RGBA8 bytes (native: `block`; browser: `finish().await`).
    pub fn read_rgba(&self, slot: usize) -> Result<Readback<Vec<u8>>> {
        self.check_out(slot)?;
        let (w, h) = (self.opts.width, self.opts.height);
        let bpr = (w * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let stage = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fh_read_rgba"),
            size: (bpr * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.ext.outs[slot].0,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &stage,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.gpu.queue.submit(Some(enc.finish()));
        Ok(Readback::new(
            stage,
            Box::new(move |d: &[u8]| {
                let row = (w * 4) as usize;
                let mut v = Vec::with_capacity(row * h as usize);
                for y in 0..h as usize {
                    let o = y * bpr as usize;
                    v.extend_from_slice(&d[o..o + row]);
                }
                v
            }),
            None,
        ))
    }

    /// The motion of the interval (`a`, `b`): the expensive, t-independent half of the network.
    pub fn pair(&self, a: usize, b: usize) -> Result<()> {
        self.check_in(a)?;
        self.check_in(b)?;
        self.pair_views((a, &self.ext.ins[a].1), (b, &self.ext.ins[b].1));
        self.pair.set(Some((a, b)));
        Ok(())
    }

    /// The pair on two views of the host's own textures, keyed by the host's slot ids (a view behind an id must never
    /// change: the steps are cached per id).
    pub fn pair_views(&self, a: (usize, &wgpu::TextureView), b: (usize, &wgpu::TextureView)) {
        self.gpu.replay(&self.ext.pair_steps_views(&self.gpu, a, b));
        self.half_done.set(false);
    }

    /// Phase `t` of the last `pair` into output slot `out`: the cheap, t-dependent half plus the render.
    pub fn mid(&self, t: f32, out: usize) -> Result<()> {
        self.check_out(out)?;
        let (a, b) = self.pair.get().context("pair two input slots first")?;
        self.mid_views(t, (a, &self.ext.ins[a].1), (b, &self.ext.ins[b].1), (out, &self.ext.outs[out].1));
        Ok(())
    }

    /// Phase `t` of the pair (`a`, `b`) last run with `pair_views`, into the output view `out`.
    pub fn mid_views(
        &self,
        t: f32,
        a: (usize, &wgpu::TextureView),
        b: (usize, &wgpu::TextureView),
        out: (usize, &wgpu::TextureView),
    ) {
        if !self.ext.is_fast() {
            self.ext.mid(&self.gpu, t, out.0, out.1);
        } else if self.opts.linear_t {
            // the network's t-dependent stage once per pair; each phase is then only a render
            if !self.half_done.get() {
                self.ext.mid_maps_half(&self.gpu);
                self.half_done.set(true);
            }
            self.gpu.replay(&[self.ext.render_tex_step(&self.gpu, a, b, out, Some(t))]);
        } else {
            self.ext.mid_fast(&self.gpu, t, a, b, out);
        }
    }

    /// `pair` and `mid` in one call.
    pub fn generate(&self, a: usize, b: usize, t: f32, out: usize) -> Result<()> {
        self.pair(a, b)?;
        self.mid(t, out)
    }

    fn pair_steps(&self, a: usize, b: usize) -> Vec<Step> {
        self.ext.pair_steps_views(&self.gpu, (a, &self.ext.ins[a].1), (b, &self.ext.ins[b].1))
    }

    fn mid_steps(&self, t: f32, out: usize) -> Result<Vec<Step>> {
        let (a, b) = self.pair.get().context("pair two input slots first")?;
        let (av, bv, ov) = ((a, &self.ext.ins[a].1), (b, &self.ext.ins[b].1), (out, &self.ext.outs[out].1));
        self.ext.set_t(&self.gpu, t);
        Ok(if self.ext.is_fast() && self.opts.linear_t {
            vec![self.ext.render_tex_step(&self.gpu, av, bv, ov, Some(t))]
        } else if self.ext.is_fast() {
            self.ext.mid_steps_fast(&self.gpu, av, bv, ov)
        } else {
            self.ext.mid_steps(&self.gpu, out, ov.1)
        })
    }

    /// GPU time of the pair stage of (`a`, `b`) in ms (timestamp queries; None without them).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn pair_timed(&self, a: usize, b: usize) -> Result<Option<f64>> {
        self.check_in(a)?;
        self.check_in(b)?;
        Ok(self.gpu.timestamps.then(|| self.gpu.gpu_time_ms_n(&self.pair_steps(a, b), 3)))
    }

    /// GPU time of one more generated frame of the last pair in ms (timestamp queries; None without them).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn mid_timed(&self, t: f32, out: usize) -> Result<Option<f64>> {
        self.check_out(out)?;
        let steps = self.mid_steps(t, out)?;
        Ok(self.gpu.timestamps.then(|| self.gpu.gpu_time_ms_n(&steps, 5)))
    }

    /// The pair stage of (`a`, `b`) submitted with timestamps; `TimedRun::finish` awaits its GPU milliseconds, which a
    /// browser cannot block for. None without timestamp queries.
    pub fn pair_timed_begin(&self, a: usize, b: usize) -> Result<Option<TimedRun>> {
        self.check_in(a)?;
        self.check_in(b)?;
        Ok(self.gpu.begin_timed(&self.pair_steps(a, b)))
    }

    /// One more generated frame of the last pair, submitted with timestamps (see `pair_timed_begin`).
    pub fn mid_timed_begin(&self, t: f32, out: usize) -> Result<Option<TimedRun>> {
        self.check_out(out)?;
        let steps = self.mid_steps(t, out)?;
        Ok(self.gpu.begin_timed(&steps))
    }

    /// Compiles every kernel the session runs, before the first real frame: one pair and one mid on slots added for
    /// the purpose (they stay as slots 0 and 1 / output 0 if the session had none). Browsers compile pipelines on
    /// first use (Firefox serially, 0.1 to 2 s each), so a host calls this and awaits the queue before playback.
    pub fn warm_up(&mut self) -> Result<()> {
        while self.ext.ins.len() < 2 {
            self.add_input();
        }
        if self.ext.outs.is_empty() {
            self.add_output();
        }
        self.pair(0, 1)?;
        self.mid(0.5, 0)
    }

    pub fn report(&self) -> SessionReport {
        SessionReport {
            width: self.opts.width,
            height: self.opts.height,
            internal_width: self.internal.0,
            internal_height: self.internal.1,
            k: self.opts.k,
            repair: self.ext.net.repair,
            render: if self.ext.net.cubic { "bicubic" } else { "bilinear" },
            fast_render: self.ext.is_fast(),
            linear_t: self.ext.is_fast() && self.opts.linear_t,
            static_protection: self.opts.static_protection,
            protection_mask: self.opts.protection_mask,
            inputs: self.ext.ins.len(),
            outputs: self.ext.outs.len(),
            pair_steps: self.ext.plan_pair.len(),
            mid_steps: self.ext.plan_mid.len(),
        }
    }
}
