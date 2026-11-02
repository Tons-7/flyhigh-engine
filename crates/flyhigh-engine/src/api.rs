//! Library contract for hosts ("plug the runtime in and frame generation works"): an `Engine` holds the GPU and the
//! model once; a `Generator` takes source frames in display order and returns the frames to display, in order, with
//! their phase. Two modes:
//! - `Mode::Realtime`: no lookahead. `push(f_n)` returns the in-betweens of (f_{n-1}, f_n) followed by f_n itself, so
//!   the host presents them at the source interval / factor. Games, capture.
//! - `Mode::Video`: one source frame of lookahead. `push(f_n)` returns the outputs of the interval [f_{n-2}, f_{n-1}):
//!   the real slot and the in-betweens. With `cadence` on, elements animated on 2s over a pan on 1s (anime) are
//!   re-timed onto their smooth trajectory using the neighbouring pairs (every displayed frame is then synthesised,
//!   see `gmhs.rs` / vault "Cadence smoothing"). Video players, the browser extension on videos.
//!
//! Frames are CPU RGBA8 here (`w * h * 4` bytes) on top of the texture-path generator (`gen_tex.rs`), which GPU hosts
//! (the browser, the desktop player) use directly without any readback.
use crate::gen_tex::{GenTex, TexMode, TexOptions};
pub use crate::gmhs::PairStats;
pub use crate::{default_internal, EngineOptions, DEFAULT_REPAIR, DEFAULT_STATIC_FRAMES, DEFAULT_STATIC_TOLERANCE};
use crate::{map_read, Blob, Gpu, Model};
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::VecDeque;
use std::path::Path;
use std::rc::Rc;

/// The GPU and the model in one, for the CPU path. GPU hosts use `crate::Engine` and `crate::Model` separately;
/// `from_core` bundles those for a `Generator`.
pub struct Engine {
    pub gpu: Rc<Gpu>,
    pub blob: Rc<Blob>,
    pub ops: Rc<Vec<Value>>,
}
#[derive(Clone, Debug)]
pub struct EngineInfo {
    pub gpu: String,
    pub tensor_cores: bool,
    pub timestamps: bool,
    pub graph_ops: usize,
    pub model: String,
    pub train_step: i64,
}

impl Engine {
    /// Native: model directory with `flyhighfg.bin`, `flyhighfg.json`, `graph.json`; default `EngineOptions`.
    pub fn load(model_dir: &Path) -> Result<Engine> {
        Engine::load_with(model_dir, &EngineOptions::default())
    }
    pub fn load_with(model_dir: &Path, opts: &EngineOptions) -> Result<Engine> {
        let gpu = Gpu::new_with(opts)?;
        let blob = Blob::load(model_dir)?;
        let graph: Value =
            serde_json::from_slice(&std::fs::read(model_dir.join("graph.json")).context("read graph.json")?)?;
        Engine::from_parts(gpu, blob, &graph)
    }
    pub fn from_parts(gpu: Gpu, blob: Blob, graph: &Value) -> Result<Engine> {
        let ops: Vec<Value> = graph["ops"].as_array().context("graph.json: no ops")?.clone();
        Ok(Engine { gpu: Rc::new(gpu), blob: Rc::new(blob), ops: Rc::new(ops) })
    }
    /// The same device and model as a `crate::Engine` and a `crate::Model` (shared, not copied).
    pub fn from_core(engine: &crate::Engine, model: &Model) -> Engine {
        Engine { gpu: engine.shared(), blob: model.shared_blob(), ops: model.shared_ops() }
    }
    pub fn info(&self) -> EngineInfo {
        EngineInfo {
            gpu: self.gpu.name.clone(),
            tensor_cores: self.gpu.coop,
            timestamps: self.gpu.timestamps,
            graph_ops: self.ops.len(),
            model: self.blob.man["meta"]["model"].as_str().unwrap_or("?").to_string(),
            train_step: self.blob.man["meta"]["train_step"].as_i64().unwrap_or(-1),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Realtime,
    Video,
}

/// Generator options. `Options::new(w, h)` gives the defaults (see docs/options.md); change fields as needed.
#[derive(Clone, Debug)]
pub struct Options {
    pub width: u32,
    pub height: u32,
    /// fine-flow iterations, 1..3 (default 1 = fastest; 2 = a little better on hard motion)
    pub k: u32,
    /// network resolution ("rung"); None = the frame size. Default: `default_internal` (480-line rung above 960 px).
    /// Keep it <= ~540p-equivalent: the model is scale-specific
    pub internal: Option<(u32, u32)>,
    /// default Realtime
    pub mode: Mode,
    /// periodic-structure repair: 0 off, 1 safe (default), 2 aggressive (drawn content)
    pub repair: u32,
    /// Video mode only: re-time elements animated on 2s (anime) onto a smooth trajectory (default off)
    pub cadence: bool,
    /// render source sampler: None = the model's (manifest), Some(true) bicubic, Some(false) bilinear
    pub cubic: Option<bool>,
    /// static protection: pixels equal in both source frames (HUD, burnt-in subtitles, logos) are kept as they are
    /// instead of being warped with the motion behind them (default off)
    pub keep_static: bool,
    /// static protection: unchanged source frames before a pixel is protected (default 8)
    pub static_frames: u32,
    /// static protection: largest per-channel change, 0..255 levels, still counted as unchanged (default 4)
    pub static_tolerance: u32,
    /// fast render: sample the source frames with hardware bilinear filtering instead of the model's bicubic render
    /// (same picture, much cheaper per generated frame); realtime mode
    pub fast_render: bool,
    /// with fast_render: the network's t-dependent stage runs once per source pair (t = 0.5) and the other phases scale
    /// its motion linearly; each further generated frame then costs only the render (3x / 4x and up)
    pub linear_t: bool,
    /// persisted autotune table for this frame size ("" = default kernel routing)
    pub tune: String,
}
impl Options {
    pub fn new(width: u32, height: u32) -> Options {
        let internal = if width.max(height) > 960 { Some(default_internal(width, height)) } else { None };
        Options {
            width,
            height,
            k: 1,
            internal,
            mode: Mode::Realtime,
            repair: DEFAULT_REPAIR,
            cadence: false,
            cubic: None,
            keep_static: false,
            static_frames: crate::DEFAULT_STATIC_FRAMES,
            static_tolerance: crate::DEFAULT_STATIC_TOLERANCE,
            fast_render: false,
            linear_t: false,
            tune: String::new(),
        }
    }
    pub fn tex_options(&self) -> TexOptions {
        let (iw, ih) = self.internal.unwrap_or((self.width, self.height));
        TexOptions {
            w: self.width,
            h: self.height,
            k: self.k,
            iw,
            ih,
            mode: if self.mode == Mode::Video { TexMode::Video } else { TexMode::Realtime },
            repair: self.repair,
            cadence: self.cadence,
            cubic: self.cubic,
            keep_static: self.keep_static,
            static_frames: self.static_frames,
            static_tolerance: self.static_tolerance,
            fast_render: self.fast_render,
            linear_t: self.linear_t,
            protection_mask: false,
        }
    }
}
/// the 480-line rung for a frame size (aspect kept, multiples of 8), regardless of the 960 px threshold
pub fn rung_for(w: u32, h: u32) -> (u32, u32) {
    let s = (480.0 / h as f64).min(1.0);
    let rw = ((w as f64 * s / 8.0).round() as u32 * 8).max(64);
    let rh = ((h as f64 * s / 8.0).round() as u32 * 8).max(64);
    (rw, rh)
}

/// One frame to display: RGBA8 bytes, its phase inside the source interval it belongs to (0 = the source frame's own
/// time), and whether it is a source frame (`real`) or synthesised.
pub struct OutFrame {
    pub rgba: Vec<u8>,
    pub phase: f32,
    pub real: bool,
}

const IN_RING: usize = 4;

pub struct Generator {
    pub opts: Options,
    gen: GenTex,
    next_slot: usize,
    frames: VecDeque<Vec<u8>>, // last source frames (bytes), newest last, <= 3
    out: usize,                // the generator's output slot used for every synthesised frame
    stage: wgpu::Buffer,       // readback staging, rows padded to `bpr`
    bpr: u32,                  // bytes per row of the staging copy (texture -> buffer copies need a 256-byte multiple)
    pub last_pair_ms: f64,
    pub last_mid_ms: f64,
}

impl Generator {
    pub fn new(engine: &Engine, opts: Options) -> Result<Generator> {
        anyhow::ensure!(opts.width >= 64 && opts.height >= 64, "frame size too small");
        let gpu = &engine.gpu;
        let tune = crate::tune_table_from(&opts.tune);
        let mut gen = GenTex::new(gpu, &engine.blob, &engine.ops, opts.tex_options(), tune.as_ref())?;
        for _ in 0..IN_RING {
            gen.add_input(gpu);
        }
        let out = gen.add_output(gpu);
        let bpr = (opts.width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let stage = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fh_api_stage"),
            size: (bpr * opts.height) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Generator {
            opts,
            gen,
            next_slot: 0,
            frames: VecDeque::new(),
            out,
            stage,
            bpr,
            last_pair_ms: 0.0,
            last_mid_ms: 0.0,
        })
    }
    pub fn mode(&self) -> Mode {
        self.opts.mode
    }
    /// pushes until the first push that returns frames (Realtime 1: the source frame itself; Video 3: the interval
    /// [f1, f2) needs f3 as lookahead)
    pub fn warmup(&self) -> usize {
        if self.opts.mode == Mode::Video {
            3
        } else {
            1
        }
    }
    pub fn stats(&self) -> [PairStats; 3] {
        self.gen.stats()
    }
    /// The cadence decision for phase `t` of the current interval (diagnostics).
    pub fn retime(&self, t: f32) -> Option<crate::gen_tex::Retime> {
        self.gen.retime(t)
    }

    fn read_out(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        let (w, h) = (self.opts.width, self.opts.height);
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.gen.outs[self.out].0,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.stage,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.bpr),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        gpu.queue.submit(Some(enc.finish()));
        let fut = map_read(&self.stage);
        gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        pollster::block_on(fut).map_err(|e| anyhow::anyhow!("map_async: {e:?}"))?;
        let v = {
            let d = self.stage.slice(..).get_mapped_range().map_err(|e| anyhow::anyhow!("mapped range: {e:?}"))?;
            let row = (w * 4) as usize;
            let mut v = Vec::with_capacity(row * h as usize);
            for y in 0..h as usize {
                let o = y * self.bpr as usize;
                v.extend_from_slice(&d[o..o + row]);
            }
            v
        };
        self.stage.unmap();
        Ok(v)
    }
    fn synth(&mut self, gpu: &Gpu, t: f32) -> Result<Option<Vec<u8>>> {
        let t0 = std::time::Instant::now();
        if !self.gen.generate(gpu, t, self.out)? {
            return Ok(None);
        }
        let v = self.read_out(gpu)?;
        self.last_mid_ms = t0.elapsed().as_secs_f64() * 1000.0;
        Ok(Some(v))
    }

    /// Push the next source frame; returns the frames to display, in order. `factor` = output frames per source
    /// frame (1 = none generated). Realtime: in-betweens of (prev, this) then this frame. Video: outputs of the
    /// interval that starts two frames back (one frame of lookahead), nothing until the warm-up is over.
    pub fn push(&mut self, engine: &Engine, rgba: &[u8], factor: u32) -> Result<Vec<OutFrame>> {
        anyhow::ensure!(rgba.len() == (self.opts.width * self.opts.height * 4) as usize, "frame size mismatch");
        let gpu = &engine.gpu;
        let factor = factor.max(1);
        let (w, h) = (self.opts.width, self.opts.height);
        let slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % IN_RING;
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.gen.ins[slot].0,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let t0 = std::time::Instant::now();
        let info = self.gen.push_slot(gpu, slot)?;
        self.gen.resolve_stats(gpu)?;
        self.last_pair_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.frames.push_back(rgba.to_vec());
        if self.frames.len() > 3 {
            self.frames.pop_front();
        }
        let mut out = Vec::new();
        match self.opts.mode {
            Mode::Realtime => {
                if info.ready {
                    for i in 1..factor {
                        let t = i as f32 / factor as f32;
                        if let Some(v) = self.synth(gpu, t)? {
                            out.push(OutFrame { rgba: v, phase: t, real: false });
                        }
                    }
                }
                out.push(OutFrame { rgba: rgba.to_vec(), phase: 0.0, real: true });
            }
            Mode::Video => {
                if !info.ready {
                    return Ok(out);
                }
                match self.synth(gpu, 0.0)? {
                    Some(v) => out.push(OutFrame { rgba: v, phase: 0.0, real: false }),
                    None => out.push(OutFrame { rgba: self.frames[0].clone(), phase: 0.0, real: true }),
                }
                for i in 1..factor {
                    let t = i as f32 / factor as f32;
                    if let Some(v) = self.synth(gpu, t)? {
                        out.push(OutFrame { rgba: v, phase: t, real: false });
                    }
                }
            }
        }
        Ok(out)
    }
}
