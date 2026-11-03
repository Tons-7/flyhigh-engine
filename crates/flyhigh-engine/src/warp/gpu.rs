//! The camera warp on the GPU: each source frame's global image shift against the previous one, the screen-locked mask
//! (warp_lock, warp_dilate) and the turned frame (warp_render). The shift is searched on a pyramid of cell grids
//! (warp_cells): every candidate on the coarsest (warp_shift), then a few cells around the result on each finer one
//! (warp_refine), the minimum of each taken by warp_best. It reaches the calibrator through a 16-byte readback a frame
//! or so later; nothing here waits on the GPU.
use super::calib::{CalibOptions, Calibration, InputCalibrator};
use crate::gpu::readback::Readback;
use crate::Gpu;
use bytemuck::{Pod, Zeroable};
use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

/// The pyramid, coarse to fine: cells per frame (each spanning the same angle of the view: 3.3, 0.83 and 0.21 degrees
/// across at the default field of view), the search either way in cells, and the stride of the cells compared. The
/// coarsest level searches +-40 degrees of yaw and +-20 of pitch between two frames: 2400 degrees a second at 60 fps,
/// beyond the fastest flick.
const LEVELS: [Level; 3] = [
    Level { grid: (32, 18), range: (12, 6), stride: 1 },
    Level { grid: (128, 72), range: (6, 6), stride: 1 },
    Level { grid: (512, 288), range: (6, 6), stride: 2 },
];
/// Readbacks in flight at once; beyond that a frame's shift goes unmeasured.
const IN_FLIGHT: usize = 4;
/// The mask's block, in pixels (warp_lock.wgsl).
const BLOCK: u32 = 8;
/// Bind-group ids of the warp's own two output textures, out of the way of the host's.
const OWN_KEY: usize = usize::MAX;

#[derive(Clone, Copy)]
struct Level {
    grid: (u32, u32),
    range: (i32, i32),
    stride: u32,
}

impl Level {
    fn candidates(&self) -> (u32, u32) {
        (2 * self.range.0 as u32 + 1, 2 * self.range.1 as u32 + 1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WarpOptions {
    /// The game's horizontal field of view: how a shift at the centre maps to a turn of the camera, and so how the
    /// rest of the frame moves with it. 106 is a 90 degree vertical view at 16:9, the default of most shooters.
    pub hfov_deg: f64,
    pub calib: CalibOptions,
    /// A block is screen-locked when standing still explains it better than the camera motion by this much (0..1 of
    /// the mean channel difference) and by this share of the moved difference.
    pub lock_margin: f32,
    pub lock_rel: f32,
}

impl Default for WarpOptions {
    fn default() -> Self {
        WarpOptions { hfov_deg: 106.0, calib: CalibOptions::default(), lock_margin: 6.0 / 255.0, lock_rel: 0.3 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GridP {
    gw: u32,
    gh: u32,
    _a: u32,
    _b: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CellsP {
    gw: u32,
    gh: u32,
    w: u32,
    h: u32,
    f: f32,
    ha: f32,
    he: f32,
    _a: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShiftP {
    gw: u32,
    gh: u32,
    rx: i32,
    ry: i32,
    ncx: u32,
    _a: u32,
    _b: u32,
    _c: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RefineP {
    gw: u32,
    gh: u32,
    rx: i32,
    ry: i32,
    ncx: u32,
    mgw: u32,
    mgh: u32,
    stride: u32,
    cw: f32,
    ch: f32,
    ha: f32,
    he: f32,
    w: u32,
    h: u32,
    _a: u32,
    _b: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BestP {
    fine: u32,
    _a: u32,
    rx: i32,
    ry: i32,
    ncx: u32,
    ncy: u32,
    cw: f32,
    ch: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LockP {
    w: u32,
    h: u32,
    gw: u32,
    gh: u32,
    margin: f32,
    rel: f32,
    f: f32,
    _b: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RenderP {
    w: u32,
    h: u32,
    gw: u32,
    gh: u32,
    sx: f32,
    sy: f32,
    f: f32,
    _a: f32,
}

type Shift = Option<(f64, f64)>;

/// One pyramid level's buffers: the cell grids of the last two frames (`CameraWarp::cur` is the newer one's), its
/// search result and the minimum taken from it.
struct LevelBufs {
    cells: [wgpu::Buffer; 2],
    p_cells: wgpu::Buffer,
    p_search: wgpu::Buffer,
    p_best: wgpu::Buffer,
    best: wgpu::Buffer,
    bg_cells: HashMap<(usize, usize), wgpu::BindGroup>,
    bg_search: [Option<wgpu::BindGroup>; 2],
    bg_best: Option<wgpu::BindGroup>,
}

pub struct CameraWarp {
    w: u32,
    h: u32,
    focal: f32,
    calib: InputCalibrator,
    sampler: wgpu::Sampler,
    levels: Vec<LevelBufs>,
    cur: usize,
    prev: Option<(usize, wgpu::TextureView)>,
    /// The search differences of the level being searched (levels run in order, so they share it).
    sad: wgpu::Buffer,
    zero: wgpu::Buffer,
    mask: wgpu::Buffer,
    grown: wgpu::Buffer,
    p_lock: wgpu::Buffer,
    p_grow: wgpu::Buffer,
    p_render: [wgpu::Buffer; 2],
    out: [(wgpu::Texture, wgpu::TextureView); 2],
    out_i: usize,
    stages: Vec<(wgpu::Buffer, Rc<Cell<bool>>)>,
    /// Frames in arrival order with their shift on its way; the calibrator takes them in that order.
    pending: VecDeque<(f64, Option<Readback<Shift>>)>,
    bg_lock: HashMap<(usize, usize), wgpu::BindGroup>,
    bg_grow: Option<wgpu::BindGroup>,
    bg_render: HashMap<(usize, usize, usize), wgpu::BindGroup>,
    last_shift: Shift,
}

fn uniform<T: Pod>(gpu: &Gpu, label: &str, v: &T) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(v),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

fn storage(gpu: &Gpu, label: &str, bytes: u64, extra: wgpu::BufferUsages) -> wgpu::Buffer {
    gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE | extra,
        mapped_at_creation: false,
    })
}

fn tex(view: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(view)
}

fn bind(gpu: &Gpu, pipe: &wgpu::ComputePipeline, label: &str, list: &[wgpu::BindingResource]) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> =
        list.iter().enumerate().map(|(i, r)| wgpu::BindGroupEntry { binding: i as u32, resource: r.clone() }).collect();
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pipe.get_bind_group_layout(0),
        entries: &entries,
    })
}

impl CameraWarp {
    pub fn new(gpu: &Gpu, w: u32, h: u32, opts: WarpOptions) -> CameraWarp {
        let (mgw, mgh) = (w.div_ceil(BLOCK), h.div_ceil(BLOCK));
        let ha = (opts.hfov_deg.clamp(10.0, 170.0) * 0.5).to_radians();
        let f = (0.5 * w as f64 / ha.tan()) as f32;
        let (ha, he) = (ha as f32, (0.5 * h as f32 / f).atan());
        let none = wgpu::BufferUsages::empty();
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fh_warp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let levels = LEVELS
            .iter()
            .enumerate()
            .map(|(i, l)| {
                let (gw, gh) = l.grid;
                let (ncx, ncy) = l.candidates();
                // a cell in pixels at the centre: the angle it spans times the focal length
                let (cw, ch) = (f * 2.0 * ha / gw as f32, f * 2.0 * he / gh as f32);
                let (rx, ry) = l.range;
                let p_search = if i == 0 {
                    uniform(gpu, "fh_warp_shift_p", &ShiftP { gw, gh, rx, ry, ncx, _a: 0, _b: 0, _c: 0 })
                } else {
                    let stride = l.stride;
                    let p = RefineP { gw, gh, rx, ry, ncx, mgw, mgh, stride, cw, ch, ha, he, w, h, _a: 0, _b: 0 };
                    uniform(gpu, "fh_warp_refine_p", &p)
                };
                let fine = (i > 0) as u32;
                LevelBufs {
                    cells: [0, 1].map(|_| storage(gpu, "fh_warp_cells", (gw * gh * 4) as u64, none)),
                    p_cells: uniform(gpu, "fh_warp_cells_p", &CellsP { gw, gh, w, h, f, ha, he, _a: 0.0 }),
                    p_search,
                    p_best: uniform(gpu, "fh_warp_best_p", &BestP { fine, _a: 0, rx, ry, ncx, ncy, cw, ch }),
                    best: storage(gpu, "fh_warp_best", 16, wgpu::BufferUsages::COPY_SRC),
                    bg_cells: HashMap::new(),
                    bg_search: [None, None],
                    bg_best: None,
                }
            })
            .collect();
        let most = LEVELS.iter().map(|l| l.candidates().0 * l.candidates().1).max().unwrap_or(1);
        let out = [0, 1].map(|_| {
            let t = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("fh_warp_out"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let v = t.create_view(&Default::default());
            (t, v)
        });
        let stages = (0..IN_FLIGHT)
            .map(|_| {
                let b = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("fh_warp_stage"),
                    size: 16,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                (b, Rc::new(Cell::new(false)))
            })
            .collect();
        let lock_p = LockP { w, h, gw: mgw, gh: mgh, margin: opts.lock_margin, rel: opts.lock_rel, f, _b: 0.0 };
        let render_p = RenderP { w, h, gw: mgw, gh: mgh, sx: 0.0, sy: 0.0, f: 0.0, _a: 0.0 };
        let mask_bytes = (mgw * mgh * 4) as u64;
        CameraWarp {
            w,
            h,
            focal: f,
            calib: InputCalibrator::new(opts.calib),
            sampler,
            levels,
            cur: 0,
            prev: None,
            sad: storage(gpu, "fh_warp_sad", (most * 2 * 4) as u64, none),
            zero: storage(gpu, "fh_warp_zero", 16, none),
            // zeroed at creation: nothing is screen-locked until a pair with motion says so
            mask: storage(gpu, "fh_warp_mask", mask_bytes, wgpu::BufferUsages::COPY_SRC),
            grown: storage(gpu, "fh_warp_grown", mask_bytes, none),
            p_lock: uniform(gpu, "fh_warp_lock_p", &lock_p),
            p_grow: uniform(gpu, "fh_warp_grow_p", &GridP { gw: mgw, gh: mgh, _a: 0, _b: 0 }),
            p_render: [0, 1].map(|_| uniform(gpu, "fh_warp_render_p", &render_p)),
            out,
            out_i: 0,
            stages,
            pending: VecDeque::new(),
            bg_lock: HashMap::new(),
            bg_grow: None,
            bg_render: HashMap::new(),
            last_shift: None,
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.w, self.h)
    }

    pub fn report(&self) -> Calibration {
        self.calib.report()
    }

    /// The last measured shift between consecutive source frames (pixels), None when it could not be measured.
    pub fn last_shift(&self) -> Shift {
        self.last_shift
    }

    /// Mouse counts at host time `t` (ms), as the game receives them.
    pub fn mouse(&mut self, t: f64, dx: f64, dy: f64) {
        self.calib.mouse(t, dx, dy);
    }

    /// Whether the host has the mouse captured (a game camera) or released (a cursor): released turns the warp off at
    /// once.
    pub fn set_captured(&mut self, captured: bool) {
        self.calib.set_captured(captured);
    }

    /// The next frame starts afresh: no shift is measured against the one before (a seek, a gap in the source).
    pub fn break_pair(&mut self) {
        self.prev = None;
    }

    /// The source textures were recreated: the cached bind groups refer to the old ones.
    pub fn forget_textures(&mut self) {
        for l in &mut self.levels {
            l.bg_cells.clear();
        }
        self.bg_lock.clear();
        self.bg_render.clear();
        self.prev = None;
    }

    /// Hands every finished shift to the calibrator, oldest first; one still in flight holds the later ones back.
    /// `frame` and `render` do this themselves; a host calls it to see the calibration move without either.
    pub fn collect(&mut self, gpu: &Gpu) {
        #[cfg(not(target_arch = "wasm32"))]
        gpu.device.poll(wgpu::PollType::Poll).ok();
        #[cfg(target_arch = "wasm32")]
        let _ = gpu;
        while let Some((t, rb)) = self.pending.pop_front() {
            let shift = match rb {
                None => None,
                Some(rb) => match rb.try_finish() {
                    Ok(r) => r.ok().flatten(),
                    Err(rb) => {
                        self.pending.push_front((t, Some(rb)));
                        return;
                    }
                },
            };
            self.last_shift = shift;
            self.calib.frame(t, shift);
        }
    }

    /// A source frame arrived at host time `t` (ms): its shift against the previous source frame is measured, and
    /// the screen-locked mask updated when that pair moved. `src.0` is a stable id of the texture (bind groups are
    /// cached per id).
    pub fn frame(&mut self, gpu: &Gpu, src: (usize, &wgpu::TextureView), t: f64) {
        self.collect(gpu);
        self.cur ^= 1;
        let cur = self.cur;
        let p_cells = gpu.pipeline("warp_cells");
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fh_warp") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&p_cells);
            for (l, lv) in self.levels.iter_mut().zip(&LEVELS) {
                let bg = l.bg_cells.entry((src.0, cur)).or_insert_with(|| {
                    let r = [
                        tex(src.1),
                        wgpu::BindingResource::Sampler(&self.sampler),
                        l.cells[cur].as_entire_binding(),
                        l.p_cells.as_entire_binding(),
                    ];
                    bind(gpu, &p_cells, "fh_warp_cells", &r)
                });
                pass.set_bind_group(0, &*bg, &[]);
                pass.dispatch_workgroups(lv.grid.0.div_ceil(8), lv.grid.1.div_ceil(8), 1);
            }
        }
        let stage = self.stages.iter().find(|(_, busy)| !busy.get()).map(|(s, b)| (s.clone(), b.clone()));
        let measured = match (self.prev.clone(), stage) {
            (Some((pk, pv)), Some((stage, busy))) => {
                busy.set(true);
                self.measure(gpu, &mut enc, (pk, &pv), src);
                let last = &self.levels[self.levels.len() - 1].best;
                enc.copy_buffer_to_buffer(last, 0, &stage, 0, 16);
                Some((stage, busy))
            }
            _ => None,
        };
        gpu.queue.submit(Some(enc.finish()));
        let rb = measured.map(|(stage, busy)| {
            let decode = |d: &[u8]| {
                let b: &[f32] = bytemuck::cast_slice(&d[..16]);
                (b[2] > 0.5).then_some((b[0] as f64, b[1] as f64))
            };
            Readback::new(stage, Box::new(decode), Some(Box::new(move || busy.set(false))))
        });
        self.pending.push_back((t, rb));
        self.prev = Some((src.0, src.1.clone()));
    }

    // The pyramid search of the pair (prev, src), then the mask from its result.
    fn measure(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        prev: (usize, &wgpu::TextureView),
        src: (usize, &wgpu::TextureView),
    ) {
        let cur = self.cur;
        let (p_shift, p_refine, p_best) =
            (gpu.pipeline("warp_shift"), gpu.pipeline("warp_refine"), gpu.pipeline("warp_best"));
        let (p_lock, p_grow) = (gpu.pipeline("warp_lock"), gpu.pipeline("warp_dilate"));
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        for (i, lv) in LEVELS.iter().enumerate() {
            let (above, rest) = self.levels.split_at_mut(i);
            let l = &mut rest[0];
            let prior = above.last().map_or(&self.zero, |a| &a.best);
            let (a, b) = (&l.cells[cur ^ 1], &l.cells[cur]);
            let search = l.bg_search[cur].get_or_insert_with(|| {
                if i == 0 {
                    let r = [a, b, &self.sad, &l.p_search].map(|x| x.as_entire_binding());
                    bind(gpu, &p_shift, "fh_warp_shift", &r)
                } else {
                    let r = [a, b, &self.sad, &l.p_search, prior, &self.mask].map(|x| x.as_entire_binding());
                    bind(gpu, &p_refine, "fh_warp_refine", &r)
                }
            });
            let best = l.bg_best.get_or_insert_with(|| {
                let r = [&self.sad, &l.best, &l.p_best, prior].map(|x| x.as_entire_binding());
                bind(gpu, &p_best, "fh_warp_best", &r)
            });
            let (ncx, ncy) = lv.candidates();
            pass.set_pipeline(if i == 0 { &p_shift } else { &p_refine });
            pass.set_bind_group(0, &*search, &[]);
            pass.dispatch_workgroups(ncx * ncy, 1, 1);
            pass.set_pipeline(&p_best);
            pass.set_bind_group(0, &*best, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        let result = &self.levels[self.levels.len() - 1].best;
        let lock = self.bg_lock.entry((prev.0, src.0)).or_insert_with(|| {
            let r = [
                tex(prev.1),
                tex(src.1),
                wgpu::BindingResource::Sampler(&self.sampler),
                self.mask.as_entire_binding(),
                self.p_lock.as_entire_binding(),
                result.as_entire_binding(),
            ];
            bind(gpu, &p_lock, "fh_warp_lock", &r)
        });
        let grow = self.bg_grow.get_or_insert_with(|| {
            let r = [&self.mask, &self.grown, &self.p_grow].map(|x| x.as_entire_binding());
            bind(gpu, &p_grow, "fh_warp_dilate", &r)
        });
        let (mgw, mgh) = (self.w.div_ceil(BLOCK), self.h.div_ceil(BLOCK));
        pass.set_pipeline(&p_lock);
        pass.set_bind_group(0, &*lock, &[]);
        pass.dispatch_workgroups(mgw.div_ceil(8), mgh.div_ceil(8), 1);
        pass.set_pipeline(&p_grow);
        pass.set_bind_group(0, &*grow, &[]);
        pass.dispatch_workgroups(mgw.div_ceil(8), mgh.div_ceil(8), 1);
    }

    /// The shift `render` would apply now to a frame showing the moment of host time `t_frame`, None while off.
    pub fn shift(&self, t_frame: f64, now: f64) -> Shift {
        self.calib.shift(t_frame, now)
    }

    /// `src` (a frame showing the moment of host time `t_frame`, ms) turned by the camera motion the mouse made since,
    /// for presenting at `now`; None while the warp is off or the turn is under a quarter pixel (present `src` as it
    /// is). The texture is the warp's own (rgba8unorm) and stays valid until the render after next.
    pub fn render(
        &mut self,
        gpu: &Gpu,
        src: (usize, &wgpu::TextureView),
        t_frame: f64,
        now: f64,
    ) -> Option<&wgpu::Texture> {
        let shift = self.decide(gpu, t_frame, now)?;
        Some(self.render_shift(gpu, src, shift))
    }

    /// The shift to apply now (finished measurements collected first), None while off or under a quarter pixel.
    pub fn decide(&mut self, gpu: &Gpu, t_frame: f64, now: f64) -> Shift {
        self.collect(gpu);
        self.calib.shift(t_frame, now).filter(|s| s.0.hypot(s.1) >= 0.25)
    }

    /// `src` turned by a given image shift at the centre (pixels) into the warp's own texture.
    pub fn render_shift(&mut self, gpu: &Gpu, src: (usize, &wgpu::TextureView), shift: (f64, f64)) -> &wgpu::Texture {
        self.out_i ^= 1;
        let i = self.out_i;
        let own = self.out[i].1.clone();
        self.draw(gpu, src, (OWN_KEY - i, &own), shift);
        &self.out[i].0
    }

    /// `src` turned by `shift` into `dst`, an rgba8unorm storage texture of the warp's size (another frame than `src`).
    /// `dst.0` is a stable id like `src.0`, from the same space.
    pub fn render_into(
        &mut self,
        gpu: &Gpu,
        src: (usize, &wgpu::TextureView),
        dst: (usize, &wgpu::TextureView),
        shift: (f64, f64),
    ) {
        self.out_i ^= 1;
        self.draw(gpu, src, dst, shift);
    }

    fn draw(
        &mut self,
        gpu: &Gpu,
        src: (usize, &wgpu::TextureView),
        dst: (usize, &wgpu::TextureView),
        shift: (f64, f64),
    ) {
        let i = self.out_i;
        let (mgw, mgh) = (self.w.div_ceil(BLOCK), self.h.div_ceil(BLOCK));
        let (sx, sy) = (shift.0 as f32, shift.1 as f32);
        let p = RenderP { w: self.w, h: self.h, gw: mgw, gh: mgh, sx, sy, f: self.focal, _a: 0.0 };
        gpu.queue.write_buffer(&self.p_render[i], 0, bytemuck::bytes_of(&p));
        let pipe = gpu.pipeline("warp_render");
        let bg = self.bg_render.entry((src.0, dst.0, i)).or_insert_with(|| {
            let r = [
                tex(src.1),
                wgpu::BindingResource::Sampler(&self.sampler),
                self.grown.as_entire_binding(),
                tex(dst.1),
                self.p_render[i].as_entire_binding(),
            ];
            bind(gpu, &pipe, "fh_warp_render", &r)
        });
        let mut enc =
            gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fh_warp_render") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &*bg, &[]);
            pass.dispatch_workgroups(self.w.div_ceil(8), self.h.div_ceil(8), 1);
        }
        gpu.queue.submit(Some(enc.finish()));
    }

    /// The screen-locked mask, 1 per locked 8 x 8 block, row-major (a blocking readback, for tests and tools).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_mask(&self, gpu: &Gpu) -> Vec<f32> {
        let bytes = (self.w.div_ceil(BLOCK) * self.h.div_ceil(BLOCK) * 4) as u64;
        let stage = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        // the lock kernel's own mask, before growing: what was detected
        enc.copy_buffer_to_buffer(&self.mask, 0, &stage, 0, bytes);
        gpu.queue.submit(Some(enc.finish()));
        let rb = Readback::new(stage, Box::new(|d: &[u8]| bytemuck::cast_slice::<u8, f32>(d).to_vec()), None);
        rb.block(gpu).unwrap_or_default()
    }
}
