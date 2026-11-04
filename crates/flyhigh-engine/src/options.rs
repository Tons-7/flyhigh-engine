//! Engine-wide options (`EngineOptions`, `Precision`), per-network options (`NetOpts`) and their defaults; the option
//! declarations' reference (`OptionInfo`, `groups`) behind `#[derive(Options)]` (ARCHITECTURE.md).

use crate::Blob;
pub use flyhigh_options::Options;
use std::fmt;

/// One option as the reference shows it (`#[derive(Options)]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptionInfo {
    pub name: &'static str,
    /// The default, as text.
    pub default: String,
    /// The range or the choices; empty when the type says it all.
    pub values: &'static str,
    /// The field of the C struct.
    pub c: &'static str,
    pub doc: &'static str,
}

/// A layer's options: the group, its introduction and its options with the defaults filled in.
pub struct OptionGroup {
    pub name: &'static str,
    pub doc: &'static str,
    pub options: Vec<OptionInfo>,
}

/// Every group in layer order, with the defaults a host gets when it passes nothing (the session at 1920 x 1080).
pub fn groups() -> Vec<OptionGroup> {
    use crate::{pacer::PacerOptions, stream::StreamOptions, SessionOptions};
    vec![
        OptionGroup {
            name: EngineOptions::GROUP,
            doc: EngineOptions::GROUP_DOC,
            options: EngineOptions::default().describe(),
        },
        OptionGroup {
            name: SessionOptions::GROUP,
            doc: SessionOptions::GROUP_DOC,
            options: SessionOptions::new(1920, 1080).describe(),
        },
        OptionGroup {
            name: StreamOptions::GROUP,
            doc: StreamOptions::GROUP_DOC,
            options: StreamOptions::default().describe(),
        },
        OptionGroup {
            name: PacerOptions::GROUP,
            doc: PacerOptions::GROUP_DOC,
            options: PacerOptions::default().describe(),
        },
    ]
}

/// A default's `Debug` text as the reference shows it: `None` is off, `Some(x)` is x, booleans are on and off, a
/// struct loses its name and braces (`Some(SceneCut { threshold: 0.08 })` reads "threshold 0.08"), a whole number
/// drops its `.0`.
pub fn default_text(debug: &str) -> String {
    let s = debug.trim();
    match s {
        "None" | "false" => return "off".into(),
        "true" => return "on".into(),
        _ => {}
    }
    let s = s.strip_prefix("Some(").and_then(|r| r.strip_suffix(')')).unwrap_or(s);
    if let Some(whole) = s.strip_suffix(".0").filter(|n| n.parse::<i64>().is_ok()) {
        return whole.to_string();
    }
    match s.find(" { ") {
        Some(i) => s[i + 3..].trim_end_matches(" }").replace(": ", " "),
        None => s.to_string(),
    }
}

/// An option outside its range: which field and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptionsError {
    pub field: &'static str,
    pub reason: String,
}

impl fmt::Display for OptionsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "option {}: {}", self.field, self.reason)
    }
}

impl std::error::Error for OptionsError {}

/// Fixed when the engine (the device) is created; the sessions and streams on it inherit them. A host that passes
/// nothing gets the platform's best backend and adapter, the f16 kernels, the tensor-core kernels where the GPU exposes
/// them and an autotune at the first build of every frame size. In the browser only `precision`, `tensor_cores` and
/// `autotune` apply: the browser picks the backend and the adapter. `from_env` adds the developer overrides (`FH_*`,
/// docs/development.md) the CLI tools honour; library hosts pass the options explicitly.
#[derive(Clone, Debug, Options)]
pub struct EngineOptions {
    /// wgpu backend. Automatic picks the platform's best (Vulkan on Windows and Linux, Metal on macOS, WebGPU in the
    /// browser). Force DX12 when presenting through DirectComposition or sharing D3D12 resources; it needs the DXC DLLs
    /// beside the library (`dxc_path`, docs/platforms.md).
    #[option(
        default = "automatic",
        values = "automatic, Vulkan, DX12, Metal, OpenGL",
        c = "backend (0 automatic, 1 Vulkan, 2 DX12, 3 Metal, 4 OpenGL)"
    )]
    pub backend: Option<wgpu::Backends>,
    /// Use the cooperative-matrix kernels where the GPU and driver expose them (NVIDIA on Vulkan today). Off = the
    /// portable kernels, the same ones the browser runs, within a few percent on this network.
    #[option(values = "on, off", c = "tensor_cores (0, 1)")]
    pub tensor_cores: bool,
    /// When a session is built, time every applicable conv kernel per layer shape and keep the fastest: a few seconds per
    /// new frame size at build time (0.9 s at 1080p on an RTX 4060 Ti over Vulkan, 9.6 s over DX12), faster frames after.
    /// Off = the fixed kernel routing. Native only; the browser tunes explicitly.
    #[option(values = "on, off", c = "autotune (0, 1)")]
    pub autotune: bool,
    /// Which GPU. `low-power` for the integrated one, `#N` for an index in `list_adapters()`, `vvvv:dddd` for a PCI
    /// vendor and device in hex, any other text for a case-insensitive part of the adapter name (`4060`, `radeon`). An
    /// unknown selector fails with the list of adapters. Native only.
    #[option(
        default = "the high-performance GPU",
        values = "low-power, #N, vvvv:dddd, part of the name",
        c = "adapter (NULL = default)"
    )]
    pub adapter: Option<String>,
    /// Tensor precision. Automatic uses f16 where the GPU has shader-f16 and the f32 path elsewhere (GTX 10xx and other
    /// GPUs without f16 arithmetic, WebGPU without the feature), so the engine runs there instead of failing. f32 moves
    /// twice the bytes and costs a few percent on the fast render path (1080p pair 1.45 against 1.40 ms on an RTX 4060
    /// Ti); the pictures match within rounding (full-graph parity 84 dB against 72 dB with f16). Tensor cores need f16.
    #[option(default = "automatic", values = "automatic, f16, f32", c = "precision (0, 1, 2)")]
    pub precision: Precision,
    /// Where the autotune results persist, one file per GPU, driver, precision, kernel set and engine version. The first
    /// build on a GPU times the kernels (a few seconds; 20 to 30 s on a laptop GTX 1650), later starts reuse the choices
    /// (1080p plan build 1.6 s to 10 ms). None = in memory for this process only. Native only.
    #[option(default = "none (in memory)", values = "a directory", c = "tune_cache (NULL = none)")]
    pub tune_cache: Option<std::path::PathBuf>,
    /// Small blocks from the GPU memory allocator. At 1080p the engine's buffers take about 100 MB; with large blocks
    /// (off) the allocator keeps 512 MB reserved and never hands it back, with small blocks about 190 MB. Matters when a
    /// game fills the video memory: the driver starts moving memory and both stutter.
    #[option(values = "on, off", c = "compact_memory (0, 1)")]
    pub compact_memory: bool,
    /// Windows, DX12 only: the directory (or the file) of `dxcompiler.dll`, with `dxil.dll` beside it. By default the
    /// engine looks beside its own module (the executable, or flyhigh.dll), then wgpu looks on the DLL search path, and
    /// without the DLL the kernels go through FXC: the set compiles in minutes instead of seconds and has no shader-f16,
    /// so the f32 kernels run. `DeviceInfo::shader_compiler` says which one loaded.
    #[option(
        default = "beside the library",
        values = "a directory, or dxcompiler.dll itself",
        c = "dxc_path (NULL = default)"
    )]
    pub dxc_path: Option<std::path::PathBuf>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Precision {
    #[default]
    Auto,
    F16,
    F32,
}
impl Default for EngineOptions {
    fn default() -> Self {
        EngineOptions {
            backend: None,
            tensor_cores: true,
            autotune: true,
            adapter: None,
            precision: Precision::Auto,
            tune_cache: None,
            compact_memory: true,
            dxc_path: None,
        }
    }
}
impl EngineOptions {
    pub fn from_env() -> EngineOptions {
        let backend = match std::env::var("FH_BACKEND").ok().as_deref() {
            Some("vulkan") => Some(wgpu::Backends::VULKAN),
            Some("dx12") => Some(wgpu::Backends::DX12),
            Some("metal") => Some(wgpu::Backends::METAL),
            Some("gl") => Some(wgpu::Backends::GL),
            _ => None,
        };
        let mut o = EngineOptions { backend, ..EngineOptions::default() };
        if std::env::var_os("FH_NO_COOP").is_some() {
            o.tensor_cores = false;
        }
        if std::env::var_os("FH_NO_AUTOTUNE").is_some() {
            o.autotune = false;
        }
        o.adapter = std::env::var("FH_ADAPTER").ok().filter(|s| !s.is_empty());
        o.tune_cache = std::env::var_os("FH_TUNE_CACHE").map(std::path::PathBuf::from);
        o.dxc_path = std::env::var_os("FH_DXC_PATH").map(std::path::PathBuf::from);
        if std::env::var_os("FH_MEM_PERF").is_some() {
            o.compact_memory = false;
        }
        o.precision = match std::env::var("FH_PRECISION").ok().as_deref() {
            Some("f32") => Precision::F32,
            Some("f16") => Precision::F16,
            _ => Precision::Auto,
        };
        o
    }
}

/// Per-session network options (part of the static plans, so they are fixed when a session is built).
#[derive(Clone, Copy, Debug)]
pub struct NetOpts {
    /// periodic-structure repair of the pair flow (GMHS, gmhs.rs): 0 off, 1 safe, 2 aggressive (drawn content)
    pub repair: u32,
    /// keep the GMHS cost maps alive after the pair stage (cadence smoothing reads them; needs repair > 0)
    pub keep_maps: bool,
    /// bicubic source sampling in the render kernel (false = bilinear)
    pub cubic: bool,
    /// static protection: pixels that stayed the same for the last `STATIC_MIN_AGE` source pairs (a HUD, subtitles burnt
    /// into the picture, a logo) are taken from the sources instead of the network (keep_static.wgsl). The age lives in
    /// the session, so it counts the pairs that session ran (video mode rotates three sessions: every third pair).
    pub keep_static: bool,
    /// static protection: consecutive unchanged source pairs before a pixel is protected (1..255)
    pub static_frames: u32,
    /// static protection: largest per-channel change (0..255 levels) still counted as unchanged
    pub static_tolerance: u32,
    /// fast render (texture path): the in-between is sampled from the source
    /// textures with hardware bilinear filtering straight into the output texture (render_tex.wgsl); false = the exact
    /// bicubic render of the model. Honoured by `SessionExt` only.
    pub fast_render: bool,
    /// the host's protection mask (`SessionExt::write_mask`): the pixels it marks are taken from the sources like
    /// statically protected ones, with or without `keep_static`
    pub protection_mask: bool,
    /// decisive blend in the render: where the two warped frames disagree, one of them instead of their blend
    pub decisive: bool,
    /// still test in the render: a pixel that stayed (a HUD over a moving world) keeps the still pixels
    pub still: bool,
    /// nearest-source regions after the render: where the warped frames disagree over a region, the nearer frame alone
    pub near: bool,
}
/// The decisive blend: off by default (on screen its patches at a failed motion were no better than the blend's
/// fog); FH_BLEND=decisive turns it on for comparisons.
pub fn decisive_default() -> bool {
    std::env::var("FH_BLEND").is_ok_and(|v| v.trim() == "decisive")
}
/// Nearest-source regions for hosts (Session, Stream realtime); FH_NEAR=0 turns them off for comparisons. On a flick the
/// network cannot follow, the blend of two misplaced copies reads as fog: the nearer frame, warped, reads as a short
/// hold.
pub fn near_default() -> bool {
    std::env::var("FH_NEAR").map_or(true, |v| v.trim() != "0")
}
/// The still test for hosts (Session, Stream realtime); FH_STILL=0 turns it off for comparisons. The model's reference
/// render (`NetOpts::new`, the parity fixtures) has none.
pub fn still_default() -> bool {
    std::env::var("FH_STILL").map_or(true, |v| v.trim() != "0")
}
/// Default `static_tolerance`: below 4 levels a pixel counts as unchanged (lossless capture: exactly equal; video: codec noise).
pub const DEFAULT_STATIC_TOLERANCE: u32 = 4;
/// Default `static_frames`: about 130 ms of a 60 fps source, long enough that an object passing over a background
/// does not qualify.
pub const DEFAULT_STATIC_FRAMES: u32 = 8;
/// The library-wide default repair level (safe: natural video and games unchanged, fixes fences / window grids).
pub const DEFAULT_REPAIR: u32 = 1;
impl NetOpts {
    /// defaults for a model: repair 1, no kept maps, the checkpoint's render sampler
    pub fn new(blob: &Blob) -> NetOpts {
        NetOpts {
            repair: DEFAULT_REPAIR,
            keep_maps: false,
            cubic: blob.render_cubic,
            keep_static: false,
            static_frames: DEFAULT_STATIC_FRAMES,
            static_tolerance: DEFAULT_STATIC_TOLERANCE,
            fast_render: false,
            protection_mask: false,
            decisive: false,
            still: false,
            near: false,
        }
    }
    /// The render's mode word (render.wgsl / render_tex.wgsl): bit 0 decisive blend, bit 1 still test, bit 2
    /// nearest-source regions (`ops::render`, the exact render only).
    pub fn render_flags(&self) -> u32 {
        self.decisive as u32 | (self.still as u32) << 1 | (self.near as u32) << 2
    }
    /// Something protects pixels: the automatic detection, the host's mask or both (the age pass runs).
    pub fn protects(&self) -> bool {
        self.keep_static || self.protection_mask
    }
    /// The age at which a pixel is protected: `static_frames` with the automatic detection, else 2, which only the
    /// mask's 255 reaches while age 0 stays clear of the apply pass's two-pair ramp.
    pub fn protecting_age(&self) -> u32 {
        if self.keep_static {
            self.static_frames.clamp(1, 255)
        } else {
            2
        }
    }
}

/// The internal network size the runtime picks when the host gives none: the frame itself up to 960 px on the long
/// side, above that the 480-line rung (aspect kept, multiples of 8). The model is scale-specific; this is the size the
/// extension and the benchmarks use.
pub fn default_internal(w: u32, h: u32) -> (u32, u32) {
    if w.max(h) <= 960 {
        return (w, h);
    }
    let s = (480.0 / h as f64).min(1.0);
    let rw = ((w as f64 * s / 8.0).round() as u32 * 8).max(64);
    let rh = ((h as f64 * s / 8.0).round() as u32 * 8).max(64);
    (rw.min(w), rh.min(h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_group_and_option_is_documented() {
        let groups = groups();
        assert_eq!(groups.iter().map(|g| g.name).collect::<Vec<_>>(), ["Engine", "Session", "Stream", "Pacer"]);
        for g in &groups {
            assert!(!g.doc.is_empty(), "{} has no introduction", g.name);
            let mut seen = std::collections::HashSet::new();
            for o in &g.options {
                assert!(!o.doc.is_empty(), "{}.{} has no doc", g.name, o.name);
                assert!(!o.default.is_empty(), "{}.{} has no default text", g.name, o.name);
                assert!(!o.doc.contains('|') && !o.values.contains('|'), "{}.{} would break the table", g.name, o.name);
                assert!(seen.insert(o.name), "{}.{} twice", g.name, o.name);
            }
        }
        let session = &groups[1].options;
        assert_eq!(session.iter().find(|o| o.name == "k").map(|o| o.default.as_str()), Some("1"));
        assert_eq!(session.iter().find(|o| o.name == "static_protection").map(|o| o.default.as_str()), Some("off"));
    }

    #[test]
    fn default_text_reads_like_the_reference() {
        assert_eq!(default_text("None"), "off");
        assert_eq!(default_text("true"), "on");
        assert_eq!(default_text("Some(SceneCut { threshold: 0.08 })"), "threshold 0.08");
        assert_eq!(default_text("Some(HeldFrames { mean: 0.0033, max: 0.06 })"), "mean 0.0033, max 0.06");
        assert_eq!(default_text("Auto"), "Auto");
        assert_eq!(default_text("0.97"), "0.97");
        assert_eq!(default_text("60.0"), "60");
    }
}
