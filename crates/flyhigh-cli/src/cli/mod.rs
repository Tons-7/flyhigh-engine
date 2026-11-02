//! The subcommands of the developer binary and what they share: the model directory, the CLI network options, fixture readers, the WxH parser.

pub mod bench;
pub mod coop_tests;
pub mod frames;
pub mod microbench;
pub mod options_doc;
pub mod phases;
pub mod play;
pub mod quality;
pub mod replay;
pub mod selftest;
pub mod texbench;

use anyhow::{Context, Result};
use flyhigh_engine::{Blob, NetOpts, DEFAULT_REPAIR};
use std::path::{Path, PathBuf};

/// `WxH` -> (w, h), for the `--internal` options.
pub fn parse_size(s: &str) -> Result<(u32, u32), String> {
    let (w, h) = s.split_once('x').ok_or_else(|| format!("'{s}' is not WxH"))?;
    Ok((w.parse().map_err(|e| format!("width: {e}"))?, h.parse().map_err(|e| format!("height: {e}"))?))
}

pub fn read_f32(p: &Path) -> Result<Vec<f32>> {
    Ok(bytemuck::cast_slice(&std::fs::read(p).with_context(|| format!("read {:?}", p))?).to_vec())
}
pub fn psnr(a: &[f32], b: &[f32]) -> f32 {
    let mse: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / a.len() as f32;
    if mse == 0.0 {
        99.0
    } else {
        10.0 * (1.0 / mse).log10()
    }
}

/// The repository root (two levels above this crate).
pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Model directory (graph.json + flyhighfg.json/.bin): FH_MODEL_DIR, else `models/` of this repository.
pub fn model_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("FH_MODEL_DIR") {
        return PathBuf::from(d);
    }
    repo_root().join("models")
}

// Network options of the CLI sessions: the library defaults plus the developer overrides FH_GMHS=0|1|2 (repair level)
// and FH_CUBIC=0|1 (render sampler).
pub fn net_from_env(blob: &Blob) -> NetOpts {
    let mut n = NetOpts::new(blob);
    if let Some(v) = std::env::var_os("FH_GMHS") {
        n.repair = v.to_string_lossy().trim().parse::<u32>().unwrap_or(DEFAULT_REPAIR).min(2);
    }
    if let Some(v) = std::env::var_os("FH_CUBIC") {
        n.cubic = v.to_string_lossy().trim() != "0";
    }
    n
}
