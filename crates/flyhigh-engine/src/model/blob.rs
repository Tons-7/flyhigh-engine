//! The weight blob (`flyhighfg.bin`) and its manifest (`flyhighfg.json`).

use anyhow::{Context, Result};
use serde_json::Value;

// `render_cubic`: the render sampler the checkpoint was trained with (manifest meta.render_sampling = "bicubic"; absent =
// bilinear, models up to v6p). It is the default of NetOpts::cubic; sessions may override it.
pub struct Blob {
    pub bytes: Vec<u8>,
    pub man: Value,
    pub render_cubic: bool,
}
impl Blob {
    pub fn load(dir: &std::path::Path) -> Result<Blob> {
        Blob::from_parts(
            std::fs::read(dir.join("flyhighfg.bin")).context("read .bin")?,
            &std::fs::read(dir.join("flyhighfg.json"))?,
        )
    }
    // in-memory form (browser: fetched bytes + manifest text)
    pub fn from_parts(bytes: Vec<u8>, manifest_json: &[u8]) -> Result<Blob> {
        let man: Value = serde_json::from_slice(manifest_json).context("parse manifest")?;
        anyhow::ensure!(man["tensors"].is_object(), "manifest has no tensors");
        let render_cubic = man["meta"]["render_sampling"].as_str() == Some("bicubic");
        Ok(Blob { bytes, man, render_cubic })
    }
    // `shape` and `weight` take a tensor the loader checked (`Model::from_shared`: every dimension an integer of at
    // least 1, the tensor inside the blob, every name the graph reads present), so they index without a second look.
    pub fn shape(&self, name: &str) -> Vec<usize> {
        self.man["tensors"][name]["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect()
    }
    pub fn weight(&self, name: &str) -> Vec<f32> {
        let off = self.man["tensors"][name]["offset"].as_u64().unwrap() as usize;
        let n = self.shape(name).iter().product::<usize>();
        (0..n)
            .map(|i| {
                let b = (off + i) * 2;
                half::f16::from_bits(u16::from_le_bytes([self.bytes[b], self.bytes[b + 1]])).to_f32()
            })
            .collect()
    }
}
