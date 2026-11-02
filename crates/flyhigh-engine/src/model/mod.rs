//! The model: the weight blob with its manifest and the graph of operations. `Model` is CPU data; sessions upload
//! what they use, and several models may live beside one engine.

pub mod blob;
pub mod graph;

use crate::Blob;
use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::path::Path;
use std::rc::Rc;

#[derive(Clone, Debug, Serialize)]
pub struct ModelInfo {
    /// The checkpoint's name (manifest `meta.model`).
    pub name: String,
    pub train_step: i64,
    pub graph_ops: usize,
    pub tensors: usize,
    /// The render sampler the checkpoint was trained with: "bicubic" or "bilinear".
    pub render_sampling: &'static str,
    /// The manifest's format version (1 when absent).
    pub format_version: u32,
}

pub struct Model {
    blob: Rc<Blob>,
    ops: Rc<Vec<Value>>,
    info: ModelInfo,
}

/// Format versions this engine reads.
const FORMAT_VERSIONS: std::ops::RangeInclusive<u32> = 1..=1;

impl Model {
    /// A directory with `graph.json`, `flyhighfg.json` and `flyhighfg.bin`.
    pub fn from_dir(dir: &Path) -> Result<Model> {
        let graph = std::fs::read(dir.join("graph.json")).context("read graph.json")?;
        let manifest = std::fs::read(dir.join("flyhighfg.json")).context("read flyhighfg.json")?;
        let weights = std::fs::read(dir.join("flyhighfg.bin")).context("read flyhighfg.bin")?;
        Model::from_bytes(&graph, &manifest, weights)
    }

    /// The same three files as bytes (the browser fetches them; an application may embed them).
    pub fn from_bytes(graph_json: &[u8], manifest_json: &[u8], weights: Vec<u8>) -> Result<Model> {
        let blob = Blob::from_parts(weights, manifest_json)?;
        let graph: Value = serde_json::from_slice(graph_json).context("parse graph.json")?;
        let ops: Vec<Value> = graph["ops"].as_array().context("graph.json: no ops")?.clone();
        Model::from_parts(blob, ops)
    }

    pub fn from_parts(blob: Blob, ops: Vec<Value>) -> Result<Model> {
        Model::from_shared(Rc::new(blob), Rc::new(ops))
    }

    /// The parts an `api::Engine` already holds.
    pub fn from_shared(blob: Rc<Blob>, ops: Rc<Vec<Value>>) -> Result<Model> {
        anyhow::ensure!(!ops.is_empty(), "graph.json: no ops");
        let meta = &blob.man["meta"];
        let format_version = meta["format_version"].as_u64().unwrap_or(1) as u32;
        anyhow::ensure!(
            FORMAT_VERSIONS.contains(&format_version),
            "model format_version {format_version} not supported, this engine reads {}..={}",
            FORMAT_VERSIONS.start(),
            FORMAT_VERSIONS.end()
        );
        let tensors = blob.man["tensors"].as_object().context("manifest has no tensors")?;
        for (name, t) in tensors {
            check_tensor(name, t, blob.bytes.len())?;
        }
        graph::validate(&ops, tensors)?;
        let info = ModelInfo {
            name: meta["model"].as_str().unwrap_or("?").to_string(),
            train_step: meta["train_step"].as_i64().unwrap_or(-1),
            graph_ops: ops.len(),
            tensors: blob.man["tensors"].as_object().map_or(0, |m| m.len()),
            render_sampling: if blob.render_cubic { "bicubic" } else { "bilinear" },
            format_version,
        };
        Ok(Model { blob, ops, info })
    }

    pub fn info(&self) -> &ModelInfo {
        &self.info
    }
}

/// A tensor entry of the manifest: an unsigned offset, integer dimensions of at least 1 (a scalar has none), and the
/// whole tensor inside the f16 blob. Every arithmetic step is checked: the manifest is a file a host may have been
/// handed, and the blob accessors index the bytes without a second look.
fn check_tensor(name: &str, t: &Value, n_bytes: usize) -> Result<()> {
    let off = t["offset"].as_u64().with_context(|| format!("tensor {name}: offset is not an unsigned integer"))?;
    let shape = t["shape"].as_array().with_context(|| format!("tensor {name}: shape is not an array"))?;
    let mut count: u64 = 1;
    for d in shape {
        let d = d.as_u64().with_context(|| format!("tensor {name}: shape has a non-integer dimension"))?;
        anyhow::ensure!(d >= 1, "tensor {name}: shape has a zero dimension");
        count = count.checked_mul(d).with_context(|| format!("tensor {name}: shape overflows"))?;
    }
    let end = off
        .checked_add(count)
        .and_then(|e| e.checked_mul(2))
        .with_context(|| format!("tensor {name}: offset overflows"))?;
    anyhow::ensure!(end <= n_bytes as u64, "tensor {name} runs past the end of the weight blob");
    Ok(())
}

impl Model {
    pub fn blob(&self) -> &Blob {
        &self.blob
    }

    pub fn ops(&self) -> &[Value] {
        &self.ops
    }

    pub fn shared_blob(&self) -> Rc<Blob> {
        self.blob.clone()
    }

    pub fn shared_ops(&self) -> Rc<Vec<Value>> {
        self.ops.clone()
    }
}
