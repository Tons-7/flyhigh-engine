//! `quality`: the quality regression on ground-truth triplets through `api::Generator`, against a saved baseline.

use super::model_dir;
use anyhow::{Context, Result};
use serde_json::Value;
use std::path::PathBuf;

// Quality regression: `flyhigh-dev quality <sets_dir> [baseline.json] [save=out.json] [k=1] [static] [fast]` — runs every triplet
// of <sets_dir>/manifest.json (entries {set, name}; files <set>/<name>_1.png, _2.png, _3.png) through the library's
// Generator with default options, generates the middle of _1 and _3 and scores it against _2 (PSNR, 8-bit RGB).
// Prints per-set means. With a baseline (a file written by save=) it fails when a set mean drops by more than 0.05 dB
// or a single triplet by more than 0.5 dB.
#[derive(clap::Args)]
pub struct Args {
    /// directory with manifest.json and the `<set>/<name>_{1,2,3}.png` triplets
    pub sets: PathBuf,
    /// a baseline written by --save: fail when a set mean drops by more than 0.05 dB or one triplet by more than 0.5 dB
    #[arg(long)]
    pub baseline: Option<PathBuf>,
    /// write the per-set and per-triplet PSNR as JSON (a new baseline)
    #[arg(long)]
    pub save: Option<String>,
    /// fine iterations of the network
    #[arg(long, default_value_t = 1)]
    pub k: u32,
    /// static protection on
    #[arg(long = "static")]
    pub keep_static: bool,
    /// fast render (texture-sampled, linear t)
    #[arg(long)]
    pub fast: bool,
    /// internal network size as WxH (experiment hook, FH_Q_INTERNAL)
    #[arg(long)]
    pub internal: Option<String>,
    /// bilinear render instead of the model's sampler (experiment hook, FH_Q_BILINEAR)
    #[arg(long)]
    pub bilinear: bool,
}

pub fn run(cmd: Args) -> Result<bool> {
    use flyhigh_engine::api::{Engine, EngineOptions, Generator, Options};
    let dir = cmd.sets.clone();
    let (save, k, keep_static, fast) = (cmd.save, cmd.k, cmd.keep_static, cmd.fast);
    if let Some(v) = &cmd.internal {
        std::env::set_var("FH_Q_INTERNAL", v);
    }
    if cmd.bilinear {
        std::env::set_var("FH_Q_BILINEAR", "1");
    }
    let baseline: Option<Value> = match &cmd.baseline {
        Some(p) => {
            Some(serde_json::from_slice(&std::fs::read(p).with_context(|| format!("baseline {}", p.display()))?)?)
        }
        None => None,
    };
    let man: Vec<Value> = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).context("manifest.json")?)?;
    let engine = Engine::load_with(&model_dir(), &EngineOptions::from_env())?;
    println!(
        "quality: {} triplets, k={} keep_static={} fast_render={} | {:?}",
        man.len(),
        k,
        keep_static,
        fast,
        engine.info()
    );
    let load = |set: &str, name: &str, i: u32| -> Result<image::RgbaImage> {
        let p = dir.join(set).join(format!("{name}_{i}.png"));
        Ok(image::open(&p).with_context(|| format!("open {}", p.display()))?.to_rgba8())
    };
    let mut per: std::collections::BTreeMap<String, Vec<(String, f64)>> = Default::default();
    for e in &man {
        let (set, name) = (e["set"].as_str().unwrap(), e["name"].as_str().unwrap());
        let (a, gt, b) = (load(set, name, 1)?, load(set, name, 2)?, load(set, name, 3)?);
        let (w, h) = a.dimensions();
        // a fresh generator per triplet: no repair history carries over from the previous clip
        let mut o = Options::new(w, h);
        o.k = k;
        o.keep_static = keep_static;
        o.fast_render = fast;
        if std::env::var_os("FH_Q_BILINEAR").is_some() {
            o.cubic = Some(false);
        }
        if let Some((iw, ih)) = std::env::var("FH_Q_INTERNAL").ok().and_then(|v| {
            let (a, b) = v.split_once('x')?;
            Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?))
        }) {
            o.internal = Some((iw.min(w), ih.min(h)));
        }
        let mut gen = Generator::new(&engine, o)?;
        gen.push(&engine, a.as_raw(), 2)?;
        let outs = gen.push(&engine, b.as_raw(), 2)?;
        let mid = outs.iter().find(|o| !o.real).context("no in-between")?;
        let (mut se, mut n) = (0f64, 0u64);
        for (p, q) in mid.rgba.as_chunks::<4>().0.iter().zip(gt.as_raw().as_chunks::<4>().0) {
            for c in 0..3 {
                let d = p[c] as f64 - q[c] as f64;
                se += d * d;
            }
            n += 3;
        }
        let mse = se / n as f64;
        let db = if mse == 0.0 { 99.0 } else { 10.0 * (255.0f64 * 255.0 / mse).log10() };
        per.entry(set.to_string()).or_default().push((name.to_string(), db));
    }
    let mut ok = true;
    let mut out = serde_json::Map::new();
    for (set, v) in &per {
        let mean = v.iter().map(|x| x.1).sum::<f64>() / v.len() as f64;
        let mut line = format!("  {:<8} {:>3} triplets  {:.3} dB", set, v.len(), mean);
        if let Some(b) = &baseline {
            if let Some(bm) = b[set]["mean"].as_f64() {
                let d = mean - bm;
                line += &format!("  (baseline {:.3}, {:+.3})", bm, d);
                if d < -0.05 {
                    ok = false;
                    line += "  REGRESSION";
                }
                for (name, db) in v {
                    if let Some(bt) = b[set]["triplets"][name].as_f64() {
                        if db - bt < -0.5 {
                            ok = false;
                            println!("  {set}/{name}: {db:.2} dB, baseline {bt:.2} -> REGRESSION");
                        }
                    }
                }
            }
        }
        println!("{line}");
        let tri: serde_json::Map<String, Value> =
            v.iter().map(|(n, d)| (n.clone(), serde_json::json!((d * 1000.0).round() / 1000.0))).collect();
        out.insert(set.clone(), serde_json::json!({ "mean": (mean * 1000.0).round() / 1000.0, "triplets": tri }));
    }
    if let Some(p) = save {
        std::fs::write(&p, serde_json::to_string_pretty(&Value::Object(out))?)?;
        println!("  saved {p}");
    }
    if baseline.is_some() {
        println!("  ==> {}", if ok { "no regression" } else { "REGRESSION" });
    }
    Ok(ok)
}
