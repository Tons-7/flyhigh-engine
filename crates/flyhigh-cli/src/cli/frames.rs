//! `frames`: one generated frame between two images through the PlanSession path.

use super::{model_dir, net_from_env, parse_size};
use anyhow::{Context, Result};
use flyhigh_engine::*;
use serde_json::Value;

// Real-frame mode: `flyhigh-dev frames A.png B.png out.png [t=0.5] [K=2] [IW IH]` — generate the frame at phase t
// between two images of any size through the PlanSession path (RGBA8 in/out on the GPU, replicate pad to /32, crop), the
// same path the browser and the app use. IW IH = internal network resolution (default: the frame size).
#[derive(clap::Args)]
pub struct Args {
    /// first source frame (any image format)
    pub a: String,
    /// second source frame, the same size
    pub b: String,
    /// output image
    pub out: String,
    /// phase of the generated frame between the two sources
    #[arg(long, default_value_t = 0.5)]
    pub t: f32,
    /// fine iterations of the network
    #[arg(long, default_value_t = 2)]
    pub k: u32,
    /// internal network size as WxH (default: the frame size)
    #[arg(long, value_parser = parse_size)]
    pub internal: Option<(u32, u32)>,
}

pub fn run(cmd: Args) -> Result<()> {
    let (pa, pb, po) = (&cmd.a, &cmd.b, &cmd.out);
    let (t, k) = (cmd.t, cmd.k);
    let ia = image::open(pa).with_context(|| format!("open {}", pa))?.to_rgba8();
    let ib = image::open(pb).with_context(|| format!("open {}", pb))?.to_rgba8();
    anyhow::ensure!(ia.dimensions() == ib.dimensions(), "frame sizes differ");
    let (w, h) = ia.dimensions();
    let (iw, ih) = cmd.internal.unwrap_or((w, h));
    let blob = Blob::load(&model_dir())?;
    let graph: Value = serde_json::from_slice(&std::fs::read(model_dir().join("graph.json"))?)?;
    let ops: Vec<Value> = graph["ops"].as_array().unwrap().clone();
    let gpu = Gpu::new()?;
    let net = net_from_env(&blob);
    let sess = PlanSession::new_int(&gpu, &blob, &ops, w, h, k, t, iw, ih, &net);
    sess.submit(&gpu, ia.as_raw(), ib.as_raw());
    gpu.wait(); // warm
    let t0 = std::time::Instant::now();
    sess.submit(&gpu, ia.as_raw(), ib.as_raw());
    gpu.wait();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let gms = gpu.gpu_time_ms_n(&sess.plan, 20);
    let out = pollster::block_on(sess.read_rgba(&gpu))?;
    image::RgbaImage::from_raw(w, h, out).unwrap().save(po).with_context(|| format!("save {}", po))?;
    println!("frames: {}x{} internal {}x{} t={} K={} repair={} cubic={} -> {} | {} steps | GPU {:.2} ms | {:.2} ms wall (submit+wait, 1 frame)", w, h, iw, ih, t, k, net.repair, net.cubic, po, sess.plan.len(), gms, ms);
    Ok(())
}
