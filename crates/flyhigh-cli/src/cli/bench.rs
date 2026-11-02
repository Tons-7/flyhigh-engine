//! `bench`: GPU time per generated frame on procedural frames through a PlanSession, for K = 1, 2, 3.

use super::{model_dir, net_from_env, parse_size};
use anyhow::Result;
use flyhigh_engine::*;
use serde_json::Value;

// Resolution sweep: `flyhigh-dev bench W H [K]` — GPU time per generated frame at W x H through a PlanSession (RGBA8
// in -> graph -> RGBA8 out, exactly the browser/app path) for K = 1, 2, 3 (or the given K). Inputs are procedural
// frames (smooth pattern + noise, B shifted by a few pixels so the flow and the warps are non-trivial).
#[derive(clap::Args)]
pub struct Args {
    /// frame width
    pub w: u32,
    /// frame height
    pub h: u32,
    /// fine iterations (0 = sweep 1, 2, 3)
    #[arg(long, default_value_t = 0)]
    pub k: u32,
    /// internal network size as WxH (default: the frame size)
    #[arg(long, value_parser = parse_size)]
    pub internal: Option<(u32, u32)>,
}

pub fn run(cmd: Args) -> Result<()> {
    let (w, h) = (cmd.w, cmd.h);
    let ks: Vec<u32> = if cmd.k == 0 { vec![1, 2, 3] } else { vec![cmd.k] };
    let (iw, ih) = cmd.internal.unwrap_or((w, h));
    let blob = Blob::load(&model_dir())?;
    let graph: Value = serde_json::from_slice(&std::fs::read(model_dir().join("graph.json"))?)?;
    let ops: Vec<Value> = graph["ops"].as_array().unwrap().clone();
    let gpu = Gpu::new()?;
    let frame = |dx: i32, dy: i32| -> Vec<u8> {
        let mut v = vec![0u8; (w * h * 4) as usize];
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let (sx, sy) = ((x + dx) as f32, (y + dy) as f32);
                let hsh = (((x + dx) * 73856093) ^ ((y + dy) * 19349663)) as u32;
                let n = (hsh.wrapping_mul(2654435761) >> 24) as f32 / 255.0;
                let r = 0.5 + 0.35 * (sx * 0.013).sin() * (sy * 0.017).cos() + 0.1 * n;
                let g = 0.5 + 0.35 * (sx * 0.009 + 1.0).cos() * (sy * 0.021).sin() + 0.1 * n;
                let b = 0.5 + 0.35 * ((sx + sy) * 0.011).sin() + 0.1 * n;
                let o = ((y as u32 * w + x as u32) * 4) as usize;
                v[o] = (r.clamp(0.0, 1.0) * 255.0) as u8;
                v[o + 1] = (g.clamp(0.0, 1.0) * 255.0) as u8;
                v[o + 2] = (b.clamp(0.0, 1.0) * 255.0) as u8;
                v[o + 3] = 255;
            }
        }
        v
    };
    let (a, b) = (frame(0, 0), frame(-5, 3));
    let net = net_from_env(&blob);
    println!("bench {}x{} internal {}x{} repair={} ({}): GPU time per generated frame, PlanSession path (rgba_in -> graph -> rgba_out)", w, h, iw, ih, net.repair, gpu.name);
    for k in ks {
        let t0 = std::time::Instant::now();
        let alloc0 = gpu.alloc_bytes.get();
        let sess = PlanSession::new_int(&gpu, &blob, &ops, w, h, k, 0.5, iw, ih, &net);
        let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
        for _ in 0..10 {
            sess.submit(&gpu, &a, &b);
            gpu.wait();
        }
        let gms = gpu.gpu_time_ms(&sess.plan);
        let mut wall: Vec<f64> = (0..100)
            .map(|_| {
                let s = std::time::Instant::now();
                sess.submit(&gpu, &a, &b);
                gpu.wait();
                s.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        wall.sort_by(|x, y| x.partial_cmp(y).unwrap());
        println!("  K={} GPU-only {:.3} ms | wall(submit+upload+wait) min {:.2} / p50 {:.2} ms | {} steps | plan built in {:.0} ms | tensor buffers created {:.1} MB",
                 k, gms, wall[0], wall[50], sess.plan.len(), build_ms, (gpu.alloc_bytes.get() - alloc0) as f64 / 1048576.0);
        if std::env::var_os("FH_SPLIT").is_some() {
            // pair / mid split with dynamic t: cost of the two stages, and parity against the single plan at t = 0.5 and
            // against a single plan built for t = 0.25 after retargeting the mid stage with set_t
            let (hp, wp) = (h.div_ceil(32) * 32, w.div_ceil(32) * 32);
            let alloc1 = gpu.alloc_bytes.get();
            let ta = Tensor { buf: gpu.alloc((3 * hp * wp) as usize), c: 3, h: hp, w: wp, c0: 0, scale: 1.0 };
            let tb = Tensor { buf: gpu.alloc((3 * hp * wp) as usize), c: 3, h: hp, w: wp, c0: 0, scale: 1.0 };
            let rp = RgbaParams { w, h, wp, hp };
            let mut pp = vec![
                gpu.compute_step("rgba_in", &[&sess.in_a], bytemuck::bytes_of(&rp), &ta.buf, workgroups_2d(wp, hp, 1)),
                gpu.compute_step("rgba_in", &[&sess.in_b], bytemuck::bytes_of(&rp), &tb.buf, workgroups_2d(wp, hp, 1)),
            ];
            let mut pm = Vec::new();
            let (r, _env, deps) =
                build_network_ex(&gpu, &blob, &ops, &ta, &tb, k, 0.5, iw, ih, &mut pp, Some(&mut pm), None, &net);
            let split_mb = (gpu.alloc_bytes.get() - alloc1) as f64 / 1048576.0;
            gpu.queue.write_buffer(&sess.in_a, 0, &a);
            gpu.queue.write_buffer(&sess.in_b, 0, &b);
            gpu.replay(&pp);
            gpu.replay(&pm);
            gpu.wait();
            let full = gpu.readback_t(&sess.env["render"]);
            let split = gpu.readback_t(&r);
            let maxd = full.iter().zip(&split).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            let (tp, tm) = (gpu.gpu_time_ms(&pp), gpu.gpu_time_ms(&pm));
            println!("    split: pair {:.3} ms + mid {:.3} ms -> per generated frame 2x {:.3} / 3x {:.3} / 4x {:.3} ms | {}+{} steps | {} t-dependent buffers | buffers {:.1} MB | max|render diff| vs single plan (t=0.5) {:.2e}",
                     tp, tm, tp + tm, tp / 2.0 + tm, tp / 3.0 + tm, pp.len(), pm.len(), deps.len(), split_mb, maxd);
            if std::env::var_os("FH_PROFILE").is_some() {
                for (name, pl) in [("pair", &pp), ("mid", &pm)] {
                    let prof = gpu.profile_plan(pl);
                    let tot: f64 = prof.iter().map(|x| x.1).sum();
                    println!("    {name} profile ({:.3} ms summed):", tot);
                    for (k, ms) in prof.iter().take(8) {
                        println!("      {:<14} {:>6.3} ms  {:>5.1}%", k, ms, 100.0 * ms / tot);
                    }
                }
            }
            set_t(&gpu, &deps, 0.25);
            gpu.replay(&pm);
            gpu.wait();
            let split25 = gpu.readback_t(&r);
            let s25 = PlanSession::new_int(&gpu, &blob, &ops, w, h, k, 0.25, iw, ih, &net);
            s25.submit(&gpu, &a, &b);
            gpu.wait();
            let full25 = gpu.readback_t(&s25.env["render"]);
            let maxd25 = full25.iter().zip(&split25).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            let moved = split.iter().zip(&split25).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            println!("    set_t(0.25): max|render diff| vs single plan built for t=0.25 {:.2e} (the two mids differ by up to {:.3}, so t really moved)", maxd25, moved);
        }
    }
    Ok(())
}
