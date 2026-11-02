//! `texbench`: GPU time of the pair stage and of one generated frame on the texture path, exact or fast render.

use super::{model_dir, parse_size};
use anyhow::Result;
use flyhigh_engine::*;
use serde_json::Value;

// Texture-path timing: `flyhigh-dev texbench W H [IW IH] [fast] [static]` — GPU time of one pair and of one
// generated frame through gen_tex (the desktop / browser path), exact or fast render.
#[derive(clap::Args)]
pub struct Args {
    /// frame width
    pub w: u32,
    /// frame height
    pub h: u32,
    /// internal network size as WxH (default: the 480-line rung above 960 px)
    #[arg(long, value_parser = parse_size)]
    pub internal: Option<(u32, u32)>,
    /// fast render: sample the sources straight into the output
    #[arg(long)]
    pub fast: bool,
    /// with --fast: the t-dependent network once per pair
    #[arg(long)]
    pub linear: bool,
    /// static protection on
    #[arg(long = "static")]
    pub keep_static: bool,
    /// also print one machine-readable JSON line (tools/baseline.py)
    #[arg(long)]
    pub json: bool,
}

pub fn run(cmd: Args) -> Result<()> {
    use flyhigh_engine::gen_tex::{GenTex, TexOptions};
    let (w, h) = (cmd.w, cmd.h);
    let mut o = TexOptions::new(w, h);
    if let Some((iw, ih)) = cmd.internal {
        o.iw = iw;
        o.ih = ih;
    }
    o.fast_render = cmd.fast;
    o.keep_static = cmd.keep_static;
    o.linear_t = cmd.linear;
    let blob = Blob::load(&model_dir())?;
    let graph: Value = serde_json::from_slice(&std::fs::read(model_dir().join("graph.json"))?)?;
    let ops: Vec<Value> = graph["ops"].as_array().unwrap().clone();
    let gpu = Gpu::new()?;
    let alloc0 = gpu.alloc_bytes.get();
    let mut g = GenTex::new(&gpu, &blob, &ops, o.clone(), None)?;
    let _ = alloc0;
    let tensors_mb = gpu.live_bytes() as f64 / 1048576.0;
    if std::env::var_os("FH_MEM").is_some() {
        match gpu.device.generate_allocator_report() {
            Some(r) => {
                let mut a: Vec<(u64, String)> = r.allocations.iter().map(|x| (x.size, x.name.clone())).collect();
                a.sort_by_key(|x| std::cmp::Reverse(x.0));
                println!(
                    "  allocator: {} allocations, {:.1} MB allocated, {:.1} MB reserved; largest (MB): {}",
                    r.allocations.len(),
                    r.total_allocated_bytes as f64 / 1048576.0,
                    r.total_reserved_bytes as f64 / 1048576.0,
                    a.iter()
                        .take(30)
                        .map(|(s, _)| format!("{:.1}", *s as f64 / 1048576.0))
                        .collect::<Vec<_>>()
                        .join(" ")
                );
                let mut hist: std::collections::BTreeMap<u64, (u32, u64)> = Default::default();
                for (sz, _) in &a {
                    let e = hist.entry((*sz as f64).log2().floor() as u64).or_default();
                    e.0 += 1;
                    e.1 += sz;
                }
                for (k, (n, b)) in hist.iter().rev().take(8) {
                    println!("    2^{k:>2} B bucket: {n:>4} allocations, {:.1} MB", *b as f64 / 1048576.0);
                }
            }
            None => println!("  allocator report: not available on this backend"),
        }
        let sz = gpu.live_sizes();
        println!(
            "  live tensor buffers: {} totalling {:.1} MB; largest (MB): {}",
            sz.len(),
            tensors_mb,
            sz.iter().take(24).map(|b| format!("{:.1}", *b as f64 / 1048576.0)).collect::<Vec<_>>().join(" ")
        );
    }
    for _ in 0..2 {
        g.add_input(&gpu);
    }
    g.add_output(&gpu);
    let frame = |dx: f32| -> Vec<u8> {
        let mut v = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let o = ((y * w + x) * 4) as usize;
                let u = (x as f32 - dx) * 0.05;
                v[o] = (128.0 + 100.0 * u.sin()) as u8;
                v[o + 1] = (128.0 + 90.0 * (u * 0.7 + y as f32 * 0.03).cos()) as u8;
                v[o + 2] = (128.0 + 80.0 * (y as f32 * 0.04).sin()) as u8;
                v[o + 3] = 255;
            }
        }
        v
    };
    for (i, dx) in [0.0f32, 6.0].iter().enumerate() {
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &g.ins[i].0,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &frame(*dx),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        g.push_slot(&gpu, i)?;
    }
    g.generate(&gpu, 0.5, 0)?;
    gpu.wait();
    let s = &g.sess[0];
    let pair = gpu.gpu_time_ms_n(&s.pair_steps_views(&gpu, (0, &g.ins[0].1), (1, &g.ins[1].1)), 30);
    let mid = if s.is_fast() && o.linear_t {
        gpu.gpu_time_ms_n(
            &[s.render_tex_step(&gpu, (0, &g.ins[0].1), (1, &g.ins[1].1), (0, &g.outs[0].1), Some(0.25))],
            30,
        )
    } else if s.is_fast() {
        gpu.gpu_time_ms_n(&s.mid_steps_fast(&gpu, (0, &g.ins[0].1), (1, &g.ins[1].1), (0, &g.outs[0].1)), 30)
    } else {
        gpu.gpu_time_ms_n(&s.mid_steps(&gpu, 0, &g.outs[0].1), 30)
    };
    if std::env::var_os("FH_PROFILE").is_some() {
        let steps = if s.is_fast() {
            s.mid_steps_fast(&gpu, (0, &g.ins[0].1), (1, &g.ins[1].1), (0, &g.outs[0].1))
        } else {
            s.mid_steps(&gpu, 0, &g.outs[0].1)
        };
        let prof = gpu.profile_plan(&steps);
        let tot: f64 = prof.iter().map(|x| x.1).sum();
        for (k, ms) in prof.iter().take(10) {
            println!("    {:<14} {:>6.3} ms  {:>5.1}%", k, ms, 100.0 * ms / tot);
        }
        println!("  pair:");
        let ps = s.pair_steps_views(&gpu, (0, &g.ins[0].1), (1, &g.ins[1].1));
        let prof = gpu.profile_plan(&ps);
        let tot: f64 = prof.iter().map(|x| x.1).sum();
        for (k, ms) in prof.iter().take(14) {
            println!("    {:<14} {:>6.3} ms  {:>5.1}%", k, ms, 100.0 * ms / tot);
        }
    }
    println!("texbench {}x{} internal {}x{} fast={} static={} ({}): pair {:.3} ms, generated frame {:.3} ms | 144 Hz from 30 fps: {:.0} ms of GPU per second | tensors {:.0} MB",
             w, h, o.iw, o.ih, s.is_fast(), o.keep_static, gpu.name, pair, mid, 30.0 * pair + 144.0 * mid, tensors_mb);
    // `json`: one machine-readable line for the baseline recorder (tools/baseline.py)
    if cmd.json {
        let class =
            format!("{}-{}", if gpu.coop { "tc" } else { "portable" }, if gpu.f32_mode { "f32" } else { "f16" });
        let backend = format!("{:?}", gpu.adapter.get_info().backend).to_lowercase();
        println!(
            "{}",
            serde_json::json!({ "gpu": gpu.name, "backend": backend, "compiler": gpu.shader_compiler, "class": class, "w": w, "h": h, "iw": o.iw, "ih": o.ih,
                                            "fast": s.is_fast(), "linear": o.linear_t, "static": o.keep_static, "pair_ms": (pair * 1e4).round() / 1e4,
                                            "mid_ms": (mid * 1e4).round() / 1e4, "tensors_mb": (tensors_mb * 10.0).round() / 10.0 })
        );
    }
    Ok(())
}
