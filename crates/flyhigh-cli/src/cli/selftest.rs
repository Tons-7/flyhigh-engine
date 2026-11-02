//! Kernel self-tests against the PyTorch fixture dumps and the full-graph parity golden (the default command).

use super::{coop_tests, model_dir, psnr, read_f32, repo_root};
use anyhow::Result;
use flyhigh_engine::*;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;

fn test_kernels(gpu: &Gpu, blob: &Blob, fixdir: &std::path::Path) -> Result<bool> {
    let cases: Value = serde_json::from_slice(&std::fs::read(fixdir.join("kernels.json"))?)?;
    let mut all = true;
    println!("-- kernel self-tests --");
    for c in cases.as_array().unwrap() {
        let tag = c["tag"].as_str().unwrap();
        let kind = c["kind"].as_str().unwrap();
        let ish: Vec<u32> = c["in_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let x = Tensor {
            buf: gpu.buf(&read_f32(&fixdir.join(format!("{}_in.f32", tag)))?),
            c: ish[1],
            h: ish[2],
            w: ish[3],
            c0: 0,
            scale: 1.0,
        };
        let bf = fixdir.join(format!("{}_b.f32", tag));
        let b = if bf.exists() {
            let bsh: Vec<u32> = c["b_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            Some(Tensor { buf: gpu.buf(&read_f32(&bf)?), c: bsh[1], h: bsh[2], w: bsh[3], c0: 0, scale: 1.0 })
        } else {
            None
        };
        let (out, steps) = match kind {
            "conv" => {
                let wp = c["weight"].as_str().unwrap();
                // the conv fixtures were dumped with one checkpoint's weights: another model (other channel widths) can't be checked by them
                let wsh = blob.shape(&format!("{}.0.weight", wp));
                let want_n = std::fs::metadata(fixdir.join(format!("{}_out.f32", tag)))?.len() as usize / 4;
                let hout = (x.h + 2 - 3) / c["stride"].as_u64().unwrap() as u32 + 1;
                let wout = (x.w + 2 - 3) / c["stride"].as_u64().unwrap() as u32 + 1;
                if wsh[1] as u32 != x.c || wsh[0] * (hout * wout) as usize != want_n {
                    println!("  {:<12} {:<8} SKIP (fixture dumped for another model: blob conv {}->{}, fixture cin {}, {} outputs)", tag, kind, wsh[1], wsh[0], x.c, want_n);
                    continue;
                }
                let (wt, bias, slope) = (
                    gpu.buf(&blob.weight(&format!("{}.0.weight", wp))),
                    gpu.buf(&blob.weight(&format!("{}.0.bias", wp))),
                    gpu.buf(&blob.weight(&format!("{}.1.weight", wp))),
                );
                let cout = blob.shape(&format!("{}.0.weight", wp))[0] as u32;
                let stride = c["stride"].as_u64().unwrap() as u32;
                gpu.conv(
                    &ConvArgs {
                        x: &x,
                        weights: &wt,
                        bias: &bias,
                        slope: &slope,
                        cin: x.c,
                        cout,
                        kp: 0,
                        np: 0,
                        cinp: 0,
                        stride,
                    },
                    1,
                )
            }
            "interp" => {
                let osh: Vec<u32> =
                    c["out_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
                gpu.interp(&x, osh[2], osh[3], c["mode"].as_str() == Some("aa"))
            }
            "avgpool" => gpu.avgpool(&x, c["k"].as_u64().unwrap() as u32),
            "warp" => gpu.warp(&x, b.as_ref().unwrap()),
            "sad" => gpu.sad(&x, b.as_ref().unwrap(), c["radius"].as_u64().unwrap() as u32),
            "convex" => gpu.convex(&x, b.as_ref().unwrap()),
            "film" => gpu.film(&x, b.as_ref().unwrap()),
            "blend" => gpu.blend(&x),
            "luma_op" => gpu.luma(&x, c["mode"].as_str() == Some("grad")),
            "binary" => gpu.binary(&x, b.as_ref().unwrap(), if c["op"].as_str() == Some("mul") { 1 } else { 0 }),
            "unary" => gpu.unary(&x, 2, 0.0, 0.0),
            _ => {
                all = false;
                continue;
            }
        };
        gpu.replay(&steps);
        let got = gpu.readback(&out.buf, out.n());
        let want = read_f32(&fixdir.join(format!("{}_out.f32", tag)))?;
        let maxd = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let ok = got.len() == want.len() && maxd < 2e-2;
        all &= ok; // f16 precision
        println!("  {:<12} {:<8} maxΔ {:.2e}  -> {}", tag, kind, maxd, if ok { "PASS" } else { "FAIL" });
    }
    Ok(all)
}

fn test_parity(gpu: &Gpu, blob: &Blob, fixdir: &std::path::Path) -> Result<bool> {
    let meta: Value = serde_json::from_slice(&std::fs::read(fixdir.join("parity.json"))?)?;
    let (hp, wp) = (meta["Hp"].as_u64().unwrap() as u32, meta["Wp"].as_u64().unwrap() as u32);
    let (hh, ww) = (meta["H"].as_u64().unwrap() as u32, meta["W"].as_u64().unwrap() as u32);
    let k = meta["K"].as_u64().unwrap() as u32;
    let graph: Value = serde_json::from_slice(&std::fs::read(model_dir().join("graph.json"))?)?;
    let ops: Vec<Value> = graph["ops"].as_array().unwrap().clone();

    let mut env: HashMap<String, Tensor> = HashMap::new();
    env.insert(
        "A".into(),
        Tensor { buf: gpu.buf(&read_f32(&fixdir.join("parity_A.f32"))?), c: 3, h: hp, w: wp, c0: 0, scale: 1.0 },
    );
    env.insert(
        "B".into(),
        Tensor { buf: gpu.buf(&read_f32(&fixdir.join("parity_B.f32"))?), c: 3, h: hp, w: wp, c0: 0, scale: 1.0 },
    );
    let mut plan: Vec<Step> = Vec::new();
    // the golden records the sampler its render used; older dumps predate the field and are bilinear
    let cubic = meta["render_sampling"].as_str() == Some("bicubic");
    if cubic != blob.render_cubic {
        println!(
            "  note: golden render is {}, the model's manifest says {}",
            if cubic { "bicubic" } else { "bilinear" },
            if blob.render_cubic { "bicubic" } else { "bilinear" }
        );
    }
    let mut cx = BuildCtx::new(gpu, blob, k, meta["t"].as_f64().unwrap() as f32, hp, wp);
    cx.cubic = cubic;
    build(&ops, &mut env, &mut plan, &mut cx);

    gpu.replay(&plan);
    gpu.wait();
    println!("\n-- full-graph parity (static plan, {} steps) --", plan.len());
    let ftq = &env["flow_tq"];
    let got_f = gpu.readback_t(ftq); // flow_tq is a view of the convex-upsample output
    let want_f = read_f32(&fixdir.join("parity_flow_tq.f32"))?;
    let fl1 = got_f.iter().zip(&want_f).map(|(a, b)| (a - b).abs()).sum::<f32>() / want_f.len() as f32;
    let r = &env["render"];
    let got_full = gpu.readback(&r.buf, r.n());
    let mut got_r = Vec::with_capacity((3 * hh * ww) as usize);
    for c in 0..3 {
        for y in 0..hh {
            for x in 0..ww {
                got_r.push(got_full[((c * hp + y) * wp + x) as usize]);
            }
        }
    }
    let want_r = read_f32(&fixdir.join("parity_render.f32"))?;
    let rp = psnr(&got_r, &want_r);
    println!("  flow_tq L1 = {:.5} px (tol<=0.05) -> {}", fl1, if fl1 <= 0.05 { "PASS" } else { "FAIL" });
    println!("  render PSNR = {:.2} dB (tol>=~50) -> {}", rp, if rp >= 50.0 { "PASS" } else { "FAIL" });
    let ok = fl1 <= 0.05 && rp >= 50.0;
    println!("  ==> runtime vs torch golden: {}", if ok { "PARITY" } else { "MISMATCH" });

    // timing: min = clean GPU latency without OS/clock noise (the yardstick for optimization)
    let timeit = |pl: &[Step]| -> (f64, f64, f64) {
        for _ in 0..15 {
            gpu.replay(pl);
            gpu.wait();
        }
        let mut t: Vec<f64> = (0..200)
            .map(|_| {
                let s = std::time::Instant::now();
                gpu.replay(pl);
                gpu.wait();
                s.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (t[0], t[100], t[190])
    };
    // latency Pareto: sweep K=1,2,3 at the fixture resolution. The fine block is weight-shared
    // (re-dispatched K times), so this is free tiering from ONE set of weights: Performance/Balanced/Quality.
    println!("  latency Pareto @{}x{} (Tensor-Core path):", wp, hp);
    for ksweep in [1u32, 2, 3] {
        let mut envk: HashMap<String, Tensor> = HashMap::new();
        envk.insert("A".into(), env["A"].clone());
        envk.insert("B".into(), env["B"].clone());
        let mut plank: Vec<Step> = Vec::new();
        let mut cxk = BuildCtx::new(gpu, blob, ksweep, meta["t"].as_f64().unwrap() as f32, hp, wp);
        cxk.cubic = cubic;
        build(&ops, &mut envk, &mut plank, &mut cxk);
        let (mn, md, p95) = timeit(&plank);
        let gms = gpu.gpu_time_ms(&plank);
        let over = mn - gms;
        let tier = match ksweep {
            1 => "Performance",
            2 => "Balanced  ",
            _ => "Quality   ",
        };
        println!("    K={} {} wall min {:.2} / p50 {:.2} / p95 {:.2} ms | GPU-only {:.2} ms | CPU-overhead {:.2} ms ({:.0}%)  ({} ops, {:.0} fps)",
                 ksweep, tier, mn, md, p95, gms, over, 100.0 * over / mn, plank.len(), 1000.0 / mn);
        if ksweep == 2 {
            let prof = gpu.profile_plan(&plank);
            let tot: f64 = prof.iter().map(|x| x.1).sum();
            println!(
                "  per-op GPU profile K=2 (by pipeline; summed {:.2} ms incl. pass-split gaps — read the shares):",
                tot
            );
            for (k, ms) in prof.iter().take(10) {
                println!("    {:<10} {:>6.3} ms  {:>5.1}%", k, ms, 100.0 * ms / tot);
            }
        }
    }
    Ok(ok)
}

/// Kernel self-tests against the PyTorch fixture dumps and the full-graph parity golden; the tensor-core probes first
/// unless FH_NO_COOP is set. Exits 1 on a failure.
pub fn run() -> Result<()> {
    if std::env::var_os("FH_NO_COOP").is_none() {
        coop_tests::coop_shared_test();
        coop_tests::coop_test();
        coop_tests::coop_gemm_test();
        coop_tests::conv_tc_test();
    }
    let blob = Blob::load(&model_dir())?;
    // kernel fixtures are model-independent (fixtures/kernels); the full-graph parity golden belongs to one checkpoint:
    // FH_FIXTURES pins it, else the catalogue names the shipped model's (`parity_fixture` in models/models.json)
    let fixdir = repo_root().join("fixtures").join("kernels");
    let paritydir = std::env::var_os("FH_FIXTURES").map(PathBuf::from).unwrap_or_else(|| {
        let catalogue: Option<Value> = std::fs::read(repo_root().join("models").join("models.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok());
        catalogue
            .as_ref()
            .and_then(|c| c["models"].as_array())
            .and_then(|m| m.iter().find(|m| m["shipped"].as_bool() == Some(true)))
            .and_then(|m| m["parity_fixture"].as_str())
            .map(|f| repo_root().join(f))
            .unwrap_or_else(|| fixdir.clone())
    });
    let gpu = Gpu::new()?;
    println!("wgpu adapter: {}\nblob: {} tensors\n", gpu.name, blob.man["tensors"].as_object().map_or(0, |t| t.len()));
    // the fixtures are dumped from PyTorch on the research rig and are not in git (see docs/development.md)
    let k_ok = if fixdir.join("kernels.json").exists() {
        test_kernels(&gpu, &blob, &fixdir)?
    } else {
        println!(
            "-- kernel self-tests SKIPPED: no {} (fixtures are not in git)",
            fixdir.join("kernels.json").display()
        );
        true
    };
    let p_ok = if paritydir.join("parity.json").exists() {
        test_parity(&gpu, &blob, &paritydir)?
    } else {
        println!(
            "-- full-graph parity SKIPPED: no {} (set FH_FIXTURES=<dir> to a golden dump of this model)",
            paritydir.join("parity.json").display()
        );
        true
    };
    if !(k_ok && p_ok) {
        std::process::exit(1);
    }
    Ok(())
}
