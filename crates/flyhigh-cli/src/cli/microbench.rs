//! Conv kernel microbenchmarks: the fused tensor-core kernels, or the portable tiled kernels against the naive conv.

use anyhow::Result;
use flyhigh_engine::*;

// Kernel microbench on the fine.r1 shape (cin 235 -> 80 @ 160x92): fused (16-px tile) vs fused4 (64-px tile)
// plus fused4 ablations (mode bits: 1 skip A gather, 2 skip B staging, 4 skip MMA) to attribute per-K-step
// time on the real GPU. Also checks fused4 against fused on identical inputs. Run with FH_BENCH=1.
fn fused_bench(gpu: &Gpu) {
    let (cin, cout, h, w) = (235u32, 80u32, 92u32, 160u32);
    let xin: Vec<f32> = (0..(cin * h * w)).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
    let x = Tensor { buf: gpu.buf(&xin), c: cin, h, w, c0: 0, scale: 1.0 };
    let wf: Vec<f32> = (0..(cout * cin * 9)).map(|i| (((i * 104729) % 1000) as f32 / 1000.0 - 0.5) * 0.05).collect();
    let bias = gpu.buf(&vec![0.01f32; cout as usize]);
    let slope = gpu.buf(&vec![0.2f32; cout as usize]);
    let (bb, kp, np) = gpu.buildb(&wf, cin, cout);
    let (bb4, kp4, np4, cinp) = gpu.buildb_tap(&wf, cin, cout);
    let rep = |st: &Step, n: usize| -> Vec<Step> {
        (0..n)
            .map(|_| match st {
                Step::Compute { .. } => flyhigh_engine::clone_step(st),
                Step::Copy { .. } => unreachable!(),
            })
            .collect()
    };
    const N: usize = 40;
    let tc = ConvArgs { x: &x, weights: &bb, bias: &bias, slope: &slope, cin, cout, kp, np, cinp: 0, stride: 1 };
    let (o_ref, s_ref) = gpu.conv_tc(&tc);
    let t_ref = gpu.gpu_time_ms(&rep(&s_ref[0], N)) / N as f64;
    println!(
        "\n-- fused kernel microbench: fine.r1 shape cin {} -> {} @ {}x{} (per conv, best of 60 x {}) --",
        cin, cout, w, h, N
    );
    println!("  {:<24} {:.3} ms", "fused (16-px tile)", t_ref);
    gpu.replay(&s_ref);
    gpu.wait();
    let r = gpu.readback(&o_ref.buf, o_ref.n());
    {
        // shifted-input direct kernel: prep (xshift3) + conv, timed together, checked against fused
        let direct =
            ConvArgs { x: &x, weights: &bb4, bias: &bias, slope: &slope, cin, cout, kp: kp4, np: np4, cinp, stride: 1 };
        let (o_d, s_d) = gpu.conv_direct(&direct);
        let plan_d: Vec<Step> = (0..N)
            .flat_map(|_| {
                s_d.iter().map(|st| match st {
                    Step::Compute { .. } => flyhigh_engine::clone_step(st),
                    Step::Copy { .. } => unreachable!(),
                })
            })
            .collect();
        let t_d = gpu.gpu_time_ms(&plan_d) / N as f64;
        let t_prep = gpu.gpu_time_ms(&rep(&s_d[0], N)) / N as f64;
        gpu.replay(&s_d);
        gpu.wait();
        let gd = gpu.readback(&o_d.buf, o_d.n());
        let maxd_d = r.iter().zip(&gd).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!(
            "  {:<24} {:.3} ms   max|d| vs fused = {:.2e}   (of which xshift3 prep {:.3} ms)",
            "fdirect (no gather)", t_d, maxd_d, t_prep
        );
    }
    for (mode, label) in [
        (0u32, "fused4 full"),
        (1, "fused4 no-gather"),
        (2, "fused4 no-Bstage"),
        (4, "fused4 no-MMA"),
        (3, "fused4 MMA only"),
        (7, "fused4 empty loop"),
    ] {
        let tc4 =
            ConvArgs { x: &x, weights: &bb4, bias: &bias, slope: &slope, cin, cout, kp: kp4, np: np4, cinp, stride: 1 };
        let (o, s) = gpu.conv_tc4(&tc4, mode);
        let t = gpu.gpu_time_ms(&rep(&s[0], N)) / N as f64;
        let mut extra = String::new();
        if mode == 0 {
            gpu.replay(&s);
            gpu.wait();
            let g = gpu.readback(&o.buf, o.n());
            let maxd = r.iter().zip(&g).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            extra = format!("   max|d| vs fused = {:.2e}", maxd);
        }
        println!("  {:<24} {:.3} ms{}", label, t, extra);
    }
}

// Portable-path microbench (any device): naive `conv` vs tiled `tconv`/`tconv32` on three real layer shapes,
// with max|d| between them. Runs when FH_BENCH=1 and the device has no cooperative matrix (or FH_NO_COOP=1).
fn tiled_bench(gpu: &Gpu) {
    let rep = |st: &Step, n: usize| -> Vec<Step> {
        (0..n)
            .map(|_| match st {
                Step::Compute { .. } => flyhigh_engine::clone_step(st),
                Step::Copy { .. } => unreachable!(),
            })
            .collect()
    };
    println!("\n-- portable conv microbench: naive vs tiled (GPU time per conv, best of 60 x N) --");
    for (cin, cout, h, w, stride, label) in [
        (3u32, 60u32, 736u32, 1280u32, 1u32, "enc0 3->60 @1280x736 s1"),
        (60, 120, 736, 1280, 2, "enc1 60->120 @1280x736 s2"),
        (235, 80, 92, 160, 1, "fine.r1 235->80 @160x92 s1"),
        (80, 24, 92, 160, 1, "head 80->24 @160x92 s1"),
    ] {
        let xin: Vec<f32> = (0..(cin * h * w)).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
        let x = Tensor { buf: gpu.buf(&xin), c: cin, h, w, c0: 0, scale: 1.0 };
        let wf: Vec<f32> =
            (0..(cout * cin * 9)).map(|i| (((i * 104729) % 1000) as f32 / 1000.0 - 0.5) * 0.05).collect();
        let wt = gpu.buf(&wf);
        let bias = gpu.buf(&vec![0.01f32; cout as usize]);
        let slope = gpu.buf(&vec![0.2f32; cout as usize]);
        let (bb, kp, np) = gpu.buildb(&wf, cin, cout);
        let n = if cin * cout * h * w > 100_000_000 { 4 } else { 20 };
        let naive =
            ConvArgs { x: &x, weights: &wt, bias: &bias, slope: &slope, cin, cout, kp: 0, np: 0, cinp: 0, stride };
        let (o_n, s_n) = gpu.conv(&naive, 1);
        let t_n = gpu.gpu_time_ms(&rep(&s_n[0], n)) / n as f64;
        let tiled = ConvArgs { x: &x, weights: &bb, bias: &bias, slope: &slope, cin, cout, kp, np, cinp: 0, stride };
        let (o_t, s_t) = gpu.conv_tiled(&tiled); // 1 step, or 2 with split-K
        let plan_t: Vec<Step> = (0..n).flat_map(|_| s_t.iter().flat_map(|st| rep(st, 1))).collect();
        let t_t = gpu.gpu_time_ms(&plan_t) / n as f64;
        gpu.replay(&s_n);
        gpu.replay(&s_t);
        gpu.wait();
        let (rn, rt) = (gpu.readback(&o_n.buf, o_n.n()), gpu.readback(&o_t.buf, o_t.n()));
        let maxd = rn.iter().zip(&rt).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let gflop = 2.0 * (cout as f64) * (cin as f64) * 9.0 * (o_t.h as f64) * (o_t.w as f64) / 1e9;
        println!(
            "  {:<30} naive {:>8.3} ms | tiled {:>7.3} ms ({:>5.1}x, {:>6.1} GFLOP/s, {} step(s)) | max|d| {:.2e}",
            label,
            t_n,
            t_t,
            t_n / t_t,
            gflop / (t_t / 1e3),
            s_t.len(),
            maxd
        );
    }
}

/// The conv microbenchmarks of this device's kernel class: fused tensor-core kernels where cooperative matrices exist,
/// else the portable tiled kernels against the naive conv.
pub fn run() -> Result<()> {
    let gpu = Gpu::new()?;
    if gpu.coop {
        fused_bench(&gpu);
    } else {
        tiled_bench(&gpu);
    }
    Ok(())
}
