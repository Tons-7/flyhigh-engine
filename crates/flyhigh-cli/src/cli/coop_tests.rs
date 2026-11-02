//! Tensor-core probes: cooperative-matrix availability, the GEMM and conv kernels against CPU references, shared-memory staging.

use bytemuck::{Pod, Zeroable};
use flyhigh_engine::*;
use wgpu::util::DeviceExt;

pub fn coop_test() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("adapter");
    if !adapter.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        println!("COOP TEST: EXPERIMENTAL_COOPERATIVE_MATRIX not supported here");
        return;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("coop"),
        required_features: wgpu::Features::SHADER_F16 | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        trace: wgpu::Trace::Off,
    }))
    .expect("device");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("coop_test"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../../../shaders/coop_test.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("coop_test"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut ab = vec![half::f16::from_f32(0.0).to_bits(); 256];
    let mut bb = vec![0u16; 256];
    for i in 0..16 {
        ab[i * 16 + i] = half::f16::from_f32(2.0).to_bits();
    } // A = 2*I
    for (i, b) in bb.iter_mut().enumerate() {
        *b = half::f16::from_f32((i % 7) as f32).to_bits();
    } // B = ramp
    let mk = |data: &[u16], usage| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage,
        })
    };
    let abuf = mk(&ab, wgpu::BufferUsages::STORAGE);
    let bbuf = mk(&bb, wgpu::BufferUsages::STORAGE);
    let cbuf = mk(&vec![0u16; 256], wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 512,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: abuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: cbuf.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        cp.set_pipeline(&pipeline);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&cbuf, 0, &stage, 0, 512);
    queue.submit(Some(enc.finish()));
    let slice = stage.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
    rx.recv().unwrap().unwrap();
    let mapped = slice.get_mapped_range().unwrap();
    let got: Vec<f32> =
        bytemuck::cast_slice::<u8, u16>(&mapped).iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
    drop(mapped);
    let af: Vec<f32> = ab.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
    let bf: Vec<f32> = bb.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
    let mut want = vec![0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            let mut s = 0.0;
            for k in 0..16 {
                s += af[i * 16 + k] * bf[k * 16 + j];
            }
            want[i * 16 + j] = s;
        }
    }
    let maxd = got.iter().zip(&want).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    println!("\n==== TENSOR-CORE PROOF (cooperative matrix: WGSL -> SPIR-V -> Vulkan) ====");
    println!(
        "  16x16x16 f16 matmul on {}: maxD vs CPU = {:.3} -> {}",
        adapter.get_info().name,
        maxd,
        if maxd < 0.5 { "WORKS -- Tensor Cores run from WGSL" } else { "ran but wrong" }
    );
    println!("========================================================================\n");
}

pub fn coop_gemm_test() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("adapter");
    if !adapter.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gemm"),
        required_features: wgpu::Features::SHADER_F16 | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        trace: wgpu::Trace::Off,
    }))
    .expect("device");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("coop_gemm"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../../../../shaders/coop_gemm.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("coop_gemm"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct Dims {
        m: u32,
        n: u32,
        k: u32,
        _p: u32,
    }

    println!("==== TENSOR-CORE GEMM (cooperative matrix, f16 x f16 -> f32) ====");
    let run = |m: u32, n: u32, k: u32, validate: bool| {
        let af: Vec<f32> = (0..(m * k)).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let bf: Vec<f32> = (0..(k * n)).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
        let a16: Vec<u16> = af.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
        let b16: Vec<u16> = bf.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
        let si = |data: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let abuf = si(bytemuck::cast_slice(&a16));
        let bbuf = si(bytemuck::cast_slice(&b16));
        let zbuf = si(bytemuck::cast_slice(&vec![0f32; 256]));
        let cbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (m * n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&Dims { m, n, k, _p: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: abuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: bbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: zbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: cbuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dbuf.as_entire_binding() },
            ],
        });
        let dispatch = || {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cp =
                    enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                cp.set_pipeline(&pipeline);
                cp.set_bind_group(0, &bg, &[]);
                cp.dispatch_workgroups(n / 16, m / 16, 1);
            }
            queue.submit(Some(enc.finish()));
            device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        };
        for _ in 0..10 {
            dispatch();
        }
        let mut ts: Vec<f64> = (0..50)
            .map(|_| {
                let s = std::time::Instant::now();
                dispatch();
                s.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ms = ts[0];
        let gflops = 2.0 * (m as f64) * (n as f64) * (k as f64) / (ms / 1000.0) / 1e9;
        let mut verdict = String::new();
        if validate {
            let stage = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (m * n * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.copy_buffer_to_buffer(&cbuf, 0, &stage, 0, (m * n * 4) as u64);
            queue.submit(Some(enc.finish()));
            let slice = stage.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
            rx.recv().unwrap().unwrap();
            let got: Vec<f32> = {
                let mp = slice.get_mapped_range().unwrap();
                bytemuck::cast_slice::<u8, f32>(&mp).to_vec()
            };
            stage.unmap();
            let mut maxd = 0f32;
            for i in 0..(m as usize) {
                for j in 0..(n as usize) {
                    let mut s = 0f32;
                    for kk in 0..(k as usize) {
                        s += af[i * k as usize + kk] * bf[kk * n as usize + j];
                    }
                    maxd = maxd.max((s - got[i * n as usize + j]).abs());
                }
            }
            verdict = format!("  maxD vs CPU = {:.4} -> {}", maxd, if maxd < 0.1 { "CORRECT" } else { "WRONG" });
        }
        println!("  {}x{}x{}: {:.3} ms  {:.0} GFLOP/s (f16){}", m, n, k, ms, gflops, verdict);
    };
    run(256, 256, 256, true);
    run(1024, 1024, 1024, false);
    run(2048, 2048, 2048, false);
    println!("================================================================\n");
}

pub fn conv_tc_test() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("adapter");
    if !adapter.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return;
    }
    let (dev, q) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("convtc"),
        required_features: wgpu::Features::SHADER_F16 | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        trace: wgpu::Trace::Off,
    }))
    .expect("device");
    let pipe = |src: &str| {
        let m = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &m,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let p_im2col = pipe(include_str!("../../../../shaders/im2col.wgsl"));
    let p_gemm = pipe(include_str!("../../../../shaders/coop_gemm.wgsl"));
    let p_epi = pipe(include_str!("../../../../shaders/conv_epilogue.wgsl"));
    let p_naive = pipe(include_str!("../../../../shaders/conv.wgsl"));
    let p_implicit = pipe(include_str!("../../../../shaders/conv_implicit.wgsl"));
    let p_implicit2 = pipe(include_str!("../../../../shaders/conv_implicit2.wgsl"));

    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct Im2colP {
        cin: u32,
        h: u32,
        w: u32,
        hout: u32,
        wout: u32,
        stride: u32,
        pad: u32,
        k: u32,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct GemmD {
        m: u32,
        n: u32,
        k: u32,
        _p: u32,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Pod, Zeroable)]
    struct ConvParams {
        cin: u32,
        cout: u32,
        h: u32,
        w: u32,
        hout: u32,
        wout: u32,
        stride: u32,
        pad: u32,
        act: u32,
        _a: u32,
        _b: u32,
        _c: u32,
    }

    let (cin, cout, h, w): (u32, u32, u32, u32) = (208, 64, 46, 80);
    let (hout, wout, mm, kk, nn) = (h, w, h * w, cin * 9, cout); // stride 1 pad 1; all multiples of 16

    let inpf: Vec<f32> = (0..(cin * h * w)).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let wgtf: Vec<f32> = (0..(cout * cin * 9)).map(|i| ((i % 13) as f32 - 6.0) * 0.04).collect(); // [cout,cin,3,3]
    let biasf: Vec<f32> = (0..cout).map(|i| (i as f32 % 5.0 - 2.0) * 0.1).collect();
    let slopef: Vec<f32> = (0..cout).map(|_| 0.1f32).collect();
    // B [K,N]: B[k*N+co] = wgt[co*K + k]
    let mut bmat = vec![0f32; (kk * nn) as usize];
    for co in 0..cout as usize {
        for k in 0..kk as usize {
            bmat[k * nn as usize + co] = wgtf[co * kk as usize + k];
        }
    }
    let h16 = |v: &[f32]| -> Vec<u16> { v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect() };
    let sb = |data: &[u8], rw| {
        dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: data, usage: rw })
    };
    let st = wgpu::BufferUsages::STORAGE;
    let stc = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    let inp = sb(bytemuck::cast_slice(&h16(&inpf)), st);
    let abuf = dev.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (mm * kk * 2) as u64,
        usage: st,
        mapped_at_creation: false,
    });
    let bbuf = sb(bytemuck::cast_slice(&h16(&bmat)), st);
    let zero = sb(bytemuck::cast_slice(&vec![0f32; 256]), st);
    let cbuf = dev.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (mm * nn * 4) as u64,
        usage: st,
        mapped_at_creation: false,
    });
    let bias = sb(bytemuck::cast_slice(&h16(&biasf)), st);
    let slope = sb(bytemuck::cast_slice(&h16(&slopef)), st);
    let outb = dev.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (cout * hout * wout * 2) as u64,
        usage: stc,
        mapped_at_creation: false,
    });
    let wgt = sb(bytemuck::cast_slice(&h16(&wgtf)), st);
    let outn = dev.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (cout * hout * wout * 2) as u64,
        usage: stc,
        mapped_at_creation: false,
    });
    let ub = |b: &[u8]| {
        dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: b,
            usage: wgpu::BufferUsages::UNIFORM,
        })
    };
    let u_im = ub(bytemuck::bytes_of(&Im2colP { cin, h, w, hout, wout, stride: 1, pad: 1, k: kk }));
    let u_gm = ub(bytemuck::bytes_of(&GemmD { m: mm, n: nn, k: kk, _p: 0 }));
    let u_ep = ub(bytemuck::bytes_of(&EpilogueParams { cout, hout, wout, npad: nn }));
    let u_nv = ub(bytemuck::bytes_of(&ConvParams {
        cin,
        cout,
        h,
        w,
        hout,
        wout,
        stride: 1,
        pad: 1,
        act: 1,
        _a: 0,
        _b: 0,
        _c: 0,
    }));
    let u_imp = ub(bytemuck::bytes_of(&ImplicitGemmParams {
        cin,
        h,
        w,
        hout,
        wout,
        stride: 1,
        pad: 1,
        k: kk,
        kp: kk,
        n: nn,
        np: nn,
        m: mm,
        cinp: 0,
        ksplit: 0,
        residual: 0,
        _pad0: 0,
    }));
    let bg = |pl: &wgpu::ComputePipeline, res: &[&wgpu::Buffer]| {
        let e: Vec<_> = res
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pl.get_bind_group_layout(0),
            entries: &e,
        })
    };
    let bg_im = bg(&p_im2col, &[&inp, &abuf, &u_im]);
    let bg_gm = bg(&p_gemm, &[&abuf, &bbuf, &zero, &cbuf, &u_gm]);
    let bg_ep = bg(&p_epi, &[&cbuf, &bias, &slope, &outb, &u_ep]);
    let bg_nv = bg(&p_naive, &[&inp, &wgt, &bias, &slope, &outn, &u_nv]);
    let bg_imp = bg(&p_implicit, &[&inp, &bbuf, &zero, &cbuf, &u_imp]);
    let bg_imp2 = bg(&p_implicit2, &[&inp, &bbuf, &zero, &cbuf, &u_imp]);
    let workgroups_2d = |a: u32, b: u32, c: u32| (a.div_ceil(8), b.div_ceil(8), c);

    let tc = || {
        let mut e = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut c = e.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            c.set_pipeline(&p_im2col);
            c.set_bind_group(0, &bg_im, &[]);
            let g = workgroups_2d(wout, hout, 1);
            c.dispatch_workgroups(g.0, g.1, g.2);
        }
        {
            let mut c = e.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            c.set_pipeline(&p_gemm);
            c.set_bind_group(0, &bg_gm, &[]);
            c.dispatch_workgroups(nn / 16, mm / 16, 1);
        }
        {
            let mut c = e.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            c.set_pipeline(&p_epi);
            c.set_bind_group(0, &bg_ep, &[]);
            let g = workgroups_2d(wout, hout, cout);
            c.dispatch_workgroups(g.0, g.1, g.2);
        }
        q.submit(Some(e.finish()));
        dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
    };
    let tci = || {
        let mut e = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut c = e.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            c.set_pipeline(&p_implicit);
            c.set_bind_group(0, &bg_imp, &[]);
            c.dispatch_workgroups(nn / 16, mm / 16, 1);
        }
        {
            let mut c = e.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            c.set_pipeline(&p_epi);
            c.set_bind_group(0, &bg_ep, &[]);
            let g = workgroups_2d(wout, hout, cout);
            c.dispatch_workgroups(g.0, g.1, g.2);
        }
        q.submit(Some(e.finish()));
        dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
    };
    let naive = || {
        let mut e = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut c = e.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            c.set_pipeline(&p_naive);
            c.set_bind_group(0, &bg_nv, &[]);
            let g = workgroups_2d(wout, hout, cout);
            c.dispatch_workgroups(g.0, g.1, g.2);
        }
        q.submit(Some(e.finish()));
        dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
    };

    let tci2 = || {
        let mut e = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut c = e.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            c.set_pipeline(&p_implicit2);
            c.set_bind_group(0, &bg_imp2, &[]);
            c.dispatch_workgroups(nn / 64, mm / 16, 1);
        }
        {
            let mut c = e.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            c.set_pipeline(&p_epi);
            c.set_bind_group(0, &bg_ep, &[]);
            let g = workgroups_2d(wout, hout, cout);
            c.dispatch_workgroups(g.0, g.1, g.2);
        }
        q.submit(Some(e.finish()));
        dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
    };
    let rb = |buf: &wgpu::Buffer, n: usize| -> Vec<f32> {
        let s = dev.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (n * 2) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut e = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        e.copy_buffer_to_buffer(buf, 0, &s, 0, (n * 2) as u64);
        q.submit(Some(e.finish()));
        let sl = s.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        sl.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        rx.recv().unwrap().unwrap();
        let mp = sl.get_mapped_range().unwrap();
        let v: Vec<f32> =
            bytemuck::cast_slice::<u8, u16>(&mp).iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();
        drop(mp);
        s.unmap();
        v
    };

    tc();
    tci2();
    tci();
    naive();
    let got = rb(&outb, (cout * hout * wout) as usize);
    // CPU reference
    let mut want = vec![0f32; (cout * hout * wout) as usize];
    for co in 0..cout as usize {
        for oy in 0..hout as usize {
            for ox in 0..wout as usize {
                let mut s = biasf[co];
                for ci in 0..cin as usize {
                    for ky in 0..3usize {
                        for kx in 0..3usize {
                            let yi = oy as i32 + ky as i32 - 1;
                            let xi = ox as i32 + kx as i32 - 1;
                            if yi >= 0 && yi < h as i32 && xi >= 0 && xi < w as i32 {
                                s += inpf[(ci * h as usize + yi as usize) * w as usize + xi as usize]
                                    * wgtf[((co * cin as usize + ci) * 3 + ky) * 3 + kx];
                            }
                        }
                    }
                }
                if s <= 0.0 {
                    s *= slopef[co];
                }
                want[(co * hout as usize + oy) * wout as usize + ox] = s;
            }
        }
    }
    let maxd = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);

    let timeit = |f: &dyn Fn()| -> f64 {
        for _ in 0..10 {
            f();
        }
        let mut t: Vec<f64> = (0..50)
            .map(|_| {
                let s = std::time::Instant::now();
                f();
                s.elapsed().as_secs_f64() * 1000.0
            })
            .collect();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        t[0]
    };
    let got2 = {
        tci2();
        rb(&outb, (cout * hout * wout) as usize)
    };
    let md2 = got2.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let tc_ms = timeit(&tc);
    let tci_ms = timeit(&tci);
    let tci2_ms = timeit(&tci2);
    let nv_ms = timeit(&naive);
    println!("==== TENSOR-CORE CONV (im2col + coop-GEMM + epilogue) vs naive ====");
    println!(
        "  conv {}x{} @{}x{}: maxD vs CPU = {:.4} -> {}",
        cin,
        cout,
        h,
        w,
        maxd,
        if maxd < 0.1 { "CORRECT" } else { "WRONG" }
    );
    println!("  implicit2 maxD = {:.4} -> {}", md2, if md2 < 0.1 { "CORRECT" } else { "WRONG" });
    println!(
        "  naive {:.3} | im2col-TC {:.3} ({:.1}x) | IMPLICIT {:.3} ({:.1}x) | IMPLICIT2 {:.3} ({:.1}x)",
        nv_ms,
        tc_ms,
        nv_ms / tc_ms,
        tci_ms,
        nv_ms / tci_ms,
        tci2_ms,
        nv_ms / tci2_ms
    );
    println!("==================================================================\n");
}

pub fn coop_shared_test() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .expect("adapter");
    if !adapter.features().contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return;
    }
    let (dev, q) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("sh"),
        required_features: wgpu::Features::SHADER_F16 | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        trace: wgpu::Trace::Off,
    }))
    .expect("device");
    let m = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("../../../../shaders/coop_shared.wgsl").into()),
    });
    let pl = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &m,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let af: Vec<f32> = (0..256).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let bf: Vec<f32> = (0..256).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();
    let h16 = |v: &[f32]| -> Vec<u16> { v.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect() };
    let sb = |d: &[u8]| {
        dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: d,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let srcb = sb(bytemuck::cast_slice(&h16(&af)));
    let bb = sb(bytemuck::cast_slice(&h16(&bf)));
    let zb = sb(bytemuck::cast_slice(&vec![0f32; 256]));
    let db = dev.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pl.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: srcb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: bb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: zb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: db.as_entire_binding() },
        ],
    });
    let mut e = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut c = e.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        c.set_pipeline(&pl);
        c.set_bind_group(0, &bg, &[]);
        c.dispatch_workgroups(1, 1, 1);
    }
    let st = dev.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1024,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    e.copy_buffer_to_buffer(&db, 0, &st, 0, 1024);
    q.submit(Some(e.finish()));
    let sl = st.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    sl.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    dev.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
    rx.recv().unwrap().unwrap();
    let got: Vec<f32> = {
        let mp = sl.get_mapped_range().unwrap();
        bytemuck::cast_slice::<u8, f32>(&mp).to_vec()
    };
    let mut want = vec![0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            let mut s = 0f32;
            for k in 0..16 {
                s += af[i * 16 + k] * bf[k * 16 + j];
            }
            want[i * 16 + j] = s;
        }
    }
    let maxd = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("==== IMPLICIT-GEMM KEY PROBE: coopLoad from SHARED memory ====");
    println!(
        "  A-from-shared @ B -> C: maxD vs CPU = {:.4} -> {}",
        maxd,
        if maxd < 0.1 { "WORKS -- implicit GEMM is viable" } else { "ran but wrong" }
    );
    println!(
        "==============================================================
"
    );
}
