// The Core API: options validate without a GPU; on a real GPU an Engine, a Model and a Session generate
// an in-between on their own slots, a host's own wgpu device drives the engine through `Engine::with_device`, and the
// reports say what was built. GPU tests are ignored by default (`cargo test --release -- --ignored`).
use flyhigh_engine::{Engine, EngineOptions, Internal, KernelClass, Model, Session, SessionOptions, StaticProtection};
use std::path::PathBuf;

const W: u32 = 256;
const H: u32 = 192;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("models")
}

#[test]
fn session_options_validate() {
    let o = SessionOptions::new(1920, 1080);
    assert!(o.validate().is_ok());
    assert_eq!(o.internal_size(), (856, 480), "the 480-line rung above 960 px");
    assert_eq!(SessionOptions::new(640, 480).internal_size(), (640, 480));
    let small = SessionOptions::new(32, 32);
    assert_eq!(small.validate().unwrap_err().field, "width");
    let k = SessionOptions { k: 4, ..SessionOptions::new(W, H) };
    assert_eq!(k.validate().unwrap_err().field, "k");
    let lin = SessionOptions { linear_t: true, ..SessionOptions::new(W, H) };
    assert_eq!(lin.validate().unwrap_err().field, "linear_t");
    let tiny = SessionOptions { internal: Internal::Size(16, 16), ..SessionOptions::new(W, H) };
    assert_eq!(tiny.validate().unwrap_err().field, "internal");
    let sp = SessionOptions {
        static_protection: Some(StaticProtection { frames: 0, tolerance: 4 }),
        ..SessionOptions::new(W, H)
    };
    assert_eq!(sp.validate().unwrap_err().field, "static_protection");
    let tex = SessionOptions { fast_render: true, linear_t: true, ..SessionOptions::new(1280, 720) }.tex_options();
    assert!(tex.fast_render && tex.linear_t && (tex.iw, tex.ih) == (856, 480));
}

#[test]
fn model_loads_and_describes_itself() {
    let m = Model::from_dir(&model_dir()).expect("the shipped model");
    let i = m.info();
    assert!(i.graph_ops > 50 && i.tensors > 50, "{i:?}");
    assert_eq!(i.format_version, 1);
    assert!(i.render_sampling == "bicubic" || i.render_sampling == "bilinear");
    let bad = Model::from_bytes(b"{\"ops\": []}", b"{\"tensors\": {}}", Vec::new());
    assert!(bad.is_err(), "an empty graph is refused");
}

fn engine() -> Option<Engine> {
    match Engine::new(&EngineOptions::default()) {
        Ok(e) => Some(e),
        Err(e) if format!("{e:#}").contains("adapter") || format!("{e:#}").contains("device") => {
            eprintln!("skipped: {e:#}");
            None
        }
        Err(e) => panic!("engine: {e:#}"),
    }
}

// a soft pattern shifted `dx` px to the right: the in-between of two shifts should sit between them
fn frame(dx: f32) -> Vec<u8> {
    let mut v = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let u = (x as f32 - dx) * 0.07;
            let w = y as f32 * 0.05;
            let o = ((y * W + x) * 4) as usize;
            v[o] = (128.0 + 100.0 * u.sin()) as u8;
            v[o + 1] = (128.0 + 100.0 * (u * 0.7 + w).cos()) as u8;
            v[o + 2] = (128.0 + 90.0 * (u + w).sin()) as u8;
            v[o + 3] = 255;
        }
    }
    v
}

fn mae(a: &[u8], b: &[u8]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).sum::<f64>() / a.len() as f64
}

fn check_inbetween(e: &Engine, s: &mut Session) {
    let (a, b) = (s.add_input(), s.add_input());
    let out = s.add_output();
    s.write_rgba(a, &frame(0.0)).unwrap();
    s.write_rgba(b, &frame(8.0)).unwrap();
    s.pair(a, b).unwrap();
    s.mid(0.5, out).unwrap();
    let got = s.read_rgba(out).unwrap().block(e.gpu()).unwrap();
    let (f0, f1, mid) = (frame(0.0), frame(8.0), frame(4.0));
    let (to_mid, to_f0, to_f1) = (mae(&got, &mid), mae(&got, &f0), mae(&got, &f1));
    eprintln!("MAE to the true middle {to_mid:.2}, to f0 {to_f0:.2}, to f1 {to_f1:.2}");
    assert!(to_mid < to_f0 && to_mid < to_f1 && to_mid < 6.0, "the in-between lies between its neighbours");
}

#[test]
#[ignore]
fn engine_model_session_generate_an_inbetween() {
    let Some(e) = engine() else { return };
    let m = Model::from_dir(&model_dir()).unwrap();
    let info = e.info();
    eprintln!("{info:?}");
    assert!(matches!(info.class, KernelClass::TcF16 | KernelClass::PortableF16 | KernelClass::PortableF32));
    let mut s = Session::new(&e, &m, &SessionOptions::new(W, H)).unwrap();
    check_inbetween(&e, &mut s);
    let r = s.report();
    assert_eq!((r.width, r.height, r.inputs, r.outputs, r.k), (W, H, 2, 1, 1));
    assert!(r.pair_steps > 10 && r.mid_steps > 3, "{r:?}");
    let er = e.report();
    assert!(er.kernels_compiled > 10 && er.tensor_bytes > 0, "{er:?}");
    assert!(s.generate(0, 1, 0.25, 0).is_ok());
    assert!(s.mid(0.5, 7).is_err(), "a slot the session does not have");
    // a second session of another size on the same engine shares the compiled kernels
    let before = e.report().kernels_compiled;
    let mut s2 = Session::new(&e, &m, &SessionOptions::new(320, 240)).unwrap();
    s2.warm_up().unwrap();
    assert!(e.report().kernels_compiled <= before + 4, "the same kernel set, a few size-specific variants at most");
}

// the moving pattern with a sharp vertical bar (a HUD element the host knows the place of) at column `bar`
fn frame_with_bar(dx: f32, bar: u32) -> Vec<u8> {
    let mut v = frame(dx);
    for y in 40..H - 40 {
        for x in bar..bar + 3 {
            let o = ((y * W + x) * 4) as usize;
            v[o..o + 3].fill(255);
        }
    }
    v
}

#[test]
#[ignore]
fn the_protection_mask_takes_pixels_from_the_sources() {
    let Some(e) = engine() else { return };
    let m = Model::from_dir(&model_dir()).unwrap();
    // the bar moves with the pattern: interpolation puts one bar at 104, the sources' mean two half bars at 100 and 108
    let (img_a, img_b) = (frame_with_bar(0.0, 100), frame_with_bar(8.0, 108));
    let mean: Vec<u8> = img_a.iter().zip(&img_b).map(|(x, y)| ((*x as u32 + *y as u32).div_ceil(2)) as u8).collect();
    let rect = |x: u32, y: u32| (90..130).contains(&x) && (30..H - 30).contains(&y);
    let far = |x: u32, y: u32| !((82..138).contains(&x) && (22..H - 22).contains(&y));
    let run = |protection_mask: bool| -> Vec<u8> {
        let opts = SessionOptions { protection_mask, ..SessionOptions::new(W, H) };
        let mut s = Session::new(&e, &m, &opts).unwrap();
        let (a, b, out) = (s.add_input(), s.add_input(), s.add_output());
        if protection_mask {
            let mask: Vec<u8> = (0..H).flat_map(|y| (0..W).map(move |x| if rect(x, y) { 255 } else { 0 })).collect();
            s.write_mask(&mask).unwrap();
        }
        s.write_rgba(a, &img_a).unwrap();
        s.write_rgba(b, &img_b).unwrap();
        s.pair(a, b).unwrap();
        s.mid(0.5, out).unwrap();
        s.read_rgba(out).unwrap().block(e.gpu()).unwrap()
    };
    let (plain, masked) = (run(false), run(true));
    let mae_where = |img: &[u8], reference: &[u8], where_: &dyn Fn(u32, u32) -> bool| -> f64 {
        let (mut sum, mut n) = (0.0, 0.0);
        for y in 0..H {
            for x in 0..W {
                if where_(x, y) {
                    let o = ((y * W + x) * 4) as usize;
                    sum += (0..3).map(|c| (img[o + c] as f64 - reference[o + c] as f64).abs()).sum::<f64>();
                    n += 3.0;
                }
            }
        }
        sum / n
    };
    let inside_masked = mae_where(&masked, &mean, &rect);
    let inside_plain = mae_where(&plain, &mean, &rect);
    let outside = mae_where(&masked, &plain, &far);
    eprintln!("MAE to the sources' mean inside the mask: masked {inside_masked:.2}, plain {inside_plain:.2}; masked to plain far from it {outside:.2}");
    assert!(inside_masked < 1.0, "masked pixels are the sources' mean");
    assert!(inside_plain > 4.0, "without the mask the bar is interpolated: another picture");
    assert!(outside < 1.0, "far from the mask the two sessions agree");
    let plain_session = Session::new(&e, &m, &SessionOptions::new(W, H)).unwrap();
    assert!(plain_session.write_mask(&vec![0; (W * H) as usize]).is_err(), "no mask without the option");
}

#[test]
#[ignore]
fn a_host_device_drives_the_engine() {
    // the host's own wgpu stack, as a game or a player would have it
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("skipped: no adapter ({e:?})");
            return;
        }
    };
    let mut features = wgpu::Features::empty();
    if adapter.features().contains(wgpu::Features::SHADER_F16) {
        features |= wgpu::Features::SHADER_F16;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("host"),
        required_features: features,
        required_limits: wgpu::Limits::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        trace: wgpu::Trace::Off,
    }))
    .expect("device");
    let e = Engine::with_device(instance, adapter, device, queue, &EngineOptions::default()).unwrap();
    let info = e.info();
    eprintln!("{info:?}");
    assert!(!info.tensor_cores, "the host did not enable cooperative matrices, so the engine does not use them");
    assert!(!info.timestamps, "nor timestamp queries");
    let m = Model::from_dir(&model_dir()).unwrap();
    let mut s = Session::new(&e, &m, &SessionOptions::new(W, H)).unwrap();
    check_inbetween(&e, &mut s);
    assert_eq!(s.pair_timed(0, 1).unwrap(), None, "no timestamp queries on this device");
}

#[test]
#[ignore]
fn fast_render_with_linear_t_on_own_slots() {
    let Some(e) = engine() else { return };
    let m = Model::from_dir(&model_dir()).unwrap();
    let opts = SessionOptions { fast_render: true, linear_t: true, ..SessionOptions::new(1280, 720) };
    let mut s = Session::new(&e, &m, &opts).unwrap();
    let r = s.report();
    assert!(r.fast_render && r.linear_t && (r.internal_width, r.internal_height) == (856, 480), "{r:?}");
    let (a, b) = (s.add_input(), s.add_input());
    let out = s.add_output();
    let f = |dx: f32| -> Vec<u8> {
        let mut v = vec![0u8; 1280 * 720 * 4];
        for y in 0..720u32 {
            for x in 0..1280u32 {
                let o = ((y * 1280 + x) * 4) as usize;
                let u = (x as f32 - dx) * 0.05;
                v[o] = (128.0 + 100.0 * u.sin()) as u8;
                v[o + 1] = (128.0 + 90.0 * (u * 0.7 + y as f32 * 0.03).cos()) as u8;
                v[o + 2] = (128.0 + 80.0 * (y as f32 * 0.04).sin()) as u8;
                v[o + 3] = 255;
            }
        }
        v
    };
    s.write_rgba(a, &f(0.0)).unwrap();
    s.write_rgba(b, &f(16.0)).unwrap();
    s.pair(a, b).unwrap();
    for (t, dx) in [(0.25, 4.0), (0.5, 8.0), (0.75, 12.0)] {
        s.mid(t, out).unwrap();
        let got = s.read_rgba(out).unwrap().block(e.gpu()).unwrap();
        let err = mae(&got, &f(dx));
        eprintln!("phase {t}: MAE {err:.2}");
        assert!(err < 4.0, "linear t stays close on linear motion at phase {t}: {err}");
    }
    let (pair, mid) = (s.pair_timed(a, b).unwrap(), s.mid_timed(0.5, out).unwrap());
    eprintln!("GPU time: pair {pair:?} ms, mid {mid:?} ms");
    if let (Some(p), Some(m)) = (pair, mid) {
        assert!(m < p, "a linear-t render is far cheaper than the pair");
    }
}
