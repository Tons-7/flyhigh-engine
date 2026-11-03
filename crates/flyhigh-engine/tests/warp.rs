// The camera warp's input calibration on synthetic play: a camera turned by the mouse (flicks and tracking), the picture
// showing the input 45 ms late, measured image shifts with noise and outliers, and a menu in which the mouse moves but
// the camera does not.
use flyhigh_engine::warp::{CalibOptions, CameraWarp, InputCalibrator, WarpOptions};

const FPS: f64 = 60.0;
const LAG: f64 = 45.0;
const GAIN: f64 = 0.138; // image pixels per count

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn n(&mut self) -> f64 {
        (0..6).map(|_| self.f()).sum::<f64>() - 3.0
    }
}

/// Mouse counts per millisecond over `secs`, and whether the camera follows them at each millisecond.
fn play(secs: f64, menu: (f64, f64)) -> (Vec<f64>, Vec<bool>) {
    let mut r = Lcg(7);
    let n = (secs * 1000.0) as usize;
    let mut rate = vec![0.0; n];
    for _ in 0..(secs * 1.6) as usize {
        let t0 = 500.0 + r.f() * (secs * 1000.0 - 1000.0);
        let d = 40.0 + 80.0 * r.f();
        let amp = if r.f() < 0.5 { -1.0 } else { 1.0 } * (2.0 + 9.0 * r.f());
        for (i, v) in rate.iter_mut().enumerate() {
            let t = i as f64;
            if t >= t0 && t < t0 + d {
                *v += amp * (std::f64::consts::PI * (t - t0) / d).sin();
            }
        }
    }
    for v in rate.iter_mut() {
        *v += 0.2 * r.n();
    }
    // a flick right after the menu opens: the moment a wrong warp would show
    if menu.0 < secs * 1000.0 {
        for (i, v) in rate.iter_mut().enumerate() {
            let t = i as f64 - (menu.0 + 300.0);
            if (0.0..100.0).contains(&t) {
                *v += 8.0 * (std::f64::consts::PI * t / 100.0).sin();
            }
        }
    }
    let follows = (0..n).map(|i| !((i as f64) >= menu.0 && (i as f64) < menu.1)).collect();
    (rate, follows)
}

/// Feeds the calibrator; `captured(t)` is what the host reports. Returns per frame (t, on, the shift the warp would
/// apply to the newest frame right then).
fn run(secs: f64, menu: (f64, f64), captured: impl Fn(f64) -> bool) -> (InputCalibrator, Vec<(f64, bool, f64)>) {
    let (counts, follows) = play(secs, menu);
    let mut cam = vec![0.0; counts.len() + 1];
    for i in 0..counts.len() {
        cam[i + 1] = cam[i] + if follows[i] { counts[i] * GAIN } else { 0.0 };
    }
    let cam_at = |t: f64| cam[(t.max(0.0) as usize).min(cam.len() - 1)];
    let mut r = Lcg(11);
    let mut c = InputCalibrator::new(CalibOptions::default());
    let mut log = Vec::new();
    let mut prev_shown: Option<f64> = None;
    let mut next_frame = 0.0;
    for (i, &k) in counts.iter().enumerate() {
        let t = i as f64;
        c.set_captured(captured(t));
        c.mouse(t, k, 0.0);
        if t >= next_frame {
            next_frame += 1000.0 / FPS;
            let shown = cam_at(t - LAG);
            let shift = prev_shown.map(|p| {
                let mut s = shown - p + 0.4 * r.n();
                if r.f() < 0.03 {
                    s += 15.0 * r.n();
                }
                (s, 0.0)
            });
            prev_shown = Some(shown);
            c.frame(t, shift);
            let applied = c.shift(t, t + 1000.0 / FPS).map_or(0.0, |s| s.0);
            log.push((t, c.report().on, applied));
        }
    }
    (c, log)
}

fn share_on(log: &[(f64, bool, f64)], a: f64, b: f64) -> f64 {
    let s: Vec<bool> = log.iter().filter(|(t, _, _)| *t >= a && *t < b).map(|(_, o, _)| *o).collect();
    s.iter().filter(|o| **o).count() as f64 / s.len().max(1) as f64
}

#[test]
fn the_calibrator_finds_the_lag_and_the_gain_of_a_mouse_turned_camera() {
    let (c, log) = run(12.0, (1e9, 1e9), |_| true);
    let r = c.report();
    eprintln!("lag {:.1} ms, gain {:.4} px per count (true {GAIN}), r {:.3}", r.lag_ms, r.gain_x, r.r);
    assert!((r.lag_ms - LAG).abs() <= 4.0, "lag {}", r.lag_ms);
    assert!((r.gain_x / GAIN - 1.0).abs() < 0.05, "gain {}", r.gain_x);
    assert!(share_on(&log, 3000.0, 12000.0) > 0.95, "on while playing");
}

#[test]
fn a_released_mouse_turns_the_warp_off_at_once_and_a_menu_without_it_soon() {
    // the host reports the cursor: off for the whole menu
    let (_, log) = run(16.0, (8000.0, 10000.0), |t| !(8000.0..10000.0).contains(&t));
    assert_eq!(share_on(&log, 8000.0, 10000.0), 0.0, "never on with a released mouse");
    assert!(share_on(&log, 12500.0, 16000.0) > 0.9, "and back on after the menu");
    // a host that cannot tell: the picture not following the flick turns it off within the lag and a few frames
    let (_, log) = run(16.0, (8000.0, 10000.0), |_| true);
    let wrong: Vec<f64> =
        log.iter().filter(|(t, _, s)| (8000.0..10000.0).contains(t) && s.abs() > 3.0).map(|e| e.2).collect();
    eprintln!(
        "menu with the mouse still captured: {} frames shifted by more than 3 px, worst {:.0} px",
        wrong.len(),
        wrong.iter().fold(0.0f64, |a, b| a.max(b.abs()))
    );
    assert!(wrong.len() <= 7, "about the lag plus three frames of a wrong shift, then off");
    assert!(share_on(&log, 9000.0, 10000.0) == 0.0, "and off for the rest of the menu");
    assert!(share_on(&log, 12500.0, 16000.0) > 0.8);
}

#[test]
fn the_shift_is_the_counts_since_the_input_the_frame_shows() {
    let (c, _) = run(8.0, (1e9, 1e9), |_| true);
    let r = c.report();
    let (cx, _) = c.counts_between(7900.0 - r.lag_ms, 7990.0);
    let (sx, _) = c.shift(7900.0, 7990.0).expect("on");
    assert!((sx - r.gain_x * cx).abs() < 1e-6);
    assert!(sx.abs() <= CalibOptions::default().max_shift_px);
}

// --- on a real GPU (ignored by default, like gpu_smoke: `cargo test --release -- --ignored`) ---

const W: u32 = 640;
const H: u32 = 360;
/// The weapon in hand: a block in the lower right that stays put while the world pans.
const LOCK: (u32, u32, u32, u32) = (440, 230, 600, 360);

fn engine() -> Option<flyhigh_engine::api::Engine> {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..").join("models");
    match flyhigh_engine::api::Engine::load(&dir) {
        Ok(e) => Some(e),
        Err(e) if format!("{e:#}").contains("shader-f16") || format!("{e:#}").contains("adapter") => {
            eprintln!("skipped: {e:#}");
            None
        }
        Err(e) => panic!("engine: {e:#}"),
    }
}

fn world(x: f64, y: f64) -> [f64; 3] {
    let u = x * 0.043;
    let v = y * 0.061;
    [
        128.0 + 60.0 * (u + 0.7 * v).sin() + 40.0 * (2.3 * u - 1.1 * v).cos(),
        128.0 + 70.0 * (1.7 * u).cos() * (0.9 * v).sin() + 30.0 * (3.1 * v + u).sin(),
        128.0 + 50.0 * (0.6 * u - 1.9 * v).sin() + 45.0 * (2.9 * u + 0.3 * v).cos(),
    ]
}

fn focal() -> f64 {
    0.5 * W as f64 / (53.0f64).to_radians().tan()
}

/// The world seen by a camera turned by `turn` (pixels at the centre: yaw and pitch times the focal length of the
/// warp's default 106 degree view; positive moves the content right and down), as a game renders it, with the
/// screen-locked block drawn over it. The same turn as warp_render.wgsl's.
fn frame(turn: (f64, f64)) -> Vec<u8> {
    let f = focal();
    let (ty, tp) = (turn.0 / f, turn.1 / f);
    let mut v = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let o = ((y * W + x) * 4) as usize;
            let locked = x >= LOCK.0 && x < LOCK.2 && y >= LOCK.1 && y < LOCK.3;
            let c = if locked {
                // not periodic: a periodic weapon would match itself shifted by a period
                let (u, v) = (x as f64 * 0.05, y as f64 * 0.07);
                let k = (u * u * 0.3 + v).sin() * (1.3 * v - 0.2 * u * v).cos();
                [90.0 + 80.0 * k, 70.0 + 60.0 * k, 60.0 - 40.0 * k]
            } else {
                let (dx, dy, dz) = (x as f64 + 0.5 - 0.5 * W as f64, y as f64 + 0.5 - 0.5 * H as f64, f);
                let (dx, dz) = (dx * ty.cos() - dz * ty.sin(), dx * ty.sin() + dz * ty.cos());
                let (dy, dz) = (dy * tp.cos() - dz * tp.sin(), dy * tp.sin() + dz * tp.cos());
                let az = dx.atan2(dz);
                let el = dy.atan2((dx * dx + dz * dz).sqrt());
                world(az * f, el * f)
            };
            for ch in 0..3 {
                v[o + ch] = c[ch].clamp(0.0, 255.0) as u8;
            }
            v[o + 3] = 255;
        }
    }
    v
}

fn texture(e: &flyhigh_engine::api::Engine, rgba: &[u8]) -> (wgpu::Texture, wgpu::TextureView) {
    let t = e.gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    e.gpu.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &t,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: Some(H) },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    let v = t.create_view(&Default::default());
    (t, v)
}

fn read(e: &flyhigh_engine::api::Engine, t: &wgpu::Texture) -> Vec<u8> {
    let stage = e.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (W * H * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = e.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: t,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &stage,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: Some(H) },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    e.gpu.queue.submit(Some(enc.finish()));
    flyhigh_engine::gpu::readback::Readback::new(stage, Box::new(|d: &[u8]| d.to_vec()), None).block(&e.gpu).unwrap()
}

/// The shift measured between two frames panned by `a` and `b`.
fn measure(e: &flyhigh_engine::api::Engine, a: (f64, f64), b: (f64, f64)) -> (Option<(f64, f64)>, CameraWarp) {
    let mut w = CameraWarp::new(&e.gpu, W, H, WarpOptions::default());
    let (_ta, va) = texture(e, &frame(a));
    let (_tb, vb) = texture(e, &frame(b));
    w.frame(&e.gpu, (0, &va), 0.0);
    w.frame(&e.gpu, (1, &vb), 16.7);
    e.gpu.wait();
    w.collect(&e.gpu);
    (w.last_shift(), w)
}

#[test]
#[ignore]
fn the_shift_of_a_panned_frame_is_measured_past_the_screen_locked_weapon() {
    let Some(e) = engine() else { return };
    let mut w = CameraWarp::new(&e.gpu, W, H, WarpOptions::default());
    // the first turn teaches the mask; until then a weapon with more detail than the world pulls a small shift
    // towards zero
    // back to level yaw before the pitch steps, so that each of them is a pure turn about the camera's x axis
    let pans = [
        (40.0, 0.0),
        (23.4, 0.0),
        (-61.0, 0.0),
        (3.2, 0.0),
        (0.0, 0.0),
        (-1.4, 0.0),
        (140.0, 0.0),
        (-144.2, 0.0),
        (0.0, 6.0),
        (0.0, -25.0),
        (0.0, 1.5),
    ];
    let mut at = (0.0, 0.0);
    let (_t0, v0) = texture(&e, &frame(at));
    w.frame(&e.gpu, (0, &v0), 0.0);
    let mut keep = vec![_t0];
    for (i, pan) in pans.into_iter().enumerate() {
        at = (at.0 + pan.0, at.1 + pan.1);
        let (t, v) = texture(&e, &frame(at));
        w.frame(&e.gpu, (i + 1, &v), 16.7 * (i + 1) as f64);
        e.gpu.wait();
        w.collect(&e.gpu);
        keep.push(t);
        let s = w.last_shift().unwrap_or_else(|| panic!("pan {pan:?}: no shift measured"));
        eprintln!("pan {pan:?}: measured ({:.2}, {:.2})", s.0, s.1);
        // the angular grid is exact for yaw; pitch bends its rows a little away from the centre
        let tol = if pan.1 == 0.0 { 0.75 } else { 0.75 + 0.04 * pan.1.abs() };
        assert!((s.0 - pan.0).abs() < tol && (s.1 - pan.1).abs() < tol, "pan {pan:?} measured {s:?}");
    }
}

#[test]
#[ignore]
fn a_flat_picture_has_no_shift() {
    let Some(e) = engine() else { return };
    let mut w = CameraWarp::new(&e.gpu, W, H, WarpOptions::default());
    let grey = vec![128u8; (W * H * 4) as usize];
    let (_a, va) = texture(&e, &grey);
    let (_b, vb) = texture(&e, &grey);
    w.frame(&e.gpu, (0, &va), 0.0);
    w.frame(&e.gpu, (1, &vb), 16.7);
    e.gpu.wait();
    w.collect(&e.gpu);
    assert_eq!(w.last_shift(), None, "nothing to measure is not a still camera");
}

#[test]
#[ignore]
fn the_weapon_is_masked_and_the_world_is_not() {
    let Some(e) = engine() else { return };
    let (_, w) = measure(&e, (0.0, 0.0), (23.4, 0.0));
    let m = w.read_mask(&e.gpu);
    let gw = W.div_ceil(8);
    let (mut lock_in, mut n_in, mut lock_out, mut n_out) = (0.0, 0.0, 0.0, 0.0);
    for (i, v) in m.iter().enumerate() {
        let (bx, by) = ((i as u32 % gw) * 8, (i as u32 / gw) * 8);
        // blocks wholly inside the weapon, and those clear of it by a block
        if bx >= LOCK.0 && bx + 8 <= LOCK.2 && by >= LOCK.1 && by + 8 <= LOCK.3 {
            lock_in += v;
            n_in += 1.0;
        } else if bx + 16 <= LOCK.0 || by + 16 <= LOCK.1 {
            lock_out += v;
            n_out += 1.0;
        }
    }
    let (inside, outside) = (lock_in / n_in, lock_out / n_out);
    eprintln!("locked: {:.1} % of the weapon's blocks, {:.2} % of the world's", 100.0 * inside, 100.0 * outside);
    assert!(inside > 0.9, "the weapon is locked to the screen");
    assert!(outside < 0.02, "the world is not");
}

#[test]
#[ignore]
fn the_render_turns_the_world_by_the_shift_and_keeps_the_weapon() {
    let Some(e) = engine() else { return };
    let pan = (23.4, 0.0);
    let (_, mut w) = measure(&e, (0.0, 0.0), pan);
    // the newest frame turned 30 px further right: near the centre it is the world panned by pan + 30
    let (_tb, vb) = texture(&e, &frame(pan));
    let out = read(&e, w.render_shift(&e.gpu, (1, &vb), (30.0, 0.0)));
    let want = frame((pan.0 + 30.0, pan.1));
    let src = frame(pan);
    let region = |x0: u32, y0: u32, x1: u32, y1: u32, a: &[u8], b: &[u8]| {
        let mut s = 0.0;
        let mut n = 0.0;
        for y in y0..y1 {
            for x in x0..x1 {
                for c in 0..3 {
                    let o = ((y * W + x) * 4) as usize + c;
                    s += (a[o] as f64 - b[o] as f64).abs();
                    n += 1.0;
                }
            }
        }
        s / n
    };
    let (cx, cy) = (W / 2, H / 2);
    let centre_turned = region(cx - 40, cy - 40, cx + 40, cy + 40, &out, &want);
    let centre_kept = region(cx - 40, cy - 40, cx + 40, cy + 40, &out, &src);
    let weapon = region(LOCK.0 + 16, LOCK.1 + 16, LOCK.2 - 16, LOCK.3, &out, &src);
    eprintln!(
        "centre vs turned world {centre_turned:.2}, vs the source {centre_kept:.2}; weapon vs source {weapon:.2}"
    );
    assert!(centre_turned < 3.0 && centre_kept > 15.0, "the centre shows the world turned the right way");
    assert!(weapon < 1.0, "the weapon keeps its own pixels");
}

#[test]
#[ignore]
fn the_stream_learns_the_mouse_and_shows_a_late_frame_where_the_camera_is_now() {
    use flyhigh_engine::gen_tex::TexOptions;
    use flyhigh_engine::stream::{Shown, Stream, StreamOptions};
    let Some(e) = engine() else { return };
    let mut s = Stream::new(&e.gpu, &e.blob, &e.ops, TexOptions::new(W, H), &StreamOptions::default(), None).unwrap();
    let secs = 5.0;
    let (counts, _) = play(secs, (1e9, 1e9));
    // the camera in pixels after each millisecond of input (cam[k] holds the counts before ms k); the picture shows it
    // LAG ms late
    let mut cam = vec![0.0; counts.len() + 1];
    for i in 0..counts.len() {
        cam[i + 1] = cam[i] + counts[i] * GAIN;
    }
    let mut next_frame = 0.0;
    let mut last: Option<(usize, f64)> = None;
    let mut checked = 0;
    for (i, &k) in counts.iter().enumerate() {
        let t = i as f64;
        s.mouse(&e.gpu, t, k, 0.0);
        if t < next_frame {
            continue;
        }
        next_frame += 1000.0 / 60.0;
        let slot = s.next_input(|_| false).unwrap();
        let rgba = frame((cam[(t - LAG).max(0.0) as usize], 0.0));
        e.gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: s.input_texture(slot),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: Some(H) },
            wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );
        s.push(&e.gpu, slot, t).unwrap();
        // the frame before this one, presented now: where the camera is now, not where it was LAG ms ago
        if let Some((prev, _)) = last {
            if t > 3500.0 && (cam[i] - cam[(t - 1000.0 / 60.0 - LAG) as usize]).abs() > 8.0 && checked < 6 {
                e.gpu.wait();
                assert!(s.warp(&e.gpu, Shown::Input(prev), 0, t).unwrap(), "the warp is on at {t} ms");
                let out = {
                    let tex = s.output_texture(0);
                    read(&e, tex)
                };
                // the mouse event at `t` itself is in: cam[i + 1]
                let now_cam = cam[i + 1];
                let shown = cam[(t - 1000.0 / 60.0 - LAG) as usize];
                let (cx, cy) = (W / 2, H / 2);
                let err = |want: &[u8]| {
                    let mut d = 0.0;
                    let mut n = 0.0;
                    // the centre of the view, clear of the weapon and its soft mask edge
                    for y in cy - 80..cy + 40 {
                        for x in cx - 120..cx + 80 {
                            for c in 0..3 {
                                let o = ((y * W + x) * 4) as usize + c;
                                d += (out[o] as f64 - want[o] as f64).abs();
                                n += 1.0;
                            }
                        }
                    }
                    d / n
                };
                let (to_now, to_late) = (err(&frame((now_cam, 0.0))), err(&frame((shown, 0.0))));
                eprintln!(
                    "t {t:.0}: camera now {now_cam:.1} px, frame shows {shown:.1}; the warped centre differs by {to_now:.2} \
                     from now, {to_late:.2} from the late frame"
                );
                assert!(to_now < 1.0 && to_now < 0.05 * to_late, "the warped frame shows the camera where it is now");
                checked += 1;
            }
        }
        last = Some((slot, t));
    }
    let r = s.report().camera_warp.expect("started by the mouse");
    eprintln!("calibration: {r:?}");
    assert!(r.on && (r.lag_ms - LAG).abs() <= 5.0 && (r.gain_x / GAIN - 1.0).abs() < 0.05, "{r:?}");
    assert!(checked >= 3, "enough turning frames were checked ({checked})");
    // a released mouse: nothing written, the frame is presented as it is
    s.set_mouse_captured(false);
    let (slot, t) = last.unwrap();
    assert!(!s.warp(&e.gpu, Shown::Input(slot), 0, t + 10.0).unwrap());
}
