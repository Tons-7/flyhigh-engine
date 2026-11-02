//! `play`: any ffmpeg input through `api::Generator` to any ffmpeg output, the host contract end to end.

use super::{model_dir, parse_size};
use anyhow::{Context, Result};

// Library path end to end: `flyhigh-dev play <input> <output> [factor=2] [realtime|video] [repair=1] [cadence] [k=1] [IWxIH] [fps=24] [max=N]`
// — decodes any ffmpeg input (video file or image pattern like dir/f%04d.png), pushes the frames through
// `api::Generator` and writes every output frame (real slots and in-betweens, in display order) to any ffmpeg output
// (mp4, or an image pattern). The host contract a desktop app would use, with ffmpeg standing in for capture/present.
#[derive(Clone, Copy, clap::ValueEnum)]
pub enum PlayMode {
    /// no lookahead: games, capture
    Realtime,
    /// one frame of lookahead: video players
    Video,
}

#[derive(clap::Args)]
pub struct Args {
    /// any ffmpeg input (a video file, or an image pattern like dir/f%04d.png)
    pub input: String,
    /// any ffmpeg output (mp4, or an image pattern)
    pub output: String,
    /// output frames per source frame
    #[arg(long, default_value_t = 2)]
    pub factor: u32,
    #[arg(long, value_enum, default_value_t = PlayMode::Realtime)]
    pub mode: PlayMode,
    /// video mode: re-time elements animated on 2s (anime)
    #[arg(long)]
    pub cadence: bool,
    /// repair level of the pair flow (0..2)
    #[arg(long, default_value_t = 1)]
    pub repair: u32,
    /// fine iterations of the network
    #[arg(long, default_value_t = 1)]
    pub k: u32,
    /// internal network size as WxH (default: the 480-line rung above 960 px)
    #[arg(long, value_parser = parse_size)]
    pub internal: Option<(u32, u32)>,
    /// source frame rate when the input has none (image patterns)
    #[arg(long, default_value_t = 0.0)]
    pub fps: f64,
    /// stop after this many source frames (0 = all)
    #[arg(long, default_value_t = 0)]
    pub max: u64,
    /// log the per-pair cadence statistics
    #[arg(long)]
    pub stats: bool,
    /// emulate a host that does not push frames across a scene cut (mean |diff| > 20)
    #[arg(long)]
    pub skip_cuts: bool,
    /// static protection on (frames 8, tolerance 4)
    #[arg(long = "static")]
    pub keep_static: bool,
    /// fast render with linear t (realtime, the texture path browser hosts run)
    #[arg(long)]
    pub fast: bool,
    /// static protection: unchanged frames before a pixel is protected
    #[arg(long, default_value_t = 8)]
    pub static_frames: u32,
    /// static protection: largest per-channel change (0..255) still counted as unchanged
    #[arg(long, default_value_t = 4)]
    pub static_tolerance: u32,
}

pub fn run(cmd: Args) -> Result<()> {
    use flyhigh_engine::api::{Engine, EngineOptions, Generator, Mode, Options};
    use std::io::{Read, Write};
    let (input, output) = (&cmd.input, &cmd.output);
    let (factor, repair, cadence, k, internal, max) =
        (cmd.factor, cmd.repair, cmd.cadence, cmd.k, cmd.internal, cmd.max);
    let (skipcuts, stats_log) = (cmd.skip_cuts, cmd.stats);
    let mut fps = cmd.fps;
    let mode = match cmd.mode {
        PlayMode::Realtime => Mode::Realtime,
        PlayMode::Video => Mode::Video,
    };
    // probe: size and rate
    let probe = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "csv=p=0",
            input,
        ])
        .output()
        .context("ffprobe")?;
    let line = String::from_utf8_lossy(&probe.stdout).trim().to_string();
    let f: Vec<&str> = line.split(',').collect();
    anyhow::ensure!(f.len() >= 3, "ffprobe gave '{}' for {}", line, input);
    let (w, h) = (f[0].parse::<u32>()?, f[1].parse::<u32>()?);
    if fps <= 0.0 {
        let (n, d) = f[2].split_once('/').unwrap_or((f[2], "1"));
        fps = n.parse::<f64>().unwrap_or(24.0) / d.parse::<f64>().unwrap_or(1.0);
    }
    if fps <= 0.0 || fps > 1000.0 {
        fps = 24.0;
    }
    let engine = Engine::load_with(&model_dir(), &EngineOptions::from_env())?;
    println!(
        "play: {}x{} @ {:.3} fps -> x{} {:?} repair={} cadence={} k={} internal={:?} | {:?}",
        w,
        h,
        fps,
        factor,
        mode,
        repair,
        cadence,
        k,
        internal,
        engine.info()
    );
    let mut opts = Options::new(w, h);
    opts.k = k;
    opts.mode = mode;
    opts.repair = repair;
    opts.cadence = cadence;
    opts.keep_static = cmd.keep_static;
    opts.static_frames = cmd.static_frames;
    opts.static_tolerance = cmd.static_tolerance;
    opts.fast_render = cmd.fast;
    opts.linear_t = cmd.fast;
    if let Some(v) = std::env::var_os("FH_CUBIC") {
        opts.cubic = Some(v.to_string_lossy().trim() != "0");
    }
    if internal.is_some() {
        opts.internal = internal;
    }
    let mut gen = Generator::new(&engine, opts)?;
    let mut dec = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i", input, "-f", "rawvideo", "-pix_fmt", "rgba", "-"])
        .stdout(std::process::Stdio::piped())
        .spawn()
        .context("ffmpeg decode")?;
    let mut enc = std::process::Command::new("ffmpeg")
        .args([
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            &format!("{}x{}", w, h),
            "-r",
            &format!("{}", fps * factor as f64),
            "-i",
            "-",
        ])
        .args(if output.ends_with(".mp4") || output.ends_with(".mkv") {
            vec!["-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p"]
        } else {
            vec![]
        })
        .arg(output)
        .stdin(std::process::Stdio::piped())
        .spawn()
        .context("ffmpeg encode")?;
    let mut src = dec.stdout.take().unwrap();
    let mut dst = enc.stdin.take().unwrap();
    let mut frame = vec![0u8; (w * h * 4) as usize];
    let (mut n_in, mut n_out) = (0u64, 0u64);
    let t_start = std::time::Instant::now();
    let (mut pair_ms, mut mid_ms, mut n_mid) = (0.0, 0.0, 0u64);
    let mut prev_frame: Vec<u8> = Vec::new();
    loop {
        if let Err(e) = src.read_exact(&mut frame) {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                break;
            }
            return Err(e.into());
        }
        n_in += 1;
        let mut cut = false;
        if skipcuts || stats_log {
            if !prev_frame.is_empty() {
                let step = 97usize;
                let mut acc = 0u64;
                let mut n = 0u64;
                let mut i = 0;
                while i + 2 < frame.len() {
                    acc += (frame[i] as i64 - prev_frame[i] as i64).unsigned_abs()
                        + (frame[i + 1] as i64 - prev_frame[i + 1] as i64).unsigned_abs()
                        + (frame[i + 2] as i64 - prev_frame[i + 2] as i64).unsigned_abs();
                    n += 3;
                    i += 4 * step;
                }
                cut = n > 0 && (acc as f64 / n as f64) > 20.0;
            }
            prev_frame.clear();
            prev_frame.extend_from_slice(&frame);
        }
        if skipcuts && cut {
            dst.write_all(&frame)?;
            n_out += 1;
            println!("  frame {} is a cut: not pushed (host behaviour), source written as is", n_in);
            continue;
        }
        let outs = gen.push(&engine, &frame, factor)?;
        if stats_log {
            let st = gen.stats();
            let rt = |t: f32| {
                gen.retime(t)
                    .map_or("-".to_string(), |r| format!("s{}@{:.2}{}", r.sess, r.te, if r.full { "F" } else { "" }))
            };
            println!("  frame {:4} cut={} outs={} | chg% prev {:6.3} cur {:6.3} next {:6.3} | jump {}{}{} | area% {:4.1} {:4.1} {:4.1} cost {:.3} cut {}{}{} | retime t0 {} t.5 {}", n_in, cut as u8, outs.len(), st[0].changed * 100.0, st[1].changed * 100.0, st[2].changed * 100.0, st[0].jump as u8, st[1].jump as u8, st[2].jump as u8, st[0].area * 100.0, st[1].area * 100.0, st[2].area * 100.0, st[1].cost, st[0].cut as u8, st[1].cut as u8, st[2].cut as u8, rt(0.0), rt(0.5));
        }
        pair_ms += gen.last_pair_ms;
        for o in &outs {
            dst.write_all(&o.rgba)?;
            n_out += 1;
            if !o.real {
                mid_ms += gen.last_mid_ms;
                n_mid += 1;
            }
        }
        if n_in % 50 == 0 {
            let st = gen.stats();
            println!("  {} in / {} out | pair {:.2} ms avg | out frame {:.2} ms avg (incl. readback) | cadence stats cur: area {:.1}% residual {:.1} px jump={}", n_in, n_out, pair_ms / n_in as f64, if n_mid > 0 { mid_ms / n_mid as f64 } else { 0.0 }, st[1].area * 100.0, st[1].residual, st[1].jump);
        }
        if max > 0 && n_in >= max {
            break;
        }
    }
    drop(dst);
    let _ = enc.wait();
    let _ = dec.kill();
    let wall = t_start.elapsed().as_secs_f64();
    println!("play done: {} source frames -> {} output frames in {:.1} s ({:.1} out fps) | pair {:.2} ms avg | synthesised frame {:.2} ms avg", n_in, n_out, wall, n_out as f64 / wall, pair_ms / n_in.max(1) as f64, if n_mid > 0 { mid_ms / n_mid as f64 } else { 0.0 });
    Ok(())
}
