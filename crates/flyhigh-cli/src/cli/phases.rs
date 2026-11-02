//! `phases`: PSNR per generated phase against the dropped real frames of a high-frame-rate video.

use super::model_dir;
use anyhow::Result;

// Phase quality: `flyhigh-dev phases <video> [factor=4] [max=N] [fast] [linear]` — keeps every factor-th frame
// of a high-frame-rate video, generates the factor-1 in-betweens of each kept pair through api::Generator and scores
// them against the dropped real frames (PSNR per phase, 8-bit RGB). Measures the phases away from the middle.
#[derive(clap::Args)]
pub struct Args {
    /// a high-frame-rate video (any ffmpeg input)
    pub video: String,
    /// keep every factor-th frame and generate the factor-1 in-betweens
    #[arg(long, default_value_t = 4)]
    pub factor: usize,
    /// number of source frames to score
    #[arg(long, default_value_t = 240)]
    pub max: usize,
    /// fast render
    #[arg(long)]
    pub fast: bool,
    /// with --fast: the t-dependent network once per pair
    #[arg(long)]
    pub linear: bool,
}

pub fn run(cmd: Args) -> Result<()> {
    use flyhigh_engine::api::{Engine, EngineOptions, Generator, Options};
    use std::io::Read;
    let input = &cmd.video;
    let (factor, max) = (cmd.factor, cmd.max);
    let probe = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            input,
        ])
        .output()?;
    let line = String::from_utf8_lossy(&probe.stdout).trim().to_string();
    let f: Vec<&str> = line.split(',').collect();
    let (w, h) = (f[0].parse::<u32>()?, f[1].parse::<u32>()?);
    let engine = Engine::load_with(&model_dir(), &EngineOptions::from_env())?;
    let mut o = Options::new(w, h);
    o.fast_render = cmd.fast;
    o.linear_t = cmd.linear;
    let mut gen = Generator::new(&engine, o.clone())?;
    let mut dec = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-i", input, "-frames:v", &max.to_string(), "-f", "rawvideo", "-pix_fmt", "rgba", "-"])
        .stdout(std::process::Stdio::piped())
        .spawn()?;
    let mut src = dec.stdout.take().unwrap();
    let mut frames: Vec<Vec<u8>> = Vec::new();
    loop {
        let mut b = vec![0u8; (w * h * 4) as usize];
        if src.read_exact(&mut b).is_err() {
            break;
        }
        frames.push(b);
    }
    let mut sums = vec![(0f64, 0usize); factor];
    let psnr = |a: &[u8], b: &[u8]| {
        let (mut se, mut n) = (0f64, 0u64);
        for (p, q) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
            for c in 0..3 {
                let d = p[c] as f64 - q[c] as f64;
                se += d * d;
            }
            n += 3;
        }
        let m = se / n as f64;
        if m == 0.0 {
            99.0
        } else {
            10.0 * (65025.0 / m).log10()
        }
    };
    let mut k = 0;
    while k + factor < frames.len() {
        if k == 0 {
            gen.push(&engine, &frames[0], factor as u32)?;
        }
        let outs = gen.push(&engine, &frames[k + factor], factor as u32)?;
        for (i, out) in outs.iter().filter(|o| !o.real).enumerate() {
            let s = &mut sums[i + 1];
            s.0 += psnr(&out.rgba, &frames[k + 1 + i]);
            s.1 += 1;
        }
        k += factor;
    }
    let _ = dec.kill();
    let per: Vec<String> = (1..factor)
        .map(|i| format!("t={:.2}: {:.2} dB", i as f64 / factor as f64, sums[i].0 / sums[i].1.max(1) as f64))
        .collect();
    println!(
        "phases {}x{} x{} fast={} linear={} ({} groups): {}",
        w,
        h,
        factor,
        o.fast_render,
        o.linear_t,
        sums[1].1,
        per.join(", ")
    );
    Ok(())
}
