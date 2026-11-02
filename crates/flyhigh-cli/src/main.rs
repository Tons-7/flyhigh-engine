//! The developer command line of the engine: self-tests against the PyTorch goldens, the quality regression, GPU
//! timings and small end-to-end runs through the library API. Hosts use the library; this binary is for the people
//! who change the engine. `flyhigh-dev --help` lists the commands, `flyhigh-dev <command> --help` their options.
mod cli;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "flyhigh-dev", version, about = "FlyHigh Engine developer tools")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Kernel self-tests and full-graph parity against the PyTorch golden dumps (the default)
    Selftest,
    /// Conv kernel microbenchmarks for this device's kernel class
    Microbench,
    /// One generated frame between two images
    Frames(cli::frames::Args),
    /// Any ffmpeg input through the library generator to any ffmpeg output
    Play(cli::play::Args),
    /// GPU time per generated frame on procedural frames (the PlanSession path)
    Bench(cli::bench::Args),
    /// Quality regression on ground-truth triplets; exits 1 on a regression
    Quality(cli::quality::Args),
    /// GPU time of the pair and mid stages on the texture path
    Texbench(cli::texbench::Args),
    /// PSNR per generated phase against the dropped frames of a high-frame-rate video
    Phases(cli::phases::Args),
    /// A recorded host trace (`trace v1`) through the Pacer: what it would present and generate, per second
    Replay(cli::replay::Args),
    /// The options reference (docs/options.md) from the option declarations; --check exits 1 when it is stale
    OptionsDoc(cli::options_doc::Args),
}

fn main() -> Result<()> {
    match Cli::parse().command.unwrap_or(Command::Selftest) {
        Command::Selftest => cli::selftest::run(),
        Command::Microbench => cli::microbench::run(),
        Command::Frames(a) => cli::frames::run(a),
        Command::Play(a) => cli::play::run(a),
        Command::Bench(a) => cli::bench::run(a),
        Command::Quality(a) => {
            if !cli::quality::run(a)? {
                std::process::exit(1);
            }
            Ok(())
        }
        Command::Texbench(a) => cli::texbench::run(a),
        Command::Phases(a) => cli::phases::run(a),
        Command::Replay(a) => cli::replay::run(a),
        Command::OptionsDoc(a) => cli::options_doc::run(a),
    }
}
