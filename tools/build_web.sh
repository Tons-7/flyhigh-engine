#!/usr/bin/env sh
# Build the browser module into web/pkg (not in git): the wasm32 crate with the size-first `wasm` profile (opt-level z,
# fat LTO: 1.02 MB), then wasm-bindgen with the CLI version that matches Cargo.lock (0.2.129), and the browser adapters
# (crates/flyhigh-web/js/adapters.js) copied next to the module. Run from
# the repository root; the demo pages in web/ and the baseline recorder use it.
set -e
cd "$(dirname "$0")/.."
cargo build -p flyhigh-web --profile wasm --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/pkg target/wasm32-unknown-unknown/wasm/flyhigh_web.wasm
cp crates/flyhigh-web/js/adapters.js web/pkg/
ls -l web/pkg/flyhigh_web_bg.wasm
