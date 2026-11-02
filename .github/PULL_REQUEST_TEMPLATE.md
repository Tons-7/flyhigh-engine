## What and why

<!-- What changes, and the reason. Link the issue or the design document when there is one. -->

## Measurements

<!-- Numbers with their setup (GPU, backend, resolution, date), or "none: documentation / refactor". -->

## Checklist

- [ ] `cargo test --release` and `cargo test --release -- --ignored` pass
- [ ] `cargo fmt --all -- --check`, `cargo clippy --release --all-targets -- -D warnings`
- [ ] the options reference is current (`flyhigh-dev options-doc --check`) if an option changed
- [ ] golden plans re-recorded and the change explained, if a plan changed
- [ ] `python tools/baseline.py --check` without a regression, if kernels or plans changed
- [ ] docs updated (design document, platforms, API guides), `CHANGELOG.md` under Unreleased
