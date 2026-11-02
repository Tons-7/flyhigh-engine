# Baselines

`baselines/<gpu>.json`: the numbers the speed and quality goals are measured against, one file per GPU,
recorded with `python tools/baseline.py` from a release build. Each file holds the GPU time of the pair stage and of
one generated frame (`texbench`) for a fixed set of configurations on every kernel class the GPU offers (`tc-f16`,
`portable-f16`, `portable-f32`) and, on Windows, the DX12 backend beside Vulkan; the quality regression means per set
(default and fast render); the parity PSNR; the wasm size and the engine commit.

`python tools/baseline.py --check` runs the same suite and compares it with the committed file of this GPU: a pair or
mid time more than 3% slower, a set mean more than 0.05 dB lower or the parity more than 0.1 dB lower is a regression
and exits 1. Every configuration is the best of three processes (`--runs`), because the autotune picks between
candidates a few percent apart and a single process swings up to 10% on the f32 path; the best of three still moves
about 2% there between runs, so a borderline verdict deserves a second run before it is called a regression. Run it before an engine commit that touches the kernels or the plans; a regression needs a documented
reason. `--check --quick` (one small configuration, no quality or parity) is the structure check CI runs on lavapipe.
Re-record after a deliberate performance change and commit the file with it. `benchmarks/` (ignored) is the private
comparison harness and is not this.
