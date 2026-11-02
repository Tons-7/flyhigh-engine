# Model format

A model is three files in one directory (`models/` ships the current one; load another with `Engine::load(dir)`,
`engine.loadModel(graph, manifest, weights)` in the browser, `FH_MODEL_DIR` for the CLI).

| File | Content |
|---|---|
| `flyhighfg.bin` | all weights, little-endian f16, 2 bytes per element, tensors back to back |
| `flyhighfg.json` | the manifest: model metadata and, per tensor, its shape and offset |
| `graph.json` | the op graph the runtime executes |

The files are produced by the export script of the research repo (`export/export_flyhighfg.py`). Weights are
licensed CC BY-NC 4.0 (`models/WEIGHTS_LICENSE.md`).

## Manifest (`flyhighfg.json`)

```json
{
  "meta": { "model": "FlyHighFG", "format_version": 1, "train_step": 110000, "render_sampling": "bicubic", ... },
  "tensors": { "enc.e1.0.weight": { "shape": [36, 3, 3, 3], "offset": 0, "dtype": "f16" }, ... }
}
```

- `tensors`: name (the PyTorch `state_dict` key) -> `shape` and `offset` **in f16 elements** from the start of the
  blob. Conv weights are `[cout, cin, 3, 3]`.
- `meta.render_sampling`: `"bicubic"` or absent (bilinear). The sampler the checkpoint was trained with; it is the
  default of the render sampler option.
- `meta.model`, `meta.train_step`: reported by `Engine::info()` / `fh_engine_info`. The other `meta` keys document the
  architecture and are not read by the runtime.

## Graph (`graph.json`)

`{ "meta": {...}, "ops": [ ... ] }`. Each op names its inputs (`in`) and outputs (`out`) as tensor names in one
environment. The graph inputs are `A` and `B` (`[3][H][W]`, values in [0, 1], BGR, H and W multiples of 32) and the
phase `t`; the result is `render`. Ops run in order.

| Op | Fields | Meaning |
|---|---|---|
| `interpolate` | `scale` or `size: "HW"`, `antialias` | bilinear resize (align_corners false); antialias = torch's antialiased downsample |
| `conv` | `w: [weight, bias, prelu]`, `stride` | 3x3 conv, pad 1, then PReLU |
| `conv2d` | `w: [weight, bias]` | 3x3 conv, pad 1, no activation |
| `avgpool` | `k` | average pool, kernel = stride = k |
| `warp` | | backward warp of `in[0]` by the 2-channel flow `in[1]` (pixels), bilinear, border clamp |
| `concat` | `axis: 1` | channel concat |
| `slice` | `a`, `b` | channels [a, b) |
| `scale` | `factor` | multiply |
| `neg` | | negate |
| `add`, `mul` | | elementwise (a 1-channel `in[1]` is broadcast over the channels) |
| `zeros` | `ch` | zeros with `ch` channels, the spatial size of `in[0]` |
| `zeros_like` | | zeros shaped like `in[0]` |
| `copy` | | alias |
| `luma` | | BGR -> luma (0.114 B + 0.587 G + 0.299 R) |
| `luma_grad` | | absolute luma gradient (forward differences) |
| `sad_volume` | `radius` | luma SAD cost volume, `(2r+1)^2` channels, 3x3 box averaged |
| `mul_t` | `mode: "fwd" \| "bwd"` | multiply by 2t (fwd) or 2(1-t) (bwd) |
| `film_mlp` | `w` (fixed names `film.film.0/2.*`) | the FiLM vector from t |
| `film_modulate` | | `h * (1 + g[:c]) + g[c:]` |
| `convex_upsample` | `w` (fixed names `up.w.*`) | 2x convex upsampling of `in[1]` with weights predicted from `in[0]` |
| `blend_weights` | | softmax of 3 visibility logits -> weights of A and B |
| `affine_out` | | `2 * sigmoid(x) - 1` |
| `clamp` | `lo`, `hi` | clamp |
| `render_pix` | | the full-resolution render: `A, B, flow_tq, vis_tq, R_t` -> `render` |
| `repeat` | `times: "K"`, `carry`, `body` | run `body` k times; names in `carry` are rebound each iteration |

Ops with `output: true` are graph outputs (kept readable). `cond` / `else_alias` describe the PyTorch model's optional
branches; the runtime always runs them (both are on in the shipped model).

## What the runtime relies on

- The first op that reads `t` (or is `mul_t` / `film_mlp`) starts the per-frame (mid) stage; everything before it is
  the pair stage.
- The repair stage looks for the 4-channel pair flow named `flow` (at 1/8 of the network input) and the inputs `A`,
  `B`; cadence needs repair.
- The last op is `render_pix` with inputs `A, B, flow_tq, vis_tq, R_t`; with an internal size it is replaced by a
  full-resolution render.
- Convs are 3x3 with pad 1, stride 1 or 2.

## Shipping another checkpoint

1. Export it with the research repo's export script (it writes all three files).
2. Put them in a directory and point the host at it; the CLI: `FH_MODEL_DIR=<dir> flyhigh-dev frames A.png B.png out.png`.
3. Check it: dump a parity golden (`parity.json` + `parity_*.f32`) with the research repo's export tooling and run `FH_MODEL_DIR=<dir> FH_FIXTURES=<golden> flyhigh-dev`
   (expect render PSNR >= 50 dB against PyTorch, about 73 dB in practice).
4. A different architecture works as long as it is expressed with the ops above and keeps the conventions of
   "What the runtime relies on".

## What the loader checks

A model is checked completely when it loads, so a directory handed to the engine from anywhere either loads or fails
with the tensor's name: every `shape` entry is an unsigned integer of at least 1 (a scalar has an empty shape), the
offset and the size are computed without overflow and the whole tensor lies inside `flyhighfg.bin`; every op of
`graph.json` has a name, its `in`, `out` and `w` lists hold strings, a `repeat` has a body, and every weight an op
reads (the four FiLM weights for `film_mlp`) exists in the manifest. The engine's tests load such files with every
one of these faults (`crates/flyhigh-engine/tests/models.rs`).
