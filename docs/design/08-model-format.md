# Design 08: model format and catalogue

Status: agreed 2026-11-01; implemented. The file format of `docs/model-format.md`
stays; this adds versioning, capabilities and the loader contract so other checkpoints and other people's models load
without engine changes.

## Files

Unchanged: `graph.json` (operations), `<name>.json` (manifest: tensor table and `meta`), `<name>.bin` (little-endian
f16 weights). The manifest's `meta.format_version` is the format version; the engine accepts versions it knows and
reports `Error::Model("format_version 3 not supported, this engine reads 1..2")` otherwise.

## Capabilities (manifest `meta`)

The loader reads, and `ModelInfo` exposes:

| Field | Meaning | Used by |
|---|---|---|
| `model`, `train_step`, `n_params` | identity | reports, catalogue test |
| `task` | `interpolation` (t in (0, 1)) or `extrapolation` (t in (1, t_max]) | Stream mode validation |
| `K_range`, `default_K` | fine iterations the graph supports | `SessionOptions.k` validation |
| `render_sampling` | the sampler the model was trained with | `RenderSampling::Model` |
| `internal_rung` (new) | the resolution the model was trained at (lines) | `Internal::Auto` |
| `t_range` (new, extrapolation) | the phases the model predicts well | Pacer insert mode |
| `side_ch`, `cost_radius`, `scales` | graph constants | the builder |

Missing new fields take today's defaults (480-line rung, interpolation), so existing manifests load.

## Catalogue

`models/models.json` (format 1) lists the shipped model and the release-asset snapshots with these capabilities; a
test checks the shipped entry against its manifest. Release assets (v5, v7x) carry their own manifests; the catalogue
entry names the asset. A host that wants to offer "quality / speed / extrapolation" choices reads the catalogue.

## Loader contract

1. Parse the manifest; check `format_version`; check every tensor named by `graph.json` exists with the declared shape.
2. Validate the graph: known ops, operands defined before use, the pair / mid boundary (`mid_start`) present.
3. Only then upload weights (per engine, on first session build).

Errors are `Error::Model` with the file and the field. A valid model never panics the builder (design 06).

## User models

The converter (`runtime/gen_parity_fixture.py` and the export tooling of the research repository) stays private; its
output format is this specification, and a public `tools/validate_model.py` checks a directory against it. A user
model is any directory that passes the validator; a worked example is still to be written.
