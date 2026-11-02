# Models

A model is one directory with three files, loaded with `Engine::load(dir)` (Rust), `fh_engine_create(dir)` (C) or
fetched by the page (browser): `graph.json` (the operation graph), `flyhighfg.json` (the manifest: tensor offsets,
shapes and the `meta` block with the training step, K range and render sampler) and `flyhighfg.bin` (little-endian f16
weights). The format is described in [docs/model-format.md](../docs/model-format.md).

`models.json` lists the models the engine knows: the shipped one (this directory) and the snapshots distributed as
release assets. The engine does not read it; it is the human-readable catalogue and the file the model test checks
against the shipped manifest, so the two cannot drift. Point `FH_MODEL_DIR` (CLI, player) or the load call at any
directory with the three files to run another checkpoint, including your own (converter specification in
docs/model-format.md).

The weights are licensed separately from the code: [WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md) (CC BY-NC 4.0).
