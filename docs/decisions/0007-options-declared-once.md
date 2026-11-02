# 0007: every option is declared once

Date: 2026-11-02 (designed 2026-11-01, design 05). Status: accepted.

## Context

The options reference was written by hand and drifted: it described the pre-layer API and lacked the Stream and the
Pacer entirely, while the Pacer alone has thirty heuristics. A reader could not trust it, and a new option needed the
same text in three places.

## Decision

An option is declared once, as a field of its layer's options struct: the doc comment is its description, the default
instance gives its default, and `#[option(default, values, c)]` adds what the type cannot say. The `Options` derive
(`crates/flyhigh-options`) turns that into `OptionInfo`; a field without a doc comment does not compile.
`flyhigh-dev options-doc` writes `docs/options.md` from it and CI fails when the file is stale.

## Consequences

Fifty-three options across the four layers in one generated reference; rustdoc shows the same text. The same
`OptionInfo` is the source for the C header comments and the TypeScript types (design 07) when those are generated.
