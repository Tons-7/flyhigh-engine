# 0006: Python is a tier-2 binding over the C ABI

Date: 2026-11-01. Status: accepted.

## Context

Research and evaluation want the engine from Python. A native extension (PyO3) means a second build system, wheels per
Python version and platform, and a second place where the API is mirrored by hand.

## Decision

Python is binding tier 2: `bindings/python/flyhigh.py` over the C ABI with `ctypes`, no native extension. Tier 1 is
Rust, C and JavaScript; tier 2 also holds the C++ header-only wrapper; tier 3 is community bindings with a guide.

## Consequences

One file, no build step, tested in CI on lavapipe against the C library. The C ABI stays the one contract every
binding follows; a new C function gets its Python and C++ line in the same change.
