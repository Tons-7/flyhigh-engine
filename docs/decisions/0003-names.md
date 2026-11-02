# 0003: names

Date: 2026-11-01. Status: accepted.

## Context

The crate was called a runtime, the repository the same, and the C library and npm package had no settled names.
Hosts and documentation referred to the same thing by several names.

## Decision

The Rust crate is `flyhigh-engine` (library `flyhigh_engine`), the npm package `@flyhigh/engine`, the C library
`libflyhigh` with the header `flyhigh.h` (C symbols `fh_*`). The repository is `getflyhigh/flyhigh-engine`.

## Consequences

The workspace crates are `flyhigh-engine`, `flyhigh-options`, `flyhigh-capi`, `flyhigh-web`, `flyhigh-cli`; the
developer binary is `flyhigh-dev`. crates.io and npm publishing wait for the accounts.
