# Design 06: errors

Status: agreed 2026-11-01; implemented.

## Rust

```rust
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("no suitable GPU adapter: {0}")]            NoAdapter(String),
    #[error("device creation failed: {0}")]             Device(String),
    #[error("device lost: {0}")]                        DeviceLost(String),
    #[error("model: {0}")]                              Model(String),            // missing file, bad manifest, unknown op, wrong version
    #[error("option {field}: {reason}")]                Options { field: &'static str, reason: String },
    #[error("{0}")]                                     State(&'static str),      // "push two frames first", "bad input slot"
    #[error("readback failed: {0}")]                    Readback(String),
    #[error("unsupported on this device: {0}")]         Unsupported(&'static str),// timestamps, f16, external memory
    #[error("internal: {0}")]                           Internal(String),         // an engine bug: what a panic used to be
}
pub type Result<T> = std::result::Result<T, Error>;
```

- `anyhow` leaves the library (stays in the CLI). Every public function returns `Result<T>`.
- `unwrap` / `expect` are forbidden in the library crate (`#![deny(clippy::unwrap_used, clippy::expect_used)]`);
  the remaining sites become either a typed error or an `Internal` error with the invariant named.
- Graph data (`graph.json`) is validated once at `Model` load: unknown ops, missing weights and shape mismatches are
  `Error::Model` there, so the builder can index without checks.
- Device loss surfaces as `Error::DeviceLost` from the next call; the host drops the engine and creates a new one
  (as today). No automatic recovery inside the engine.

## C

Every function returns a status code or a handle; `fh_last_error()` keeps the message of the last failure on the
calling thread, as today, plus `fh_last_error_code()`:

| Code | Variant |
|---|---|
| 0 | ok |
| 1 | NoAdapter |
| 2 | Device |
| 3 | DeviceLost |
| 4 | Model |
| 5 | Options |
| 6 | State |
| 7 | Readback |
| 8 | Unsupported |
| 9 | Internal |

Null handles and out-of-range arguments are `Options` / `State`, never a crash; the header documents each function's
preconditions (the `# Safety` sections of the Rust side are the source).

## JavaScript

Exported functions throw `FlyHighError` (a `DOMException`-like object) with `code` (the names above as strings),
`message` and, for `Options`, `field`. Async functions reject with the same. The classes of design 07 never throw
plain strings.

## Panics

A panic in the library is a bug. The wasm build keeps `console_error_panic_hook`; natively the C layer catches
unwinding at the boundary (`catch_unwind`) and returns `Internal`. Both are reported with the report JSON attached
when the host asks for it.
