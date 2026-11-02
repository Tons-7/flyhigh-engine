# Security

## Reporting

Report a vulnerability privately through GitHub's private vulnerability reporting on this repository (the Security
tab, "Report a vulnerability"). If that is not available to you, open an issue titled "security contact request"
without any details and a maintainer will get in touch. Please do not describe a vulnerability in a public issue.

You will get an acknowledgement within a week and a plan or a fix as soon as the problem is understood. Fixed
vulnerabilities are noted in `CHANGELOG.md`.

## Supported versions

Before 1.0 the supported version is the latest release and the `main` branch. Fixes land on `main` and in the next
release; there are no backports to earlier 0.x versions.

## What is in scope

The engine processes data it should treat as untrusted:

- model files (`graph.json`, the manifest and the weight blob): the loader validates them, and a malformed model must
  fail with an error, never read out of bounds or hang;
- frames and masks from a host, and option values from a host (C and JavaScript callers): out-of-range values must be
  refused with the field named;
- the wasm module runs inside web pages and extensions, so anything that escapes its sandbox assumptions matters.

GPU driver bugs and WebGPU implementation bugs are reported to their projects; we are glad to help narrow them down.
