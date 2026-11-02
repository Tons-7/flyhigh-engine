# 0002: dual licence for the weights

Date: 2026-11-01. Status: accepted.

## Context

The code is Apache-2.0. The shipped weights were CC BY-NC 4.0 only, which blocks commercial integrators outright and
leaves them no path to use the engine with the shipped model in a product.

## Decision

The weights are dual licensed: CC BY-NC 4.0 for everyone at no cost, and a commercial licence by agreement with the
maintainers for products that ship or run the weights commercially. The code stays Apache-2.0. A model someone trains
themselves is theirs. `models/WEIGHTS_LICENSE.md`, `NOTICE` and the README say so in one paragraph each.

## Consequences

Commercial use has a path without changing the terms for everyone else. The engine loads other checkpoints in the
same format, so a user who cannot accept either licence can bring a model of their own.
