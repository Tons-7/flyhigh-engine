# 0008: the performance regression rule

Date: 2026-11-02. Status: accepted.

## Context

The speed goal is "not below today's numbers, a regression over 3% only with a documented reason", but
nothing enforced it. A single texbench process swings up to 10% on the f32 path because the autotune picks between
candidates a few percent apart; a bare comparison would raise false alarms.

## Decision

`tools/baseline.py --check` runs the fixed texbench set, the quality regression and the parity, each configuration the
best of three processes, and compares them with the committed file of the GPU: a pair or mid time more than 3%
slower, a set mean more than 0.05 dB lower or the parity more than 0.1 dB lower is a regression and exits 1. It runs
by hand on the rig before a change to kernels or plans; CI runs a quick form on lavapipe for the structure, not the
numbers. The baseline is re-recorded after a deliberate change, with the reason in the commit.

## Consequences

The numbers in the README and the platforms matrix have a guard. The f32 path still moves about 2% between
best-of-three runs, so a borderline verdict deserves a second run before it is called a regression.
