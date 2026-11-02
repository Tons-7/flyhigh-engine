# Decision records

Decisions that shape the engine, one file each, numbered in the order they were taken. A record says what the
situation was, what was decided and what follows from it; it is not edited afterwards. When a decision is reversed,
a new record supersedes it and the old one gets a line pointing forward.

| Record | Decision |
|---|---|
| [0001](0001-the-engine-is-the-product.md) | the engine is the whole product, in layers; hosts are thin shells |
| [0002](0002-dual-licence-for-the-weights.md) | the weights are dual licensed, the code stays Apache-2.0 |
| [0003](0003-names.md) | `flyhigh-engine`, `@flyhigh/engine`, `libflyhigh` with `flyhigh.h` |
| [0004](0004-vulkan-primary-dx12-through-dxc.md) | Vulkan is the primary native backend; DX12 runs through the DXC compiler and ships it |
| [0005](0005-engine-and-session-send-sync.md) | `Engine` and `Session` become `Send + Sync` |
| [0006](0006-python-over-the-c-abi.md) | Python is a tier-2 binding over the C ABI, no native extension |
| [0007](0007-options-declared-once.md) | every option is declared once; the reference is generated and checked |
| [0008](0008-performance-regression-rule.md) | a 3% slower pair or mid time against the GPU's baseline is a regression |

Format of a record:

```
# NNNN: title

Date: YYYY-MM-DD. Status: accepted | superseded by NNNN.

## Context
## Decision
## Consequences
```
