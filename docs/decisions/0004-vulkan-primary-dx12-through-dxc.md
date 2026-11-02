# 0004: Vulkan is the primary native backend; DX12 runs through DXC and ships it

Date: 2026-11-01, amended 2026-11-02. Status: accepted.

## Context

The engine runs on every wgpu backend, but one of them has to be the one measured first and kept fastest. Vulkan
exposes the cooperative-matrix (tensor core) kernels and the D3D11 texture import works through its external memory.
DX12 matters for interop with games and overlays on Windows. wgpu compiles HLSL for DX12 through FXC unless it finds
the DXC compiler: with FXC the kernel set took 139 s to compile, the first autotuned plan 460 s, and there is no
shader-f16, so only the f32 kernels ran. The statically linked DXC needs the Visual Studio ATL component to build and
a 114 MB static library.

## Decision

Vulkan stays the primary native backend. DX12 is supported through the dynamically loaded DXC (`dxcompiler.dll` with
`dxil.dll`): the engine looks for it beside its own module, takes `EngineOptions::dxc_path`, and reports which
compiler loaded. The Windows release archive carries the two DLLs with their licence files (dxcompiler under the LLVM
licence, dxil under Microsoft's terms for the DirectX Shader Compiler, Windows only).

## Consequences

DX12 compiles the 24 kernels in 4 s and a plan with the autotune in 9.6 s, runs the f16 kernels (parity 72.2 dB, pair
1.76 / mid 0.74 ms at 720p on an RTX 4060 Ti) and is listed as experimental until the game pacing lands. Without the
DLLs the backend still works on the slow FXC path, and the report says so.
