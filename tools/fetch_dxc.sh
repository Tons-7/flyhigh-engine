#!/usr/bin/env sh
# Fetch the DirectX Shader Compiler DLLs the DX12 backend needs (docs/platforms.md): dxcompiler.dll and dxil.dll from
# the pinned Microsoft release, checksum verified, into the directory given (default target/release, beside
# flyhigh-dev.exe and flyhigh.dll), with the licence files that must travel with them (LICENSE-LLVM.txt for
# dxcompiler, LICENSE-MS.txt for dxil). Without the DLLs wgpu compiles the kernels through FXC: 139 s instead of 4 s,
# no shader-f16. Needs curl, sha256sum and unzip (Git for Windows has all three).
set -e
DEST="${1:-target/release}"
TAG="v1.9.2609"
ZIP="dxc_2026_09_29.zip"
SHA256="ad31b1fc8443175d204f77a611fdb3ef2ec42759bdc2f1167368de24a4a7e7f1"
ARCH="${DXC_ARCH:-x64}"   # x64 | arm64 | x86

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -sSL -o "$tmp/$ZIP" "https://github.com/microsoft/DirectXShaderCompiler/releases/download/$TAG/$ZIP"
echo "$SHA256 *$tmp/$ZIP" | sha256sum -c - > /dev/null
# the archive's member names use backslashes: unzip converts them (exit 1 = warnings only) and selecting members by name fails
unzip -q -o "$tmp/$ZIP" -d "$tmp/x" || [ $? -eq 1 ]
mkdir -p "$DEST"
cp "$tmp/x/bin/$ARCH/dxcompiler.dll" "$tmp/x/bin/$ARCH/dxil.dll" "$DEST/"
cp "$tmp/x/LICENSE-LLVM.txt" "$DEST/LICENSE-dxcompiler.txt"
cp "$tmp/x/LICENSE-MS.txt" "$DEST/LICENSE-dxil.txt"
ls -l "$DEST/dxcompiler.dll" "$DEST/dxil.dll"
