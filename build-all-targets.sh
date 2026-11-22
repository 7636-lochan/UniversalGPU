#!/bin/bash
# Quad-vendor build: x86, NVIDIA PTX, AMD GCN, Intel SPIR-V
# Usage: ./build-all-targets.sh <file.cu>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <file.cu>" >&2
    exit 1
fi

INPUT="$(realpath "$1")"
BASENAME=$(basename "$INPUT" .cu)
BUILD_DIR="/tmp/ugpu_build_$$"
mkdir -p "$BUILD_DIR"

echo "=== Compiling $INPUT ==="
cd ~/ugpu/frontend/build
./cuda-launch-parser "$INPUT" -- > "$BUILD_DIR/$BASENAME.mlir"

echo "=== Lowering MLIR to LLVM IR ==="
mlir-opt-22 "$BUILD_DIR/$BASENAME.mlir" \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts \
  -o "$BUILD_DIR/${BASENAME}_llvm.mlir"

mlir-translate-22 "$BUILD_DIR/${BASENAME}_llvm.mlir" --mlir-to-llvmir -o "$BUILD_DIR/$BASENAME.ll"

echo "=== x86 assembly ==="
llc-22 "$BUILD_DIR/$BASENAME.ll" -o "$BUILD_DIR/$BASENAME.x86.s"

echo "=== NVIDIA PTX ==="
llc-22 "$BUILD_DIR/$BASENAME.ll" -march=nvptx64 -mcpu=sm_70 -o "$BUILD_DIR/$BASENAME.ptx"

echo "=== AMD GCN ==="
llc-22 "$BUILD_DIR/$BASENAME.ll" -march=amdgcn -mcpu=gfx90a -o "$BUILD_DIR/$BASENAME.gcn.s"

echo "=== Intel SPIR-V assembly ==="
llc-22 "$BUILD_DIR/$BASENAME.ll" -march=spirv64 -o "$BUILD_DIR/$BASENAME.spvasm"

echo "=== Intel SPIR-V binary ==="
spirv-as "$BUILD_DIR/$BASENAME.spvasm" -o "$BUILD_DIR/$BASENAME.spv"

echo ""
echo "=== Build complete! ==="
echo "Artifacts:"
echo "  $BUILD_DIR/$BASENAME.x86.s    (x86 CPU)"
echo "  $BUILD_DIR/$BASENAME.ptx      (NVIDIA GPU)"
echo "  $BUILD_DIR/$BASENAME.gcn.s    (AMD GPU)"
echo "  $BUILD_DIR/$BASENAME.spv      (Intel GPU, binary)"
echo "  $BUILD_DIR/$BASENAME.spvasm   (Intel SPIR-V assembly)"
echo "  $BUILD_DIR/$BASENAME.ll       (LLVM IR)"
