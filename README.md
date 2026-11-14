# UniversalGPU

Breaking the CUDA Moat. A cross-vendor GPU compiler frontend that translates NVIDIA CUDA C++ into vendor-neutral MLIR, lowers to LLVM IR, and generates native x86 machine code.

## What It Does

- Parses real CUDA C++ using Clang LibTooling (AST)
- Extracts grid/block dimensions from launch syntax
- Translates math, memory, control flow into MLIR (arith, memref, scf)
- Lowers MLIR to LLVM IR to x86 assembly
- Handles for, while, if/else, switch, shared memory
- Supports f16, f32, f64, i32 types
- Runtime shim intercepts cudaMalloc, cudaMemcpy, cudaLaunchKernel
- Workload splitter routes kernels to simulated AMD/Intel backends

## Verified Example

Input: tests/matmul.cu (4x4 matrices, A=1, B=2)

Output:
8.0 8.0 8.0 8.0
8.0 8.0 8.0 8.0
8.0 8.0 8.0 8.0
8.0 8.0 8.0 8.0

## Build

Requires: clang-22, mlir-22, g++

./build.sh

## Known Limitations

- f16 math is promoted to f32
- No NVIDIA PTX or AMD GCN backend yet
- No full PyTorch integration

## License

MIT

