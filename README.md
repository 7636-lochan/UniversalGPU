# UniversalGPU

**Breaking the CUDA Moat.** A cross-vendor GPU compiler frontend that translates NVIDIA CUDA C++ into vendor-neutral MLIR, lowers to LLVM IR, and generates native machine code for x86, NVIDIA PTX, and AMD GCN.

## What It Does

- Parses real CUDA C++ using Clang LibTooling (AST)
- Extracts grid/block dimensions from launch syntax
- Translates math, memory, and control flow into MLIR (arith, memref, scf)
- Lowers MLIR to LLVM IR
- Generates **x86 assembly**, **NVIDIA PTX**, and **AMD GCN** from a single source file
- Handles for, while, if/else, switch, and __shared__ memory
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

## Installation

### Prerequisites (Ubuntu 22.04+)
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 22
sudo apt install libclang-22-dev libmlir-22-dev mlir-22-tools clang-22 g++ cmake make -y

### Build
git clone https://github.com/7636-lochan/UniversalGPU.git
cd UniversalGPU
./build.sh

## Quick Start

# 1. Compile the matmul CUDA kernel to MLIR
cd frontend/build
./cuda-launch-parser ../../tests/matmul.cu -- > matmul.mlir

# 2. Lower MLIR to LLVM IR
mlir-opt-22 matmul.mlir --convert-scf-to-cf --convert-func-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts -o matmul_llvm.mlir
mlir-translate-22 matmul_llvm.mlir --mlir-to-llvmir -o matmul.ll

# 3. Generate NVIDIA PTX
llc-22 matmul.ll -march=nvptx64 -mcpu=sm_70 -o matmul.ptx

# 4. Generate AMD GCN
llc-22 matmul.ll -march=amdgcn -mcpu=gfx90a -o matmul.amdgcn

# 5. Compile to native x86 binary and run it
cd ../../runtime
clang-22 driver.c ../frontend/build/matmul.ll -o test_matmul -lm
./test_matmul

## Verified Backends

From a single CUDA source file, UniversalGPU generates:
- **x86 assembly** — verified running correctly with 8.0 x 16 output
- **NVIDIA PTX** — .target sm_70, ld.global.f32, fma.rn.f32
- **AMD GCN** — v_mul_f32, v_rcp_f32, buffer_load_dword

### Example PTX Output
.version 6.0
.target sm_70
.visible .func matmul(...)
    ld.param.b64 %rd8, [matmul_param_15];
    mul.lo.s64 %rd1, %rd8, %rd8;

### Example GCN Output
.globl matmul
matmul:
    v_mad_u64_u32 v[4:5], s[4:5], v30, v30, 0
    v_mul_f32_e32 v1, 0x5f7ffffc, v1

## Known Limitations
- f16 math is promoted to f32 (loads/stores correctly f16)
- No full PyTorch integration yet (needs cuBLAS/cuDNN shims)
- No C++ templates, warp primitives, or inline PTX

## Roadmap
- [x] Phase 1: Clang AST parsing (CUDA)
- [x] Phase 2: MLIR generation and LLVM IR lowering
- [x] Phase 3: Runtime shim (libcudart.so interception)
- [x] Phase 4: Workload splitter (AMD/Intel routing)
- [x] Verified PTX and GCN backend generation
- [ ] Full PyTorch integration
- [ ] f16 math promotion fix

## License
MIT
