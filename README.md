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
  
### Quick Start

```bash
# 1. Compile the matmul CUDA kernel to MLIR
cd frontend/build
./cuda-launch-parser ../../tests/matmul.cu -- > matmul.mlir

# 2. Lower MLIR to LLVM IR
mlir-opt-22 matmul.mlir \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts \
  -o matmul_llvm.mlir

mlir-translate-22 matmul_llvm.mlir --mlir-to-llvmir -o matmul.ll

# 3. Generate NVIDIA PTX
llc-22 matmul.ll -march=nvptx64 -mcpu=sm_70 -o matmul.ptx

# 4. Generate AMD GCN
llc-22 matmul.ll -march=amdgcn -mcpu=gfx90a -o matmul.amdgcn

# 5. Compile to native x86 binary and run it
cd ../../runtime
clang-22 driver.c ../frontend/build/matmul.ll -o test_matmul -lm
./test_matmul
./build.sh

## Known Limitations

- f16 math is promoted to f32
- No NVIDIA PTX or AMD GCN backend yet
- No full PyTorch integration

## License

MIT

## Verified Backends

From a single CUDA source file, UniversalGPU generates:

- **x86 assembly** (verified running correctly with 8.0 x 16 output)
- **NVIDIA PTX** (.target sm_70, ld.global.f32, fma.rn.f32)
- **AMD GCN** (v_mul_f32, v_rcp_f32, buffer_load_dword)

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
