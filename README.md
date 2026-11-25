# UniversalGPU

Breaking the CUDA Moat. An open-source, quad-vendor CUDA compiler that translates NVIDIA CUDA C++ into vendor-neutral MLIR, lowers to LLVM IR, and generates native machine code for x86, NVIDIA PTX, AMD GCN, and Intel SPIR-V.

## What It Does

- Parses real CUDA C++ using Clang LibTooling (AST)
- Extracts grid/block dimensions from launch syntax
- Translates math, memory, and control flow into MLIR (arith, memref, scf)
- Lowers MLIR to LLVM IR
- Generates machine code for four vendors: x86 (CPU), NVIDIA PTX, AMD GCN, Intel SPIR-V
- Handles N-dimensional tensors (1D through 6D)
- Supports C++ templates (function specializations)
- Supports f16, f32, f64, i32 types
- Handles for, while, if/else, switch, and __shared__ memory
- Handles __syncthreads() (gpu.barrier) and atomicAdd() (memref.atomic_rmw)
- Handles __device__ function calls and multi-file compilation
- Resolves #include directives via a preprocessing wrapper
- Implements warp32 emulation for AMD 64-lane targets (masked shuffles)
- Intercepts inline PTX (bar.sync, shfl.sync, fma.rn.f32, mov, ld/st, setp, selp, and, or, xor, shl, shr)
- Runtime shim intercepts cudaMalloc, cudaMemcpy, cudaLaunchKernel, cudaGetDeviceProperties, and cublasSgemm

## Verification Status (Honest)

- x86 (Intel CPU): Verified on Intel i5-13th Gen HX (8.0 x 16 matmul output)
- Intel SPIR-V (Iris Xe): Verified (SPIR-V loaded and built via OpenCL)
- NVIDIA PTX (RTX 4050): Generated (needs ptxas from CUDA toolkit to execute)
- AMD GCN: Generated (needs AMD GPU + ROCm to execute)

## Quick Start

git clone https://github.com/7636-lochan/UniversalGPU.git
cd UniversalGPU
./build.sh

### Compile CUDA to MLIR

cd frontend/build
./cuda-launch-parser ../../tests/matmul.cu -- > matmul.mlir

### Lower to LLVM IR

mlir-opt-22 matmul.mlir --convert-scf-to-cf --convert-func-to-llvm --convert-arith-to-llvm --finalize-memref-to-llvm --convert-cf-to-llvm --reconcile-unrealized-casts -o matmul_llvm.mlir

mlir-translate-22 matmul_llvm.mlir --mlir-to-llvmir -o matmul.ll

### Quad-Vendor Build (x86, PTX, GCN, SPIR-V)

./build-all-targets.sh tests/matmul.cu

## Multi-File Compilation

For projects with #include directives, use the wrapper script:

./ugpu-compile-with-includes.sh kernel.cu -I./include

## Runtime Shim

cd runtime
g++ -shared -fPIC cudart_shim.cpp -o libcudart.so -ldl
g++ -shared -fPIC libcublas_shim.cpp -o libcublas.so
LD_LIBRARY_PATH=. python3 test_mock_ai.py

## Known Limitations

- Inline PTX: 20 instructions intercepted. Full PTX grammar (200+ instructions) is not implemented.
- No cuBLAS / cuDNN / cuFFT full library shims yet.
- No full PyTorch integration.
- Warp32 emulation is structurally correct but needs validation on real AMD hardware.
- No Intel backend execution beyond SPIR-V load/build.

## Roadmap

- [x] Clang AST parsing (CUDA)
- [x] MLIR generation and LLVM IR lowering
- [x] x86 machine code (verified)
- [x] NVIDIA PTX assembly
- [x] AMD GCN assembly
- [x] Intel SPIR-V binary (verified on Iris Xe)
- [x] N-dimensional tensors (1D-6D)
- [x] C++ templates
- [x] __shared__, __syncthreads, atomicAdd
- [x] 20 inline PTX instructions
- [x] Warp32 emulation
- [x] __device__ function calls
- [x] Multi-file compilation
- [x] #include resolution
- [x] Runtime shims (libcudart.so, libcublas.so)
- [ ] Full inline PTX (200 instructions)
- [ ] Full PyTorch integration
- [ ] cuDNN shim
- [ ] AMD hardware validation
- [ ] Performance optimization

## License

MIT
