# UniversalGPU

**Breaking the CUDA Moat.** An open-source cross-vendor GPU compiler frontend.

## What It Does

- Parses CUDA C++ using Clang LibTooling
- Generates MLIR (arith, memref, scf)
- Lowers to LLVM IR
- Generates x86, NVIDIA PTX, and AMD GCN machine code
- Handles N-dimensional arrays (1D-6D)
- Supports f16, f32, f64
- Handles for, while, if/else, switch
- Supports __shared__ memory + __syncthreads()
- Supports atomicAdd() reductions
- Supports C++ templates
- Supports warp-size emulation (--warp-size=32 / 64)
- Intercepts inline PTX (bar.sync, shfl.sync, fma.rn.f32)

## Verified

- x86 machine code (matmul: 8.0 x 16 output)
- NVIDIA PTX assembly
- AMD GCN assembly

## Quick Start

git clone https://github.com/7636-lochan/UniversalGPU.git
cd UniversalGPU
./build.sh

## Known Limitations

- Inline PTX: top 5 instructions only
- No multi-file compilation
- No cuBLAS / cuDNN shims yet
- Warp emulation untested on real AMD hardware

## License

MIT
