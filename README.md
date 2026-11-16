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

## Inline PTX Instructions Supported (20)

shfl.sync.down, shfl.sync.up, shfl.sync.bfly, shfl.sync, bar.sync, fma.rn.f32, mad.lo.s32, mad.lo.u32, ld.global.f32, ld.shared.f32, ld.global, st.global.f32, st.shared.f32, st.global, mov.b32, mov.f32, mov.b64, mov.u64, setp.lt.f32, setp.eq.f32, setp.lt.s32, setp.lt.u32, setp.eq.s32, setp.eq.u32, selp.f32, and.b32, or.b32, xor.b32, shl.b32, shr.b32

## Warp32 Emulation

The `--warp-size=64` flag (for AMD CDNA) implements warp32 emulation:
- Computes `lane = tx % 32` and `half = tx / 32`
- Masks `shfl.sync` with `scf.if` to stay within the 32-lane half
- Preserves CUDA's 32-thread logical warp semantics on 64-lane AMD wavefronts
- Needs validation on real AMD hardware (MI200/MI300)
