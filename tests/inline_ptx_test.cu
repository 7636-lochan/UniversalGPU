#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
struct uint3 { unsigned int x, y, z; };
struct dim3 { unsigned int x, y, z; dim3(unsigned int vx=1, unsigned int vy=1, unsigned int vz=1) : x(vx), y(vy), z(vz) {} };
extern "C" { __device__ uint3 blockIdx, threadIdx, blockDim; }
extern "C" int cudaConfigureCall(dim3 gridDim, dim3 blockDim, unsigned long sharedMem = 0, void *stream = 0) { return 0; }

__global__ void ptx_kernel(float* input, float* output, int N) {
    int tid = threadIdx.x;
    float val = input[tid];
    
    asm volatile("bar.sync 0;");
    
    float shuffled;
    asm volatile("shfl.sync.down.b32 %0, %1, 0x1, 0x1f, 0xffffffff;"
                 : "=f"(shuffled) : "f"(val));
    
    float fma_result;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;"
                 : "=f"(fma_result) : "f"(val), "f"(shuffled), "f"(val));
    
    if (tid == 0) {
        output[blockIdx.x] = fma_result;
    }
}
