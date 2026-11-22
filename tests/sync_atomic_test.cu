#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
struct uint3 { unsigned int x, y, z; };
struct dim3 { unsigned int x, y, z; dim3(unsigned int vx=1, unsigned int vy=1, unsigned int vz=1) : x(vx), y(vy), z(vz) {} };
extern "C" { __device__ uint3 blockIdx, threadIdx, blockDim; }
extern "C" int cudaConfigureCall(dim3 gridDim, dim3 blockDim, unsigned long sharedMem = 0, void *stream = 0) { return 0; }

__device__ float atomicAdd(float* addr, float val) { return 0.0f; }

__global__ void tiled_reduction(float* input, float* output, int N) {
    __shared__ float tile[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    tile[tid] = input[idx];
    __syncthreads();
    
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += tile[i];
        }
        atomicAdd(&output[blockIdx.x], sum);
    }
}
