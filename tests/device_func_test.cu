#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
struct uint3 { unsigned int x, y, z; };
struct dim3 { unsigned int x, y, z; dim3(unsigned int vx=1, unsigned int vy=1, unsigned int vz=1) : x(vx), y(vy), z(vz) {} };
extern "C" { __device__ uint3 blockIdx, threadIdx, blockDim; }
extern "C" int cudaConfigureCall(dim3 gridDim, dim3 blockDim, unsigned long sharedMem = 0, void *stream = 0) { return 0; }

__device__ float warp_reduce(float val) {
    return val + 1.0f;
}

__global__ void myKernel(float* data, int N) {
    int i = threadIdx.x;
    if (i < N) {
        float x = warp_reduce(data[i]);
        data[i] = x;
    }
}
