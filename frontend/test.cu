#define __global__ __attribute__((global))
#define __device__ __attribute__((device))

struct uint3 { unsigned int x, y, z; };
struct dim3 { unsigned int x, y, z; dim3(unsigned int vx=1, unsigned int vy=1, unsigned int vz=1) : x(vx), y(vy), z(vz) {} };
extern "C" {
__device__ uint3 blockIdx;
__device__ uint3 threadIdx;
__device__ uint3 blockDim;
}

extern "C" int cudaConfigureCall(dim3 gridDim, dim3 blockDim, unsigned long sharedMem = 0, void *stream = 0) { return 0; }

__global__ void myKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = 1.0f;
}

void host_launcher(float* d_data, int n) {
    dim3 gridDim(64);   // Changed from (n + 255) / 256
    dim3 blockDim(512); // Changed from 256
    myKernel<<<gridDim, blockDim>>>(d_data, n);
}