#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
struct uint3 { unsigned int x, y, z; };
struct dim3 { unsigned int x, y, z; dim3(unsigned int vx=1, unsigned int vy=1, unsigned int vz=1) : x(vx), y(vy), z(vz) {} };
extern "C" { __device__ uint3 blockIdx, threadIdx, blockDim; }
extern "C" int cudaConfigureCall(dim3 gridDim, dim3 blockDim, unsigned long sharedMem = 0, void *stream = 0) { return 0; }

__global__ void relu(float* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

void launch_relu(float* data, int n) {
    dim3 grid(1024);
    dim3 block(256);
    relu<<<grid, block>>>(data, n);
}
