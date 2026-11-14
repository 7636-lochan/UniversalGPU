#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
struct uint3 { unsigned int x, y, z; };
struct dim3 { unsigned int x, y, z; dim3(unsigned int vx=1, unsigned int vy=1, unsigned int vz=1) : x(vx), y(vy), z(vz) {} };
extern "C" { __device__ uint3 blockIdx, threadIdx, blockDim; }
extern "C" int cudaConfigureCall(dim3 gridDim, dim3 blockDim, unsigned long sharedMem = 0, void *stream = 0) { return 0; }

// 4D tensor access: N (batch), C (channel), H (height), W (width)
__global__ void conv4d(float**** input, float**** output, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    int h = threadIdx.z;
    int w = blockIdx.z;
    if (n < N && c < N) {
        output[n][c][h][w] = input[n][c][h][w] * 2.0f;
    }
}
