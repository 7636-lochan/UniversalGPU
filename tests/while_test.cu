#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
struct uint3 { unsigned int x, y, z; };
struct dim3 { unsigned int x, y, z; dim3(unsigned int vx=1, unsigned int vy=1, unsigned int vz=1) : x(vx), y(vy), z(vz) {} };
extern "C" { __device__ uint3 blockIdx, threadIdx, blockDim; }
extern "C" int cudaConfigureCall(dim3 gridDim, dim3 blockDim, unsigned long sharedMem = 0, void *stream = 0) { return 0; }

__global__ void while_sum(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    while (i < N) {
        sum += A[i] * B[i];
        i = i + 1;
    }
    C[0] = sum;
}

void launch_while(float* A, float* B, float* C, int N) {
    dim3 grid(4);
    dim3 block(4);
    while_sum<<<grid, block>>>(A, B, C, N);
}
