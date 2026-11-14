#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __shared__ __attribute__((shared))
struct uint3 { unsigned int x, y, z; };
struct dim3 { unsigned int x, y, z; dim3(unsigned int vx=1, unsigned int vy=1, unsigned int vz=1) : x(vx), y(vy), z(vz) {} };
extern "C" { __device__ uint3 blockIdx, threadIdx, blockDim; }
extern "C" int cudaConfigureCall(dim3 gridDim, dim3 blockDim, unsigned long sharedMem = 0, void *stream = 0) { return 0; }

__global__ void tiled_matmul(float* A, float* B, float* C, int N) {
    __shared__ float tileA[256];
    __shared__ float tileB[256];
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            tileA[k] = A[row * N + k];
            tileB[k] = B[k * N + col];
            sum += tileA[k] * tileB[k];
        }
        C[row * N + col] = sum;
    }
}

void launch_tiled(float* A, float* B, float* C, int N) {
    dim3 grid(32, 32);
    dim3 block(16, 16);
    tiled_matmul<<<grid, block>>>(A, B, C, N);
}
