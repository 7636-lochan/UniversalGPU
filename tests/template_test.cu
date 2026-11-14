#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
struct dim3 { unsigned int x, y, z; dim3(unsigned int vx=1, unsigned int vy=1, unsigned int vz=1) : x(vx), y(vy), z(vz) {} };
extern "C" { __device__ unsigned int threadIdx_x; }
extern "C" int cudaConfigureCall(dim3 gridDim, dim3 blockDim, unsigned long sharedMem = 0, void *stream = 0) { return 0; }

template <typename T>
__global__ void vector_add(T* A, T* B, T* C, int N) {
    int i = 0;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void launch_float(float* A, float* B, float* C, int N) {
    dim3 grid(4);
    dim3 block(4);
    vector_add<float><<<grid, block>>>(A, B, C, N);
}

void launch_half(__fp16* A, __fp16* B, __fp16* C, int N) {
    dim3 grid(4);
    dim3 block(4);
    vector_add<__fp16><<<grid, block>>>(A, B, C, N);
}
