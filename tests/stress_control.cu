#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
struct uint3 { unsigned int x, y, z; };
struct dim3 { unsigned int x, y, z; dim3(unsigned int vx=1, unsigned int vy=1, unsigned int vz=1) : x(vx), y(vy), z(vz) {} };
extern "C" { __device__ uint3 blockIdx, threadIdx, blockDim; }
extern "C" int cudaConfigureCall(dim3 gridDim, dim3 blockDim, unsigned long sharedMem = 0, void *stream = 0) { return 0; }

__global__ void control_flow(float* data, int N) {
    int i = threadIdx.x;
    if (i < N) {
        int j = 0;
        while (j < 10) {
            for (int k = 0; k < 5; k++) {
                if (k % 2 == 0) {
                    data[i] += 1.0f;
                } else {
                    data[i] -= 1.0f;
                }
            }
            j++;
        }
        switch (i) {
            case 0: data[i] = 1.0f; break;
            case 1: data[i] = 2.0f; break;
            default: data[i] = 3.0f;
        }
    }
}
