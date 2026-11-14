#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
struct uint3 { unsigned int x, y, z; };
struct dim3 { unsigned int x, y, z; dim3(unsigned int vx=1, unsigned int vy=1, unsigned int vz=1) : x(vx), y(vy), z(vz) {} };
extern "C" { __device__ uint3 blockIdx, threadIdx, blockDim; }
extern "C" int cudaConfigureCall(dim3 gridDim, dim3 blockDim, unsigned long sharedMem = 0, void *stream = 0) { return 0; }

// 6D tensor: [batch][seq][head][dim][subdim][channel]
__global__ void tensor6d(float****** input, float****** output, int N) {
    int b = blockIdx.x;
    int s = threadIdx.x;
    int h = threadIdx.y;
    int d = threadIdx.z;
    int sd = blockIdx.y;
    int ch = blockIdx.z;
    if (b < N && s < N && h < N && d < N && sd < N && ch < N) {
        output[b][s][h][d][sd][ch] = input[b][s][h][d][sd][ch] + 1.0f;
    }
}
