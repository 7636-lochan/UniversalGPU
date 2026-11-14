#include <iostream>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

typedef int CUresult;
typedef void* CUdevice;
typedef void* CUcontext;
typedef unsigned long long CUdeviceptr;

CUresult (*p_cuDeviceGet)(CUdevice*, int) = nullptr;
CUresult (*p_cuCtxCreate)(CUcontext*, unsigned int, CUdevice) = nullptr;
CUresult (*p_cuMemAlloc)(CUdeviceptr*, size_t) = nullptr;
CUresult (*p_cuMemFree)(CUdeviceptr) = nullptr;

static bool initialized = false;
static CUcontext global_ctx = nullptr;

void init_hardware() {
    if (initialized) return;
    initialized = true;

    setenv("CUDA_VISIBLE_DEVICES", "0", 1);

    void* handle = dlopen("/usr/lib/wsl/lib/libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!handle) return;

    p_cuDeviceGet = (CUresult(*)(CUdevice*, int))dlsym(handle, "cuDeviceGet");
    p_cuCtxCreate = (CUresult(*)(CUcontext*, unsigned int, CUdevice))dlsym(handle, "cuCtxCreate_v2");
    p_cuMemAlloc = (CUresult(*)(CUdeviceptr*, size_t))dlsym(handle, "cuMemAlloc_v2");
    p_cuMemFree = (CUresult(*)(CUdeviceptr))dlsym(handle, "cuMemFree_v2");

    if (p_cuDeviceGet && p_cuCtxCreate) {
        CUdevice dev = 0;
        int r = p_cuDeviceGet(&dev, 0);
        if (r == 0) {
            p_cuCtxCreate(&global_ctx, 0, dev);
        }
    }
}

extern "C" {
    int cudaMalloc(void** devPtr, size_t size) {
        init_hardware();
        if (p_cuMemAlloc && global_ctx) {
            CUdeviceptr dptr = 0;
            int result = p_cuMemAlloc(&dptr, size);
            *devPtr = (void*)dptr;
            return result;
        }
        *devPtr = malloc(size);
        return 0;
    }

    int cudaFree(void* devPtr) {
        if (p_cuMemFree && global_ctx) return p_cuMemFree((CUdeviceptr)devPtr);
        free(devPtr);
        return 0;
    }

    int cudaMemcpy(void* dst, const void* src, size_t count, int kind) {
        memcpy(dst, src, count);
        return 0;
    }

    int cudaGetDeviceCount(int* count) {
        *count = 1;
        return 0;
    }

    int cudaSetDevice(int device) { return 0; }
    int cudaGetDevice(int* device) { *device = 0; return 0; }

    struct cudaDeviceProp {
        char name[256];
        size_t totalGlobalMem;
        int major;
        int minor;
    };

    int cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device) {
        strcpy(prop->name, "UniversalGPU (RTX 4050)");
        prop->totalGlobalMem = 6ULL * 1024 * 1024 * 1024;
        prop->major = 8;
        prop->minor = 9;
        return 0;
    }

    int cudaLaunchKernel(const void* func, int gridDimX, int blockDimX, void** args, size_t sharedMem) {
        return 0;
    }
}