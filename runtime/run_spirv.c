#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc < 2) { printf("Usage: %s <file.spv>\n", argv[0]); return 1; }
    
    // 1. Read SPIR-V binary
    FILE* f = fopen(argv[1], "rb");
    if (!f) { printf("Cannot open %s\n", argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* buf = malloc(size);
    fread(buf, 1, size, f);
    fclose(f);
    
    // 2. Get OpenCL platform (Intel)
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    
    // 3. Get Intel GPU device
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    // 4. Create context and queue
    cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, NULL);
    
    // 5. Create program from SPIR-V binary
    cl_int err;
    cl_program program = clCreateProgramWithIL(ctx, buf, size, &err);
    if (err != CL_SUCCESS) { printf("clCreateProgramWithIL failed: %d\n", err); return 1; }
    
    // 6. Build program
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Build failed: %d\n%s\n", err, log);
        return 1;
    }
    
    printf("SUCCESS: SPIR-V loaded and built on Intel GPU!\n");
    free(buf);
    return 0;
}
