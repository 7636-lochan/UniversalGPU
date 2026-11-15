#include <iostream>
#include <cstdlib>
#include <cstring>

typedef void* cublasHandle_t;
typedef int cublasStatus_t;
typedef int cublasOperation_t;

#define CUBLAS_STATUS_SUCCESS 0

static void cpu_sgemm(
    int M, int N, int K,
    float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float beta,
    float* C, int ldc
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[j * ldc + i] = alpha * sum + beta * C[j * ldc + i];
        }
    }
}

extern "C" {
    cublasStatus_t cublasCreate(cublasHandle_t* handle) {
        std::cout << "[libcublas shim] cublasCreate\n";
        *handle = (void*)1;
        return CUBLAS_STATUS_SUCCESS;
    }
    cublasStatus_t cublasDestroy(cublasHandle_t handle) {
        return CUBLAS_STATUS_SUCCESS;
    }
    cublasStatus_t cublasSetStream(cublasHandle_t handle, void* stream) {
        return CUBLAS_STATUS_SUCCESS;
    }
    cublasStatus_t cublasSgemm(
        cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const float* alpha,
        const float* A, int lda, const float* B, int ldb,
        const float* beta, float* C, int ldc
    ) {
        std::cout << "[libcublas shim] cublasSgemm M=" << m << " N=" << n << " K=" << k << "\n";
        cpu_sgemm(m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
        return CUBLAS_STATUS_SUCCESS;
    }
    cublasStatus_t cublasLtCreate(void** ltHandle) {
        *ltHandle = (void*)1;
        return CUBLAS_STATUS_SUCCESS;
    }
    cublasStatus_t cublasLtDestroy(void* ltHandle) {
        return CUBLAS_STATUS_SUCCESS;
    }
}
