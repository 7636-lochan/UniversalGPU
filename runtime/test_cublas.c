#include <stdio.h>
extern int cublasCreate(void** handle);
extern int cublasSgemm(void* handle, int transa, int transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
int main() {
    void* handle = NULL;
    cublasCreate(&handle);
    int M = 2, N = 2, K = 2;
    float alpha = 1.0f, beta = 0.0f;
    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float C[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    printf("Testing cublasSgemm shim...\n");
    cublasSgemm(handle, 0, 0, M, N, K, &alpha, A, K, B, N, &beta, C, N);
    printf("Result C:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%.1f ", C[i*2 + j]);
        }
        printf("\n");
    }
    return 0;
}
