#include <stdio.h>
#include <stdlib.h>

extern void matmul(
    float* A_alloc, float* A_aligned, long A_offset, long A_size, long A_stride,
    float* B_alloc, float* B_aligned, long B_offset, long B_size, long B_stride,
    float* C_alloc, float* C_aligned, long C_offset, long C_size, long C_stride,
    long N
);

int main() {
    int N = 4;
    float A[16], B[16], C[16];
    for (int i = 0; i < 16; i++) { A[i] = 1.0f; B[i] = 2.0f; C[i] = 0.0f; }
    matmul(A, A, 0, 16, 1, B, B, 0, 16, 1, C, C, 0, 16, 1, N);
    printf("Result C:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) printf("%.1f ", C[i*4+j]);
        printf("\n");
    }
    return 0;
}
