// #include "include/cblas.h"

#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <random>
#include <algorithm>

#pragma GCC optimize("Ofast")
#pragma GCC target("avx2")

using namespace std;
const int MAX_SIZE = 2e7;

void naive(double *A, double *B, double *C, int m, int n, int k)
{
    for (int i = 0; i < m; ++i)    
        for (int j = 0; j < n; ++j)
            for (int v = 0; v < k; ++v)
                C[i * n + j] += A[i * k + v] * B[j * k + v];
}


/*
 * DGEMM CODE
 *
 */


// use vec as data type for 4-double vector
typedef double vec __attribute__ ((vector_size(32)));

// update C[x:x+8][0:4] using
// A[x:x+8][l:r] and B[l:r][0:4]
void kernel_8x4(double *A, vec *B, vec *C, int x, int l, int r, int K)
{
    // vectors to store block of matrix C
    vec c1, c2, c3, c4, c5, c6, c7, c8;

    // load 8x4 sub matrix of C into SIMD registers
    c1 = C[x  ];
    c2 = C[x+1];
    c3 = C[x+2];
    c4 = C[x+3];
    c5 = C[x+4];
    c6 = C[x+5];
    c7 = C[x+6];
    c8 = C[x+7];

    // iterate through both columns of A and rows of B
    for(int k = l; k < r; k++)
    {
        // repeatedly load an element of A into a register, then multiply it by a row of matrix B
        // add the result to corresponding row of matrix C

        // fills all four slots of a 4-double vector register with a value from A
        vec a1 = vec{} + A[(x + 0) * K + k];
        c1 += a1 * B[k];

        vec a2 = vec{} + A[(x + 1) * K + k];
        c2 += a2 * B[k];

        vec a3 = vec{} + A[(x + 2) * K + k];
        c3 += a3 * B[k];

        vec a4 = vec{} + A[(x + 3) * K + k];
        c4 += a4 * B[k];

        vec a5 = vec{} + A[(x + 4) * K + k];
        c5 += a5 * B[k];

        vec a6 = vec{} + A[(x + 5) * K + k];
        c6 += a6 * B[k];

        vec a7 = vec{} + A[(x + 6) * K + k];
        c7 += a7 * B[k];

        vec a8 = vec{} + A[(x + 7) * K + k];
        c8 += a8 * B[k];
    }

    // store results back into matrix C
    C[x  ] = c1;
    C[x+1] = c2;
    C[x+2] = c3;
    C[x+3] = c4;
    C[x+4] = c5;
    C[x+5] = c6;
    C[x+6] = c7;
    C[x+7] = c8;
}

// update C[x:x+4][0:4] using
// A[x:x+4][l:r] and B[l:r][0:4]
void kernel_4x4(double *A, vec *B, vec *C, int x, int l, int r, int K)
{
    // works in the same way as above kernel
    vec c1, c2, c3, c4;
    c1 = C[x  ];
    c2 = C[x+1];
    c3 = C[x+2];
    c4 = C[x+3];

    for(int k = l; k < r; k++)
    {
        vec a1 = vec{} + A[(x + 0) * K + k];
        c1 += a1 * B[k];
        vec a2 = vec{} + A[(x + 1) * K + k];
        c2 += a2 * B[k];
        vec a3 = vec{} + A[(x + 2) * K + k];
        c3 += a3 * B[k];
        vec a4 = vec{} + A[(x + 3) * K + k];
        c4 += a4 * B[k];
    }

    C[x  ] = c1;
    C[x+1] = c2;
    C[x+2] = c3;
    C[x+3] = c4;
}

// update C[x:x+2][0:4] using
// A[x:x+2][l:r] and B[l:r][0:4]
void kernel_2x4(double *A, vec *B, vec *C, int x, int l, int r, int K)
{
    // works in the same way as above kernel
    vec c1, c2;
    c1 = C[x  ];
    c2 = C[x+1];

    for(int k = l; k < r; k++)
    {
        vec a1 = vec{} + A[(x + 0) * K + k];
        c1 += a1 * B[k];
        vec a2 = vec{} + A[(x + 1) * K + k];
        c2 += a2 * B[k];
    }

    C[x  ] = c1;
    C[x+1] = c2;
}

// update C[x:x+1][0:4] using
// A[x:x+1][l:r] and B[l:r][0:4]
void kernel_1x4(double *A, vec *B, vec *C, int x, int l, int r, int K)
{
    // works in the same way as above kernel
    vec c1;
    c1 = C[x];

    for(int k = l; k < r; k++)
    {
        vec a1 = vec{} + A[(x + 0) * K + k];
        c1 += a1 * B[k];
    }

    C[x] = c1;
}


// memory aligned 4-double vectors to store temporary data for gemm calls
// faster to use these instead of allocating memory each time, but comes at the cost
// of knowing the MAX_SIZE of the arrays beforehand
alignas(64) vec _B[MAX_SIZE / 4], _C[MAX_SIZE / 4];

// matrix multiplication when B is transposed and N <= 4
void multiply_N4(double *A, double *B, double *C, int M, int N, int K)
{
    // allocate K memory-aligned 4-double vectors to store matrix B
//    vec *_B = alloc(K);

    // undo Transpose on B and store it in memory aligned array
    for (int i = 0; i < N; i++)
    {
        for(int j = 0; j < K; j++) {
            _B[j][i] = B[i * K + j];
        }
    }

    // allocate M memory-aligned 4-double vectors to store matrix C
//    vec *_C = alloc(M);

    const int lr = 240;

    // go through matrix C 8 rows at a time, and apply 8 row kernel

    int row = 0;
    for(; row + 8 <= M; row += 8) {
        // select 120 columns of A and 120 rows of B to call with the kernel
        for(int vals = 0; vals < K; vals += lr) {
            // call kernel to update _C[vals:vals+8][0:4]
            // using A[row:row+8][vals:vals+120] and _B[vals:vals+120][0:4]
            kernel_8x4(A, _B, _C, row, vals, min(vals + lr, K), K);
        }
    }

    // now there is up to 7 rows of C still not calculated
    // if there are 4 rows not calculated, apply 4 row kernel
    if(row + 4 <= M) {
        // select 120 columns of A and 120 rows of B to call with the kernel
        for(int vals = 0; vals < K; vals += lr) {
            // call kernel to update _C[vals:vals+8][0:4]
            // using A[row:row+8][vals:vals+120] and _B[vals:vals+120][0:4]
            kernel_4x4(A, _B, _C, row, vals, min(vals + lr, K), K);
        }
        row += 4;
    }

    // if there are 2 rows not calculated, apply 2 row kernel
    if(row + 2 <= M) {
        // select 120 columns of A and 120 rows of B to call with the kernel
        for(int vals = 0; vals < K; vals += lr) {
            // call kernel to update _C[vals:vals+8][0:4]
            // using A[row:row+8][vals:vals+120] and _B[vals:vals+120][0:4]
            kernel_2x4(A, _B, _C, row, vals, min(vals + lr, K), K);
        }
        row += 2;
    }

    // if there is one row not calculated, apply 1 row kernel
    if(row + 1 <= M) {
        // select 120 columns of A and 120 rows of B to call with the kernel
        for(int vals = 0; vals < K; vals += lr) {
            // call kernel to update _C[vals:vals+8][0:4]
            // using A[row:row+8][vals:vals+120] and _B[vals:vals+120][0:4]
            kernel_1x4(A, _B, _C, row, vals, min(vals + lr, K), K);
        }
    }

    // matrix _C is fully calculated
    // copy results from _C to result matrix C
    for (int i = 0; i < M; i++)
        memcpy(&C[i * N], &_C[i], sizeof(double) * N);

    // free memory in allocated _B and _C matrices
//    _aligned_free(_B);
//    _aligned_free(_C);
}


void multiply_tiling(double *A, double *B, double *C, int M, int N, int K)
{
    for (int n = 0; n < N; n += 4)
    {
        memset(_C, 0, sizeof(double) * 4 * M);

        // undo Transpose on B and store it in memory aligned array
        int rem = min(4, N - n);
        if (rem != 4)
            memset(_B, 0, sizeof(double) * 4 * K);

        for (int i = 0; i < rem; i++)
        {
            for(int j = 0; j < K; j++) {
                _B[j][i] = B[(i + n) * K + j];
            }
        }

        // allocate M memory-aligned 4-double vectors to store matrix C
        // vec *_C = alloc(M);

        const int lr = 240;

        // go through matrix C 8 rows at a time, and apply 8 row kernel

        int row = 0;
        for(; row + 8 <= M; row += 8) {
            // select 120 columns of A and 120 rows of B to call with the kernel
            for(int vals = 0; vals < K; vals += lr) {
                // call kernel to update _C[vals:vals+8][0:4]
                // using A[row:row+8][vals:vals+120] and _B[vals:vals+120][0:4]
                kernel_8x4(A, _B, _C, row, vals, min(vals + lr, K), K);
            }
        }

        // now there is up to 7 rows of C still not calculated
        // if there are 4 rows not calculated, apply 4 row kernel
        if(row + 4 <= M) {
            // select 120 columns of A and 120 rows of B to call with the kernel
            for(int vals = 0; vals < K; vals += lr) {
                // call kernel to update _C[vals:vals+8][0:4]
                // using A[row:row+8][vals:vals+120] and _B[vals:vals+120][0:4]
                kernel_4x4(A, _B, _C, row, vals, min(vals + lr, K), K);
            }
            row += 4;
        }

        // if there are 2 rows not calculated, apply 2 row kernel
        if(row + 2 <= M) {
            // select 120 columns of A and 120 rows of B to call with the kernel
            for(int vals = 0; vals < K; vals += lr) {
                // call kernel to update _C[vals:vals+8][0:4]
                // using A[row:row+8][vals:vals+120] and _B[vals:vals+120][0:4]
                kernel_2x4(A, _B, _C, row, vals, min(vals + lr, K), K);
            }
            row += 2;
        }

        // if there is one row not calculated, apply 1 row kernel
        if(row + 1 <= M) {
            // select 120 columns of A and 120 rows of B to call with the kernel
            for(int vals = 0; vals < K; vals += lr) {
                // call kernel to update _C[vals:vals+8][0:4]
                // using A[row:row+8][vals:vals+120] and _B[vals:vals+120][0:4]
                kernel_1x4(A, _B, _C, row, vals, min(vals + lr, K), K);
            }
        }

        // matrix _C is fully calculated
        // copy results from _C to result matrix C
        for (int i = 0; i < M; i++)
        {
            memcpy(&C[i * N + n], &_C[i], sizeof(double) * rem);
        }
    }
}


/*
 *  BENCHMARKING CODE
 *
 */

const double EPS = 1e-9;
double A[MAX_SIZE], B[MAX_SIZE], C[MAX_SIZE], D[MAX_SIZE];
void verify(double *_C, double *_D, int N, int M) {
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            if(fabs(_C[i * M + j] - _D[i * M + j]) > EPS) {
                printf("mismatch detected, %.10f\n", _C[i * M + j] - _D[i * M + j]);
                // printf("mismatch detected, %.10f %.10f\n", _C[i * M + j], _D[i * M + j]);
            }
        }
    }
}




double test(int m, int k, int n)
{
    struct timeval start, finish;

    // if (n <= 4)
    {
        gettimeofday(&start, NULL);
        // multiply_N4(A, B, C, m, n, k); 
        multiply_tiling(A, B, C, m, n, k);
        gettimeofday(&finish, NULL);

        naive(A, B, D, m, n, k);
    }
    // else
    // {
    //     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, A, k, B, k, 0, C, n);
    // }

    double duration = ((double)(finish.tv_sec - start.tv_sec) * 1000000 +
                       (double)(finish.tv_usec - start.tv_usec)) / 1000000;
    return duration;
}

int getint(void)
{
    int res = 0; char c;
    while (isdigit(c = getchar()))
        res = res * 10 + c - '0';
    return res;
}

int main(int argc, char const *argv[])
{
    // restrict openblast to a single thread
    // openblas_set_num_threads(1);

    // fill A and B with random data
    uniform_real_distribution<double> unif(1.0,1000.0);
    default_random_engine re;
    for (int i = 0; i < MAX_SIZE; i += 1)
    {
        A[i] = (double) unif(re);
        B[i] = (double) unif(re);

        A[i] = int(A[i]) / 100;
        B[i] = int(B[i]) / 100;
    }

    // for (int i = 0; i < 5; ++i)
    //     cerr << A[i] << " \n"[i == 4];
    // for (int i = 0; i < 5; ++i)
    //     cerr << B[i] << " \n"[i == 4];

    double tot_duration = 0.0;
    while(true) {
        int M = getint();
        if(!M) break;

        int K = getint();
        int N = getint();

        for (int i = 0; i < M * N; ++i)
            C[i] = D[i] = 0;
        for (int i = 0; i < M; ++i)
            _C[i] = vec{};

        tot_duration += test(M, K, N);

        // for (int i = 0; i < M * N; ++i)
        //     cerr << C[i] << " \n"[i == M * N - 1];
        // for (int i = 0; i < M * N; ++i)
        //     cerr << D[i] << " \n"[i == M * N - 1];

        verify(C, D, N, M);
    }


    printf("%f\n", tot_duration);
    return 0;
}