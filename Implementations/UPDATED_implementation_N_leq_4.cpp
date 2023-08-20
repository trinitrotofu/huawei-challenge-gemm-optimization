#include "include/cblas.h"

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
    for(int i=0; i<m; i++)
        for(int v=0; v<k; v++)
            for(int j=0; j<n; j++)
                C[i*n+j]+=A[i*k+v]*B[v*n+j];
}



/*
 * DGEMM CODE
 *
 */


// use vec as data type for 4-double vector
typedef double vec __attribute__ ((vector_size(32)));

// allocate n 4-double vectors
vec* alloc(int n)
{
    vec* ptr = (vec*) _aligned_malloc(32 * n, 32);
    memset(ptr, 0, 32 * n);
    return ptr;
}

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



// matrix multiplication when B is transposed and N <= 4
void multiply_N4(double *A, double *B, double *C, int M, int N, int K)
{
    // allocate K memory-aligned 4-double vectors to store matrix B
    vec *_B = alloc(K);

    // undo Transpose on B and store it in memory aligned array
    for (int i = 0; i < N; i++)
    {
        for(int j = 0; j < K; j++) {
            _B[j][i] = B[i * K + j];
        }
    }

    // allocate M memory-aligned 4-double vectors to store matrix C
    vec *_C = alloc(M);

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
    _aligned_free(_B);
    _aligned_free(_C);
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
            if(fabs(_C[i * N + j] - _D[i * N + j]) > EPS) {
                printf("mismatch detected, %.10f\n", _C[i * N + j] - _D[i * N + j]);
            }
        }
    }
}




double test(int m, int k, int n)
{
    struct timeval start, finish;
    gettimeofday(&start, NULL);

    if(n > 4)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, A, k, B, k, 0, C, n);
    else
        multiply_N4(A, B, C, m, n, k);

    gettimeofday(&finish, NULL);

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
    openblas_set_num_threads(1);

    // fill A and B with random data
    uniform_real_distribution<double> unif(1.0,1000.0);
    default_random_engine re;
    for (int i = 0; i < MAX_SIZE; i += 1)
    {
        A[i] = (double) unif(re);
        B[i] = (double) unif(re);
    }

    double tot_duration = 0.0;
    while(true) {
        int M = getint();
        if(!M) break;

        int K = getint();
        int N = getint();

        tot_duration += test(M, K, N);
    }

    // verify(C, D, N, M);

    printf("%f\n", tot_duration);
    return 0;
}

















//
//void kernel_K1_OLD(const double *A, vec *B, vec *C, int n, int m) {
//    for(int i = 0; i < m; i++) {
//        vec av = vec{} + A[i];
//
//        for(int j = 0; j < n / 4; j++) {
//            C[(i * n) / 4 + j] += av * B[j];
//        }
//    }
//}
//
//void kernel_K1(const double *A, vec *B, vec *C, int n, int m) {
//
//    for(int bBlk = 0; bBlk < n / 4; bBlk += 12) {
//        vec bv[12]{};
//        for(int j = 0; j < min(bBlk + 12, n / 4) - bBlk; j++) bv[j] = B[j];
//
//        for(int i = 0; i < m; i++) {
//            vec av = vec{} + A[i];
//
//            for(int j = 0; j < min(bBlk + 12, n / 4) - bBlk; j++) {
//                C[(i * n) / 4 + j] += av * bv[j];
//            }
//        }
//    }
//}
//
//void multiply_K1(double * __restrict__ A, double * __restrict__ _B, double * __restrict__ _C, int m, int n)
//{
//    int n4 = (n + 3) / 4;
//
//    vec *B = alloc(n4);
//    vec *C = alloc(m * n4);
//
//    memcpy(B, _B, 8 * n);
//
//    kernel_K1(A, B, C, n4 * 4, m);
//
//    for(int i = 0; i < m; i++)
//        memcpy(&_C[i * n], &C[i * n4], 8 * n);
//
//    _aligned_free(B);
//    _aligned_free(C);
//}
//
// update 6x8 submatrix C[x:x+6][y:y+8]
// using A[x:x+6][l:r] and B[l:r][y:y+8]
//void kernel(double *A, vec *B, vec *C, int x, int y, int l, int r, int K, int N) {
//    vec tmp[6][2] = {0};
//
//    for(int k = l; k < r; k++) {
//        __builtin_prefetch(&B[(N * k + y) / 4 + 1]);
//        for(int i = 0; i < 6; i++) {
//            vec av = vec{} + A[(x + i) * K + k];
//
//            for(int j = 0; j < 2; j++) {
//                tmp[i][j] += av * B[(N * k + y) / 4 + j];
//            }
//        }
//    }
//
//    for(int i = 0; i < 6; i++) {
//        for(int j = 0; j < 2; j++) {
//            C[((x + i) * N + y) / 4 + j] += tmp[i][j];
//        }
//    }
//}
//
//void kernel(double *a, vec *b, vec *c, int x, int y, int l, int r, int K, int N) {
//    vec t00, t01, t10, t11, t20, t21, t30, t31, t40, t41, t50, t51;
//
//    t00 = c[((x + 0) * N + y) / 4 + 0];
//    t01 = c[((x + 0) * N + y) / 4 + 1];
//
//    t10 = c[((x + 1) * N + y) / 4 + 0];
//    t11 = c[((x + 1) * N + y) / 4 + 1];
//
//    t20 = c[((x + 2) * N + y) / 4 + 0];
//    t21 = c[((x + 2) * N + y) / 4 + 1];
//
//    t30 = c[((x + 3) * N + y) / 4 + 0];
//    t31 = c[((x + 3) * N + y) / 4 + 1];
//
//    t40 = c[((x + 4) * N + y) / 4 + 0];
//    t41 = c[((x + 4) * N + y) / 4 + 1];
//
//    t50 = c[((x + 5) * N + y) / 4 + 0];
//    t51 = c[((x + 5) * N + y) / 4 + 1];
//
//    for (int k = l; k < r; k++) {
//        vec a0 = vec{} + a[(x + 0) * K + k];
//        t00 += a0 * b[(k * N + y) / 4];
//        t01 += a0 * b[(k * N + y) / 4 + 1];
//
//        vec a1 = vec{} + a[(x + 1) * K + k];
//        t10 += a1 * b[(k * N + y) / 4];
//        t11 += a1 * b[(k * N + y) / 4 + 1];
//
//        vec a2 = vec{} + a[(x + 2) * K + k];
//        t20 += a2 * b[(k * N + y) / 4];
//        t21 += a2 * b[(k * N + y) / 4 + 1];
//
//        vec a3 = vec{} + a[(x + 3) * K + k];
//        t30 += a3 * b[(k * N + y) / 4];
//        t31 += a3 * b[(k * N + y) / 4 + 1];
//
//        vec a4 = vec{} + a[(x + 4) * K + k];
//        t40 += a4 * b[(k * N + y) / 4];
//        t41 += a4 * b[(k * N + y) / 4 + 1];
//
//        vec a5 = vec{} + a[(x + 5) * K + k];
//        t50 += a5 * b[(k * N + y) / 4];
//        t51 += a5 * b[(k * N + y) / 4 + 1];
//    }
//
//    c[((x + 0) * N + y) / 4 + 0] = t00;
//    c[((x + 0) * N + y) / 4 + 1] = t01;
//
//    c[((x + 1) * N + y) / 4 + 0] = t10;
//    c[((x + 1) * N + y) / 4 + 1] = t11;
//
//    c[((x + 2) * N + y) / 4 + 0] = t20;
//    c[((x + 2) * N + y) / 4 + 1] = t21;
//
//    c[((x + 3) * N + y) / 4 + 0] = t30;
//    c[((x + 3) * N + y) / 4 + 1] = t31;
//
//    c[((x + 4) * N + y) / 4 + 0] = t40;
//    c[((x + 4) * N + y) / 4 + 1] = t41;
//
//    c[((x + 5) * N + y) / 4 + 0] = t50;
//    c[((x + 5) * N + y) / 4 + 1] = t51;
//}
//
//
//alignas(64) double _A[MAX_SIZE], _B[MAX_SIZE], _C[MAX_SIZE];
//void multiply(double *A, double *B, double *C, int M, int N, int K)
//{
//    // rounded K and N to nearest 8
//    int K8 = (K + 7) / 8 * 8;
//    int N8 = (N + 7) / 8 * 8;
//
//    // rounded M to nearest 6
//    int M6 = (M + 5) / 6 * 6;
//
//
//    // load A into memory aligned region
//    for(int i = 0; i < M; i++) {
//        memcpy(&_A[i * K8], &A[i * K], 8 * K);
//    }
//
//    // load B into memory aligned region
//    for(int i = 0; i < K; i++) {
//        memcpy(&_B[i * N8], &B[i * N], 8 * N);
//    }
//
//    const int s3 = 48;
//    const int s2 = 96;
//    const int s1 = 192;
//
//    for(int i3 = 0; i3 < N8; i3 += s3) {
//        for(int i2 = 0; i2 < M6; i2 += s2) {
//            for(int i1 = 0; i1 < K8; i1 += s1) {
//
//                for(int row = i2; row < min(i2 + s2, M6); row += 6) {
//                    for(int col = i3; col < min(i3 + s3, K8); col += 8) {
//                        kernel(_A, (vec*) _B, (vec*)_C, row, col, i1, min(i1 + s1, K8), K, N8);
//                    }
//                }
//            }
//        }
//    }
//
//    for (int i = 0; i < M; i++)
//        memcpy(&C[i * N], &_C[i * N8], 8 * N);
//}
