#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/time.h>
#include <immintrin.h>

#define USING_OPENBLAS 0

#define MAX_SIZE 10000000

#if ( USING_OPENBLAS == 1 )

#include <cblas.h>

#else

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define KX 8
#define KY 16
#define S3 64     // # of columns of B to select
#define S2 120    // # of rows of A to select 
#define S1 240    // # of rows of B to select
#define VEC_LEN 4
#define KVY (KY / VEC_LEN)

typedef double vec __attribute__ (( vector_size(VEC_LEN * sizeof(double)) ));

void *vec_alloc(int n)
{
    void* ptr = (void*) aligned_alloc(
        VEC_LEN * sizeof(double),
        VEC_LEN * sizeof(double) * n
    );
    memset(ptr, 0, VEC_LEN * sizeof(double) * n);
    return ptr;
}

// A[x:x+6][l:r], B[l:r][y:y+16] => C[x:x+6][y:y+16]
void kernel(double *A, vec *B, vec *C, int x, int y, int l, int r, int Np, int K)
{
    vec t[KX][KVY] = {0};

    for (int k = l; k < r; k++) {
        for (int i = 0; i < KX; i++) {
            // A[x + i][k]
            vec a = (vec){0} + A[(x + i) * K + k];
            // A[x + i][k] * B[k][y:y+16] => t[i][0:16]
            for (int j = 0; j < KVY; j++)
                t[i][j] += a * B[(k * Np + y) / VEC_LEN + j];
        }
    }

    for (int i = 0; i < KX; i++)
        for (int j = 0; j < KVY; j++)
            C[((x + i) * Np + y) / VEC_LEN + j] += t[i][j];
}

void multiply(const double *_a, const double *_b, double *_c, int M, int N, int K)
{
    int Mp = (M + KX - 1) / KX * KX;
    int Np = (N + KY - 1) / KY * KY;

    double *a = (double *) vec_alloc(Mp * K / VEC_LEN);
    double *b = (double *) vec_alloc(K * Np / VEC_LEN);
    double *c = (double *) vec_alloc(Mp * Np / VEC_LEN);

    for (int i = 0; i < M; i += 1)
        memcpy(&a[i * K], &_a[i * K], sizeof(double) * K);

    for (int i = 0; i < N; i += 1)
    {
        // Tanspose B
        for (int j = 0; j < K; j += 1)
            b[(j * Np) + i] = _b[i * K + j];
        // memcpy(&b[i * Np], &_b[i * N], 8 * N);
    }

    for (int i3 = 0; i3 < Np; i3 += S3)
    {
        int lm3 = MIN(i3 + S3, Np);
        for (int i2 = 0; i2 < Mp; i2 += S2)
        {
            int lm2 = MIN(i2 + S2, Mp);
            for (int i1 = 0; i1 < K; i1 += S1)
            {
                int lm1 = MIN(i1 + S1, K);
                for (int x = i2; x < lm2; x += KX)
                    for (int y = i3; y < lm3; y += KY)
                        kernel(a, (vec*) b, (vec*) c, x, y, i1, lm1, Np, K);
            }
        }
    }

    for (int i = 0; i < M; i++)
        memcpy(&_c[i * N], &c[i * Np], sizeof(double) * N);
    
    free(a);
    free(b);
    free(c);
}

#endif

double A[MAX_SIZE], B[MAX_SIZE], C[MAX_SIZE];

double test(int m, int k, int n)
{
    struct timeval start, finish;
    gettimeofday(&start, NULL);
#if ( USING_OPENBLAS == 1 )
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, A, k, B, k, 0, C, n);
#else
    multiply(A, B, C, m, n, k);
#endif
    // printf("%d %d %d\n", m, n, k);
    gettimeofday(&finish, NULL);
    double duration = ((double)(finish.tv_sec - start.tv_sec) * 1000000 +
                       (double)(finish.tv_usec - start.tv_usec)) /
                      1000000;
    // printf("%f\n", duration);
    return duration;
}

int getint(void)
{
    int res = 0;
    char c;
    while (isdigit(c = getchar()))
        res = res * 10 + c - '0';
    return res;
}

int main(int argc, char const *argv[])
{
#if ( USING_OPENBLAS == 1 )
    openblas_set_num_threads(1);
#endif
    srand(19260817);
    for (int i = 0; i < MAX_SIZE; i += 1)
    {
        A[i] = (rand() % 100) / 10.0;
        B[i] = (rand() % 100) / 10.0;
    }
    int m, k, n;
    double tot_duration = 0;
    while (1)
    {
        m = getint();
        if (!m)
            break;
        k = getint();
        n = getint();
        tot_duration += test(m, k, n);
    }
    printf("%f\n", tot_duration);
    return 0;
}
