// gcc -o test_openblas main.c -mavx2
// run: ./test_openblas < input.in > result.out

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/time.h>
#include <immintrin.h>
#include <math.h>

#define MAX_SIZE 11398392
#define BLOCK_SIZE 32
#define EPS 1e-9

int M, K, N;
double A[MAX_SIZE], B[MAX_SIZE], C[MAX_SIZE], D[MAX_SIZE];

int getint(void)
{
    int res = 0;
    char c;
    while (isdigit(c = getchar())) res = res * 10 + c - '0';
    return res;
}

void naive()
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            for (int k = 0; k < K; ++k)
            {
                D[i * M + j] += A[i * M + k] * B[j * N + k];
            }
        }
    }
}

void kernel(int r1, int r2, int c1, int c2)
{
    // printf("%d %d - %d %d\n", r1, r2, c1, c2);
    for (int i = r1; i < r2; ++i)
    {
        for (int j = c1; j < c2; ++j)
        {
            for (int k = 0; k < K; ++k)
            {
                C[i * M + j] += A[i * M + k] * B[j * N + k];
            }
        }
    }
}

void matmul_tiling() {
    int r = 0;
    for (; r + 8 <= M; r += 8)
    {
        int c = 0;
        for (; c + 4 <= N; c += 4)
        {
            kernel(r, r + 8, c, c + 4);
        }
        if (c < N)
        {
            kernel(r, M, c, N);
        }
    }

    if (r + 4 <= M)
    {
        int c = 0;
        for (; c + 8 <= N; c += 8)
        {
            kernel(r, r + 4, c, c + 8);
        }
        if (c + 4 <= N)
        {
            kernel(r, r + 4, c, c + 4);
            c += 4;
        }
        if (c < N)
        {
            kernel(r, r + 4, c, N);
        }
        r += 4;
    }

    if (r < M)
    {
        int c = 0;
        for (; c + 8 <= N; c += 8)
        {
            kernel(r, M, c, c + 8);
        }
        if (c + 4 <= N)
        {
            kernel(r, M, c, c + 4);
            c += 4;
        }
        if (c < N)
        {
            kernel(r, M, c, N);
        }
    }
}

int main(int argc, char const* argv[]) 
{
    srand(0);
    for (int i = 0; i < MAX_SIZE; i++)
    {
        A[i] = (rand() % 100 + 1) / 10.0;
        B[i] = (rand() % 100 + 1) / 10.0;
    }
    int m, k, n;
    double duration_naive = 0, duration_tiling = 0;
    // while (1)
    for (int rep = 0; rep < 5000; ++rep)
    {
        M = getint();
        if (!M) break;
        K = getint();
        N = getint();

        for (int i = 0; i < m * n; ++i)
            C[i] = D[i] = 0;

        double duration;
        struct timeval start, finish;
        gettimeofday(&start, NULL);
        naive(m, k, n);
        gettimeofday(&finish, NULL);

        duration = ((double)(finish.tv_sec - start.tv_sec) * 1000000 +
            (double)(finish.tv_usec - start.tv_usec)) / 1000000;
        duration_naive += duration;

        gettimeofday(&start, NULL);
        matmul_tiling(m, k, n);
        gettimeofday(&finish, NULL);

        duration = ((double)(finish.tv_sec - start.tv_sec) * 1000000 +
            (double)(finish.tv_usec - start.tv_usec)) / 1000000;
        duration_tiling += duration;

        for (int i = 0; i < m * n; ++i)
        {
            if (fabs(C[i] - D[i]) > EPS)
            {
                printf("error: %d: %f and %f\n", i, C[i], D[i]);
            }
        }
    }
    printf("naive: %lf\n", duration_naive);
    printf("tiling: %lf\n", duration_tiling);

    return 0;
}