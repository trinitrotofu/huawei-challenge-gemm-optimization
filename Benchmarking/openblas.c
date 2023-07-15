// gcc -o test_openblas openblas.c libopenblas_vortexp-r0.3.23.dev.a
// the openblas library is installed using homebrew, might need to reinstall openblas if you're not using mac
// run: ./test_openblas < gemm_inputs/0.txt >> blas_out.txt
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/time.h>
#include "OpenBLAS/cblas.h"
//#include "cblas.h"
#define MAX_SIZE 11398392

double A[MAX_SIZE], B[MAX_SIZE], C[MAX_SIZE];

double test(int m, int k, int n)
{
//    double A[m * k], B[k * n], C[m * n];
//    double* A = (double*)malloc(sizeof(double) * m * k);
//    double* B = (double*)malloc(sizeof(double) * k * n);
//    double* C = (double*)malloc(sizeof(double) * m * n);
//    for (int i = 0; i < m * k; i++) A[i] = (rand() % 100) / 10.0;
//    for (int i = 0; i < k * n; i++) B[i] = (rand() % 100) / 10.0;
//    for (int i = 0; i < m * n; i++) C[i] = (rand() % 100) / 10.0;
    struct timeval start, finish;
    gettimeofday(&start, NULL);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, A, k, B, k, 0, C, n);
    gettimeofday(&finish, NULL);
    double duration = ((double)(finish.tv_sec - start.tv_sec) * 1000000 +
                       (double)(finish.tv_usec - start.tv_usec)) / 1000000;

//    free(A);
//    free(B);
//    free(C);
    return duration;
}

int getint(void)
{
    int res = 0;
    char c;
    while (isdigit(c = getchar())) res = res * 10 + c - '0';
    return res;
}

int main(int argc, char const* argv[])
{
//    freopen("outputs/blas_out.txt", "a", stdout);
    srand(0);
    for (int i = 0; i < MAX_SIZE; i++)
    {
        A[i] = (rand() % 100 + 1) / 10.0;
        B[i] = (rand() % 100 + 1) / 10.0;
    }
    int m, k, n;
    double tot_duration = 0;
    while (1)
    {
        m = getint();
        if (!m) break;
        k = getint();
        n = getint();
        tot_duration += test(m, k, n);
    }
    printf("%f\n", tot_duration);
    return 0;
}

/*
gcc -o test_openblas test_openblas.c -I /opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib -lopenblas -lpthread -lgfortran
gcc -o test_openblas openblas.c libopenblas_vortexp-r0.3.23.dev.a
 */
