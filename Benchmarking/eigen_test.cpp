// Either uncomment the freopens below and run in vscode or something
// or run: gcc -o eigen_test eigen_test.cpp
// then: ./eigen_test < gemm_inputs/0.txt >> eigen_out.txt

// this is designed to only run on the first 100,000 inputs: uncomment line 70 to run on everything
#include <iostream>
#include "Eigen/Dense"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/time.h>

using Eigen::MatrixXd;
void init_mat(MatrixXd &mat, int n, int m);

double test(int m, int k, int n)
{
    MatrixXd A(m, k), B(k, n), C(m, n);

    struct timeval start, finish;
    gettimeofday(&start, NULL);
    C.noalias() += A * B;
    gettimeofday(&finish, NULL);
    double duration = ((double)(finish.tv_sec - start.tv_sec) * 1000000 +
                       (double)(finish.tv_usec - start.tv_usec)) / 1000000;

    return duration;
}


int getint(void)
{
    int res = 0;
    char c;
    while (isdigit(c = getchar())) res = res * 10 + c - '0';
    return res;
}


int main()
{
//    freopen("gemm_inputs/0.txt", "r", stdin);

//    freopen("test.txt", "r", stdin);
//    freopen("outputs/blas_out.txt", "a", stdout);
    srand(0);
//    for (int i = 0; i < MAX_SIZE; i++)
//    {
//        A[i] = (rand() % 100 + 1) / 10.0;
//        B[i] = (rand() % 100 + 1) / 10.0;
//    }
    int m, k, n;
    double tot_duration = 0;
    int cnt = 0;

    struct timeval start, finish;
    gettimeofday(&start, NULL);

    while (1)
    {
        m = getint();
        if (!m) break;
        k = getint();
        n = getint();
        tot_duration += test(m, k, n);

        cnt++;
        if (cnt % 500 == 0) std::cout << cnt << ", " << tot_duration << '\n';
        if (cnt == 100000) break;
    }
    printf("%f\n", tot_duration);

    gettimeofday(&finish, NULL);
    double duration = ((double)(finish.tv_sec - start.tv_sec) * 1000000 +
                       (double)(finish.tv_usec - start.tv_usec)) / 1000000;

//    freopen("outputs/eigen_out.txt", "a", stdout);
    std::cout << duration << '\n';
    return 0;
}