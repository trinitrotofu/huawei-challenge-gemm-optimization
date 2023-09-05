#include <vector>
#include <sys/time.h>
#include <iostream>
#include <cstring>
#include <random>
#include <algorithm>

#include <libxsmm.h>
#include <cblas.h>
#include "utils/matrix.cpp"

#pragma GCC optimize("Ofast")
#pragma GCC target("avx2")

#define MAX(a, b) ((a) > (b) ? (a) : (b)
#define MIN(a, b) ((a) < (b) ? (a) : (b))

using namespace std;
const int MAX_SIZE = 2e7;
bool OURS = true;

const double EPS = 1e-9;
alignas(64) double A[MAX_SIZE], B[MAX_SIZE], C[MAX_SIZE], D[MAX_SIZE];
alignas(64) double A2[MAX_SIZE], B2[MAX_SIZE], C2[MAX_SIZE];
alignas(64) double A3[MAX_SIZE], B3[MAX_SIZE], C3[MAX_SIZE];

void tile_single_dim(int n, int row_size, int *arr, int index) {
    /*
     * index: the index we are working on in arr (points to 0, 1, or 2)
     * index points to which row we're on (0 indicates column tiling of row tiling 0,
     * 1 indicates column tiling of row tiling 1, etc)
     *
     * after this function call,
     * arr[3 * index] will store the size of the first column tiling
     * arr[3 * index + 1] will store the number of leftover columns, which can always be tiled by one block
     */

    // Get the max column tilings that match the row size tiling
    int mx = 4;
    if (row_size <= 4) mx = 8;
    if (row_size <= 2) mx = 20;

    int max_element = min(mx, n);
    index *= 2;

    // Tile as many columns of size max_element as possible
    arr[index] = max_element;

    // Tile the remaining amount of columns
    n -= n / max_element * max_element;
    arr[index + 1] = n;
}

void tile_greedy(int m, int n, int k, int tile_row[], int tile_col[]) {
    /*
     * m, n: Sizes of the array C
     * tile_row: Pointer to an array of length 3 (must be initialized to 0)
     * tile_col: Pointer to an array of length 6 (must be initialized to 0)
     *
     * Postconditions:
     * tile_row = {tile size 1, tile size 2 (if needed), tile size 3 (if needed)}
     * tile_col = {(column tile size 1, tile size 2 (if needed)),
     *            repeated 3 times for each row tiling}
     *
     */
    if (m >= 8) {
        // If m >= 8, as many 8s as possible - store the number of times we used row 8's in tile_row[1]
        tile_row[0] = 8;
        m -= m / 8 * 8;
        tile_single_dim(n, 8, tile_col, 0);
    }

    if (m >= 4) {
        // For m >= 4, tile a row of 4
        tile_row[1] = 4, tile_row[2] = m - 4;

        tile_single_dim(n, 4, tile_col, 1);
        // If there's still rows remaining, tile that from blocks of row size{1, 2, 3}
        if (m - 4 > 0) tile_single_dim(n, m - 4, tile_col, 2);
    } else {
        // If m <= 3, then we can tile the entire remaining rows directly
        tile_row[1] = m;
        tile_single_dim(n, m, tile_col, 1);
    }
}

int cnt = 0;
void restore_columns(int curr_row, int row_size, int index, int n, const int *tile_col, vector<vector<char> > &v) {
    int curr_col = 0;
    index *= 2;

    if (tile_col[index] != 0) {
        for (int i = 0; i < n / tile_col[index]; i++) {
            for (int row = curr_row; row < curr_row + row_size; row++) {
                for (int col = curr_col; col < curr_col + tile_col[index]; col++) {
                    v[row][col] = (char) ('A' + cnt);
                }
            }
            cnt++;
            curr_col += tile_col[index];
        }
    }

    if (tile_col[index + 1] != 0) {
        for (int row = curr_row; row < curr_row + row_size; row++) {
            for (int col = curr_col; col < curr_col + tile_col[index + 1]; col++) {
                v[row][col] = (char) ('A' + cnt);
            }
        }
        cnt++;
    }
}

void restore(int m, int n, const int *tile_row, const int *tile_col) {
    // For testing purposes - prints out the array with corresponding tiles
    vector<vector<char> > v(m, vector<char>(n));
    cnt = 0;
    int curr_row = 0;

    if (tile_row[0] != 0) {
        for (int i = 0; i < m / tile_row[0]; i++) {
            restore_columns(curr_row, tile_row[0], 0, n, tile_col, v);
            curr_row += tile_row[0];
        }
    }
    if (tile_row[1] != 0) {
        restore_columns(curr_row, tile_row[1], 1, n, tile_col, v);
        curr_row += tile_row[1];
    }

    if (tile_row[2] != 0) {
        restore_columns(curr_row, tile_row[2], 2, n, tile_col, v);
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << v[i][j] << ' ';
        }
        cout << '\n';
    }
}

void debug(double *A, int n, int m) {
    cout.precision(5);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cout << fixed << A[i * m + j] << ' ';
        }
        cout << '\n';
    }
}

// Computes C[x:x+mc][y:y+nc]
void call_kernel(const int mc, const int nc, const int m, const int n, const int k, const int x, const int y, double *__restrict__ const A, double *__restrict__ const B, double *__restrict__ const C) {
    // mc, nc: kernel size
    // m, n, k: size of input matrices
    // x, y: x is the topmost row, y is the leftmost column (points to the top-left corner)
    // A, B, C: the matrices
    // A2, B2 are transposed versions of A, B

    // Copy A2[][x:x+mc] to A3
    for (int i = 0; i < k; i++) {
        memcpy(&A3[i * mc], &A2[i * m + x], sizeof(double) * mc);
    }
    // Copy B2[y:y+nc][] to B3
    for (int i = 0; i < nc; i++) {
        memcpy(&B3[i * k], &B2[(y + i) * k], sizeof(double) * k);
    }

    memset(C2, 0, sizeof(double) * mc * nc);

    typedef libxsmm_mmfunction<double> kernel_type;
    kernel_type kernel(LIBXSMM_GEMM_FLAG_NONE, mc, nc, k, 1.0 /*alpha*/, 1.0 /*beta*/);
    kernel(&A3[0], &B3[0], &C2[0]);

    for (int i = 0; i < mc; i += 8) {
        for (int j = 0; j < nc; j += 8) {
            for (int ii = i; ii < i + 8 && ii < mc; ii++) {
                for (int jj = j; jj < j + 8 && jj < nc; jj++) {
                    C[(x + ii) * n + y + jj] = C2[jj * mc + ii];
                }
            }
        }
    }
}

void dgemm(const int m, const int n, const int k, double *__restrict__ const A, double *__restrict__ const B, double *__restrict__ const C) {
    int tile_row[3];
    int tile_col[6];
    memset(tile_row, 0, sizeof(tile_row));
    memset(tile_col, 0, sizeof(tile_col));

    tile_greedy(m, n, k, tile_row, tile_col);

    transpose(A, A2, m, k, ROW_MAJOR);
    transpose(B, B2, k, n, ROW_MAJOR);

    int curr_row = 0;
    for (int i = 0; i < 3; i++) {
        if (!tile_row[i]) continue;
        for (; curr_row + tile_row[i] <= m; curr_row += tile_row[i]) {
            // Call kernels based on the column tiling
            int curr_col = 0;
            if (tile_col[2 * i]) {
                for (; curr_col + tile_col[2 * i] <= n; curr_col += tile_col[2 * i]) {
                    call_kernel(tile_row[i], tile_col[2 * i], m, n, k, curr_row, curr_col, A, B, C);
                }
            }
            // tile the remaining columns
            if (tile_col[2 * i + 1]) {
                call_kernel(tile_row[i], tile_col[2 * i + 1], m, n, k, curr_row, curr_col, A, B, C);
            }
        }
    }
}

void naive(int m, int n, int k, double *A, double *B, double *C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int l = 0; l < k; l++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

void check_correctness() {
    int tests = 100;
    cout.precision(1);
    srand(1249322);
    for (int i = 0; i < tests; i++) {
        int n = rand() % 100, m = rand() % 100, k = rand() % 100;
        double A[m * k], B[k * n], C[m * n];
        double A2[m * k], B2[k * n], C2[m * n];
        for (int j = 0; j < m * k; j++) {
            A[j] = A2[j] = (double) (rand() % 100) / 10.0;
        }

        for (int j = 0; j < k * n; j++) B[j] = B2[j] = (double) (rand() % 100) / 10.0;
        for (int j = 0; j < m * n; j++) C[j] = C2[j] = (double) (rand() % 100) / 10.0;

        naive(m, n, k, A, B, C);
        dgemm(m, n, k, A2, B2, C2);

        for (int j = 0; j < m * n; j++) {
            if (C[j] - C2[j] > 1e-5) {
                cout << fixed << "mismatch detected " << abs(C2[j] - C[j]) << '\n';
            }
        }
    }
}

/*
 *  ANTHONY'S BENCHMARKING CODE
 */

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
    
    if (OURS) dgemm(m, n, k, A, B, C);
    else cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, A, k, B, k, 0, C, n);

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

/*
g++ tiling_iaat_freeman_copy.cpp -o bill -I/home/freeman/anaconda3/include -L/home/freeman/anaconda3/lib -l:libxsmm.a -lopenblas -ldl -lpthread -Ofast -march=native
*/
