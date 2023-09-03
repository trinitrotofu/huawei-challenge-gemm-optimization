#include <iostream>
#include <vector>

using namespace std;

//int count_memops(int n, int m, int k) {
// //TODO: implement this and compare memops saved compared to traditional implementation
//}

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

// Naively computes C[x:x+mc][y:y+nc]
void call_kernel(int mc, int nc, int m, int n, int k, int x, int y, double *A, double *B, double *C) {
    // mc, nc: kernel size
    // m, n, k: size of input matrices
    // x, y: x is the topmost row, y is the leftmost column (points to the top-left corner)
    // A, B, C: the matrices
    // TODO: replace this function with libxsmm kernel calls
    for (int i = x; i < x + mc; i++) {
        for (int j = y; j < y + nc; j++) {
            C[i * n + j] = 0;
            for (int l = 0; l < k; l++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
//                cout << "Adding A[" << i * k + l << "] * B[" << l * n + j << "] to C[" << i * n + j << "]\n";
            }
        }
    }
}

void dgemm(int m, int n, int k, double *A, double *B, double *C) {
    int tile_row[3];
    int tile_col[6];
    memset(tile_row, 0, sizeof(tile_row));
    memset(tile_col, 0, sizeof(tile_col));

    tile_greedy(m, n, k, tile_row, tile_col);

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
//        cout << m << ' ' << n << ' ' << k << '\n';
        double A[m * k], B[k * n], C[m * n];
        double A2[m * k], B2[k * n], C2[m * n];
        for (int j = 0; j < m * k; j++) {
            A[j] = A2[j] = (double) (rand() % 100) / 10.0;
        }

//        cout << '\n';

        for (int j = 0; j < k * n; j++) B[j] = B2[j] = (double) (rand() % 100) / 10.0;
        for (int j = 0; j < m * n; j++) C[j] = C2[j] = (double) (rand() % 100) / 10.0;

//        debug(A, m, k);
//        debug(B, k, n);
//        debug(C, m, n);

        naive(m, n, k, A, B, C);
        dgemm(m, n, k, A2, B2, C2);

        for (int j = 0; j < m * n; j++) {
            if (C[j] != C2[j]) cout << fixed << "mismatch detected " << abs(C2[j] - C[j]) << '\n';
        }

//        debug(C, m, n);
//        debug(C2, m, n);

    }
}

int main() {
//    int tile_row[3];
//    int tile_col[6];
//    memset(tile_row, 0, sizeof(tile_row));
//    memset(tile_col, 0, sizeof(tile_col));
//    int k = 5;

    // m = num rows, n = num cols
//    int m = 16, n = 24;
//    cout << "m = " << m << ", " << "n = " << n << '\n';
//    tile_greedy(m, n, k, tile_row, tile_col);
//    restore(m, n, tile_row, tile_col);

    check_correctness();



}
