#include "benchmarking.cpp"

enum storage_order { ROW_MAJOR, COLUMN_MAJOR };

// TODO: replace with SIMD
void transpose(double *__restrict__ const A, double *__restrict__ const T, const int m, const int n, const storage_order STORAGE_ORDER_FLAG, Stopwatch *sw) {
    if (sw != nullptr) sw->start();
    
    for (int i = 0; i < m; i += 8) {
        for (int j = 0; j < n; j += 8) {
            for (int ii = i; ii < i + 8 && ii < m; ii++) {
                for (int jj = j; jj < j + 8 && jj < n; jj++) {
                    if (STORAGE_ORDER_FLAG == ROW_MAJOR) {
                        T[jj * m + ii] = A[ii * n + jj];
                    } else {
                        T[ii * n + jj] = A[jj * m + ii];
                    }
                }
            }
        }
    }

    if (sw != nullptr) sw->stop();
}

void transpose(double *__restrict__ const A, double *__restrict__ const T, const int m, const int n, const storage_order STORAGE_ORDER_FLAG) {
    transpose(A, T, m, n, STORAGE_ORDER_FLAG, nullptr);
}
