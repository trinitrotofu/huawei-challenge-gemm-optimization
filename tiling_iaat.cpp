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
     * arr[3 * index + 1] will store the number of times the first column tiling is used
     * arr[3 * index + 2] will store the number of leftover columns, which can always be tiled by one block
     */

    // Get the max column tilings that match the row size tiling
    int mx = 4;
    if (row_size <= 4) mx = 8;
    if (row_size <= 2) mx = 20;

    int max_element = min(mx, n);
    index *= 3;

    // Tile as many columns of size max_element as possible
    arr[index] = max_element;
    arr[index + 1] = n / max_element;

    // Tile the remaining amount of columns
    n -= n / max_element * max_element;
    arr[index + 2] = n;
}

void tile_greedy(int m, int n, int k, int tile_row[], int tile_col[]) {
    /*
     * n, m: Sizes of the array C
     * tile_row: Pointer to an array of length 4 (must be initialized to 0)
     * tile_col: Pointer to an array of length 12 (must be initialized to 0)
     *
     * Postconditions:
     * tile_row = {tile size 1, amount of times tile size 1 is used, tile size 2 (if needed), tile size 3 (if needed)}
     * tile_col = {(column tile size 1, amt of times tile size 1 is used, tile size 2 (if needed)),
     *            repeated 3 times for each row tiling}
     *
     */
    if (m >= 8) {
        // If n >= 8, as many 8s as possible - store the number of times we used row 8's in tile_row[1]
        tile_row[0] = 8, tile_row[1] = m / 8;
        m -= m / 8 * 8;
        tile_single_dim(n, 8, tile_col, 0);
    }

    if (n >= 4) {
        // For n >= 4, tile a row of 4
        tile_row[2] = 4, tile_row[3] = m - 4;

        tile_single_dim(n, 4, tile_col, 1);
        // If there's still rows remaining, tile that from blocks of row size{1, 2, 3}
        if (m - 4 > 0) tile_single_dim(n, m - 4, tile_col, 2);
    } else {
        // If n <= 3, then we can tile the entire remaining rows directly
        tile_row[2] = m;
        tile_single_dim(n, m, tile_col, 1);
    }
}

int cnt = 0;
void restore_columns(int curr_row, int row_size, int index, const int *tile_col, vector<vector<char>> &v) {
    int curr_col = 0;
    index *= 3;

    if (tile_col[index] != 0) {
        for (int i = 0; i < tile_col[index + 1]; i++) {
            for (int row = curr_row; row < curr_row + row_size; row++) {
                for (int col = curr_col; col < curr_col + tile_col[index]; col++) {
                    v[row][col] = (char) ('A' + cnt);
                }
            }
            cnt++;
            curr_col += tile_col[index];
        }
    }

    if (tile_col[index + 2] != 0) {
        for (int row = curr_row; row < curr_row + row_size; row++) {
            for (int col = curr_col; col < curr_col + tile_col[index + 2]; col++) {
                v[row][col] = (char) ('A' + cnt);
            }
        }
        cnt++;
    }
}

void restore(int m, int n, const int *tile_row, const int *tile_col) {
    // For testing purposes - prints out the array with corresponding tiles
    vector<vector<char>> v(m, vector<char>(n));
    cnt = 0;
    int curr_row = 0;

    if (tile_row[0] != 0) {
        for (int i = 0; i < tile_row[1]; i++) {
            restore_columns(curr_row, tile_row[0], 0, tile_col, v);
            curr_row += tile_row[0];
        }
    }
    if (tile_row[2] != 0) {
        restore_columns(curr_row, tile_row[2], 1, tile_col, v);
        curr_row += tile_row[2];
    }

    if (tile_row[3] != 0) {
        restore_columns(curr_row, tile_row[3], 2, tile_col, v);
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << v[i][j] << ' ';
        }
        cout << '\n';
    }
}


int main() {
    int tile_row[4];
    int tile_col[12];
    memset(tile_row, 0, sizeof(tile_row));
    memset(tile_col, 0, sizeof(tile_col));
    int k = 5;

    // m = num rows, n = num cols
    int m = 15, n = 21;
    cout << "m = " << m << ", " << "n = " << n << '\n';
    tile_greedy(m, n, k, tile_row, tile_col);
    restore(m, n, tile_row, tile_col);

}