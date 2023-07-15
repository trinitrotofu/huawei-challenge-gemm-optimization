# Example: python3 aten.py gemm_inputs/0.txt aten_output.txt
# you can replace the input/output file names with whatever

import torch
import numpy as np
import sys

import time

in_file = sys.argv[1]
out_file = sys.argv[2]

with open(in_file) as f:
    for line in f:
        calls = line.split(",")

    for i in range(len(calls)):
        calls[i] = int(calls[i])

    total_time = 0
    total_calls = len(calls) // 3
    # total_calls = 1
    # total_calls = 100000
    # calls = [10, 10, 10]

    for curr_call in range(0, total_calls):
        m = calls[curr_call * 3]
        n = calls[curr_call * 3 + 1]
        k = calls[curr_call * 3 + 2]

        list_a = np.random.uniform(low=0.0, high=10.0, size=(m, k)) # Generate random 2d array of size m * k
        list_b = np.random.uniform(low=0.0, high=10.0, size=(k, n)) # size k * n

        A = torch.tensor(list_a)
        B = torch.tensor(list_b)

        start = time.time()
        result = torch.matmul(A, B)
        end = time.time()
        total_time += end - start

        # if curr_call % 1000 == 0:
        #     print(total_time)

    out = open(out_file, "a")
    out.write(str(total_time) + "\n")
