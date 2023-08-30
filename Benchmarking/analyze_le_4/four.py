import os
import numpy as np
from matplotlib import pyplot as plt

parent = "/home/freeman/huawei_challenge/huawei-challenge-gemm-optimization/Benchmarking/gemm_inputs"
for _, _, f in os.walk(parent):
    for name in f:
        with open(f"{parent}/{name}", "r") as fp:
            data = np.loadtxt(fp, delimiter=",").astype(np.int32)
        n_data = data.size
        assert n_data % 3 == 0, "n_data % 3 != 0"
        data = data.reshape(n_data // 3, 3)
        data = data[data[:, 2] <= 4]
        # plot distribution of data[:, 0]
        fig, ax = plt.subplots()
        ax.hist(data[:, 0], bins=100)
        ax.set_title(f"Distribution of M for {name}")
        ax.set_xlabel("M")
        ax.set_ylabel("Frequency")
        fig.savefig(f"{name}_M.png")
        # plot distribution of data[:, 1]
        fig, ax = plt.subplots()
        ax.hist(data[:, 1], bins=100)
        ax.set_title(f"Distribution of K for {name}")
        ax.set_xlabel("K")
        ax.set_ylabel("Frequency")
        fig.savefig(f"{name}_K.png")
    
