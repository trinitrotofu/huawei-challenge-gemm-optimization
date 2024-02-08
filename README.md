# huawei-challenge-gemm-optimization

Implementation 1: ([NO_ALLOC_implementation_N_leq_4.cpp](https://github.com/trinitrotofu/huawei-challenge-gemm-optimization/blob/main/Implementations/UPDATED_implementation_N_leq_4.cpp)) specifically optimized for N <= 4

Implementation 2: ([tiling_anthony.cpp](https://github.com/trinitrotofu/huawei-challenge-gemm-optimization/blob/main/Implementations/Tiling/tiling_anthony.cpp)) a generalized version of implementation 1 that works for N > 4

Implementation 3: ([tiling_iaat_freeman_copy.cpp](https://github.com/trinitrotofu/huawei-challenge-gemm-optimization/blob/main/Implementations/Tiling/tiling_iaat_freeman_copy.cpp)) IAAT tiling with LIBXSMM kernels

Our main implementation and the one with the best results is Implementation 1.

Test data, provided by Huawei is located [here](https://github.com/trinitrotofu/huawei-challenge-gemm-optimization/tree/main/Benchmarking/gemm_inputs)

Implementation 1 achieved an average performance improvement of 5.46% over OpenBLAS on the given input data.
