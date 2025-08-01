# Nsight Profile Analysis Report

## How to run
```bash 
uv run analyze_all_nsight_profiles.py ../outputs/nsys/ --debug
```

## COMPREHENSIVE NSIGHT PROFILING ANALYSIS


Analyzed 20 profile files:

## Question (a): Forward Pass Timing

| Model   |   ctx128 |   ctx256 |   ctx512 |   ctx1024 |
|:--------|---------:|---------:|---------:|----------:|
| small   |    28.97 |    27.17 |    27.79 |     45.54 |
| medium  |    52.87 |    53.47 |    55.52 |    124.74 |
| large   |    79.57 |    81.03 |    98.3  |    242.02 |
| xl      |   105.6  |   108.24 |   174.69 |    441.22 |
| 2.7B    |    72.52 |   104.47 |   219.19 |    509.53 |

These timings should be compared with your Python benchmarking results.
Any differences might be due to:
- CUDA synchronization differences
- Overhead from profiling
- Warm-up effects

## Question (b): Most Time-Consuming CUDA Kernels

### Summary (Kernel Type by Model/Context):
| Model   | ctx128         | ctx256         | ctx512         | ctx1024        |
|:--------|:---------------|:---------------|:---------------|:---------------|
| small   | Elem (14.7ms)  | Elem (16.5ms)  | Elem (38.9ms)  | Elem (102.7ms) |
| medium  | Elem (43.0ms)  | Elem (47.4ms)  | Elem (102.9ms) | Elem (271.6ms) |
| large   | Elem (91.7ms)  | GEMM (111.8ms) | Elem (191.8ms) | Elem (506.9ms) |
| xl      | Elem (175.3ms) | Elem (204.3ms) | GEMM (371.7ms) | GEMM (217.4ms) |
| 2.7B    | Elem (299.0ms) | GEMM (457.3ms) | GEMM (552.5ms) | GEMM (504.3ms) |

### Detailed View (All Models):

#### SMALL Model:
| Model   |   Context | Top Kernel                                                   |   Time (ms) |   Count |   Avg (μs) |
|:--------|----------:|:-------------------------------------------------------------|------------:|--------:|-----------:|
| small   |       128 | void at::native::elementwise_kernel<(int)128, (int)2, voi... |       14.69 |    1446 |       10.2 |
| small   |       256 | void at::native::elementwise_kernel<(int)128, (int)2, voi... |       16.54 |    1446 |       11.4 |
| small   |       512 | void at::native::vectorized_elementwise_kernel<(int)4, at... |       38.87 |    1200 |       32.4 |
| small   |      1024 | void at::native::vectorized_elementwise_kernel<(int)4, at... |      102.69 |    1200 |       85.6 |

#### MEDIUM Model:
| Model   |   Context | Top Kernel                                                   |   Time (ms) |   Count |   Avg (μs) |
|:--------|----------:|:-------------------------------------------------------------|------------:|--------:|-----------:|
| medium  |       128 | void at::native::elementwise_kernel<(int)128, (int)2, voi... |       42.97 |    2886 |       14.9 |
| medium  |       256 | void at::native::elementwise_kernel<(int)128, (int)2, voi... |       47.43 |    2886 |       16.4 |
| medium  |       512 | void at::native::vectorized_elementwise_kernel<(int)4, at... |      102.89 |    2376 |       43.3 |
| medium  |      1024 | void at::native::vectorized_elementwise_kernel<(int)4, at... |      271.64 |    2376 |      114.3 |

#### LARGE Model:
| Model   |   Context | Top Kernel                                                   |   Time (ms) |   Count |   Avg (μs) |
|:--------|----------:|:-------------------------------------------------------------|------------:|--------:|-----------:|
| large   |       128 | void at::native::elementwise_kernel<(int)128, (int)2, voi... |       91.67 |    4326 |       21.2 |
| large   |       256 | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x1... |      111.76 |     763 |      146.5 |
| large   |       512 | void at::native::vectorized_elementwise_kernel<(int)4, at... |      191.8  |    3552 |       54   |
| large   |      1024 | void at::native::vectorized_elementwise_kernel<(int)4, at... |      506.89 |    3552 |      142.7 |

#### XL Model:
| Model   |   Context | Top Kernel                                                   |   Time (ms) |   Count |   Avg (μs) |
|:--------|----------:|:-------------------------------------------------------------|------------:|--------:|-----------:|
| xl      |       128 | void at::native::elementwise_kernel<(int)128, (int)2, voi... |      175.27 |    5766 |       30.4 |
| xl      |       256 | void at::native::elementwise_kernel<(int)128, (int)2, voi... |      204.3  |    5766 |       35.4 |
| xl      |       512 | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x1... |      371.74 |     864 |      430.2 |
| xl      |      1024 | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x1... |      217.42 |     281 |      773.7 |

#### 2.7B Model:
| Model   |   Context | Top Kernel                                                   |   Time (ms) |   Count |   Avg (μs) |
|:--------|----------:|:-------------------------------------------------------------|------------:|--------:|-----------:|
| 2.7B    |       128 | void at::native::elementwise_kernel<(int)128, (int)2, voi... |      299.01 |    3846 |       77.7 |
| 2.7B    |       256 | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x1... |      457.32 |    1575 |      290.4 |
| 2.7B    |       512 | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x1... |      552.54 |    1158 |      477.2 |
| 2.7B    |      1024 | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x1... |      504.29 |     547 |      921.9 |

### Analysis:
- 6/20 configurations have GEMM as the top kernel
- Smaller models are dominated by elementwise operations
- Larger models shift to GEMM-dominated computation
- The transition occurs around the 'large' model size

Legend: GEMM=Matrix Multiply, Elem=Elementwise, Soft=Softmax, Red=Reduction

## Question (c): Non-Matrix-Multiply Kernels

| Model   |   Context |   Total (ms) |   GEMM % |   Softmax % |   Elementwise % |   Other % |
|:--------|----------:|-------------:|---------:|------------:|----------------:|----------:|
| small   |       128 |        107.9 |     37.1 |         3.3 |            55.3 |       3.3 |
| small   |       256 |        169.8 |     41.4 |         2.6 |            52.6 |       2.6 |
| small   |       512 |        343.6 |     35.3 |         3.1 |            57.9 |       3.1 |
| small   |      1024 |        857   |     29.7 |         3   |            63.8 |       3   |
| medium  |       128 |        296.5 |     42.4 |         2.4 |            52   |       2.4 |
| medium  |       256 |        466.7 |     44.1 |         2   |            51.2 |       2   |
| medium  |       512 |        984.1 |     41.2 |         2.5 |            53.3 |       2.5 |
| medium  |      1024 |       2451.1 |     35.1 |         2.6 |            59.3 |       2.6 |
| large   |       128 |        583.3 |     42.8 |         2   |            52.6 |       2   |
| large   |       256 |        945.4 |     45.2 |         1.7 |            50.8 |       1.7 |
| large   |       512 |       1948.4 |     43.1 |         2.2 |            52   |       2.2 |
| large   |      1024 |       4719.5 |     35.9 |         2.5 |            58.7 |       2.5 |
| xl      |       128 |       1006.7 |     41.7 |         1.6 |            54.5 |       1.7 |
| xl      |       256 |       1766.5 |     47.3 |         1.6 |            49   |       1.6 |
| xl      |       512 |       3574.4 |     46.6 |         1.9 |            49.1 |       1.9 |
| xl      |      1024 |       1251.9 |     42.9 |         3.3 |            50.5 |       3.3 |
| 2.7B    |       128 |       1522.5 |     45.8 |         0.9 |            52   |       0.9 |
| 2.7B    |       256 |       2469   |     53.1 |         1   |            44.5 |       1   |
| 2.7B    |       512 |       4664.7 |     55   |         1.3 |            42.1 |       1.3 |
| 2.7B    |      1024 |       1391   |     54.1 |         2.5 |            40.9 |       2.5 |

Significant non-GEMM kernels across models:

ELEMENTWISE kernels:
  - void at::native::vectorized_elementwise_kernel<(int)4, at::n... (506.89 ms)
  - void at::native::elementwise_kernel<(int)128, (int)2, void a... (479.70 ms)
  - void at::native::elementwise_kernel<(int)128, (int)2, void a... (390.90 ms)

OTHER kernels:
  - void at::native::reduce_kernel<(int)512, (int)1, at::native:... (153.48 ms)
  - void at::native::reduce_kernel<(int)512, (int)1, at::native:... (60.61 ms)
  - void at::native::reduce_kernel<(int)128, (int)4, at::native:... (10.66 ms)

## Question (d): Training Step Breakdown

| Model   |   Context |   Forward (ms) |   Train Step (ms) |   Forward % | Backward (ms)   |   Optimizer (ms) |
|:--------|----------:|---------------:|------------------:|------------:|:----------------|-----------------:|
| small   |       128 |          28.97 |             99.87 |        29   | 298.58          |            33.76 |
| small   |       256 |          27.17 |             91.06 |        29.8 | 296.62          |            23.88 |
| small   |       512 |          27.79 |             91.6  |        30.3 | 310.17          |            22.59 |
| small   |      1024 |          45.54 |            162.48 |        28   | 365.91          |            23.13 |
| medium  |       128 |          52.87 |            207.79 |        25.4 | 480.10          |            80.49 |
| medium  |       256 |          53.47 |            190.25 |        28.1 | 508.86          |            58.05 |
| medium  |       512 |          55.52 |            214.94 |        25.8 | 543.51          |            45.06 |
| medium  |      1024 |         124.74 |            439.28 |        28.4 | 1516.65         |            45.42 |
| large   |       128 |          79.57 |            322.39 |        24.7 | 732.32          |           124.73 |
| large   |       256 |          81.03 |            325.64 |        24.9 | 734.68          |           122.12 |
| large   |       512 |          98.3  |            378.15 |        26   | 1104.45         |            72.76 |
| large   |      1024 |         242.02 |            821.19 |        29.5 | 3044.81         |            73.24 |
| xl      |       128 |         105.6  |            428.41 |        24.6 | 946.53          |           165.77 |
| xl      |       256 |         108.24 |            474.99 |        22.8 | 1041.70         |           163.46 |
| xl      |       512 |         174.69 |            708.12 |        24.7 | 2270.51         |           151.15 |
| xl      |      1024 |         441.22 |            272.19 |       162.1 | 0.00*           |             0.76 |
| 2.7B    |       128 |          72.52 |            464.48 |        15.6 | 766.62          |           232.6  |
| 2.7B    |       256 |         104.47 |            590.47 |        17.7 | 1362.63         |           231.94 |
| 2.7B    |       512 |         219.19 |            938.14 |        23.4 | 2885.45         |           231.36 |
| 2.7B    |      1024 |         509.53 |            455.95 |       111.8 | 0.00*           |             0.87 |

Matrix multiplication fraction (estimated):
- Forward only: 43.0%
- Full training step: 34.4%
The fraction decreases because optimizer operations are mostly element-wise.

## Question (e): Attention Layer Analysis

| Model   |   Context |   NVTX Softmax (ms) |   Kernel Softmax (ms) |   Est. MatMul (ms) |   Ratio |
|:--------|----------:|--------------------:|----------------------:|-------------------:|--------:|
| 2.7B    |      1024 |               74.59 |                 34.56 |             890.89 |   0.084 |
| 2.7B    |       128 |               44.5  |                 13.91 |             715.39 |   0.062 |
| 2.7B    |       256 |               50.59 |                 25.19 |             731.71 |   0.069 |
| 2.7B    |       512 |               66.57 |                 58.68 |             761.71 |   0.087 |
| large   |      1024 |               81.05 |                116.29 |             937.82 |   0.086 |
| large   |       128 |               42.09 |                 11.57 |             751.48 |   0.056 |
| large   |       256 |               54.92 |                 16.05 |             782.7  |   0.07  |
| large   |       512 |               55.47 |                 42.26 |             806.48 |   0.069 |
| medium  |      1024 |               41.32 |                 63.77 |             608.72 |   0.068 |
| medium  |       128 |               30.64 |                  7.09 |             592.95 |   0.052 |
| medium  |       256 |               39.56 |                  9.31 |             595.19 |   0.066 |
| medium  |       512 |               40.15 |                 24.13 |             620.87 |   0.065 |
| small   |      1024 |               26.93 |                 25.8  |             446.38 |   0.06  |
| small   |       128 |               21.63 |                  3.51 |             449.23 |   0.048 |
| small   |       256 |               24.87 |                  4.38 |             448.63 |   0.055 |
| small   |       512 |               26.57 |                 10.65 |             455.3  |   0.058 |
| xl      |      1024 |              270.83 |                 41.38 |             897.45 |   0.302 |
| xl      |       128 |               57.11 |                 16.54 |             908.25 |   0.063 |
| xl      |       256 |               69.61 |                 27.76 |             941.09 |   0.074 |
| xl      |       512 |               69.67 |                 67.67 |             960.84 |   0.073 |

Note: These are estimates. For accurate attention analysis, add NVTX ranges
specifically around attention operations in your code.

Softmax has higher time/FLOP ratio than MatMul because:
- Softmax is memory-bandwidth bound (low arithmetic intensity)
- MatMul can utilize tensor cores efficiently (high arithmetic intensity)
- Softmax requires exp() operations which are more expensive than multiply-add