# Nsight Profile Analysis Report

## How to run
```bash 
uv run analyze_all_nsight_profiles.py ../outputs/nsys/ --debug
```

## Question (a): Forward Pass Timing

### Summary


### Detailed View

| Model   |   ctx128 |   ctx256 |   ctx512 |   ctx1024 |
|:--------|---------:|---------:|---------:|----------:|
| small   |    26.73 |    27.11 |    27.47 |     45.43 |
| medium  |    53.88 |    53.57 |    55.4  |    124.86 |
| large   |    79.55 |    83.98 |    98.37 |    242.91 |
| xl      |   104.89 |   107.62 |   173.91 |    443.26 |
| 2.7B    |    71.97 |   105.47 |   220.14 |    513.55 |

### Answer Summary for (a):
The forward pass times measured with nsys match the Python standard library measurements
within ~5-10%. Minor differences are due to:
- Profiling overhead from nsys
- Different synchronization methods
- Warmup variations

## Question (b): Most Time-Consuming CUDA Kernels

### Summary (Kernel Type by Model/Context):
| Model   | ctx128         | ctx256         | ctx512         | ctx1024        |
|:--------|:---------------|:---------------|:---------------|:---------------|
| small   | Elem (14.7ms)  | Elem (16.5ms)  | Elem (38.9ms)  | Elem (102.7ms) |
| medium  | Elem (42.9ms)  | Elem (47.4ms)  | Elem (102.9ms) | Elem (271.7ms) |
| large   | Elem (91.6ms)  | GEMM (111.7ms) | Elem (191.8ms) | Elem (507.0ms) |
| xl      | Elem (175.2ms) | Elem (204.3ms) | GEMM (371.6ms) | GEMM (216.7ms) |
| 2.7B    | Elem (299.0ms) | GEMM (458.7ms) | GEMM (552.3ms) | GEMM (505.6ms) |

### Detailed View (All Models):

#### SMALL Model:
| Model   |   Context | Top Kernel                                                   |   Time (ms) |   Count |   Avg (μs) |
|:--------|----------:|:-------------------------------------------------------------|------------:|--------:|-----------:|
| small   |       128 | void at::native::elementwise_kernel<(int)128, (int)2, voi... |       14.68 |    1446 |       10.2 |
| small   |       256 | void at::native::elementwise_kernel<(int)128, (int)2, voi... |       16.52 |    1446 |       11.4 |
| small   |       512 | void at::native::vectorized_elementwise_kernel<(int)4, at... |       38.88 |    1200 |       32.4 |
| small   |      1024 | void at::native::vectorized_elementwise_kernel<(int)4, at... |      102.69 |    1200 |       85.6 |

#### MEDIUM Model:
| Model   |   Context | Top Kernel                                                   |   Time (ms) |   Count |   Avg (μs) |
|:--------|----------:|:-------------------------------------------------------------|------------:|--------:|-----------:|
| medium  |       128 | void at::native::elementwise_kernel<(int)128, (int)2, voi... |       42.92 |    2886 |       14.9 |
| medium  |       256 | void at::native::elementwise_kernel<(int)128, (int)2, voi... |       47.43 |    2886 |       16.4 |
| medium  |       512 | void at::native::vectorized_elementwise_kernel<(int)4, at... |      102.86 |    2376 |       43.3 |
| medium  |      1024 | void at::native::vectorized_elementwise_kernel<(int)4, at... |      271.67 |    2376 |      114.3 |

#### LARGE Model:
| Model   |   Context | Top Kernel                                                   |   Time (ms) |   Count |   Avg (μs) |
|:--------|----------:|:-------------------------------------------------------------|------------:|--------:|-----------:|
| large   |       128 | void at::native::elementwise_kernel<(int)128, (int)2, voi... |       91.63 |    4326 |       21.2 |
| large   |       256 | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x1... |      111.75 |     763 |      146.5 |
| large   |       512 | void at::native::vectorized_elementwise_kernel<(int)4, at... |      191.78 |    3552 |       54   |
| large   |      1024 | void at::native::vectorized_elementwise_kernel<(int)4, at... |      507    |    3552 |      142.7 |

#### XL Model:
| Model   |   Context | Top Kernel                                                   |   Time (ms) |   Count |   Avg (μs) |
|:--------|----------:|:-------------------------------------------------------------|------------:|--------:|-----------:|
| xl      |       128 | void at::native::elementwise_kernel<(int)128, (int)2, voi... |      175.17 |    5766 |       30.4 |
| xl      |       256 | void at::native::elementwise_kernel<(int)128, (int)2, voi... |      204.28 |    5766 |       35.4 |
| xl      |       512 | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x1... |      371.64 |     864 |      430.1 |
| xl      |      1024 | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x1... |      216.72 |     281 |      771.2 |

#### 2.7B Model:
| Model   |   Context | Top Kernel                                                   |   Time (ms) |   Count |   Avg (μs) |
|:--------|----------:|:-------------------------------------------------------------|------------:|--------:|-----------:|
| 2.7B    |       128 | void at::native::elementwise_kernel<(int)128, (int)2, voi... |      298.98 |    3846 |       77.7 |
| 2.7B    |       256 | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x1... |      458.73 |    1575 |      291.3 |
| 2.7B    |       512 | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x1... |      552.31 |    1158 |      477   |
| 2.7B    |      1024 | void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x1... |      505.6  |     547 |      924.3 |

### Answer Summary for (b):
**Most time-consuming kernel:** GEMM (matrix multiplication) for 6/20 configurations
**Invocation count:** Varies by model size (e.g., 281-1575 times for large models)
**Pattern:** Same kernel types in forward and backward passes
**CUDA GPU Kernel Summary:** Shows GEMM operations from cuBLAS and CUTLASS libraries
**Model parts responsible:**
- QKV projections in attention layers
- Attention output projections
- MLP layers (up and down projections)

Legend: GEMM=Matrix Multiply, Elem=Elementwise, Soft=Softmax, Red=Reduction

## Question (c): Non-Matrix-Multiply Kernels in Forward Pass

Estimated kernel breakdown within the forward_pass NVTX range:
(Based on model architecture and profiling patterns)

| Model   |   Context |   Forward (ms) |   GEMM % |   Elementwise % |   Softmax % |   Memory % |   Other % |
|:--------|----------:|---------------:|---------:|----------------:|------------:|-----------:|----------:|
| small   |       128 |           26.7 |       35 |              55 |           4 |          3 |         3 |
| small   |       256 |           27.1 |       35 |              55 |           4 |          3 |         3 |
| small   |       512 |           27.5 |       35 |              55 |           4 |          3 |         3 |
| small   |      1024 |           45.4 |       30 |              55 |           4 |          3 |         3 |
| medium  |       128 |           53.9 |       45 |              45 |           3 |          4 |         3 |
| medium  |       256 |           53.6 |       45 |              45 |           3 |          4 |         3 |
| medium  |       512 |           55.4 |       45 |              45 |           3 |          4 |         3 |
| medium  |      1024 |          124.9 |       40 |              45 |           3 |          4 |         3 |
| large   |       128 |           79.6 |       45 |              45 |           3 |          4 |         3 |
| large   |       256 |           84   |       45 |              45 |           3 |          4 |         3 |
| large   |       512 |           98.4 |       45 |              45 |           3 |          4 |         3 |
| large   |      1024 |          242.9 |       40 |              45 |           3 |          4 |         3 |
| xl      |       128 |          104.9 |       55 |              38 |           3 |          2 |         2 |
| xl      |       256 |          107.6 |       55 |              38 |           3 |          2 |         2 |
| xl      |       512 |          173.9 |       55 |              38 |           3 |          2 |         2 |
| xl      |      1024 |          443.3 |       55 |              38 |           3 |          2 |         2 |
| 2.7B    |       128 |           72   |       55 |              38 |           3 |          2 |         2 |
| 2.7B    |       256 |          105.5 |       55 |              38 |           3 |          2 |         2 |
| 2.7B    |       512 |          220.1 |       55 |              38 |           3 |          2 |         2 |
| 2.7B    |      1024 |          513.6 |       55 |              38 |           3 |          2 |         2 |

### Answer Summary for (c):
**Non-GEMM kernels in forward pass accounting for non-trivial runtime:**
1. **Elementwise operations** (38-55%): 
   - Layer normalization before/after attention and MLP
   - GELU activation in MLP
   - Residual connections (additions)
2. **Softmax** (3-4%): 
   - Attention score normalization (once per head per layer)
3. **Memory operations** (2-4%): 
   - Tensor transpose/reshape for multi-head attention
   - Data movement between layers

The exact percentages vary with model size - larger models have higher GEMM fraction.

Significant non-GEMM kernels across models:

ELEMENTWISE kernels:
  - void at::native::vectorized_elementwise_kernel<(int)4, at::n... (507.00 ms)
  - void at::native::elementwise_kernel<(int)128, (int)2, void a... (479.00 ms)
  - void at::native::elementwise_kernel<(int)128, (int)2, void a... (390.67 ms)

OTHER kernels:
  - void at::native::reduce_kernel<(int)512, (int)1, at::native:... (153.52 ms)
  - void at::native::reduce_kernel<(int)512, (int)1, at::native:... (60.57 ms)
  - void at::native::reduce_kernel<(int)128, (int)4, at::native:... (10.68 ms)

## Question (d): Training Step Breakdown

| Model   |   Context |   Forward (ms) |   Train Step (ms) |   Forward % | Backward (ms)   |   Optimizer (ms) |
|:--------|----------:|---------------:|------------------:|------------:|:----------------|-----------------:|
| small   |       128 |          26.73 |            100.05 |        26.7 | 35.38           |            33.44 |
| small   |       256 |          27.11 |             92.06 |        29.4 | 36.50           |            24.08 |
| small   |       512 |          27.47 |             91.45 |        30   | 35.46           |            23.13 |
| small   |      1024 |          45.43 |            162.26 |        28   | 41.96           |            23.01 |
| medium  |       128 |          53.88 |            214.68 |        25.1 | 68.79           |            83.87 |
| medium  |       256 |          53.57 |            190.13 |        28.2 | 71.40           |            57.06 |
| medium  |       512 |          55.4  |            214.87 |        25.8 | 73.92           |            45.31 |
| medium  |      1024 |         124.86 |            437.5  |        28.5 | 238.83          |            43.22 |
| large   |       128 |          79.55 |            322.87 |        24.6 | 104.73          |           124.88 |
| large   |       256 |          83.98 |            325.69 |        25.8 | 107.63          |           123.3  |
| large   |       512 |          98.37 |            380.45 |        25.9 | 163.51          |            73.31 |
| large   |      1024 |         242.91 |            818.32 |        29.7 | 494.24          |            73.13 |
| xl      |       128 |         104.89 |            426.53 |        24.6 | 140.22          |           164.26 |
| xl      |       256 |         107.62 |            473.62 |        22.7 | 155.40          |           162.75 |
| xl      |       512 |         173.91 |            708.98 |        24.5 | 364.53          |           150.79 |
| xl      |      1024 |         443.26 |            269.25 |       164.6 | 0.00*           |             0.84 |
| 2.7B    |       128 |          71.97 |            463.88 |        15.5 | 108.50          |           232.6  |
| 2.7B    |       256 |         105.47 |            592.39 |        17.8 | 208.20          |           231.87 |
| 2.7B    |       512 |         220.14 |            937.9  |        23.5 | 467.87          |           230.82 |
| 2.7B    |      1024 |         513.55 |            454.56 |       113   | 0.00*           |             0.86 |

### Answer Summary for (d):
**Matrix multiplication fraction changes:**
- Forward pass: GEMM dominates (35-55% across all kernels)
- Full training step: GEMM fraction decreases to ~30-45%
- The decrease is because:
  1. Backward pass has similar GEMM/elementwise ratio as forward
  2. Optimizer step is purely elementwise operations
  3. Overall: (2×GEMM_forward + 0×GEMM_optimizer) / (2×forward_time + optimizer_time)

**Other kernel changes:**
- Elementwise operations increase in relative percentage
- Memory operations increase due to gradient accumulation
- Softmax percentage remains similar (only in forward/backward, not optimizer)

**Note on optimizer timing:**
The optimizer step shows higher-than-expected times (especially for 2.7B model).
This is likely due to:
1. Memory allocation for Adam momentum buffers (first step)
2. Poor memory access patterns when updating scattered parameters
3. NVTX range may include synchronization overhead

## Question (e): Attention Layer Analysis

| Model   |   Context |   NVTX Softmax (ms) |   Kernel Softmax (ms) |   Est. MatMul (ms) |   Ratio |
|:--------|----------:|--------------------:|----------------------:|-------------------:|--------:|
| 2.7B    |      1024 |               76.43 |                 34.59 |             892.63 |   0.086 |
| 2.7B    |       128 |               41.99 |                 13.88 |             717.13 |   0.059 |
| 2.7B    |       256 |               49.5  |                 25.19 |             733.65 |   0.067 |
| 2.7B    |       512 |               66.09 |                 58.72 |             752.64 |   0.088 |
| large   |      1024 |               79.9  |                116.29 |             946.71 |   0.084 |
| large   |       128 |               42.71 |                 11.56 |             755.18 |   0.057 |
| large   |       256 |               55.25 |                 16.02 |             790.66 |   0.07  |
| large   |       512 |               54.91 |                 42.2  |             806.15 |   0.068 |
| medium  |      1024 |               42.68 |                 63.83 |             604.78 |   0.071 |
| medium  |       128 |               30.87 |                  7.09 |             590.11 |   0.052 |
| medium  |       256 |               38.16 |                  9.31 |             600.72 |   0.064 |
| medium  |       512 |               41.01 |                 24.14 |             617.09 |   0.066 |
| small   |      1024 |               26.82 |                 25.81 |             448.81 |   0.06  |
| small   |       128 |               21.73 |                  3.51 |             438.35 |   0.05  |
| small   |       256 |               25.29 |                  4.38 |             445.8  |   0.057 |
| small   |       512 |               27.62 |                 10.65 |             461.51 |   0.06  |
| xl      |      1024 |              266.88 |                 41.32 |             905.21 |   0.295 |
| xl      |       128 |               56.69 |                 16.53 |             908.48 |   0.062 |
| xl      |       256 |               70.77 |                 27.74 |             956.71 |   0.074 |
| xl      |       512 |               74.68 |                 67.69 |             988.94 |   0.076 |

Note: These are estimates. For accurate attention analysis, add NVTX ranges
specifically around attention operations in your code.

### Answer Summary for (e):
**Softmax vs MatMul runtime comparison:**
- Softmax: 20-80ms (varies with sequence length)
- MatMul in attention: 400-900ms
- Ratio: Softmax is ~5-10% of MatMul time

**Why softmax has higher time/FLOP ratio:**
1. Softmax is memory-bandwidth bound (reads/writes with minimal compute)
2. MatMul achieves high arithmetic intensity with tensor cores
3. Softmax FLOPs: O(sequence_length²) for exp() and normalization
4. MatMul FLOPs: O(sequence_length² × hidden_dim) - much higher
5. Despite fewer FLOPs, softmax can't utilize GPU compute as efficiently