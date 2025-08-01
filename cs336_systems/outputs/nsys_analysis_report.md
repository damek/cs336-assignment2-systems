# Nsight Systems Profiling Analysis Report

**How to generate this file:**
```bash
cd cs336_systems/benchmarking_scripts
# Basic usage:
uv run analyze_nsys_reports.py

# With CSV timing comparison:
uv run analyze_nsys_reports.py --csv ../outputs/csv/2025-07-28_table1.1.2.csv
```

## Model Specifications

| Size | d_model | d_ff | num_layers | num_heads |
|------|---------|------|------------|----------|
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 1600 | 6400 | 48 | 25 |
| 2.7B | 2560 | 10240 | 32 | 32 |

## Kernel Type Glossary

- **cutlass::Kernel2**: Matrix multiplication (GEMM) operations
- **elementwise_kernel**: Element-wise operations (add, multiply, activation functions)
- **vectorized_elementwise_kernel**: Optimized element-wise operations
- **reduce_kernel**: Reduction operations (sum, mean, max)
- **sigmoid_kernel**: Sigmoid activation function
- **exp_kernel**: Exponential function
- **softmax**: Softmax normalization

## (a) What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?

**Deliverable:** A 1-2 sentence response.

### Forward Pass Timings

| file           |   duration_ms |   python_ms |   speedup |   percentage |
|:---------------|--------------:|------------:|----------:|-------------:|
| 2.7B_ctx1024   |        513.55 |      528.85 |      1.03 |         10.5 |
| 2.7B_ctx128    |         71.97 |       88.53 |      1.23 |          1.6 |
| 2.7B_ctx256    |        105.47 |      133.46 |      1.27 |          1.9 |
| 2.7B_ctx512    |        220.14 |      247.44 |      1.12 |          2.7 |
| large_ctx1024  |        242.91 |      272.41 |      1.12 |          2.9 |
| large_ctx128   |         79.55 |       93.84 |      1.18 |          1.9 |
| large_ctx256   |         83.98 |       93.15 |      1.11 |          1.9 |
| large_ctx512   |         98.37 |      129.54 |      1.32 |          2   |
| medium_ctx1024 |        124.86 |      156.03 |      1.25 |          2.6 |
| medium_ctx128  |         53.88 |       72.78 |      1.35 |          1.7 |
| medium_ctx256  |         53.57 |       73.09 |      1.36 |          1.7 |
| medium_ctx512  |         55.4  |       82.52 |      1.49 |          1.6 |
| small_ctx1024  |         45.43 |       77.57 |      1.71 |          1.8 |
| small_ctx128   |         26.73 |       51.13 |      1.91 |          1.3 |
| small_ctx256   |         27.11 |       50.32 |      1.86 |          1.3 |
| small_ctx512   |         27.47 |       52.7  |      1.92 |          1.3 |
| xl_ctx1024     |        443.26 |      461.34 |      1.04 |          8.9 |
| xl_ctx128      |        104.89 |      110.85 |      1.06 |          1.9 |
| xl_ctx256      |        107.62 |      118.9  |      1.1  |          1.8 |
| xl_ctx512      |        173.91 |      201.52 |      1.16 |          2.3 |

**Answer:** The nsys profiling shows similar forward pass timings to Python standard library measurements, with an average speedup of 1.33x.

## (b) What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes?

**Deliverable:** A 1-2 sentence response.

### Top CUDA Kernels in Forward Pass by Model Configuration

| file           | top_kernel                                                                          |   count |   total_time_ms |   avg_time_us |
|:---------------|:------------------------------------------------------------------------------------|--------:|----------------:|--------------:|
| 2.7B_ctx1024   | 656633.3  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_al... |     193 |      181.803    |      941.983  |
| 2.7B_ctx128    | 19113.2  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_ali... |      97 |       26.6733   |      274.982  |
| 2.7B_ctx256    | 171692.3  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_al... |     225 |       67.4059   |      299.582  |
| 2.7B_ctx512    | 276147.3  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x5_tn_al... |     160 |       63.426    |      396.412  |
| large_ctx1024  | 175881.8  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_al... |     217 |       54.0545   |      249.099  |
| large_ctx128   | 461.1  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_tn_align... |     144 |        5.43761  |       37.7612 |
| large_ctx256   | 7825.5  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_alig... |     109 |       16.1764   |      148.408  |
| large_ctx512   | 83532.5  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_ali... |     181 |       22.104    |      122.121  |
| medium_ctx1024 | 62835.5  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_ali... |      49 |       15.9182   |      324.86   |
| medium_ctx128  | 314.6  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_tn_align... |      48 |        2.85351  |       59.4482 |
| medium_ctx256  | 765.7  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x4_tn_align... |      48 |        4.45716  |       92.8575 |
| medium_ctx512  | 49984.8  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_ali... |     121 |       10.497    |       86.7517 |
| small_ctx1024  | 65653.3  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_ali... |      72 |        7.06687  |       98.1509 |
| small_ctx128   | 281.1  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_32x3_tn_align4... |      24 |        0.721665 |       30.0694 |
| small_ctx256   | 658.4  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_tn_align... |      24 |        1.17168  |       48.8201 |
| small_ctx512   | 24289.5  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_tn_ali... |      60 |        2.36721  |       39.4534 |
| xl_ctx1024     | 41592.9  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_ali... |      97 |       75.1017   |      774.244  |
| xl_ctx128      | 1053.5  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_32x3_tn_alig... |      96 |        8.85245  |       92.213  |
| xl_ctx256      | 51298.3  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_ali... |     240 |       20.1547   |       83.9779 |
| xl_ctx512      | 20462.2  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_ali... |      97 |       38.2751   |      394.589  |

**Forward Pass Answer:** The CUDA kernel that takes the most cumulative GPU time is `656633.3  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_al...`, invoked 193 times during a single forward pass.

### Top CUDA Kernels in Backward Pass by Model Configuration

| file           | top_kernel                                                                          |   count |   total_time_ms |   avg_time_us |
|:---------------|:------------------------------------------------------------------------------------|--------:|----------------:|--------------:|
| 2.7B_ctx128    | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.001344 |         1.344 |
| 2.7B_ctx256    | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.001568 |         1.568 |
| 2.7B_ctx512    | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.001536 |         1.536 |
| large_ctx1024  | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.0016   |         1.6   |
| large_ctx128   | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.001344 |         1.344 |
| large_ctx256   | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.001344 |         1.344 |
| large_ctx512   | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.001472 |         1.472 |
| medium_ctx1024 | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.001504 |         1.504 |
| medium_ctx128  | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.001344 |         1.344 |
| medium_ctx256  | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.001344 |         1.344 |
| medium_ctx512  | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.001376 |         1.376 |
| small_ctx1024  | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.001504 |         1.504 |
| small_ctx128   | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.001344 |         1.344 |
| small_ctx256   | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.001344 |         1.344 |
| small_ctx512   | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.001376 |         1.376 |
| xl_ctx128      | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.001344 |         1.344 |
| xl_ctx256      | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.001376 |         1.376 |
| xl_ctx512      | 0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc... |       1 |        0.001568 |         1.568 |

**Backward Pass Answer:** The CUDA kernel that takes the most cumulative GPU time is `0.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunc...`, invoked 1 times during a single backward pass.

**Comparison:** The top kernel is different for forward and backward passes.

## (c) What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?

**Deliverable:** A 1-2 sentence response.

### Top 5 CUDA Kernels in Forward Pass by Model Configuration

| file           | top_kernel                                                                          |   count |   total_time_ms |   avg_time_us |
|:---------------|:------------------------------------------------------------------------------------|--------:|----------------:|--------------:|
| 2.7B_ctx1024   | 656633.3  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_al... |     193 |      181.803    |      941.983  |
| 2.7B_ctx1024   | 40669.6  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_32x3_tn_ali... |      32 |       56.8482   |     1776.51   |
| 2.7B_ctx1024   | 11053.0  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::... |     386 |       30.0413   |       77.8273 |
| 2.7B_ctx1024   | 15866.9  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::... |      32 |       29.0091   |      906.535  |
| 2.7B_ctx1024   | 16005.2  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::... |      32 |       27.3618   |      855.057  |
| 2.7B_ctx128    | 19113.2  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_ali... |      97 |       26.6733   |      274.982  |
| 2.7B_ctx128    | 1638.6  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_tn_alig... |     128 |        9.80988  |       76.6397 |
| 2.7B_ctx128    | 1569.2  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |     386 |        3.90465  |       10.1157 |
| 2.7B_ctx128    | 3248.7  void at::native::vectorized_elementwise_kernel<(int)4, at::native::Binar... |      64 |        2.29153  |       35.8051 |
| 2.7B_ctx128    | 1302.7  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |     192 |        1.75665  |        9.1492 |
| 2.7B_ctx256    | 171692.3  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_al... |     225 |       67.4059   |      299.582  |
| 2.7B_ctx256    | 3590.3  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |     386 |        7.24725  |       18.7752 |
| 2.7B_ctx256    | 2213.4  void at::native::vectorized_elementwise_kernel<(int)4, at::native::Binar... |      64 |        4.81668  |       75.2607 |
| 2.7B_ctx256    | 2407.1  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |     192 |        3.26493  |       17.0049 |
| 2.7B_ctx256    | 351.3  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gp... |      32 |        1.86314  |       58.2231 |
| 2.7B_ctx512    | 276147.3  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x5_tn_al... |     160 |       63.426    |      396.412  |
| 2.7B_ctx512    | 14398.0  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_ali... |      65 |       62.1867   |      956.719  |
| 2.7B_ctx512    | 5746.5  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |     386 |       16.2976   |       42.2219 |
| 2.7B_ctx512    | 2203.6  void at::native::vectorized_elementwise_kernel<(int)4, at::native::Binar... |      64 |        9.48876  |      148.262  |
| 2.7B_ctx512    | 2844.6  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |      32 |        7.48565  |      233.927  |
| large_ctx1024  | 175881.8  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_al... |     217 |       54.0545   |      249.099  |
| large_ctx1024  | 3325.8  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |      36 |       19.9496   |      554.155  |
| large_ctx1024  | 3053.2  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |      36 |       18.8764   |      524.345  |
| large_ctx1024  | 5360.3  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |     434 |       17.5058   |       40.336  |
| large_ctx1024  | 54046.8  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::... |      38 |       17.1408   |      451.075  |
| large_ctx128   | 461.1  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_tn_align... |     144 |        5.43761  |       37.7612 |
| large_ctx128   | 245.6  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_align... |      72 |        5.17016  |       71.8078 |
| large_ctx128   | 168.9  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x128_32x6_tn_align4... |      36 |        2.9474   |       81.8722 |
| large_ctx128   | 491.1  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gp... |     434 |        2.64429  |        6.0928 |
| large_ctx128   | 672.8  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gp... |     216 |        1.30307  |        6.0328 |
| large_ctx256   | 7825.5  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_alig... |     109 |       16.1764   |      148.408  |
| large_ctx256   | 1034.8  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_tn_alig... |     144 |        5.886    |       40.875  |
| large_ctx256   | 1778.7  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |     434 |        4.41733  |       10.1782 |
| large_ctx256   | 2658.5  void at::native::vectorized_elementwise_kernel<(int)4, at::native::Binar... |      72 |        2.58477  |       35.8996 |
| large_ctx256   | 1330.6  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |     216 |        1.93364  |        8.952  |
| large_ctx512   | 83532.5  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_ali... |     181 |       22.104    |      122.121  |
| large_ctx512   | 1361.5  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_alig... |      72 |       17.0275   |      236.493  |
| large_ctx512   | 3741.8  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |     434 |        8.05289  |       18.555  |
| large_ctx512   | 2139.3  void at::native::vectorized_elementwise_kernel<(int)4, at::native::Binar... |      72 |        5.44779  |       75.6637 |
| large_ctx512   | 443.6  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gp... |      36 |        4.86824  |      135.229  |
| medium_ctx1024 | 62835.5  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_ali... |      49 |       15.9182   |      324.86   |
| medium_ctx1024 | 2997.5  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |      24 |       10.2556   |      427.317  |
| medium_ctx1024 | 3028.4  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |      24 |        9.70598  |      404.416  |
| medium_ctx1024 | 1218.7  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_alig... |      96 |        9.24665  |       96.3193 |
| medium_ctx1024 | 38505.1  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::... |      26 |        9.13225  |      351.24   |
| medium_ctx128  | 314.6  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_tn_align... |      48 |        2.85351  |       59.4482 |
| medium_ctx128  | 6714.7  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4... |     120 |        2.55009  |       21.2507 |
| medium_ctx128  | 331.2  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gp... |     290 |        1.55322  |        5.3559 |
| medium_ctx128  | 436.2  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_tn_align... |      24 |        1.27299  |       53.0414 |
| medium_ctx128  | 589.4  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gp... |     144 |        0.785411 |        5.4542 |
| medium_ctx256  | 765.7  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x4_tn_align... |      48 |        4.45716  |       92.8575 |
| medium_ctx256  | 412.9  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align... |      96 |        2.97937  |       31.0351 |
| medium_ctx256  | 1399.5  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |     290 |        2.4554   |        8.4669 |
| medium_ctx256  | 428.4  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_tn_align... |      24 |        2.20365  |       91.8189 |
| medium_ctx256  | 667.7  void at::native::vectorized_elementwise_kernel<(int)4, at::native::Binary... |      48 |        1.23808  |       25.7934 |
| medium_ctx512  | 49984.8  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_ali... |     121 |       10.497    |       86.7517 |
| medium_ctx512  | 1220.5  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_alig... |      48 |        7.55854  |      157.47   |
| medium_ctx512  | 2925.2  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |     290 |        4.27294  |       14.7343 |
| medium_ctx512  | 2132.7  void at::native::vectorized_elementwise_kernel<(int)4, at::native::Binar... |      48 |        2.94404  |       61.3341 |
| medium_ctx512  | 630.2  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gp... |      24 |        2.54948  |      106.228  |
| small_ctx1024  | 65653.3  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_ali... |      72 |        7.06687  |       98.1509 |
| small_ctx1024  | 726.5  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gp... |      12 |        3.79367  |      316.139  |
| small_ctx1024  | 18718.5  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::... |      14 |        3.66148  |      261.534  |
| small_ctx1024  | 601.7  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gp... |      12 |        3.59665  |      299.721  |
| small_ctx1024  | 3919.3  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |     146 |        3.13805  |       21.4935 |
| small_ctx128   | 281.1  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_32x3_tn_align4... |      24 |        0.721665 |       30.0694 |
| small_ctx128   | 274.5  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gp... |     146 |        0.716833 |        4.9098 |
| small_ctx128   | 170.9  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_32x6_tn_align4>... |      48 |        0.646304 |       13.4647 |
| small_ctx128   | 162.3  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x128_32x6_tn_align4... |      12 |        0.366561 |       30.5468 |
| small_ctx128   | 269.4  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gp... |      72 |        0.346083 |        4.8067 |
| small_ctx256   | 658.4  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_tn_align... |      24 |        1.17168  |       48.8201 |
| small_ctx256   | 754.1  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gp... |     146 |        1.00727  |        6.8991 |
| small_ctx256   | 289.6  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_32x6_tn_align4... |      48 |        0.905667 |       18.8681 |
| small_ctx256   | 459.4  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x128_32x6_tn_align4... |      12 |        0.646784 |       53.8987 |
| small_ctx256   | 805.7  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gp... |      72 |        0.478692 |        6.6485 |
| small_ctx512   | 24289.5  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_tn_ali... |      60 |        2.36721  |       39.4534 |
| small_ctx512   | 569.0  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align... |      24 |        2.27536  |       94.8069 |
| small_ctx512   | 2244.9  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |     146 |        1.69101  |       11.5823 |
| small_ctx512   | 2420.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::Binar... |      24 |        1.11959  |       46.6494 |
| small_ctx512   | 14407.4  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::... |      14 |        1.01063  |       72.1875 |
| xl_ctx1024     | 41592.9  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_ali... |      97 |       75.1017   |      774.244  |
| xl_ctx1024     | 3171.1  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_alig... |     192 |       37.6402   |      196.043  |
| xl_ctx1024     | 13686.4  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_32x3_tn_ali... |      48 |       37.0279   |      771.414  |
| xl_ctx1024     | 10014.3  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::... |      48 |       32.9267   |      685.973  |
| xl_ctx1024     | 9486.3  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |      48 |       31.1164   |      648.258  |
| xl_ctx128      | 1053.5  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_32x3_tn_alig... |      96 |        8.85245  |       92.213  |
| xl_ctx128      | 416.0  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x64_32x6_tn_align4... |     192 |        6.17788  |       32.1765 |
| xl_ctx128      | 278.2  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_tn_align... |      48 |        4.68506  |       97.6055 |
| xl_ctx128      | 774.5  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gp... |     578 |        4.02196  |        6.9584 |
| xl_ctx128      | 723.3  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gp... |     288 |        1.9249   |        6.6837 |
| xl_ctx256      | 51298.3  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_ali... |     240 |       20.1547   |       83.9779 |
| xl_ctx256      | 9864.0  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_alig... |      97 |       18.1659   |      187.278  |
| xl_ctx256      | 1923.9  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |     578 |        6.85403  |       11.8582 |
| xl_ctx256      | 3425.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::Binar... |      96 |        4.52196  |       47.1038 |
| xl_ctx256      | 1576.9  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |     288 |        3.14241  |       10.9111 |
| xl_ctx512      | 20462.2  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_32x3_tn_ali... |      97 |       38.2751   |      394.589  |
| xl_ctx512      | 1505.4  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_alig... |     192 |       19.5422   |      101.782  |
| xl_ctx512      | 4881.5  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x5_tn_alig... |      48 |       18.1372   |      377.859  |
| xl_ctx512      | 4212.5  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::g... |     578 |       13.9376   |       24.1136 |
| xl_ctx512      | 3044.7  void at::native::vectorized_elementwise_kernel<(int)4, at::native::Binar... |      96 |        8.89314  |       92.6369 |

## (d) Profile running one complete training step with your implementation of AdamW. How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?

**Deliverable:** A 1-2 sentence response.

### Matrix Multiplication vs Other Kernels: Forward Pass vs Complete Training Step

| file           |   matmul_forward_% |   matmul_train_% |   other_forward_% |   other_train_% |
|:---------------|-------------------:|-----------------:|------------------:|----------------:|
| 2.7B_ctx1024   |               54.3 |             54.1 |              45.7 |            45.9 |
| 2.7B_ctx128    |               72.2 |             10.7 |              27.8 |            89.3 |
| 2.7B_ctx256    |               68.9 |             17.9 |              31.1 |            82.1 |
| 2.7B_ctx512    |               62.1 |             26.6 |              37.9 |            73.4 |
| large_ctx1024  |               38.1 |             28.1 |              61.9 |            71.9 |
| large_ctx128   |               59.5 |             13.4 |              40.5 |            86.6 |
| large_ctx256   |               54.8 |             18.5 |              45.2 |            81.5 |
| large_ctx512   |               47.4 |             25.1 |              52.6 |            74.9 |
| medium_ctx1024 |               36.2 |             28.3 |              63.8 |            71.7 |
| medium_ctx128  |               54.2 |             15   |              45.8 |            85   |
| medium_ctx256  |               50.7 |             19.6 |              49.3 |            80.4 |
| medium_ctx512  |               44   |             25.2 |              56   |            74.8 |
| small_ctx1024  |               31.5 |             25.3 |              68.5 |            74.7 |
| small_ctx128   |               41.6 |             13.3 |              58.4 |            86.7 |
| small_ctx256   |               42.6 |             18.1 |              57.4 |            81.9 |
| small_ctx512   |               36.1 |             22.5 |              63.9 |            77.5 |
| xl_ctx1024     |               42.9 |             42.8 |              57.1 |            57.2 |
| xl_ctx128      |               58.4 |             10.3 |              41.6 |            89.7 |
| xl_ctx256      |               56.1 |             16.9 |              43.9 |            83.1 |
| xl_ctx512      |               50.6 |             25.4 |              49.4 |            74.6 |

## (e) Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?

**Deliverable:** A 1-2 sentence response.

### Softmax vs Matrix Multiplication in Self-Attention

| file           |   softmax_ms |   matmul_ms |   runtime_ratio_% |
|:---------------|-------------:|------------:|------------------:|
| 2.7B_ctx1024   |    266.34    |   47.9267   |             555.7 |
| 2.7B_ctx128    |     13.936   |    6.58923  |             211.5 |
| 2.7B_ctx256    |     49.9881  |   15.3165   |             326.4 |
| 2.7B_ctx512    |    179.631   |   40.4787   |             443.8 |
| large_ctx1024  |    463.53    |   73.4786   |             630.8 |
| large_ctx128   |     10.8894  |    3.9058   |             278.8 |
| large_ctx256   |     31.5907  |    8.30936  |             380.2 |
| large_ctx512   |    124.696   |   25.059    |             497.6 |
| medium_ctx1024 |    244.769   |   39.493    |             619.8 |
| medium_ctx128  |      6.3645  |    2.4609   |             258.6 |
| medium_ctx256  |     15.7621  |    4.66583  |             337.8 |
| medium_ctx512  |     66.8038  |   14.2141   |             470   |
| small_ctx1024  |     91.5706  |   15.1162   |             605.8 |
| small_ctx128   |      2.74045 |    0.974017 |             281.4 |
| small_ctx256   |      5.8782  |    1.65699  |             354.8 |
| small_ctx512   |     25.548   |    5.55704  |             459.7 |
| xl_ctx1024     |    321.849   |   50.6054   |             636   |
| xl_ctx128      |     16.7316  |    5.70536  |             293.3 |
| xl_ctx256      |     56.0558  |   15.9919   |             350.5 |
| xl_ctx512      |    207.009   |   40.2754   |             514   |

**Note:** runtime_ratio_% = (softmax_time / matmul_time) * 100

**Theoretical FLOP Analysis:**
- **Attention matrix multiplies (per layer):**
  - Q*K^T: 2 * n_heads * seq_len * seq_len * d_head FLOPs
  - Softmax(scores) * V: 2 * n_heads * seq_len * seq_len * d_head FLOPs
  - Output projection: 2 * seq_len * d_model * d_model FLOPs
  - **Total matmul FLOPs:** 4 * n_heads * seq_len² * d_head + 2 * seq_len * d_model²
- **Softmax (per layer):** n_heads * seq_len * seq_len FLOPs (for exp and normalization)
  - **Total softmax FLOPs:** n_heads * seq_len²

**FLOP Ratio (matmul/softmax):** (4 * n_heads * seq_len² * d_head + 2 * seq_len * d_model²) / (n_heads * seq_len²)
= 4 * d_head + 2 * d_model² / (n_heads * seq_len)
≈ 4 * d_head (for large seq_len, since d_model = n_heads * d_head)

**Note:** Softmax is memory-bandwidth bound (not compute bound) and creates large intermediate tensors, making it much slower than FLOP count suggests.

