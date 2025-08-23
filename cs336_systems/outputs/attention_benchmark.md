# Attention Benchmarking Results

> (a) Benchmark your attention implementation at different scales. Write a script that will:
> (a) Fix the batch size to 8 and don’t use multihead attention (i.e. remove the head dimension).
> (b) Iterate through the cartesian product of [16, 32, 64, 128] for the head embedding di-
> mension dmodel, and [256, 1024, 4096, 8192, 16384] for the sequence length.
> (c) Create random inputs Q, K, V for the appropriate size.
> (d) Time 100 forward passes through attention using the inputs.
> (e) Measure how much memory is in use before the backward pass starts, and time 100 backward
> passes.
> (f) Make sure to warm up, and to call torch.cuda.synchronize() after each forward/backward
> pass.

You can run the script with 
```bash
uv run cs336_systems/benchmarking_scripts/attention_benchmarking_script.py
```

## Timings
> Report the timings (or out-of-memory errors) you get for these configurations. 
> At what size do you get out-of-memory errors?

Timings below! We never get out-of-memory errors.

|   d_model |   seq_len |   forward_ms |   backward_ms |   mem_after_inputs_MiB |   mem_before_backward_MiB |   saved_activations_MiB |   forward_peak_MiB |   backward_peak_MiB | status   |
|----------:|----------:|-------------:|--------------:|----------------------:|-------------------------:|-----------------------:|------------------:|-------------------:|:---------|
|        16 |       256 |     0.245064 |      0.875942 |               16.7505 |                  20.7739 |                4.02344 |           24.7739 |            33.0483 | ok       |
|        16 |      1024 |     0.365331 |      1.2174   |               18.2505 |                  82.3442 |               64.0938  |          146.344  |           275.439  | ok       |
|        16 |      4096 |     4.09638  |     14.0086   |               24.2505 |                1048.63   |             1024.38    |         2072.63   |          4125      | ok       |
|        16 |      8192 |    16.3921   |     55.1659   |               32.2505 |                4129      |             4096.75    |         8225      |         16425.8    | ok       |
|        16 |     16384 |    64.5354   |    220.459    |               48.2505 |               16433.8    |            16385.5     |        32817.8    |         65603.3    | ok       |
|        32 |       256 |     0.255866 |      0.743333 |               17.2505 |                  21.2739 |                4.02344 |           25.2739 |            33.7983 | ok       |
|        32 |      1024 |     0.376353 |      1.15932  |               20.2505 |                  84.3442 |               64.0938  |          148.344  |           278.439  | ok       |
|        32 |      4096 |     4.30662  |     14.5374   |               32.2505 |                1056.63   |             1024.38    |         2080.63   |          4137      | ok       |
|        32 |      8192 |    16.7079   |     55.9166   |               48.2505 |                4145      |             4096.75    |         8241      |         16449.8    | ok       |
|        32 |     16384 |    65.8194   |    222.943    |               80.2505 |               16465.8    |            16385.5     |        32849.8    |         65651.3    | ok       |
|        64 |       256 |     0.239101 |      0.747274 |               18.2505 |                  22.2739 |                4.02344 |           26.2739 |            35.2983 | ok       |
|        64 |      1024 |     0.381877 |      1.17848  |               24.2505 |                  88.3442 |               64.0938  |          152.344  |           284.439  | ok       |
|        64 |      4096 |     4.43956  |     14.8003   |               48.2505 |                1072.63   |             1024.38    |         2096.63   |          4161      | ok       |
|        64 |      8192 |    17.3423   |     57.206    |               80.2505 |                4177      |             4096.75    |         8273      |         16497.8    | ok       |
|        64 |     16384 |    68.1324   |    228.148    |              144.25   |               16529.8    |            16385.5     |        32913.8    |         65747.3    | ok       |
|       128 |       256 |     0.239547 |      0.750159 |               20.2505 |                  24.2739 |                4.02344 |           28.2739 |            38.2983 | ok       |
|       128 |      1024 |     0.401423 |      1.2698   |               32.2505 |                  96.3442 |               64.0938  |          160.344  |           296.439  | ok       |
|       128 |      4096 |     4.64845  |     15.5428   |               80.2505 |                1104.63   |             1024.38    |         2128.63   |          4209      | ok       |
|       128 |      8192 |    18.2704   |     60.044    |              144.25   |                4241      |             4096.75    |         8337      |         16593.8    | ok       |
|       128 |     16384 |    71.7961   |    240.959    |              272.25   |               16657.8    |            16385.5     |        33041.8    |         65939.3    | ok       |

Forward (ms/iter) by seq_len x d_model:
 |   seq_len |        16 |        32 |        64 |       128 |
|----------:|----------:|----------:|----------:|----------:|
|       256 |  0.245064 |  0.255866 |  0.239101 |  0.239547 |
|      1024 |  0.365331 |  0.376353 |  0.381877 |  0.401423 |
|      4096 |  4.09638  |  4.30662  |  4.43956  |  4.64845  |
|      8192 | 16.3921   | 16.7079   | 17.3423   | 18.2704   |
|     16384 | 64.5354   | 65.8194   | 68.1324   | 71.7961   |

Backward (ms/iter) by seq_len x d_model:
 |   seq_len |         16 |         32 |         64 |        128 |
|----------:|-----------:|-----------:|-----------:|-----------:|
|       256 |   0.875942 |   0.743333 |   0.747274 |   0.750159 |
|      1024 |   1.2174   |   1.15932  |   1.17848  |   1.2698   |
|      4096 |  14.0086   |  14.5374   |  14.8003   |  15.5428   |
|      8192 |  55.1659   |  55.9166   |  57.206    |  60.044    |
|     16384 | 220.459    | 222.943    | 228.148    | 240.959    |

## Memory Usage

> Do the accounting for the memory usage of attention in one of the
> smallest configurations you find that runs out of memory (you can use the equations for memory
> usage of Transformers from Assignment 1). How does the memory saved for backward change
> with the sequence length? 

We had no OOM errors, so we'll just use the largest configuration that ran.

Our forward pass saves the activations (probs and logits). This is in the table under `saved_activations_MiB`. This amounts to 2* B * L^2 * 4 bytes. = 2* 8 * 16384^2 * 4 bytes = 17,179,869,184 bytes = 16,384 MiB, which closely matches the table. 

> What would you do to eliminate this memory cost?

Flash attention, duh.

