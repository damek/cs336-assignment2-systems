# Problem (optimizer_state_sharding_accounting): 5 points

## Question (a)
> Create a script to profile the peak memory usage when training language models with and without optimizer state sharding. Using the standard configuration (1 node, 2 GPUs, XL model size), report the peak memory usage after model initialization, directly before the optimizer step, and directly after the optimizer step. Do the results align with your expectations? Break down the memory usage in each setting (e.g., how much memory for parameters, how much for optimizer states, etc.).
> Deliverable: 2-3 sentence response with peak memory usage results and a breakdown of how
> the memory is divided between different model and optimizer components.

```bash
uv run cs336_systems/benchmarking_scripts/optimizer_state_sharding_accounting.py
```

Conclusion: Mmeory usage exactly matches expectation. Size of model is 7GB and sharded model optimizer uses 7GB than the non sharded version. This must be so since in the sharded version each rank stores one less state than in the non sharded version.

### Sharded: True
```bash
Training DDP model, local_bs: 2, seq_len: 128, sharded: True
total time train: tensor([1.0515], device='cuda:0')
Mem after model and optimizer creation: 7.4440598487854 GB
Mem before step: tensor([22.4767], device='cuda:0') GB
Mem after step: tensor([22.4767], device='cuda:0') GB
Training DDP model, local_bs: 2, seq_len: 256, sharded: True
total time train: tensor([1.0569], device='cuda:0')
Mem after model and optimizer creation: 7.444090366363525 GB
Mem before step: tensor([22.4863], device='cuda:0') GB
Mem after step: tensor([22.4863], device='cuda:0') GB
Training DDP model, local_bs: 2, seq_len: 512, sharded: True
total time train: tensor([1.0918], device='cuda:0')
Mem after model and optimizer creation: 7.444151401519775 GB
Mem before step: tensor([22.5055], device='cuda:0') GB
Mem after step: tensor([22.5055], device='cuda:0') GB
```

### Sharded: False
```bash
Training DDP model, local_bs: 2, seq_len: 128, sharded: False
total time train: tensor([0.7261], device='cuda:0')
Mem after model and optimizer creation: 7.4440598487854 GB
Mem before step: tensor([29.8016], device='cuda:0') GB
Mem after step: tensor([29.8016], device='cuda:0') GB
Training DDP model, local_bs: 2, seq_len: 256, sharded: False
total time train: tensor([0.7355], device='cuda:0')
Mem after model and optimizer creation: 7.444090366363525 GB
Mem before step: tensor([29.8111], device='cuda:0') GB
Mem after step: tensor([29.8111], device='cuda:0') GB
Training DDP model, local_bs: 2, seq_len: 512, sharded: False
total time train: tensor([0.7739], device='cuda:0')
Mem after model and optimizer creation: 7.444151401519775 GB
Mem before step: tensor([29.8303], device='cuda:0') GB
Mem after step: tensor([29.8303], device='cuda:0') GB
```

## Question (b)
> How does our implementation of optimizer state sharding affect training speed? Measure the time taken per iteration with and without optimizer state sharding for the standard configuration (1 node, 2 GPUs, XL model size).
> Deliverable: 2-3 sentence response with your timings.

Looking at the above results, the speed is slightly slower with sharding: ~1.1 seconds (sharded) vs ~.75 seconds (non sharded).

## Question (c)
> How does our approach to optimizer state sharding differ from ZeRO stage 1 (described as ZeRO-1 DP $P_{os}$ in Rajbhandari et al., 2020)?
> Deliverable: 2-3 sentence summary of any differences, especially those related to memory and communication volume.

In Zero-1 P_{os}, the DP and state sharding communications are wrapped together and so there is no extra communication. 

Why? Well at the end of DP, we do an All reduce, meaning we average the gradients across the ranks. We can split this into two pieces. Indeed, recall that an all reduce is a combination of (1) a reduce scatter followed by (2) an all gather. So we do two things: 
1. We reduce scatter the gradients across the ranks. This gives each rank the shard of the gradient it needs to perform an optimizer step. So we take the step here. 
2. Then after we perform the step, we just all gather all the shards of the gradient.

