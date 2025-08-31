# Problem naive_ddp_benchmarking on GPT2 XL model

> In this naïve DDP implementation, parameters are individually all-reduced across ranks after each
> backward pass. To better understand the overhead of data parallel training, create a script to bench-
> mark your previously-implemented language model when trained with this naïve implementation of
> DDP. Measure the total time per training step and the proportion of time spent on communicating
> gradients. Collect measurements in the single-node setting (1 node x 2 GPUs) for the XL model size
> described in §1.1.2.

How to run
```bash 
uv run cs336_systems/benchmarking_scripts/naive_ddp_benchmarking.py
```
# Deliverable
> Deliverable: A description of your benchmarking setup, along with the measured time per training iteration and time spent communicating gradients for each setting.

Conclusion: 50% of time spend on communicating gradients. 

Note: We also scatter the shards of the data to each process. This is included in the total training time.

## Batch size 2
```bash
Training DDP model, local_bs: 2, seq_len: 128
total time train: tensor([0.7833], device='cuda:0')
total time grad all reduce: tensor([0.4182], device='cuda:0')
ratio grad all reduce to train time: tensor([0.5339], device='cuda:0')
Training DDP model, local_bs: 2, seq_len: 256
total time train: tensor([0.8051], device='cuda:0')
total time grad all reduce: tensor([0.4425], device='cuda:0')
ratio grad all reduce to train time: tensor([0.5496], device='cuda:0')
Training DDP model, local_bs: 2, seq_len: 512
total time train: tensor([0.9033], device='cuda:0')
total time grad all reduce: tensor([0.4576], device='cuda:0')
ratio grad all reduce to train time: tensor([0.5066], device='cuda:0')
```

## Batch size 4
```bash
Training DDP model, local_bs: 4, seq_len: 128
total time train: tensor([0.7958], device='cuda:0')
total time grad all reduce: tensor([0.4395], device='cuda:0')
ratio grad all reduce to train time: tensor([0.5523], device='cuda:0')
Training DDP model, local_bs: 4, seq_len: 256
total time train: tensor([0.8811], device='cuda:0')
total time grad all reduce: tensor([0.4555], device='cuda:0')
ratio grad all reduce to train time: tensor([0.5170], device='cuda:0')
Training DDP model, local_bs: 4, seq_len: 512
W0829 17:22:16.079000 5638 torch/multiprocessing/spawn.py:169] Terminating process 6979 via signal SIGTERM
out of memory (skipping this config)
```