Problem (ddp_overlap_individual_parameters_benchmarking):

## Question (a)
> (a) Benchmark the performance of your DDP implementation when overlapping backward pass computation with communication of individual parameter gradients. Compare its performance with
> our previously-studied settings (the minimal DDP implementation that either issues an all-reduce
> for each parameter tensor, or a single all-reduce on the concatenation of all parameter tensors)
> with the same setup: 1 node, 2 GPUs, and the XL model size described in §1.1.2.

How to run the script:
```bash
uv run cs336_systems/benchmarking_scripts/ddp_overlap_individual_parameters_benchmarking.py
```

### Deliverable
> The measured time per training iteration when overlapping the backward pass
> with communication of individual parameter gradients, with 1-2 sentences comparing the results.

### Batch size 2

```bash
Training DDP model, local_bs: 2, seq_len: 128
total time train: tensor([0.7228], device='cuda:0')
total time grad all reduce: tensor([0.2290], device='cuda:0')
ratio grad all reduce to train time: tensor([0.3168], device='cuda:0')
Training DDP model, local_bs: 2, seq_len: 256
total time train: tensor([0.7293], device='cuda:0')
total time grad all reduce: tensor([0.1922], device='cuda:0')
ratio grad all reduce to train time: tensor([0.2635], device='cuda:0')
Training DDP model, local_bs: 2, seq_len: 512
total time train: tensor([0.7686], device='cuda:0')
total time grad all reduce: tensor([0.1336], device='cuda:0')
ratio grad all reduce to train time: tensor([0.1738], device='cuda:0')
```

### Batch size 4

```bash
Training DDP model, local_bs: 4, seq_len: 128
total time train: tensor([0.7294], device='cuda:0')
total time grad all reduce: tensor([0.1942], device='cuda:0')
ratio grad all reduce to train time: tensor([0.2662], device='cuda:0')
Training DDP model, local_bs: 4, seq_len: 256
total time train: tensor([0.7599], device='cuda:0')
total time grad all reduce: tensor([0.1426], device='cuda:0')
ratio grad all reduce to train time: tensor([0.1877], device='cuda:0')
Training DDP model, local_bs: 4, seq_len: 512
total time train: tensor([0.9500], device='cuda:0')
total time grad all reduce: tensor([0.1278], device='cuda:0')
ratio grad all reduce to train time: tensor([0.1345], device='cuda:0')
```