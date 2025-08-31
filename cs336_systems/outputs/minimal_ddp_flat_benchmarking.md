# Problem (minimal_ddp_flat_benchmarking): 

> Modify your minimal DDP implementation to communicate a tensor with flattened gradients from
> all parameters. Compare its performance with the minimal DDP implementation that issues an all-
> reduce for each parameter tensor under the previously-used conditions (1 node x 2 GPUs, XL model
> size as described in §1.1.2).

How to run the script:
```bash 
uv run cs336_systems/benchmarking_scripts/minimal_ddp_flat_benchmarking.py
```

# Deliverable: 

> The measured time per training iteration and time spent communicating gradients under distributed data parallel training with a single batched all-reduce call. 1-2 sentences comparing the results when batching vs. individually communicating gradients.

Conclusion: 
Ratio of communication time to training time tends to decrease as context length and batch size increases. Compare to [naive_ddp_benchmarking.md](cs336_systems/outputs/naive_ddp_benchmarking.md).


## Batch size 2
```bash
Training DDP model, local_bs: 2, seq_len: 128
total time train: tensor([0.8251], device='cuda:0')
total time grad all reduce: tensor([0.4273], device='cuda:0')
ratio grad all reduce to train time: tensor([0.5178], device='cuda:0')
Training DDP model, local_bs: 2, seq_len: 256
total time train: tensor([0.7995], device='cuda:0')
total time grad all reduce: tensor([0.4080], device='cuda:0')
ratio grad all reduce to train time: tensor([0.5103], device='cuda:0')
Training DDP model, local_bs: 2, seq_len: 512
total time train: tensor([0.9150], device='cuda:0')
total time grad all reduce: tensor([0.4227], device='cuda:0')
ratio grad all reduce to train time: tensor([0.4619], device='cuda:0')
```

## Batch size 4
```bash
Training DDP model, local_bs: 4, seq_len: 128
total time train: tensor([0.7978], device='cuda:0')
total time grad all reduce: tensor([0.4065], device='cuda:0')
ratio grad all reduce to train time: tensor([0.5096], device='cuda:0')
Training DDP model, local_bs: 4, seq_len: 256
total time train: tensor([0.8768], device='cuda:0')
total time grad all reduce: tensor([0.4132], device='cuda:0')
ratio grad all reduce to train time: tensor([0.4712], device='cuda:0')
Training DDP model, local_bs: 4, seq_len: 512
total time train: tensor([1.9628], device='cuda:0')
total time grad all reduce: tensor([0.4029], device='cuda:0')
ratio grad all reduce to train time: tensor([0.2053], device='cuda:0')
```