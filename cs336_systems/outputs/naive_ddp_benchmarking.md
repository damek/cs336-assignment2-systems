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
total time train: tensor([0.7894], device='cuda:0')
total time grad all reduce: tensor([0.4236], device='cuda:0')
ratio grad all reduce to train time: tensor([0.5367], device='cuda:0')
Training DDP model, local_bs: 2, seq_len: 256
total time train: tensor([0.8113], device='cuda:0')
total time grad all reduce: tensor([0.4399], device='cuda:0')
ratio grad all reduce to train time: tensor([0.5423], device='cuda:0')
Training DDP model, local_bs: 2, seq_len: 512
total time train: tensor([0.9078], device='cuda:0')
total time grad all reduce: tensor([0.4611], device='cuda:0')
ratio grad all reduce to train time: tensor([0.5079], device='cuda:0')
```

## Batch size 4
```bash
Training DDP model, local_bs: 4, seq_len: 128
total time train: tensor([0.8009], device='cuda:0')
total time grad all reduce: tensor([0.4289], device='cuda:0')
ratio grad all reduce to train time: tensor([0.5355], device='cuda:0')
Training DDP model, local_bs: 4, seq_len: 256
total time train: tensor([0.8796], device='cuda:0')
total time grad all reduce: tensor([0.4550], device='cuda:0')
ratio grad all reduce to train time: tensor([0.5173], device='cuda:0')
Training DDP model, local_bs: 4, seq_len: 512
total time train: tensor([1.1434], device='cuda:0')
total time grad all reduce: tensor([0.4933], device='cuda:0')
ratio grad all reduce to train time: tensor([0.4314], device='cuda:0')
