# Problem (ddp_overlap_individual_parameters_benchmarking):

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

Conclusion: Reduces time for the all-reduce step to ~20-30%. 

Compare to [minimal_ddp_flat_benchmarking.md](minimal_ddp_flat_benchmarking.md) (~40%) and [naive_ddp_benchmarking.md](naive_ddp_benchmarking.md) (~50%).

#### Batch size 2

```bash
Training DDP model, local_bs: 2, seq_len: 128
total time train: tensor([0.7245], device='cuda:0')
total time grad all reduce: tensor([0.2318], device='cuda:0')
ratio grad all reduce to train time: tensor([0.3199], device='cuda:0')
Training DDP model, local_bs: 2, seq_len: 256
total time train: tensor([0.7328], device='cuda:0')
total time grad all reduce: tensor([0.1831], device='cuda:0')
ratio grad all reduce to train time: tensor([0.2499], device='cuda:0')
Training DDP model, local_bs: 2, seq_len: 512
total time train: tensor([0.7731], device='cuda:0')
total time grad all reduce: tensor([0.1371], device='cuda:0')
ratio grad all reduce to train time: tensor([0.1773], device='cuda:0')
```

#### Batch size 4

```bash
Training DDP model, local_bs: 4, seq_len: 128
total time train: tensor([0.7297], device='cuda:0')
total time grad all reduce: tensor([0.2001], device='cuda:0')
ratio grad all reduce to train time: tensor([0.2742], device='cuda:0')
Training DDP model, local_bs: 4, seq_len: 256
total time train: tensor([0.7574], device='cuda:0')
total time grad all reduce: tensor([0.1393], device='cuda:0')
ratio grad all reduce to train time: tensor([0.1840], device='cuda:0')
Training DDP model, local_bs: 4, seq_len: 512
total time train: tensor([0.9468], device='cuda:0')
total time grad all reduce: tensor([0.1286], device='cuda:0')
ratio grad all reduce to train time: tensor([0.1359], device='cuda:0')
```

## Question (b)

> Instrument your benchmarking code (using the 1 node, 2 GPUs, XL model size setup) with the
> Nsight profiler, comparing between the initial DDP implementation and this DDP implementation that overlaps backward computation and communication. Visually compare the two traces,
> and provide a profiler screenshot demonstrating that one implementation overlaps compute with
> communication while the other doesn’t.

How to run the script:
```bash
bash cs336_systems/benchmarking_scripts/ddp_overlap_individual_parameters_benchmarking_nvtx.sh
```
### Deliverable
> Deliverable: 2 screenshots (one from the initial DDP implementation, and another from this
> DDP implementation that overlaps compute with communication) that visually show that communication is or isn’t overlapped with the backward pass.

In the figures below look at the pt_autograd_0 row and the nccl rows. You can see that in the naive case the all reduce and backward pass do not overlap. Whereas in the overlap case they do.


#### Naive DDP
![Naive DDP](../outputs/nsys/ddp_overlap_individual_parameters_benchmarking_nvtx/naive_ddp.png)


#### Overlap DDP
![Overlap DDP](../outputs/nsys/ddp_overlap_individual_parameters_benchmarking_nvtx/overlap.png)
