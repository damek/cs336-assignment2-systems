# How to run 

Run the the profiling script [cs336_systems/benchmarking_scripts/memory_profiling.sh](cs336_systems/benchmarking_scripts/memory_profiling.sh), 

```bash
cd cs336_systems/benchmarking_scripts
uv run memory_profiling.sh
```

This will output a tarball of the memory profiling outputs (pickle files and html) to outputs/memory_profiling_<date>_<time>.tar.gz. Then you can untar it and open the html files in your browser / upload the pickle files to [https://docs.pytorch.org/memory_viz](https://docs.pytorch.org/memory_viz)

# Results

> Profile your forward pass, backward pass, and optimizer step of the 2.7B model from Table 1 with context lengths of 128, 256, and 512.

# Question A
> Add an option to your profiling script to run your model through the memory profiler. It may be helpful to reuse some of your previous infrastructure (e.g., to activate mixed-precision, load specific model sizes, etc). Then, run your script to get a memory profile of the 2.7B model when either doing inference only (just forward pass) or a full training step. How do your memory timelines look like? Can you tell which stage is running based on the peaks you see?
> Deliverable: Two images of the “Active memory timeline” of a 2.7B model, from the memory_viz tool: one for the forward pass, and one for running a full training step (forward and backward passes, then optimizer step), and a 2–3 sentence response.

![](figures/memory_snapshot_num_layers_32_num_heads_32_d_model_2560_d_ff_10240_context_length_512_batch_size_4_only_forward_False_bfloat16_False.png)
> Figure 1: Forward, backward, optimizer step. 

![](figures/memory_snapshot_num_layers_32_num_heads_32_d_model_2560_d_ff_10240_context_length_512_batch_size_4_only_forward_True_bfloat16_False.png)

# Question B
> What is the peak memory usage of each context length when doing a forward pass? What about when doing a full training step?
> Deliverable: A table with two numbers per context length.

# Question C
> Find the peak memory usage of the 2.7B model when using mixed-precision, for both a forward pass and a full optimizer step. Does mixed-precision significantly affect memory usage?
> Deliverable: A 2–3 sentence response.

# Question D
> Consider the 2.7B model. At our reference hyperparameters, what is the size of a tensor of activations in the Transformer residual stream, in single-precision? Give this size in MB (i.e., divide the number of bytes by 1024^2).
> Deliverable: A 1–2 sentence response with your derivation.

# Question E
> Now look closely at the “Active Memory Timeline” from pytorch.org/memory_viz of a memory snapshot of the 2.7B model doing a forward pass. When you reduce the “Detail” level, the tool hides the smallest allocations to the corresponding level (e.g., putting “Detail” at 10% only shows the 10% largest allocations). What is the size of the largest allocations shown? Looking through the stack trace, can you tell where those allocations come from?
> Deliverable: A 1–2 sentence response.