# Problem (distributed_communication_single_node): 5 points
> Write a script to benchmark the runtime of the all-reduce operation in the single-node multi-process
> setup. The example code above may provide a reasonable starting point. Experiment with varying the
> following settings:
> Backend + device type: Gloo + CPU, NCCL + GPU.
> all-reduce data size: float32 data tensors ranging over 1MB, 10MB, 100MB, 1GB.
> Number of processes: 2, 4, or 6 processes.
> Resource requirements: Up to 6 GPUs. Each benchmarking run should take less than 5 minutes.

How to run the script:
```bash
uv run cs336_systems/benchmarking_scripts/distributed_communication_single_node.py
```

## Deliverable
> Deliverable: Plot(s) and/or table(s) comparing the various settings, with 2-3 sentences of com-
> mentary about your results and thoughts about how the various factors interact.

```bash
world_size: 2, MB: 1, max time taken: 0.00011833049211418256 seconds
world_size: 2, MB: 10, max time taken: 0.0005937751266174018 seconds
world_size: 2, MB: 100, max time taken: 0.005274404771625996 seconds
world_size: 2, MB: 1000, max time taken: 0.05077763646841049 seconds
world_size: 4, MB: 1, max time taken: 0.00017331502749584615 seconds
world_size: 4, MB: 10, max time taken: 0.0008651238167658448 seconds
world_size: 4, MB: 100, max time taken: 0.007902786135673523 seconds
world_size: 4, MB: 1000, max time taken: 0.07780535519123077 seconds
```

1. Communication time increases linearly with tensor size.
2. Communication time increases with world size. Makes sense: results need to be communicated to all processes.