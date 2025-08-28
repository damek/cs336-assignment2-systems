import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    try: 
        data = torch.randint(0, 10, (3,), device=f"cuda:{rank}")
        print(f"rank {rank} data (before all-reduce): {data}")
        dist.all_reduce(data, async_op=False)
        print(f"rank {rank} data (after all-reduce): {data}")
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)