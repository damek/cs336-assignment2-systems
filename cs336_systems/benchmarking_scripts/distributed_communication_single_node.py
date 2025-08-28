import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, tensor_size_mb):
    setup(rank, world_size)
    timing = [None] * world_size
    try: 
        # float32 tensor of size tensor_size_mb
        convert_tensor_size = int(tensor_size_mb * 1024 * 1024 / 4)
        data = torch.randn(convert_tensor_size, device=f"cuda:{rank}", dtype=torch.float32)
        # print the size of the tensor
        print(f"rank {rank} tensor size: {data.size()}")
        # print(f"rank {rank} data (before all-reduce): {data}")
        start_time = time.perf_counter()
        dist.all_reduce(data, async_op=False)
        # synchronize all processes
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        timing[rank] = end_time - start_time
        # dist.all_gather(data, data)
        print(f"rank {rank} data (after all-reduce): {data}")
        print(f"rank {rank} time taken: {end_time - start_time} seconds")
        # print the minimum time taken
        print(f"rank {rank} minimum time taken: {min(timing)} seconds")
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4
    tensor_size_mb = 1
    mp.spawn(fn=distributed_demo, args=(world_size, tensor_size_mb), nprocs=world_size, join=True)