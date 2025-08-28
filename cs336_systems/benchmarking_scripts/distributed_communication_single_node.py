import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, tensor_size_mb, num_iterations, num_warmup_iterations=5):
    setup(rank, world_size)
    # timing = [None] * world_size
    try: 
        convert_tensor_size = int(tensor_size_mb * 1024 * 1024 / 4)
        data = torch.randn(convert_tensor_size, device=f"cuda:{rank}", dtype=torch.float32)
        print(f"rank {rank} tensor size: {data.size()}")
        total_time = 0
        for _ in range(num_warmup_iterations):
            dist.all_reduce(data, async_op=False)
            torch.cuda.synchronize()
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            dist.all_reduce(data, async_op=False)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            total_time += end_time - start_time
        total_time /= num_iterations

        # timing[rank] = end_time - start_time
        # dist.all_gather(data, data)
        print(f"rank {rank} time taken: {total_time} seconds")
        # print the minimum time taken
        # print(f"rank {rank} minimum time taken: {min(timing)} seconds")
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4
    tensor_size_mb = 1
    num_iterations = 10
    num_warmup_iterations = 5
    mp.spawn(fn=distributed_demo, args=(world_size, tensor_size_mb, num_iterations, num_warmup_iterations), nprocs=world_size, join=True)