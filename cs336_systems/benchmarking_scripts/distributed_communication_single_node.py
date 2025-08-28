import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, MB, num_iterations, num_warmup_iterations=5):
    setup(rank, world_size)
    try: 
        convert_tensor_size = int(MB * 1024 * 1024 / 4)
        data = torch.randn(convert_tensor_size, device=f"cuda:{rank}", dtype=torch.float32)
        total_time = 0
        for _ in range(num_warmup_iterations):
            dist.all_reduce(data, async_op=False)
            torch.cuda.synchronize()
        dist.barrier()
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            dist.all_reduce(data, async_op=False)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            total_time += end_time - start_time
        total_time /= num_iterations
        max_time = torch.tensor([total_time], device="cuda")
        dist.all_reduce(max_time, op=dist.ReduceOp.MAX)
        torch.cuda.synchronize()
        if rank == 0:
            print(f"world_size: {world_size}, MB: {MB}, max time taken: {max_time[0]} seconds")
    finally:
        dist.destroy_process_group()

if __name__ == "__main__":
    for world_size in [2, 4]:
        for MB in [1, 10, 100, 1000]:
            print(f"world_size: {world_size}, MB: {MB}")
            num_warmup_iterations = 5
            num_iterations = 10
            mp.spawn(fn=distributed_demo, args=(world_size, MB, num_iterations, num_warmup_iterations), nprocs=world_size, join=True)