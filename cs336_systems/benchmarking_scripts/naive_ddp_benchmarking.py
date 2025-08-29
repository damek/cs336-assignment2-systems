import torch
import cs336_basics.model as model_class
import cs336_basics.data as data
import cs336_basics.optimizer as optimizer_class
import cs336_basics.nn_utils as nn_utils
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessRaisedException
import os
import numpy as np
import time

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def create_model_and_optimizer(model_dict, optimizer_dict, device):
    vocab_size = model_dict["vocab_size"]
    context_length = model_dict["context_length"]
    d_model = model_dict["d_model"]
    num_layers = model_dict["num_layers"]
    num_heads = model_dict["num_heads"]
    d_ff = model_dict["d_ff"]
    rope_theta = model_dict["rope_theta"]
    model = model_class.BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    model.to(device)
    # Create optimizer
    lr = optimizer_dict.get("lr", 1e-3)
    weight_decay = optimizer_dict.get("weight_decay", 0.01)
    optimizer = optimizer_class.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return model, optimizer

def train(rank, world_size, nb_iters, model_dict, optimizer_dict, local_bs, nb_warmup=10):
    setup(rank, world_size)
    device=f"cuda:{rank}"
    try: 
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        model, optimizer = create_model_and_optimizer(model_dict, optimizer_dict, device)

        dicts = [None, None]
        if rank == 0: 
            dicts = [model.state_dict(), optimizer.state_dict()]
        dist.broadcast_object_list(dicts, src=0)
        model.load_state_dict(dicts[0])
        optimizer.load_state_dict(dicts[1])

        vocab_size = model_dict["vocab_size"]
        context_length = model_dict["context_length"]
        dataset_len = 100_000
        if rank == 0:
            np.random.seed(0)   
            dataset_len = 100_000
            dataset = np.random.randint(0, vocab_size, size=(dataset_len,), dtype=np.int64)
        else:
            dataset = None  
        global_bs = local_bs*world_size

        x_local = torch.empty((local_bs, context_length), dtype=torch.long, device=device)
        y_local = torch.empty((local_bs, context_length), dtype=torch.long, device=device)

        total_time_train = torch.zeros(1, device=device)
        total_time_grad_all_reduce = torch.zeros(1, device=device)

        for iter in range(nb_iters + nb_warmup):
            start_time_train = time.perf_counter()
            if rank == 0:
                inputs, targets = data.get_batch(dataset, global_bs, context_length, device=device)
                x_list = [t.contiguous() for t in inputs.chunk(world_size, dim=0)]
                y_list = [t.contiguous() for t in targets.chunk(world_size, dim=0)]
            else:
                x_list = y_list = None
            dist.scatter(x_local, scatter_list=x_list, src=0)
            dist.scatter(y_local, scatter_list=y_list, src=0)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x_local)
            loss = nn_utils.cross_entropy(logits, y_local)
            loss.backward()
            start_time_grad_all_reduce = time.perf_counter()
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.cuda.synchronize()
            end_time_grad_all_reduce = time.perf_counter()
            if iter >= nb_warmup:
                total_time_grad_all_reduce += end_time_grad_all_reduce - start_time_grad_all_reduce
            optimizer.step()
            torch.cuda.synchronize()
            end_time_train = time.perf_counter()
            if iter >= nb_warmup:
                total_time_train += end_time_train - start_time_train

        dist.all_reduce(total_time_train, op=dist.ReduceOp.MAX)
        dist.all_reduce(total_time_grad_all_reduce, op=dist.ReduceOp.MAX)
        
        if rank == 0:
            print(f"total time train: {total_time_train/nb_iters}")
            print(f"total time grad all reduce: {total_time_grad_all_reduce/nb_iters}")
            print(f"ratio train time to grad all reduce: {total_time_grad_all_reduce/total_time_train}")

    finally: 
        if dist.is_initialized():
            dist.destroy_process_group()



if __name__ == "__main__":
    world_size = 2
    nb_iters = 10
    local_bss = [2, 4]
    seq_lengths = [128, 256, 512]
    warmup=10
    # XL model
    optimizer_dict = {
        "lr": 1e-3,
        "weight_decay": 0.01,
    }

    for local_bs in local_bss:
        for seq_len in seq_lengths:
            model_dict = {
                "vocab_size": 10000,
                "context_length": seq_len,
                "d_model": 1600,
                "num_layers": 48,
                "num_heads": 25,
                "d_ff": 6400,
                "rope_theta": 10000,    
            }
            print(f"Training DDP model, local_bs: {local_bs}, seq_len: {seq_len}")
            try: 
                mp.spawn(fn=train, args=(world_size, nb_iters, model_dict, optimizer_dict, local_bs,warmup), nprocs=world_size, join=True)
            # If out of memory error, print out of memory, skipping
            except ProcessRaisedException as e:           
                if "out of memory" in str(e).lower():    
                    print("out of memory (skipping this config)")
                    continue
                raise  

                
