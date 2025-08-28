import torch
import cs336_basics.model as model_class
import cs336_basics.data as data
import cs336_basics.optimizer as optimizer_class
import cs336_basics.nn_utils as nn_utils
import torch.distributed as dist
import torch.multiprocessing as mp
import os, tempfile
import numpy as np

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
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

def train(rank, world_size, nb_iters, model_dict, optimizer_dict, local_bs, path):
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

        for iter in range(nb_iters):
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
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

            optimizer.step()

            if iter % 10 == 0:
                loss_avg = loss.detach()
                dist.all_reduce(loss_avg, op=dist.ReduceOp.AVG)
                if rank == 0:
                    print(f"Iteration {iter} loss: {loss_avg.item()}")

        if rank == 0:
            torch.save(model.state_dict(), path)
    finally: 
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4
    nb_iters = 1000
    local_bs = 2
    global_bs = local_bs*world_size
    model_dict = {
        "vocab_size": 1024,
        "context_length": 128,
        "d_model": 128,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 4*128,
        "rope_theta": 10000,    
    }
    optimizer_dict = {
        "lr": 1e-3,
        "weight_decay": 0.01,
    }
    with tempfile.TemporaryDirectory(prefix="ddp_compare_") as outdir:
        single_path = os.path.join(outdir, "single.pt")
        ddp_path    = os.path.join(outdir, "ddp.pt")

        print("Training single GPU model")
        mp.spawn(train, args=(1, nb_iters, model_dict, optimizer_dict, global_bs, single_path),
                 nprocs=1, join=True)

        print("Training DDP model")
        mp.spawn(fn=train, args=(world_size, nb_iters, model_dict, optimizer_dict, local_bs, ddp_path), nprocs=world_size, join=True)

        s1 = torch.load(single_path, map_location="cpu")
        s2 = torch.load(ddp_path, map_location="cpu")
        same = all(torch.equal(s1[k], s2[k]) for k in s1)

        total_error = 0
        norm_of_s1 = 0
        norm_of_s2 = 0
        
        for k in s1:
            total_error += torch.sum(torch.square(s1[k] - s2[k]))
            norm_of_s1 += torch.sum(torch.square(s1[k]))
            norm_of_s2 += torch.sum(torch.square(s2[k]))
        total_error = torch.sqrt(total_error)
        norm_of_s1 = torch.sqrt(norm_of_s1)
        norm_of_s2 = torch.sqrt(norm_of_s2)
        print("same", same)
        print("total_error", total_error)
        print("relative_error", total_error/(torch.max(norm_of_s1, norm_of_s2)))