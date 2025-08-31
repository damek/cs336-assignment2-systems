import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.iteration = 0
        
        # Broadcast all parameters and buffers from rank 0 to all ranks
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
            print(f"[Rank {rank}] Initializing DDP wrapper")
            
            with torch.no_grad():
                # Broadcast parameters
                for name, p in module.named_parameters():
                    before = p.data.clone()
                    dist.broadcast(p.data, src=0)
                    after = p.data
                    changed = not torch.allclose(before, after)
                    if changed or rank == 0:
                        print(f"[Rank {rank}] Param {name}: shape={p.shape}, "
                              f"changed={changed}, requires_grad={p.requires_grad}")
                
                # Broadcast buffers  
                for name, b in module.named_buffers():
                    dist.broadcast(b, src=0)
                    print(f"[Rank {rank}] Buffer {name}: shape={b.shape}")

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """Manually synchronize all gradients after backward pass"""
        if not dist.is_initialized():
            return
            
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
        
        rank = dist.get_rank()
        self.iteration += 1
        print(f"\n[Rank {rank}] Iteration {self.iteration}: Starting gradient sync")
        
        # Synchronize all gradients
        for name, p in self.module.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    # Check gradient before sync
                    grad_norm_before = p.grad.data.norm().item()
                    grad_mean_before = p.grad.data.mean().item()
                    
                    # All-reduce to sum gradients
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    
                    # Divide by world size to average
                    p.grad.data.div_(world_size)
                    
                    # Check gradient after sync
                    grad_norm_after = p.grad.data.norm().item()
                    grad_mean_after = p.grad.data.mean().item()
                    
                    print(f"[Rank {rank}] {name}: grad_norm {grad_norm_before:.6f} -> {grad_norm_after:.6f}, "
                          f"grad_mean {grad_mean_before:.6f} -> {grad_mean_after:.6f}")
                else:
                    print(f"[Rank {rank}] {name}: NO GRADIENT")
            else:
                print(f"[Rank {rank}] {name}: requires_grad=False")
        
        print(f"[Rank {rank}] Iteration {self.iteration}: Gradient sync complete\n")
    
    # Properly expose module's parameters
    def parameters(self, recurse=True):
        return self.module.parameters(recurse=recurse)
    
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)