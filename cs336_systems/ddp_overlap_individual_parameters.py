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
                    after = p.data.clone()
                    changed = not torch.allclose(before, after)
                    
                    # Print first few values of the parameter
                    flat = p.data.flatten()
                    values = flat[:3].tolist() if len(flat) >= 3 else flat.tolist()
                    print(f"[Rank {rank}] Param {name}: shape={p.shape}, "
                          f"changed={changed}, requires_grad={p.requires_grad}, "
                          f"first_values={values}")
                
                # Broadcast buffers  
                for name, b in module.named_buffers():
                    dist.broadcast(b, src=0)

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
        
        # Only print for first 2 iterations to avoid clutter
        if self.iteration <= 2:
            print(f"\n[Rank {rank}] Iteration {self.iteration}: Starting gradient sync")
        
        # Synchronize all gradients
        for name, p in self.module.named_parameters():
            if p.requires_grad:
                if p.grad is not None:
                    # All-reduce to sum gradients
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    
                    # Divide by world size to average
                    p.grad.data.div_(world_size)
                    
                    if self.iteration <= 2:
                        grad_norm = p.grad.data.norm().item()
                        grad_mean = p.grad.data.mean().item()
                        print(f"[Rank {rank}] {name}: grad_norm={grad_norm:.6f}, grad_mean={grad_mean:.6f}")
        
        # After last iteration, print parameter values
        if self.iteration == 5:
            print(f"\n[Rank {rank}] FINAL PARAMETER VALUES:")
            for name, p in self.module.named_parameters():
                flat = p.data.flatten()
                values = flat[:3].tolist() if len(flat) >= 3 else flat.tolist()
                print(f"[Rank {rank}] {name}: first_values={values}")
    
    # Properly expose module's parameters
    def parameters(self, recurse=True):
        return self.module.parameters(recurse=recurse)
    
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)