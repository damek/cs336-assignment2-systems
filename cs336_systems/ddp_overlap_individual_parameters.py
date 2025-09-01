import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.step = 0
        
        # Broadcast all parameters and buffers from rank 0 to all ranks
        if dist.is_initialized() and dist.get_world_size() > 1:
            with torch.no_grad():
                # Broadcast parameters
                for p in module.parameters():
                    dist.broadcast(p.data, src=0)
                # Broadcast buffers  
                for b in module.buffers():
                    dist.broadcast(b, src=0)
            
            rank = dist.get_rank()
            print(f"[Rank {rank}] Initial parameter sync complete")
            # Print initial parameter values for first param
            for name, p in list(module.named_parameters())[:1]:
                flat = p.data.flatten()
                print(f"[Rank {rank}] Initial {name}[0:3]: {flat[:3].tolist()}")

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
        self.step += 1
        
        # Before sync: print first param's values and gradients
        for name, p in list(self.module.named_parameters())[:1]:
            if p.grad is not None:
                param_vals = p.data.flatten()[:3].tolist()
                grad_vals = p.grad.data.flatten()[:3].tolist()
                print(f"[Rank {rank}] Step {self.step} BEFORE sync - {name}[0:3]: param={param_vals}, grad={grad_vals}")
        
        # Synchronize all gradients
        for p in self.module.parameters():
            if p.requires_grad and p.grad is not None:
                # All-reduce to sum gradients
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                # Divide by world size to average
                p.grad.data.div_(world_size)
        
        # After sync: print first param's gradients
        for name, p in list(self.module.named_parameters())[:1]:
            if p.grad is not None:
                grad_vals = p.grad.data.flatten()[:3].tolist()
                print(f"[Rank {rank}] Step {self.step} AFTER sync - {name}[0:3]: grad={grad_vals}")
        
        # After optimizer step (which happens after this function returns),
        # the parameters should be updated. We'll check in the next forward pass.
        
    # Properly expose module's parameters
    def parameters(self, recurse=True):
        return self.module.parameters(recurse=recurse)
    
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)