import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.iteration = 0
        
        # Broadcast all parameters from rank 0 to all other ranks
        if dist.is_initialized():
            with torch.no_grad():
                for p in module.parameters():
                    dist.broadcast(p.data, src=0)
                for b in module.buffers():
                    dist.broadcast(b, src=0)
        
        # Print to confirm we're being used
        rank = dist.get_rank() if dist.is_initialized() else -1
        print(f"[Rank {rank}] DDPOverlapIndividualParameters initialized")

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """Synchronize gradients across all ranks"""
        if not dist.is_initialized():
            return
            
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
        
        rank = dist.get_rank()
        self.iteration += 1
        
        # Average all gradients across ranks
        for p in self.module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world_size)
        
        # After final iteration, print parameter values for debugging
        if self.iteration == 5 and rank == 0:
            print(f"\n[Rank 0] After iteration {self.iteration}, before optimizer step:")
            for name, p in list(self.module.named_parameters())[:2]:
                flat = p.data.flatten()
                print(f"  {name}[0:3]: {flat[:3].tolist()}")
                if p.grad is not None:
                    grad_flat = p.grad.flatten()
                    print(f"  {name}.grad[0:3]: {grad_flat[:3].tolist()}")
    
    def parameters(self, recurse=True):
        """Forward to module's parameters"""
        return self.module.parameters(recurse=recurse)
    
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        """Forward to module's named_parameters"""
        return self.module.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
    
    def state_dict(self, *args, **kwargs):
        """Forward to module's state_dict"""
        return self.module.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        """Forward to module's load_state_dict"""
        return self.module.load_state_dict(*args, **kwargs)