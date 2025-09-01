import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        
        # Sync initial parameters from rank 0
        if dist.is_initialized():
            for p in module.parameters():
                dist.broadcast(p.data, src=0)
            for b in module.buffers():
                dist.broadcast(b, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        # Only sync if distributed is initialized and we have multiple ranks
        if dist.is_initialized() and dist.get_world_size() > 1:
            for p in self.module.parameters():
                if p.requires_grad and p.grad is not None:
                    # Sum gradients across all ranks
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    # Average by dividing by world size
                    p.grad.data /= dist.get_world_size()
    
    # Delegate everything else to the wrapped module
    def __getattr__(self, name):
        return getattr(self.module, name)