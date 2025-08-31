import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    """
    Simplest possible DDP implementation.
    No hooks - all synchronization happens manually in finish_gradient_synchronization.
    """
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        
        # Broadcast parameters and buffers from rank 0 to all other ranks
        if dist.is_initialized() and dist.get_world_size() > 1:
            with torch.no_grad():
                for p in module.parameters():
                    dist.broadcast(p.data, src=0)
                for b in module.buffers():
                    dist.broadcast(b, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """Synchronize all gradients after backward pass."""
        if not dist.is_initialized():
            return
            
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
        
        # Synchronize gradients for all parameters that have them
        with torch.no_grad():
            for p in self.module.parameters():
                if p.grad is not None:
                    # Synchronously all-reduce (sum) the gradients
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    # Average by dividing by world size
                    p.grad.data.div_(world_size)
    
    # Make the wrapper completely transparent
    def __getattr__(self, name):
        # For any attribute not found in DDPOverlapIndividualParameters,
        # forward to the wrapped module
        return getattr(self.module, name)