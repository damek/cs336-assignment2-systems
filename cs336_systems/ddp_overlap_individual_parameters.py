import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        
        # Broadcast all parameters from rank 0 to all other ranks
        if dist.is_initialized():
            with torch.no_grad():
                for p in module.parameters():
                    dist.broadcast(p.data, src=0)
                for b in module.buffers():
                    dist.broadcast(b, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """Synchronize gradients across all ranks"""
        if not dist.is_initialized():
            return
            
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
        
        # Average all gradients across ranks
        for p in self.module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.div_(world_size)
    
    def parameters(self, recurse=True):
        """Forward to module's parameters"""
        return self.module.parameters(recurse=recurse)
    
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        """Forward to module's named_parameters"""
        return self.module.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)