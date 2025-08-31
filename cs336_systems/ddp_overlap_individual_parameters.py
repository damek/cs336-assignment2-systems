import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        # Store the module as an attribute
        self.module = module
        self._pending = []
        
        # Broadcast parameters and buffers from rank 0 to all other ranks
        if dist.is_initialized() and dist.get_world_size() > 1:
            with torch.no_grad():
                for p in module.parameters():
                    dist.broadcast(p.data, src=0)
                for b in module.buffers():
                    dist.broadcast(b, src=0)

            # Register hooks for gradient synchronization
            for p in module.parameters():
                if p.requires_grad:
                    if not p.is_leaf:
                        raise RuntimeError("Parameter is not a leaf tensor")
                    p.register_post_accumulate_grad_hook(self._hook)

    def _hook(self, param):   
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        
        if param.grad is None:
            return
            
        # Start async all_reduce to sum gradients across all ranks
        work = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self._pending.append((param, work))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        if not dist.is_initialized():
            return
            
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
        
        # Wait for all async operations to complete and then divide by world size
        for p, work in self._pending:
            work.wait()
            if p.grad is not None:
                p.grad.div_(world_size)
        
        # Clear pending operations for next iteration
        self._pending.clear()
    
    # Explicitly forward the methods that the test might need
    def parameters(self, recurse=True):
        return self.module.parameters(recurse=recurse)
    
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
    
    def buffers(self, recurse=True):
        return self.module.buffers(recurse=recurse)
    
    def named_buffers(self, prefix='', recurse=True, remove_duplicate=True):
        return self.module.named_buffers(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
    
    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)
    
    def train(self, mode=True):
        self.module.train(mode)
        return super().train(mode)
    
    def eval(self):
        self.module.eval()
        return super().eval()
    
