import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    """Version that requires manual call to finish_gradient_synchronization()"""
    
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._pending = []
        
        # Only proceed if distributed is available and initialized
        if not (dist.is_available() and dist.is_initialized()):
            return
        
        # Broadcast initial parameters and buffers from rank 0
        with torch.no_grad():
            for t in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(t.data, src=0)

        # Register hooks for gradient synchronization
        for p in module.parameters():
            if p.requires_grad:
                if not p.is_leaf:
                    raise RuntimeError("Parameter is not a leaf tensor")
                p.register_post_accumulate_grad_hook(self._make_hook())

    def _make_hook(self): 
        def _hook(param):   
            # Check if distributed is available and initialized
            if not (dist.is_available() and dist.is_initialized()):
                return
            
            # Get world size and check if we need to sync
            ws = dist.get_world_size()
            if ws == 1:
                return
            
            # Check if gradient exists
            if param.grad is None:
                return
            
            # Start async all-reduce
            work = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._pending.append((param, work))
            
        return _hook
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """Wait for all pending gradient synchronizations and average gradients."""
        if not (dist.is_available() and dist.is_initialized()):
            return
            
        ws = dist.get_world_size()
        if ws == 1:
            self._pending.clear()
            return
        
        # Wait for all operations and average the gradients
        for p, work in self._pending:
            work.wait()
            # Average the gradient after the sum reduction
            if p.grad is not None:
                p.grad.div_(ws)
        
        self._pending.clear()