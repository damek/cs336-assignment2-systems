import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._pending = []
        # dist.broadcast_object_list(module.state_dict(), src=0) # this pickles the entire object and then sends. Apparently you lose some benefits because we must do the pickle on the CPU and then copy back to GPU then send.
        with torch.no_grad():
            for t in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(t.data, src=0)

            for p in module.parameters():
                if p.requires_grad:
                    if not p.is_leaf:
                        raise RuntimeError("Parameter is not a leaf tensor")
                    p.register_post_accumulate_grad_hook(self._hook)


    def _hook(self, param):   
        if not (dist.is_available() and dist.is_initialized()):
            return
        # work = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True) # gloo doesn't have avg!??!?!
        work = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self._pending.append((param, work))
        return None

    
    def forward(self, *args, **kwargs):
        # Clear any pending operations from previous forward pass
        self._pending.clear()
        
        # Forward through the wrapped module
        output = self.module(*args, **kwargs)
        
        # Register a hook on the output to finish gradient sync after backward
        if torch.is_tensor(output):
            output.register_hook(lambda grad: self._backward_hook(grad))
        elif isinstance(output, tuple):
            # If output is a tuple, register on the first tensor
            for o in output:
                if torch.is_tensor(o) and o.requires_grad:
                    o.register_hook(lambda grad: self._backward_hook(grad))
                    break
        
        return output
    
    def finish_gradient_synchronization(self):
        ws = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        for p, work in self._pending:
            work.wait()
            if ws > 1 and p.grad is not None:
                p.grad.div_(ws)
        self._pending.clear()
