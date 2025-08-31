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
                    p.register_post_accumulate_grad_hook(self._make_hook(p))

    def _make_hook(self, p: torch.Tensor): 
        def _hook(param):   
            ws = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
            if param.grad is None or ws == 1:
                return
            # work = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True) # gloo doesn't have avg!??!?!
            work = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._pending.append((param, work))
        return _hook
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        ws = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        for p, work in self._pending:
            work.wait()
            if ws > 1 and p.grad is not None:
                p.grad.div_(ws)
        self._pending.clear()
