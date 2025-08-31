import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._pending = []
        # dist.broadcast_object_list(module.state_dict(), src=0) # this pickles the entire object and then sends. Apparently you lose some benefits because we must do the pickle on the CPU and then copy back to GPU then send.

        for t in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(t.data, src=0)
        for p in module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._make_hook(p))

    def _make_hook(self, p: torch.Tensor): 
        def _hook(grad):   
            if grad is None or dist.world_size == 1:
                return
            work = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True)
            self._pending.append((p, work))
        return _hook
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        for p, work in self._pending:
            work.wait()
        self._pending.clear()
