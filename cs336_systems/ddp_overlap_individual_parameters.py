import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._pending = []  # stores (param, work, g64)

        # on-device broadcast of initial state from rank 0
        with torch.no_grad():
            for t in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(t.data, src=0)

        # IMPORTANT: register hook that captures the Parameter (not the grad)
        for p in module.parameters():
            if p.requires_grad:
                if not p.is_leaf:
                    raise RuntimeError("Parameter is not a leaf tensor")
                p.register_post_accumulate_grad_hook(lambda _, p=p: self._hook(p))

    def _hook(self, p: torch.nn.Parameter):
        if not (dist.is_available() and dist.is_initialized()):
            return
        if p.grad is None:
            return
        # Reduce in float64 to remove FP32 order/rounding drift.
        g64 = p.grad.detach().to(torch.float64)
        work = dist.all_reduce(g64, op=dist.ReduceOp.SUM, async_op=True)  # AVG not portable on gloo
        self._pending.append((p, work, g64))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        if not (dist.is_available() and dist.is_initialized()):
            self._pending.clear()
            return
        ws = dist.get_world_size()
        if ws == 1:
            self._pending.clear()
            return

        # Wait for all async reductions, then average once and copy back to p.grad
        for _, work, _ in self._pending:
            work.wait()
        for p, _, g64 in self._pending:
            if p.grad is not None:
                p.grad.copy_((g64 / ws).to(p.grad.dtype))

        self._pending.clear()
