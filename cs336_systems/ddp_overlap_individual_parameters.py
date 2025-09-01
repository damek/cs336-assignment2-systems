import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        # map: Parameter -> list of (work_handle, reduced_buf64)
        self._pending = {}

        with torch.no_grad():
            for t in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(t.data, src=0)

        for p in module.parameters():
            if p.requires_grad:
                if not p.is_leaf:
                    raise RuntimeError("Parameter is not a leaf tensor")
                # capture both param and the hook's grad
                p.register_post_accumulate_grad_hook(lambda grad, p=p: self._hook(p, grad))

    def _hook(self, p: torch.nn.Parameter, grad: torch.Tensor):
        if not (dist.is_available() and dist.is_initialized()):
            return
        if grad is None:
            return
        # snapshot this contribution into a separate buffer (avoid racing on p.grad)
        g64 = grad.detach().to(torch.float64).contiguous()
        work = dist.all_reduce(g64, op=dist.ReduceOp.SUM, async_op=True)
        self._pending.setdefault(p, []).append((work, g64))

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

        # wait for all async reductions to complete
        for lst in self._pending.values():
            for work, _ in lst:
                work.wait()

        # sum all reduced contributions per parameter, average, and write back once
        for p, lst in self._pending.items():
            if p.grad is None:
                continue  # nothing to write
            total = None
            for _, g64 in lst:
                total = g64 if total is None else total.add_(g64)
            p.grad.copy_((total / ws).to(p.grad.dtype))

        self._pending.clear()
