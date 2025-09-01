import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self._pending = []  # (param, work, g64)

        # initial sync, on-device
        with torch.no_grad():
            for t in list(module.parameters()) + list(module.buffers()):
                print(t.data.dtype)
                dist.broadcast(t.data, src=0)

        # KEY CHANGE: use register_hook (final grad), not post_accumulate
        for p in module.parameters():
            if p.requires_grad:
                if not p.is_leaf:
                    raise RuntimeError("Parameter is not a leaf tensor")
                p.register_hook(lambda grad, p=p: self._hook(p, grad))

    # hook gets the FINAL grad for this param for this backward
    def _hook(self, p: torch.nn.Parameter, grad: torch.Tensor):
        if not (dist.is_available() and dist.is_initialized()):
            return
        if grad is None:
            return
        # reduce in float64 to kill FP32 order noise
        g64 = grad.detach().to(torch.float64)
        work = dist.all_reduce(g64, op=dist.ReduceOp.SUM, async_op=True)
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

        # wait for async reductions, then average and copy back to p.grad
        for _, work, _ in self._pending:
            work.wait()
        for p, _, g64 in self._pending:
            if p.grad is not None:
                p.grad.copy_((g64 / ws).to(p.grad.dtype))

        self._pending.clear()
