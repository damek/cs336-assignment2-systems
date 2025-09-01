import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

        # Build a unique, deterministic list of trainable params by storage
        # (handles tied/shared weights cleanly).
        self._params = []
        seen_ptrs = set()
        for p in module.parameters():
            if not p.requires_grad:
                continue
            ptr = p.data_ptr()
            if ptr in seen_ptrs:
                continue
            seen_ptrs.add(ptr)
            self._params.append(p)

        # One-time broadcast of initial state from rank 0 (on device)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            with torch.no_grad():
                for p in self._params:
                    dist.broadcast(p.data, src=0)
                for b in module.buffers():
                    dist.broadcast(b.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        if not (dist.is_available() and dist.is_initialized()):
            return
        ws = dist.get_world_size()
        if ws == 1:
            return

        handles = []
        # Always issue the same number of collectives in the same order
        # (the unique list above makes this trivial even with tied weights).
        for p in self._params:
            if p.grad is None:
                # Participate to keep collective count identical across ranks
                tmp = torch.zeros_like(p.data)
                handles.append((None, dist.all_reduce(tmp, op=dist.ReduceOp.SUM, async_op=True)))
            else:
                handles.append((p, dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)))

        # Wait, then average exactly once per unique parameter
        for p, h in handles:
            h.wait()
            if p is not None and p.grad is not None:
                p.grad.div_(ws)
