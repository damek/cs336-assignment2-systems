import torch
import torch.distributed as dist

class DDPOverlapIndividualParameters(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

        # --- Robust init sync: broadcast params then buffers, always in same order ---
        if dist.is_available() and dist.is_initialized():
            with torch.no_grad():
                # parameters first
                for p in self.module.parameters():
                    dist.broadcast(p.data, src=0)
                # then buffers (e.g., running stats)
                for b in self.module.buffers():
                    dist.broadcast(b.data, src=0)
            # make sure all ranks are past init sync before comparisons
            dist.barrier()

        # Keep your hooks if you want, but don’t launch comms from them
        # (reductions happen once in finish_gradient_synchronization)
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(lambda _: None)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        if not (dist.is_available() and dist.is_initialized()):
            return
        ws = dist.get_world_size()
        if ws == 1:
            return

        handles = []
        # Deterministic, identical sequence across ranks.
        # NOTE: Module.parameters() yields each Parameter once (tied weights handled).
        for p in self.module.parameters():
            if p.grad is None:
                # keep collective counts identical even if some rank has no grad
                tmp = torch.zeros_like(p.data)
                h = dist.all_reduce(tmp, op=dist.ReduceOp.SUM, async_op=True)
                handles.append((None, h))
            else:
                # reduce the actual buffer the optimizer will read
                h = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
                handles.append((p, h))

        # Wait for comms, then average exactly once per parameter
        for p, h in handles:
            h.wait()
        for p, _ in handles:
            if p is not None and p.grad is not None:
                p.grad.div_(ws)
