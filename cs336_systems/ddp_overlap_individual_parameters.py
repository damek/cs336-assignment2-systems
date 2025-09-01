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
    tmp_bufs = []  # keep 64-bit buffers alive until after wait

    # Deterministic param order; one collective per param
    for p in self.module.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            # participate to keep collective counts identical across ranks
            g64 = torch.zeros_like(p.data, dtype=torch.float64, device=p.data.device)
            handles.append(dist.all_reduce(g64, op=dist.ReduceOp.SUM, async_op=True))
            tmp_bufs.append((None, g64))
        else:
            # do the reduction in float64, then copy back
            g64 = p.grad.detach().to(torch.float64)
            handles.append(dist.all_reduce(g64, op=dist.ReduceOp.SUM, async_op=True))
            tmp_bufs.append((p, g64))

    # wait, then average in high precision and copy back to p.grad
    for h in handles:
        h.wait()
    for p, g64 in tmp_bufs:
        if p is not None:
            p.grad.copy_((g64 / ws).to(p.grad.dtype))
