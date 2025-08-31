import torch
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)

class DDPOverlapIndividualParameters(torch.nn.Module):
    """
    Debug version with logging to understand what's happening.
    """
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.step_count = 0
        
        # Broadcast parameters and buffers from rank 0 to all other ranks
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
            logger.info(f"Rank {rank}: Broadcasting parameters from rank 0")
            
            with torch.no_grad():
                for name, p in module.named_parameters():
                    before = p.data.clone()
                    dist.broadcast(p.data, src=0)
                    after = p.data.clone()
                    changed = not torch.allclose(before, after)
                    logger.debug(f"Rank {rank}: Param {name} changed after broadcast: {changed}")
                    
                for name, b in module.named_buffers():
                    dist.broadcast(b, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """Synchronize all gradients after backward pass."""
        if not dist.is_initialized():
            return
            
        world_size = dist.get_world_size()
        if world_size <= 1:
            return
        
        rank = dist.get_rank()
        self.step_count += 1
        logger.info(f"Rank {rank}: Starting gradient sync for step {self.step_count}")
        
        # Synchronize gradients for all parameters that have them
        with torch.no_grad():
            for name, p in self.module.named_parameters():
                if p.grad is not None:
                    # Log gradient stats before sync
                    grad_mean_before = p.grad.data.mean().item()
                    grad_norm_before = p.grad.data.norm().item()
                    
                    # Synchronously all-reduce (sum) the gradients
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    
                    # Average by dividing by world size
                    p.grad.data.div_(world_size)
                    
                    # Log gradient stats after sync
                    grad_mean_after = p.grad.data.mean().item()
                    grad_norm_after = p.grad.data.norm().item()
                    
                    logger.debug(f"Rank {rank}, Step {self.step_count}, Param {name}: "
                               f"grad mean {grad_mean_before:.6f} -> {grad_mean_after:.6f}, "
                               f"norm {grad_norm_before:.6f} -> {grad_norm_after:.6f}")
                else:
                    logger.debug(f"Rank {rank}, Step {self.step_count}, Param {name}: No gradient")
        
        logger.info(f"Rank {rank}: Completed gradient sync for step {self.step_count}")
    
    # Make the wrapper completely transparent
    def __getattr__(self, name):
        return getattr(self.module, name)