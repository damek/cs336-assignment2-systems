import torch
import torch.distributed as dist
from typing import Iterator, Tuple, Optional

class DDPOverlapIndividualParameters(torch.nn.Module):
    """
    Fully transparent DDP wrapper with maximum compatibility.
    Handles tied weights and provides complete transparency to the underlying module.
    """
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        # Store module as a submodule (not just an attribute)
        # This ensures it's properly registered with PyTorch
        self.module = module
        self._pending = []
        self._param_to_hook_handles = {}
        
        # Broadcast parameters and buffers from rank 0 to all other ranks
        if dist.is_initialized() and dist.get_world_size() > 1:
            with torch.no_grad():
                # Track unique tensors to handle tied weights
                seen_tensors = set()
                
                # Broadcast all unique parameters
                for p in module.parameters():
                    tensor_id = id(p.data)
                    if tensor_id not in seen_tensors:
                        dist.broadcast(p.data, src=0)
                        seen_tensors.add(tensor_id)
                
                # Broadcast all unique buffers
                for b in module.buffers():
                    tensor_id = id(b)
                    if tensor_id not in seen_tensors:
                        dist.broadcast(b, src=0)
                        seen_tensors.add(tensor_id)

            # Register hooks for gradient synchronization
            # Track which parameters we've already hooked (for tied weights)
            hooked_params = set()
            for p in module.parameters():
                param_id = id(p)
                if p.requires_grad and param_id not in hooked_params:
                    if not p.is_leaf:
                        raise RuntimeError(f"Parameter is not a leaf tensor")
                    handle = p.register_post_accumulate_grad_hook(self._make_hook(param_id))
                    self._param_to_hook_handles[param_id] = handle
                    hooked_params.add(param_id)

    def _make_hook(self, param_id):
        """Create a hook function for a specific parameter."""
        def hook(param):
            if not dist.is_initialized() or dist.get_world_size() == 1:
                return
            
            if param.grad is None:
                return
            
            # Start async all_reduce
            work = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._pending.append((param, work))
        
        return hook

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """Wait for all gradient synchronizations to complete."""
        if not dist.is_initialized():
            return
            
        world_size = dist.get_world_size()
        if world_size == 1:
            return
        
        # Wait for all async operations and average gradients
        for param, work in self._pending:
            work.wait()
            if param.grad is not None:
                param.grad.data.div_(world_size)
        
        # Clear pending list for next iteration
        self._pending.clear()
    
    # Explicitly forward all the important methods to ensure full compatibility
    
    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]:
        return self.module.parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True, 
                        remove_duplicate: bool = True) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        return self.module.named_parameters(prefix=prefix, recurse=recurse, 
                                           remove_duplicate=remove_duplicate)
    
    def buffers(self, recurse: bool = True) -> Iterator[torch.Tensor]:
        return self.module.buffers(recurse=recurse)
    
    def named_buffers(self, prefix: str = '', recurse: bool = True,
                     remove_duplicate: bool = True) -> Iterator[Tuple[str, torch.Tensor]]:
        return self.module.named_buffers(prefix=prefix, recurse=recurse,
                                        remove_duplicate=remove_duplicate)
    
    def children(self) -> Iterator[torch.nn.Module]:
        # Only return the wrapped module
        yield self.module
    
    def named_children(self) -> Iterator[Tuple[str, torch.nn.Module]]:
        yield ('module', self.module)
    
    def modules(self) -> Iterator[torch.nn.Module]:
        return self.module.modules()
    
    def named_modules(self, memo: Optional[set] = None, prefix: str = '',
                     remove_duplicate: bool = True) -> Iterator[Tuple[str, torch.nn.Module]]:
        return self.module.named_modules(memo=memo, prefix=prefix,
                                        remove_duplicate=remove_duplicate)
    
    def train(self, mode: bool = True):
        self.module.train(mode)
        return super().train(mode)
    
    def eval(self):
        self.module.eval()
        return super().eval()
    
    def zero_grad(self, set_to_none: bool = True):
        return self.module.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self, *args, **kwargs):
        # Return the underlying module's state dict
        return self.module.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, strict: bool = True):
        return self.module.load_state_dict(state_dict, strict=strict)
    
    def to(self, *args, **kwargs):
        # Move both the wrapper and the module
        super().to(*args, **kwargs)
        self.module.to(*args, **kwargs)
        return self
    
    def cuda(self, device: Optional[int] = None):
        super().cuda(device)
        self.module.cuda(device)
        return self
    
    def cpu(self):
        super().cpu()
        self.module.cpu()
        return self
    
    def __repr__(self):
        return f"DDPOverlapIndividualParameters({self.module})"
    
    def __getattr__(self, name):
        # Fallback for any other attributes
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)