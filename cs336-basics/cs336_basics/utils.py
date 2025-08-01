from contextlib import nullcontext, contextmanager # going to use this for making a range for profiling.
import torch

@contextmanager
def maybe_range(tag: str, enabled: bool = False):
    """
    Will return an empty context when set to false. 
    Otherwise, will return an nvtx annotation that we'll use in the nsys.
    Also adds start/end markers for better CSV export compatibility.
    """
    if not enabled:
        yield
        return
    
    # Add start marker
    torch.cuda.nvtx.mark(f"START:{tag}")
    
    # Create the range
    with torch.cuda.nvtx.range(tag) as nvtx_range:
        try:
            yield nvtx_range
        finally:
            # Add end marker
            torch.cuda.nvtx.mark(f"END:{tag}")