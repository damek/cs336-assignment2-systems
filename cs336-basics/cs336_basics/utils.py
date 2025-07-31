from contextlib import nullcontext # going to use this for making a range for profiling.
import torch

def maybe_range(tag: str, enabled: bool = False):
    """
    Will return an empty context when set to false. 
    Otherwise, will return an nvtx annotation that we'll use in the nsys.
    """
    return torch.cuda.nvtx.range(tag) if enabled else nullcontext()