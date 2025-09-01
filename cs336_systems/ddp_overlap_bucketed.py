import torch
import torch.distributed as dist

class Bucket:
    def __init__(self, param_list: list[torch.Tensor]):
        self.param_list = param_list
        self.grad_list = [p.grad for p in param_list]
        self.work_list = [None for _ in param_list]

class DDPOverlapBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self._pending = []
        self.bucket_size_mb = bucket_size_mb
        self.segments = [] # list of segments
        self.param_to_segment = {} # {param: segment_idx}
        self.global_flat = None

        with torch.no_grad():
            for t in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(t.data, src=0)
            print("finished broadcasting")
            total_numel = 0
            device = None
            for p in module.parameters():
                if p.requires_grad:
                    if not p.is_leaf:
                        raise RuntimeError("Parameter is not a leaf tensor")
                    total_numel += p.numel()
                    device = p.device # assuming entire model is one same device.
            print("total_numel, device", total_numel, device)
            self.global_flat = torch.zeros(total_numel, dtype=torch.float32, device=device) 
            print("finished creating global flat")
            # initialize our first segment
            self.segments.append({"params": [], "start": 0, "length": 0, "ready": 0, "handle": None})
            for p in reversed(list(module.parameters())):
                print("building segments")
                if p.requires_grad:
                    if not p.is_leaf:
                        raise RuntimeError("Parameter is not a leaf tensor")
                    
                    # Case 1: we can just add to our current segment 
                    if p.numel()*4/1024**2 > self.bucket_size_mb:
                        raise RuntimeError("Parameter is too large to fit in a single bucket")
                    
                    if (self.segments[-1]["length"] == 0) or ((self.segments[-1]["length"] + p.numel())*4/1024**2 <= self.bucket_size_mb):
                        self.segments[-1]["params"].append(p)
                        self.segments[-1]["length"] += p.numel()
                    else:
                         self.segments.append({"params": [p], "start": self.segments[-1]["start"] + self.segments[-1]["length"], "length": p.numel(), "ready": 0, "handle": None, "view": None})

                    self.param_to_segment[p] = len(self.segments) - 1

            for segment in self.segments:
                segment["view"] = self.global_flat.narrow(0, segment["start"], segment["length"])


            print("finished building segments")
            for p in module.parameters():
                if p.requires_grad:
                    if not p.is_leaf:
                        raise RuntimeError("Parameter is not a leaf tensor")
                    p.register_post_accumulate_grad_hook(self._hook)
            
    def _hook(self, param):   
        if not (dist.is_available() and dist.is_initialized()):
            return
        if param.grad is None:
            print(f"param.grad is None for {param.name}")
            return
        segment_idx = self.param_to_segment[param]
        segment = self.segments[segment_idx]
        segment["ready"] += 1
        if segment["ready"] == len(segment["params"]):
            offset = 0
            for p in segment["params"]:
                size = p.numel()
                segment["view"][offset:offset + size] = p.grad.view(-1)
                offset += size
            # send the bucket
            segment["handle"] = dist.all_reduce(segment["view"], op=dist.ReduceOp.SUM, async_op=True)
        self._pending.append(segment)
        return None

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        ws = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        ws = torch.as_tensor(ws, dtype=torch.float64)

        for segment in self._pending:
            segment.handle.wait()
            if ws > 1:
                segment.view.div_(ws)
            offset = 0
            for p in segment.params:
                size = p.numel()
                p.grad.view(-1).copy_(segment.view[offset:offset + size])
                offset += size
            segment.ready = 0
            segment.handle = None

        self._pending.clear()
