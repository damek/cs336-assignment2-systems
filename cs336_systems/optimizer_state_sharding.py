import torch
import torch.distributed as dist

class OptimizerStateSharding(torch.optim.Optimizer):

    
    def __init__(self, params, optimizer_cls, **kwargs):
        self._opt_class = optimizer_cls
        self.optimizer = None
        self._defaults = kwargs.copy()
        self._all_params = []
        self._owner = {}
        self.counter = 0
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
        super().__init__(params, self._defaults.copy())

    def step(self, closure=None, **kwargs):
        with torch.no_grad():
            self.optimizer.step(closure, **kwargs)
            if self.world_size > 1 or not dist.is_initialized():
                for p in self._all_params: 
                    dist.broadcast(p.data, src=self._owner[p])
    
    def add_param_group(self, param_group: dict[str, any]):
        super().add_param_group(param_group)
        local_params = []
        for p in param_group["params"]:
            owner = self.counter % self.world_size
            self._owner[p] = owner
            self._all_params.append(p)
            if owner == self.rank:
                local_params.append(p)
            self.counter += 1
        if local_params:
            inner_group = {k : v for k, v in param_group.items() if k != "params"}
            inner_group["params"] = local_params
            if self.optimizer is None:
                self.optimizer = self._opt_class(local_params, **self._defaults)
            else: 
                self.optimizer.add_param_group(inner_group)
