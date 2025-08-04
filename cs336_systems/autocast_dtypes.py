import torch
import torch.nn as nn
from cs336_basics.nn_utils import cross_entropy

class ToyModel(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x

model = ToyModel(10, 10)
x = torch.randn(10, 10)
model.to("cuda")
x = x.to("cuda")


dtype : torch.dtype = torch.float16

with torch.autocast(device_type="cuda", dtype=dtype):
    print(f"dtype: {dtype}")
    print(f"model.fc1.weight.dtype: {model.fc1.weight.dtype}")
    print(f"model.ln.weight.dtype: {model.ln.weight.dtype}")
    print(f"model.fc2.weight.dtype: {model.fc2.weight.dtype}")
    print(f"model.ln.bias.dtype: {model.ln.bias.dtype}")

    logits = model(x)
    print(f"logits.dtype: {logits.dtype}")
    for n, m in model.named_modules():
        if len(list(m.children())) == 0:
            m.register_forward_hook(lambda module, input, output, name=n: print(f"{name}: {output.dtype}")) 

    loss = cross_entropy(logits, torch.randint(0, 10, (10,)))
    print(f"loss.dtype: {loss.dtype}")

    loss.backward()
    print(f"model.fc1.weight.grad.dtype: {model.fc1.weight.grad.dtype}")
    print(f"model.ln.weight.grad.dtype: {model.ln.weight.grad.dtype}")
    print(f"model.fc2.weight.grad.dtype: {model.fc2.weight.grad.dtype}")
    print(f"model.ln.bias.grad.dtype: {model.ln.bias.grad.dtype}")

    
