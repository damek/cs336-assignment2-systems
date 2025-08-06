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


model = ToyModel(in_features=10, out_features=10)
x = torch.randn(10, 10)
model.to("cuda")
x = x.to("cuda")

def print_dtype(dtype: torch.dtype):
    
    with torch.autocast(device_type="cuda", dtype=dtype):
        print(f"dtype: {dtype}")
        print("Model:")
        print(f"- model.fc1.weight.dtype: {model.fc1.weight.dtype}")
        print(f"- model.ln.weight.dtype: {model.ln.weight.dtype}")
        print(f"- model.fc2.weight.dtype: {model.fc2.weight.dtype}")
        print(f"- model.ln.bias.dtype: {model.ln.bias.dtype}")

        print("Forward Pass:")
        for n, m in model.named_modules():
            if len(list(m.children())) == 0:
                m.register_forward_hook(lambda module, input, output, name=n: print(f"- {name}: {output.dtype}")) 

        logits = model(x)
        print(f"- logits.dtype: {logits.dtype}")
        loss = cross_entropy(logits, torch.randint(0, 10, (10,)).to("cuda"))
        print(f"- loss.dtype: {loss.dtype}")

        loss.backward()
        print("Gradients:")
        print(f"- model.fc1.weight.grad.dtype: {model.fc1.weight.grad.dtype}")
        print(f"- model.ln.weight.grad.dtype: {model.ln.weight.grad.dtype}")
        print(f"- model.fc2.weight.grad.dtype: {model.fc2.weight.grad.dtype}")
        print(f"- model.ln.bias.grad.dtype: {model.ln.bias.grad.dtype}")

dtype : torch.dtype = torch.float16
print_dtype(dtype)
dtype : torch.dtype = torch.bfloat16
print_dtype(dtype)