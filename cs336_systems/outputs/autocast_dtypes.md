# What does torch.autocast do?

uv run cs336_systems/autocast_dtypes.py 


dtype: torch.float16

Model weights: 
- model.fc1.weight.dtype: torch.float32
- model.ln.weight.dtype: torch.float32
- model.fc2.weight.dtype: torch.float32
- model.ln.bias.dtype: torch.float32

Intermediate outputs dtypes, e.g., fc1(x): 
- fc1: torch.float16
- relu: torch.float16
- ln: torch.float32
- fc2: torch.float16
- logits.dtype: torch.float16
- loss.dtype: torch.float32

Gradient dtypes: 
- model.fc1.weight.grad.dtype: torch.float32
- model.ln.weight.grad.dtype: torch.float32
- model.fc2.weight.grad.dtype: torch.float32
- model.ln.bias.grad.dtype: torch.float32