# What does torch.autocast do?

uv run cs336_systems/autocast_dtypes.py 

## dtype: torch.float16

Model:
- model.fc1.weight.dtype: torch.float32
- model.ln.weight.dtype: torch.float32
- model.fc2.weight.dtype: torch.float32
- model.ln.bias.dtype: torch.float32

Forward Pass:
- fc1: torch.float16
- relu: torch.float16
- ln: torch.float32
- fc2: torch.float16
- logits.dtype: torch.float16
- loss.dtype: torch.float32

Gradients:
- model.fc1.weight.grad.dtype: torch.float32
- model.ln.weight.grad.dtype: torch.float32
- model.fc2.weight.grad.dtype: torch.float32
- model.ln.bias.grad.dtype: torch.float32

## dtype: torch.bfloat16

Model:
- model.fc1.weight.dtype: torch.float32
- model.ln.weight.dtype: torch.float32
- model.fc2.weight.dtype: torch.float32
- model.ln.bias.dtype: torch.float32

Forward Pass:
- fc1: torch.bfloat16
- fc1: torch.bfloat16
- relu: torch.bfloat16
- relu: torch.bfloat16
- ln: torch.float32
- ln: torch.float32
- fc2: torch.bfloat16
- fc2: torch.bfloat16
- logits.dtype: torch.bfloat16
- loss.dtype: torch.float32

Gradients:
- model.fc1.weight.grad.dtype: torch.float32
- model.ln.weight.grad.dtype: torch.float32
- model.fc2.weight.grad.dtype: torch.float32
- model.ln.bias.grad.dtype: torch.float32