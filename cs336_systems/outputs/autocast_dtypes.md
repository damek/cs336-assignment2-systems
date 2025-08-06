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
- relu: torch.bfloat16
- ln: torch.float32
- fc2: torch.bfloat16
- logits.dtype: torch.bfloat16
- loss.dtype: torch.float32

Gradients:
- model.fc1.weight.grad.dtype: torch.float32
- model.ln.weight.grad.dtype: torch.float32
- model.fc2.weight.grad.dtype: torch.float32
- model.ln.bias.grad.dtype: torch.float32

## Layer norm and bfloat16

> You should have seen that FP16 mixed precision autocasting treats the layer normalization layer differently than the feed-forward layers. What parts of layer normalization are sensitive to mixed precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently? Why or why not?

Layer norm formula: 
$$
y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
$$
Sensitive reductions: Good to accumulate in fp32
- $\mathrm{E}[x]$
- $\mathrm{Var}[x]$

Range issues: 
- Computing square in variance. Will exceed the fp16 max $\pm 65k$ limit.
- Inverting a small standard deviation could blow up.

Although bf16 is not sensitive to range issues (since range is as high as fp32), it is sensitive to reductions. We should still accumulate in fp32.

