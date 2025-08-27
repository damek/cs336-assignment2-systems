# Cowritten with gpt5.
import math, itertools, torch, triton, os
import triton.testing as tt
import cs336_basics.model as models
import pandas as pd
from cs336_systems.flashattention import FlashAttention

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
device = "cuda"

uid = getattr(os, "getuid", lambda: os.getpid())()
cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", f"/tmp/torchinductor_{uid}")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
os.environ.setdefault("USER", f"user{uid}")   # sidestep getpass.getuser()
os.environ.setdefault("HOME", "/tmp")
os.makedirs(cache_dir, exist_ok=True)

# --- Baseline PyTorch attention (forward) ---
def attn_pytorch_forward(Q, K, V, *, is_causal=True):
    # q,k,v: [1, N, D]
    attention = models.scaled_dot_product_attention
    # if compile:
        # attention = torch.compile(attention)  
    if is_causal:
        i = torch.arange(Q.shape[-2], device=Q.device)
        mask = i[None, :] > i[:, None]
        return attention(Q, K, V, mask=mask)
    else:
        return attention(Q, K, V, mask=None)

FA_Triton = FlashAttention
def fa_triton_forward(Q, K, V, *, is_causal=True):
    return FA_Triton.apply(Q, K, V,is_causal)

def make_inputs(N, D, dtype):
    Q = torch.randn(1, N, D, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(1, N, D, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(1, N, D, device=device, dtype=dtype, requires_grad=True)
    return Q, K, V

def bench_forward(forward_impl, N, D, dtype, is_causal=True):
    def run():
        with torch.no_grad():
            Q, K, V = make_inputs(N, D, dtype)
            forward_impl(Q, K, V, is_causal=is_causal)
    return tt.do_bench(run)

def bench_backward(forward_impl, N, D, dtype, is_causal=True):
    def run():
        Q, K, V = make_inputs(N, D, dtype)
        O = forward_impl(Q, K, V, is_causal=is_causal)
        dO = torch.randn_like(O)
        O.backward(dO, retain_graph=False)
    return tt.do_bench(run)

def bench_end2end(forward_impl, N, D, dtype, is_causal=True):
    def run():
        Q, K, V = make_inputs(N, D, dtype)
        O = forward_impl(Q, K, V, is_causal=is_causal)
        dO = torch.randn_like(O)
        O.backward(dO, retain_graph=False)
    return tt.do_bench(run)

Ns = [2**n for n in range(7, 17)]           # 128 .. 65536
Ds = [16, 32, 64, 128]
dtypes = [torch.bfloat16, torch.float32]
rows = []

def would_oom(N, dtype):
    bytes_per = 2 if dtype is torch.bfloat16 else 4
    return (3 * N * N * bytes_per) > (70 * 2**30)  

for fwd in (attn_pytorch_forward, fa_triton_forward):
    Q, K, V = make_inputs(128, 64, torch.float32)
    _ = fwd(Q, K, V, is_causal=True); _.sum().backward()

count = 0
for dtype in dtypes:
    for D in Ds:
        for N in Ns:
            count+=1
            print("N, D, dtype", N, D, dtype)
            print(f"Setting {count} of {len(Ns)*len(Ds)*len(dtypes)}")
            if would_oom(N, dtype):
                continue
            # PyTorch baseline
            pt_fwd = bench_forward(attn_pytorch_forward, N, D, dtype, is_causal=True)
            pt_bwd = bench_backward(attn_pytorch_forward, N, D, dtype, is_causal=True)
            pt_end = bench_end2end(attn_pytorch_forward, N, D, dtype, is_causal=True)
            # Triton FA-2
            fa_fwd = bench_forward(fa_triton_forward, N, D, dtype, is_causal=True)
            fa_bwd = bench_backward(fa_triton_forward, N, D, dtype, is_causal=True)
            fa_end = bench_end2end(fa_triton_forward, N, D, dtype, is_causal=True)

            rows.append(dict(
                N=N, D=D, dtype=str(dtype).split(".")[-1],
                pt_fwd_ms=pt_fwd, pt_bwd_ms=pt_bwd, pt_end_ms=pt_end,
                fa_fwd_ms=fa_fwd, fa_bwd_ms=fa_bwd, fa_end_ms=fa_end,
                fwd_speedup=pt_fwd / fa_fwd if fa_fwd else float("inf"),
                bwd_speedup=pt_bwd / fa_bwd if fa_bwd else float("inf"),
                end_speedup=pt_end / fa_end if fa_end else float("inf"),
            ))

df = pd.DataFrame(rows)
print(df.to_string(index=False))
