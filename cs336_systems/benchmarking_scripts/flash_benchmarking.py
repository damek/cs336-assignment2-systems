import torch,os
import triton
import triton.testing
import math
import pandas as pd
from typing import List, Tuple, Optional, Dict
import itertools
import gc
from cs336_systems.flashattention import FlashAttention

uid = getattr(os, "getuid", lambda: os.getpid())()
cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", f"/tmp/torchinductor_{uid}")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
os.environ.setdefault("USER", f"user{uid}")   # sidestep getpass.getuser()
os.environ.setdefault("HOME", "/tmp")
os.makedirs(cache_dir, exist_ok=True)

# Compile the PyTorch attention for fair comparison
@torch.compile
def pytorch_attention(Q, K, V, is_causal=True):
    d = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
    if is_causal:
        mask = torch.triu(torch.ones(scores.shape[-2:], device=Q.device), 1).bool()
        scores.masked_fill_(mask, float('-inf'))
    return torch.matmul(torch.softmax(scores, dim=-1), V)

def benchmark_config(batch_size, seq_len, dim, dtype):
    """Benchmark one configuration - returns partial results on failure"""
    warmup=200
    rep = 100
    result = {
        'flash_fwd': None,
        'torch_fwd': None,
        'flash_bwd': None, 
        'torch_bwd': None,
        'flash_e2e': None,
        'torch_e2e': None,
    }
    
    try:
        # Create inputs
        Q = torch.randn(batch_size, seq_len, dim, dtype=dtype, device='cuda')
        K = torch.randn(batch_size, seq_len, dim, dtype=dtype, device='cuda')
        V = torch.randn(batch_size, seq_len, dim, dtype=dtype, device='cuda')
        # dO = torch.randn(batch_size, seq_len, dim, dtype=dtype, device='cuda')
    except torch.cuda.OutOfMemoryError:
        print(f"    OOM during input creation")
        torch.cuda.empty_cache()
        return result
    
    # Try each benchmark independently
    try:
        result['flash_fwd'] = triton.testing.do_bench(lambda: FlashAttention.apply(Q, K, V, True), warmup=warmup,rep=rep)
    except torch.cuda.OutOfMemoryError:
        print(f"    FlashAttention forward OOM")
        torch.cuda.empty_cache()
    
    try:
        result['torch_fwd'] = triton.testing.do_bench(lambda: pytorch_attention(Q, K, V, True), warmup=warmup,rep=rep)
    except torch.cuda.OutOfMemoryError:
        print(f"    PyTorch forward OOM")
        torch.cuda.empty_cache()
    
    # E2E benchmarks
    try:
        Qe = Q.clone().requires_grad_(True)
        Ke = K.clone().requires_grad_(True)
        Ve = V.clone().requires_grad_(True)
        def flash_e2e():
            Qe.grad = None
            Ke.grad = None
            Ve.grad = None
            O = FlashAttention.apply(Qe, Ke, Ve, True)
            O.sum().backward()
        result['flash_e2e'] = triton.testing.do_bench(flash_e2e, warmup=warmup,rep=rep)
    except torch.cuda.OutOfMemoryError:
        print(f"    FlashAttention e2e OOM")
        torch.cuda.empty_cache()
    
    try:
        Qe = Q.clone().requires_grad_(True)
        Ke = K.clone().requires_grad_(True)
        Ve = V.clone().requires_grad_(True)
        def torch_e2e():
            Qe.grad = None
            Ke.grad = None
            Ve.grad = None
            O = pytorch_attention(Qe, Ke, Ve, True)
            O.sum().backward()
        result['torch_e2e'] = triton.testing.do_bench(torch_e2e, warmup=warmup,rep=rep)
    except torch.cuda.OutOfMemoryError:
        print(f"    PyTorch e2e OOM")
        torch.cuda.empty_cache()
    
    # Calculate backward times if we have the data
    if result['flash_e2e'] is not None and result['flash_fwd'] is not None:
        result['flash_bwd'] = result['flash_e2e'] - result['flash_fwd']
    if result['torch_e2e'] is not None and result['torch_fwd'] is not None:
        result['torch_bwd'] = result['torch_e2e'] - result['torch_fwd']
    
    # Clear memory before returning
    torch.cuda.empty_cache()
    return result

# Run benchmarks
results = []
batch_size = 1
seq_lengths = [2**i for i in range(7, 17)]  # 128 to 65536
dims = [2**i for i in range(4, 8)]  # 16 to 128
dtypes = [torch.bfloat16, torch.float32]

print("Running benchmarks...")
for seq_len in seq_lengths:
    for dim in dims:
        for dtype in dtypes:
            dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"
            print(f"seq_len={seq_len}, dim={dim}, dtype={dtype_str}")
            
            timings = benchmark_config(batch_size, seq_len, dim, dtype)
            
            # Only add to results if we got at least some data
            if any(v is not None for v in timings.values()):
                # Calculate speedups where possible
                fwd_speedup = None
                bwd_speedup = None
                e2e_speedup = None
                
                if timings['flash_fwd'] and timings['torch_fwd']:
                    fwd_speedup = timings['torch_fwd'] / timings['flash_fwd']
                if timings['flash_bwd'] and timings['torch_bwd']:
                    bwd_speedup = timings['torch_bwd'] / timings['flash_bwd']
                if timings['flash_e2e'] and timings['torch_e2e']:
                    e2e_speedup = timings['torch_e2e'] / timings['flash_e2e']
                
                results.append({
                    'seq_len': seq_len,
                    'dim': dim,
                    'dtype': dtype_str,
                    'flash_fwd_ms': timings['flash_fwd'] or 'OOM',
                    'torch_fwd_ms': timings['torch_fwd'] or 'OOM',
                    'flash_bwd_ms': timings['flash_bwd'] or 'OOM',
                    'torch_bwd_ms': timings['torch_bwd'] or 'OOM',
                    'flash_e2e_ms': timings['flash_e2e'] or 'OOM',
                    'torch_e2e_ms': timings['torch_e2e'] or 'OOM',
                    'fwd_speedup': fwd_speedup,
                    'bwd_speedup': bwd_speedup,
                    'e2e_speedup': e2e_speedup,
                })

# Create DataFrame and display results
df = pd.DataFrame(results)
print("\n" + "="*60)
print("RESULTS")
print("="*60)

# Summary stats (only for valid numeric values)
numeric_df = df.copy()
for col in ['fwd_speedup', 'bwd_speedup', 'e2e_speedup']:
    numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

print(f"\nMean Forward Speedup: {numeric_df['fwd_speedup'].mean():.2f}x")
print(f"\nMean Backward Speedup: {numeric_df['bwd_speedup'].mean():.2f}x")
print(f"\nMean End-to-End Speedup: {numeric_df['e2e_speedup'].mean():.2f}x")

# Save full results
df.to_csv('../outputs/csv/flash_benchmark_results.csv', index=False)

# Show pivot tables (handling OOM cases)
for dtype in ['bf16', 'fp32']:
    print(f"\n{dtype.upper()} End-to-End Speedup:")
    dtype_df = df[df['dtype']==dtype].copy()
    dtype_df['e2e_speedup_numeric'] = pd.to_numeric(dtype_df['e2e_speedup'], errors='coerce')
    pivot = dtype_df.pivot(index='seq_len', columns='dim', values='e2e_speedup_numeric')
    print(pivot.round(2).fillna('OOM'))