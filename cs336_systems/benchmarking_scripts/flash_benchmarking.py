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
    """Benchmark one configuration"""
    # Create inputs
    Q = torch.randn(batch_size, seq_len, dim, dtype=dtype, device='cuda')
    K = torch.randn(batch_size, seq_len, dim, dtype=dtype, device='cuda')
    V = torch.randn(batch_size, seq_len, dim, dtype=dtype, device='cuda')
    dO = torch.randn(batch_size, seq_len, dim, dtype=dtype, device='cuda')
    
    # Forward benchmarks
    flash_fwd = triton.testing.do_bench(lambda: FlashAttention.apply(Q, K, V, True))
    torch_fwd = triton.testing.do_bench(lambda: pytorch_attention(Q, K, V, True))
    
    # End-to-end benchmarks (fresh tensors each time to avoid retain_graph issues)
    def flash_e2e():
        Qe = Q.clone().requires_grad_(True)
        Ke = K.clone().requires_grad_(True)
        Ve = V.clone().requires_grad_(True)
        O = FlashAttention.apply(Qe, Ke, Ve, True)
        O.backward(dO)
        
    def torch_e2e():
        Qe = Q.clone().requires_grad_(True)
        Ke = K.clone().requires_grad_(True)
        Ve = V.clone().requires_grad_(True)
        O = pytorch_attention(Qe, Ke, Ve, True)
        O.backward(dO)
    
    flash_e2e_time = triton.testing.do_bench(flash_e2e)
    torch_e2e_time = triton.testing.do_bench(torch_e2e)
    
    # Calculate backward time (e2e - forward)
    flash_bwd = flash_e2e_time - flash_fwd
    torch_bwd = torch_e2e_time - torch_fwd
    
    return {
        'flash_fwd': flash_fwd,
        'torch_fwd': torch_fwd,
        'flash_bwd': flash_bwd,
        'torch_bwd': torch_bwd,
        'flash_e2e': flash_e2e_time,
        'torch_e2e': torch_e2e_time,
    }

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
            
            try:
                timings = benchmark_config(batch_size, seq_len, dim, dtype)
                results.append({
                    'seq_len': seq_len,
                    'dim': dim,
                    'dtype': dtype_str,
                    'flash_fwd_ms': timings['flash_fwd'],
                    'torch_fwd_ms': timings['torch_fwd'],
                    'flash_bwd_ms': timings['flash_bwd'],
                    'torch_bwd_ms': timings['torch_bwd'],
                    'flash_e2e_ms': timings['flash_e2e'],
                    'torch_e2e_ms': timings['torch_e2e'],
                    'fwd_speedup': timings['torch_fwd'] / timings['flash_fwd'],
                    'bwd_speedup': timings['torch_bwd'] / timings['flash_bwd'],
                    'e2e_speedup': timings['torch_e2e'] / timings['flash_e2e'],
                })
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM - skipping")
                torch.cuda.empty_cache()
                continue

# Create DataFrame and display results
df = pd.DataFrame(results)
print("\n" + "="*60)
print("RESULTS")
print("="*60)

# Summary stats
print(f"\nMean Forward Speedup: {df['fwd_speedup'].mean():.2f}x")
print(f"Mean Backward Speedup: {df['bwd_speedup'].mean():.2f}x")
print(f"Mean End-to-End Speedup: {df['e2e_speedup'].mean():.2f}x")

# Save to CSV
df.to_csv('flash_benchmark_results.csv', index=False)

# Show pivot table
for dtype in ['bf16', 'fp32']:
    print(f"\n{dtype.upper()} End-to-End Speedup:")
    pivot = df[df['dtype']==dtype].pivot(index='seq_len', columns='dim', values='e2e_speedup')
    print(pivot.round(2))