import torch,os
import triton
import triton.testing
import math
import pandas as pd
from typing import List, Tuple
import itertools
import gc

from cs336_systems.flashattention import FlashAttention
uid = getattr(os, "getuid", lambda: os.getpid())()
cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", f"/tmp/torchinductor_{uid}")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
os.environ.setdefault("USER", f"user{uid}")   # sidestep getpass.getuser()
os.environ.setdefault("HOME", "/tmp")
os.makedirs(cache_dir, exist_ok=True)

@torch.compile
def pytorch_attention_forward(Q, K, V, is_causal=True):
    """Standard PyTorch attention implementation with proper gradient tracking"""
    d = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
    
    if is_causal:
        seq_len = Q.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out

def clear_gpu_memory():
    """Clear GPU memory between benchmarks"""
    gc.collect()
    torch.cuda.empty_cache()

def benchmark_attention(batch_size: int, seq_len: int, dim: int, dtype: torch.dtype, 
                       warmup: int = 25, rep: int = 100) -> dict:
    """Benchmark forward and backward passes for both implementations"""
    
    # Generate random inputs
    Q = torch.randn(batch_size, seq_len, dim, dtype=dtype, device='cuda', requires_grad=True)
    K = torch.randn(batch_size, seq_len, dim, dtype=dtype, device='cuda', requires_grad=True)
    V = torch.randn(batch_size, seq_len, dim, dtype=dtype, device='cuda', requires_grad=True)
    dO = torch.randn(batch_size, seq_len, dim, dtype=dtype, device='cuda')
    
    results = {}
    
    # ============ FORWARD PASS BENCHMARKS ============
    
    # Benchmark FlashAttention Forward
    def flash_fwd():
        Q_f = Q.clone().detach().requires_grad_(True)
        K_f = K.clone().detach().requires_grad_(True)
        V_f = V.clone().detach().requires_grad_(True)
        return FlashAttention.apply(Q_f, K_f, V_f, True)
    
    results['flash_fwd'] = triton.testing.do_bench(
        flash_fwd, warmup=warmup, rep=rep
    )
    
    # Benchmark PyTorch Forward
    def torch_fwd():
        Q_t = Q.clone().detach().requires_grad_(True)
        K_t = K.clone().detach().requires_grad_(True)
        V_t = V.clone().detach().requires_grad_(True)
        return pytorch_attention_forward(Q_t, K_t, V_t, True)
    
    results['torch_fwd'] = triton.testing.do_bench(
        torch_fwd, warmup=warmup, rep=rep
    )
    
    # ============ BACKWARD PASS BENCHMARKS ============
    
    # Benchmark FlashAttention Backward
    # First create the forward pass output that we'll use for backward
    Q_flash_bwd = Q.clone().detach().requires_grad_(True)
    K_flash_bwd = K.clone().detach().requires_grad_(True)
    V_flash_bwd = V.clone().detach().requires_grad_(True)
    O_flash = FlashAttention.apply(Q_flash_bwd, K_flash_bwd, V_flash_bwd, True)
    
    def flash_bwd():
        # Use retain_graph since we're calling backward multiple times
        if Q_flash_bwd.grad is not None:
            Q_flash_bwd.grad.zero_()
            K_flash_bwd.grad.zero_()
            V_flash_bwd.grad.zero_()
        O_flash.backward(dO, retain_graph=True)
    
    results['flash_bwd'] = triton.testing.do_bench(
        flash_bwd, warmup=warmup, rep=rep
    )
    
    # Benchmark PyTorch Backward
    Q_torch_bwd = Q.clone().detach().requires_grad_(True)
    K_torch_bwd = K.clone().detach().requires_grad_(True)
    V_torch_bwd = V.clone().detach().requires_grad_(True)
    O_torch = pytorch_attention_forward(Q_torch_bwd, K_torch_bwd, V_torch_bwd, True)
    
    def torch_bwd():
        # Use retain_graph since we're calling backward multiple times
        if Q_torch_bwd.grad is not None:
            Q_torch_bwd.grad.zero_()
            K_torch_bwd.grad.zero_()
            V_torch_bwd.grad.zero_()
        O_torch.backward(dO, retain_graph=True)
    
    results['torch_bwd'] = triton.testing.do_bench(
        torch_bwd, warmup=warmup, rep=rep
    )
    
    # ============ END-TO-END BENCHMARKS ============
    
    # Benchmark FlashAttention End-to-End
    def flash_e2e():
        Q_e2e = Q.clone().detach().requires_grad_(True)
        K_e2e = K.clone().detach().requires_grad_(True)
        V_e2e = V.clone().detach().requires_grad_(True)
        O = FlashAttention.apply(Q_e2e, K_e2e, V_e2e, True)
        O.backward(dO)
    
    results['flash_e2e'] = triton.testing.do_bench(
        flash_e2e, warmup=warmup, rep=rep
    )
    
    # Benchmark PyTorch End-to-End
    def torch_e2e():
        Q_e2e = Q.clone().detach().requires_grad_(True)
        K_e2e = K.clone().detach().requires_grad_(True)
        V_e2e = V.clone().detach().requires_grad_(True)
        O = pytorch_attention_forward(Q_e2e, K_e2e, V_e2e, True)
        O.backward(dO)
    
    results['torch_e2e'] = triton.testing.do_bench(
        torch_e2e, warmup=warmup, rep=rep
    )
    
    return results

def main():
    """Run benchmarks and create results table"""
    
    # Configuration
    batch_size = 1
    seq_lengths = [2**i for i in range(7, 17)]  # 128 to 65536
    dims = [2**i for i in range(4, 8)]  # 16 to 128
    dtypes = [torch.bfloat16, torch.float32]
    
    # Store results
    all_results = []
    
    print("Running benchmarks...")
    print("-" * 80)
    
    for seq_len, dim, dtype in itertools.product(seq_lengths, dims, dtypes):
        dtype_str = "bf16" if dtype == torch.bfloat16 else "fp32"
        
        try:
            print(f"Benchmarking: seq_len={seq_len}, dim={dim}, dtype={dtype_str}")
            
            results = benchmark_attention(batch_size, seq_len, dim, dtype)
            
            # Store results with configuration
            row = {
                'seq_len': seq_len,
                'dim': dim,
                'dtype': dtype_str,
                'flash_fwd_ms': results['flash_fwd'],
                'torch_fwd_ms': results['torch_fwd'],
                'flash_bwd_ms': results['flash_bwd'],
                'torch_bwd_ms': results['torch_bwd'],
                'flash_e2e_ms': results['flash_e2e'],
                'torch_e2e_ms': results['torch_e2e'],
                'fwd_speedup': results['torch_fwd'] / results['flash_fwd'],
                'bwd_speedup': results['torch_bwd'] / results['flash_bwd'],
                'e2e_speedup': results['torch_e2e'] / results['flash_e2e']
            }
            all_results.append(row)
            
        except Exception as e:
            print(f"  Skipped due to error: {e}")
            continue
    
    # Create DataFrame and save results
    df = pd.DataFrame(all_results)
    
    # Format for display
    df_display = df.round(3)
    
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print("\nSummary Statistics:")
    print("-" * 40)
    print(f"Mean Forward Speedup: {df['fwd_speedup'].mean():.2f}x")
    print(f"Mean Backward Speedup: {df['bwd_speedup'].mean():.2f}x")
    print(f"Mean End-to-End Speedup: {df['e2e_speedup'].mean():.2f}x")
    
    print("\n" + "=" * 80)
    print("DETAILED RESULTS TABLE")
    print("=" * 80)
    
    # Display table grouped by dtype
    for dtype in ['bf16', 'fp32']:
        print(f"\n--- {dtype.upper()} Results ---")
        dtype_df = df_display[df_display['dtype'] == dtype]
        
        # Create a more compact display
        display_cols = ['seq_len', 'dim', 'flash_fwd_ms', 'torch_fwd_ms', 'fwd_speedup',
                       'flash_bwd_ms', 'torch_bwd_ms', 'bwd_speedup',
                       'flash_e2e_ms', 'torch_e2e_ms', 'e2e_speedup']
        
        print(dtype_df[display_cols].to_string(index=False))
    
    # Save to CSV
    df.to_csv('flash_attention_benchmark_results.csv', index=False)
    print(f"\nResults saved to 'flash_attention_benchmark_results.csv'")
    
    # Create a summary pivot table for better readability
    print("\n" + "=" * 80)
    print("SPEEDUP SUMMARY (Flash vs PyTorch)")
    print("=" * 80)
    
    for dtype in ['bf16', 'fp32']:
        print(f"\n--- {dtype.upper()} End-to-End Speedup ---")
        dtype_df = df[df['dtype'] == dtype]
        pivot = dtype_df.pivot_table(values='e2e_speedup', index='seq_len', columns='dim')
        print(pivot.round(2).to_string())

if __name__ == "__main__":
    main()