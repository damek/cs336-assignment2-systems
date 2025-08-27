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
                       warmup: int = 25, rep: int = 100) -> Dict[str, Optional[float]]:
    """Benchmark forward and backward passes for both implementations"""
    
    results = {
        'flash_fwd': None,
        'torch_fwd': None,
        'flash_bwd': None,
        'torch_bwd': None,
        'flash_e2e': None,
        'torch_e2e': None,
        'flash_fwd_oom': False,
        'torch_fwd_oom': False,
        'flash_bwd_oom': False,
        'torch_bwd_oom': False,
        'flash_e2e_oom': False,
        'torch_e2e_oom': False,
    }
    
    # Clear memory before starting
    clear_gpu_memory()
    
    # Generate random inputs
    try:
        Q = torch.randn(batch_size, seq_len, dim, dtype=dtype, device='cuda', requires_grad=True)
        K = torch.randn(batch_size, seq_len, dim, dtype=dtype, device='cuda', requires_grad=True)
        V = torch.randn(batch_size, seq_len, dim, dtype=dtype, device='cuda', requires_grad=True)
        dO = torch.randn(batch_size, seq_len, dim, dtype=dtype, device='cuda')
    except torch.cuda.OutOfMemoryError:
        print("  OOM during input generation")
        return results
    
    # ============ FORWARD PASS BENCHMARKS ============
    
    # Benchmark FlashAttention Forward
    try:
        def flash_fwd():
            Q_f = Q.clone().detach().requires_grad_(True)
            K_f = K.clone().detach().requires_grad_(True)
            V_f = V.clone().detach().requires_grad_(True)
            return FlashAttention.apply(Q_f, K_f, V_f, True)
        
        results['flash_fwd'] = triton.testing.do_bench(
            flash_fwd, warmup=warmup, rep=rep
        )
    except torch.cuda.OutOfMemoryError:
        results['flash_fwd_oom'] = True
        print("  FlashAttention Forward: OOM")
        clear_gpu_memory()
    
    # Benchmark PyTorch Forward
    try:
        def torch_fwd():
            Q_t = Q.clone().detach().requires_grad_(True)
            K_t = K.clone().detach().requires_grad_(True)
            V_t = V.clone().detach().requires_grad_(True)
            return pytorch_attention_forward(Q_t, K_t, V_t, True)
        
        results['torch_fwd'] = triton.testing.do_bench(
            torch_fwd, warmup=warmup, rep=rep
        )
    except torch.cuda.OutOfMemoryError:
        results['torch_fwd_oom'] = True
        print("  PyTorch Forward: OOM")
        clear_gpu_memory()
    
    # ============ BACKWARD PASS BENCHMARKS ============
    
    # Benchmark FlashAttention Backward
    try:
        Q_flash_bwd = Q.clone().detach().requires_grad_(True)
        K_flash_bwd = K.clone().detach().requires_grad_(True)
        V_flash_bwd = V.clone().detach().requires_grad_(True)
        O_flash = FlashAttention.apply(Q_flash_bwd, K_flash_bwd, V_flash_bwd, True)
        
        def flash_bwd():
            if Q_flash_bwd.grad is not None:
                Q_flash_bwd.grad.zero_()
                K_flash_bwd.grad.zero_()
                V_flash_bwd.grad.zero_()
            O_flash.backward(dO, retain_graph=True)
        
        results['flash_bwd'] = triton.testing.do_bench(
            flash_bwd, warmup=warmup, rep=rep
        )
    except torch.cuda.OutOfMemoryError:
        results['flash_bwd_oom'] = True
        print("  FlashAttention Backward: OOM")
        clear_gpu_memory()
    
    # Benchmark PyTorch Backward
    try:
        Q_torch_bwd = Q.clone().detach().requires_grad_(True)
        K_torch_bwd = K.clone().detach().requires_grad_(True)
        V_torch_bwd = V.clone().detach().requires_grad_(True)
        O_torch = pytorch_attention_forward(Q_torch_bwd, K_torch_bwd, V_torch_bwd, True)
        
        def torch_bwd():
            if Q_torch_bwd.grad is not None:
                Q_torch_bwd.grad.zero_()
                K_torch_bwd.grad.zero_()
                V_torch_bwd.grad.zero_()
            O_torch.backward(dO, retain_graph=True)
        
        results['torch_bwd'] = triton.testing.do_bench(
            torch_bwd, warmup=warmup, rep=rep
        )
    except torch.cuda.OutOfMemoryError:
        results['torch_bwd_oom'] = True
        print("  PyTorch Backward: OOM")
        clear_gpu_memory()
    
    # ============ END-TO-END BENCHMARKS ============
    
    # Benchmark FlashAttention End-to-End
    try:
        def flash_e2e():
            Q_e2e = Q.clone().detach().requires_grad_(True)
            K_e2e = K.clone().detach().requires_grad_(True)
            V_e2e = V.clone().detach().requires_grad_(True)
            O = FlashAttention.apply(Q_e2e, K_e2e, V_e2e, True)
            O.backward(dO)
        
        results['flash_e2e'] = triton.testing.do_bench(
            flash_e2e, warmup=warmup, rep=rep
        )
    except torch.cuda.OutOfMemoryError:
        results['flash_e2e_oom'] = True
        print("  FlashAttention End-to-End: OOM")
        clear_gpu_memory()
    
    # Benchmark PyTorch End-to-End
    try:
        def torch_e2e():
            Q_e2e = Q.clone().detach().requires_grad_(True)
            K_e2e = K.clone().detach().requires_grad_(True)
            V_e2e = V.clone().detach().requires_grad_(True)
            O = pytorch_attention_forward(Q_e2e, K_e2e, V_e2e, True)
            O.backward(dO)
        
        results['torch_e2e'] = triton.testing.do_bench(
            torch_e2e, warmup=warmup, rep=rep
        )
    except torch.cuda.OutOfMemoryError:
        results['torch_e2e_oom'] = True
        print("  PyTorch End-to-End: OOM")
        clear_gpu_memory()
    
    # Clear memory after benchmarks
    clear_gpu_memory()
    
    return results

def format_result(value, oom_flag):
    """Format result with OOM indicator"""
    if oom_flag:
        return "OOM"
    elif value is None:
        return "-"
    else:
        return f"{value:.3f}"

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
        
        print(f"Benchmarking: seq_len={seq_len}, dim={dim}, dtype={dtype_str}")
        
        try:
            results = benchmark_attention(batch_size, seq_len, dim, dtype)
            
            # Calculate speedups only if both values exist
            fwd_speedup = None
            bwd_speedup = None
            e2e_speedup = None
            
            if results['flash_fwd'] is not None and results['torch_fwd'] is not None:
                fwd_speedup = results['torch_fwd'] / results['flash_fwd']
            if results['flash_bwd'] is not None and results['torch_bwd'] is not None:
                bwd_speedup = results['torch_bwd'] / results['flash_bwd']
            if results['flash_e2e'] is not None and results['torch_e2e'] is not None:
                e2e_speedup = results['torch_e2e'] / results['flash_e2e']
            
            # Store results with configuration
            row = {
                'seq_len': seq_len,
                'dim': dim,
                'dtype': dtype_str,
                'flash_fwd_ms': format_result(results['flash_fwd'], results['flash_fwd_oom']),
                'torch_fwd_ms': format_result(results['torch_fwd'], results['torch_fwd_oom']),
                'flash_bwd_ms': format_result(results['flash_bwd'], results['flash_bwd_oom']),
                'torch_bwd_ms': format_result(results['torch_bwd'], results['torch_bwd_oom']),
                'flash_e2e_ms': format_result(results['flash_e2e'], results['flash_e2e_oom']),
                'torch_e2e_ms': format_result(results['torch_e2e'], results['torch_e2e_oom']),
                'fwd_speedup': f"{fwd_speedup:.3f}" if fwd_speedup else "-",
                'bwd_speedup': f"{bwd_speedup:.3f}" if bwd_speedup else "-",
                'e2e_speedup': f"{e2e_speedup:.3f}" if e2e_speedup else "-"
            }
            all_results.append(row)
            
        except Exception as e:
            print(f"  Unexpected error: {e}")
            continue
    
    # Create DataFrame and save results
    df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    # Count OOMs
    flash_ooms = sum(1 for r in all_results if 'OOM' in str(r.get('flash_e2e_ms', '')))
    torch_ooms = sum(1 for r in all_results if 'OOM' in str(r.get('torch_e2e_ms', '')))
    
    print("\nOOM Statistics:")
    print("-" * 40)
    print(f"FlashAttention OOMs: {flash_ooms}")
    print(f"PyTorch OOMs: {torch_ooms}")
    
    # Calculate mean speedups (excluding OOM cases)
    valid_speedups = [r for r in all_results if r['e2e_speedup'] != '-']
    if valid_speedups:
        mean_fwd = sum(float(r['fwd_speedup']) for r in valid_speedups if r['fwd_speedup'] != '-') / len([r for r in valid_speedups if r['fwd_speedup'] != '-'])
        mean_bwd = sum(float(r['bwd_speedup']) for r in valid_speedups if r['bwd_speedup'] != '-') / len([r for r in valid_speedups if r['bwd_speedup'] != '-'])
        mean_e2e = sum(float(r['e2e_speedup']) for r in valid_speedups if r['e2e_speedup'] != '-') / len([r for r in valid_speedups if r['e2e_speedup'] != '-'])
        
        print("\nMean Speedups (excluding OOM):")
        print("-" * 40)
        print(f"Mean Forward Speedup: {mean_fwd:.2f}x")
        print(f"Mean Backward Speedup: {mean_bwd:.2f}x")
        print(f"Mean End-to-End Speedup: {mean_e2e:.2f}x")
    
    print("\n" + "=" * 80)
    print("DETAILED RESULTS TABLE")
    print("=" * 80)
    
    # Display table grouped by dtype
    for dtype in ['bf16', 'fp32']:
        print(f"\n--- {dtype.upper()} Results ---")
        dtype_df = df[df['dtype'] == dtype]
        
        # Create a more compact display
        display_cols = ['seq_len', 'dim', 'flash_fwd_ms', 'torch_fwd_ms', 'fwd_speedup',
                       'flash_bwd_ms', 'torch_bwd_ms', 'bwd_speedup',
                       'flash_e2e_ms', 'torch_e2e_ms', 'e2e_speedup']
        
        print(dtype_df[display_cols].to_string(index=False))
    
    # Save to CSV
    df.to_csv('flash_attention_benchmark_results.csv', index=False)
    print(f"\nResults saved to 'flash_attention_benchmark_results.csv'")
    
    # Note about OOM entries
    print("\n" + "=" * 80)
    print("Note: 'OOM' indicates Out-Of-Memory for that specific implementation")
    print("'-' indicates the benchmark was skipped due to a prior OOM in the pipeline")

if __name__ == "__main__":
    main()