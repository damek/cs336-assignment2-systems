import torch, time
import cs336_basics.model as models
import pandas as pd

attention = models.scaled_dot_product_attention
device = "cuda"
dtype = torch.float32
batch_size = 8
nb_warmup = 10
nb_forward_passes = 100
nb_backward_passes = nb_forward_passes

def benchmark_attention(d_models, context_lengths, nb_iter, backward=False):
    for d_model in d_models:
        for context_length in context_lengths:
            Q = torch.randn(batch_size, context_length, d_model, device="cuda")
            K = torch.randn(batch_size, context_length, d_model, device="cuda")
            V = torch.randn(batch_size, context_length, d_model, device="cuda")
            for _ in range(nb_iter):
                out = attention(Q,K,V)
                loss = out.sum()
                if backward:
                    loss.backward()
                    Q.grad = None
                    K.grad = None
                    V.grad = None
                torch.cuda.synchronize()

def time_loop(fn, iters):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters  # ms/iter

def run_config(d_model, seq_length):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    Q = torch.randn(batch_size, seq_length, d_model, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(batch_size, seq_length, d_model, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(batch_size, seq_length, d_model, device=device, dtype=dtype, requires_grad=True)

    print("Warmup for d_model: ", d_model, "and context_length: ", seq_length)
    for _ in range(nb_warmup):
        out = attention(Q, K, V)
        loss = out.sum()
        loss.backward()
        Q.grad = K.grad = V.grad = None
    torch.cuda.synchronize()

    print("Forward timing for d_model: ", d_model, "and context_length: ", seq_length)
    torch.cuda.reset_peak_memory_stats()
    fwd_ms = time_loop(lambda: (attention(Q, K, V).sum(),), nb_forward_passes) 
    fwd_peak_bytes = torch.cuda.max_memory_allocated()

    print("Memory before backward pass for d_model: ", d_model, "and context_length: ", seq_length)
    torch.cuda.synchronize()
    mem_after_inputs = torch.cuda.memory_allocated()
    out = attention(Q, K, V); loss = out.sum()
    torch.cuda.synchronize()
    mem_before_bwd = torch.cuda.memory_allocated()
    saved_activations = mem_before_bwd - mem_after_inputs  

    print("Backward timing for d_model: ", d_model, "and context_length: ", seq_length)
    torch.cuda.reset_peak_memory_stats()
    bwd_ms = None
    bwd_peak_bytes = None
    bwd_oom = False
    try:
        bwd_ms = time_loop(lambda: _step_bwd(Q, K, V), nb_backward_passes)
        bwd_peak_bytes = torch.cuda.max_memory_allocated()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            bwd_oom = True
            torch.cuda.empty_cache()
            print("Out of memory error in backward pass")
        else:
            raise

    return {
        "d_model": d_model,
        "seq_len": seq_length,
        "forward_ms": fwd_ms,
        "forward_peak_MB": fwd_peak_bytes / 1024**2,
        "mem_after_inputs_MB": mem_after_inputs / 1024**2,
        "mem_before_backward_MB": mem_before_bwd / 1024**2,
        "saved_activations_MB": saved_activations / 1024**2,
        "backward_ms": (None if bwd_oom else bwd_ms),
        "backward_peak_MB": (None if bwd_oom else bwd_peak_bytes / 1024**2),
        "status": ("OOM(backward)" if bwd_oom else "ok"),
    }

def _step_bwd(Q, K, V):
    out = attention(Q, K, V)
    loss = out.sum()
    loss.backward()
    Q.grad = K.grad = V.grad = None

d_models = [16, 32, 64, 128]
context_lengths = [256, 1024, 4096, 8192, 16384]

rows = []
for d in d_models:
    for L in context_lengths:
        try:
            res = run_config(d, L)  
        except RuntimeError as e:
            res = {"d_model": d, "seq_len": L, "status": "OOM(forward)"}
        rows.append(res)

df = pd.DataFrame(rows)

col_order = [
    "d_model", "seq_len",
    "forward_ms", "backward_ms",
    "mem_after_inputs_MB", "mem_before_backward_MB",
    "saved_activations_MB",
    "forward_peak_MB", "backward_peak_MB",
    "status",
]
df = df.reindex(columns=[c for c in col_order if c in df.columns])

df = df.sort_values(["d_model", "seq_len"]).reset_index(drop=True)

# Pretty print 
print(df.to_markdown(index=False))  # requires 'tabulate' package

# Save a CSV 
out_csv = "../outputs/csv/attention_benchmark.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved results to {out_csv}")

if {"forward_ms", "backward_ms"} <= set(df.columns):
    fwd_pivot = df.pivot(index="seq_len", columns="d_model", values="forward_ms")
    bwd_pivot = df.pivot(index="seq_len", columns="d_model", values="backward_ms")
    print("\nForward (ms/iter) by seq_len x d_model:\n", fwd_pivot.to_markdown())
    print("\nBackward (ms/iter) by seq_len x d_model:\n", bwd_pivot.to_markdown())