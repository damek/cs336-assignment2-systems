import torch, time
import cs336_basics.model as models

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

for d_model in d_models:
    for seq_length in context_lengths:
        run_config(d_model, seq_length)