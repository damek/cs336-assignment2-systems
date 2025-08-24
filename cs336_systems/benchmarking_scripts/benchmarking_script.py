import argparse, csv, datetime as _dt, os, sys, torch, timeit
import cs336_basics.model as models
import cs336_basics.nn_utils as nn_utils
import cs336_basics.optimizer as optimizer
from cs336_basics.utils import maybe_range
import torch 
import timeit
from contextlib import nullcontext
from torch.profiler import record_function

uid = getattr(os, "getuid", lambda: os.getpid())()
cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", f"/tmp/torchinductor_{uid}")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
os.environ.setdefault("USER", f"user{uid}")   # sidestep getpass.getuser()
os.environ.setdefault("HOME", "/tmp")
os.makedirs(cache_dir, exist_ok=True)


p = argparse.ArgumentParser()
p.add_argument("--num_layers", type=int, required=True)
p.add_argument("--output_csv", type=str, default=None, help="If given, append a single CSV row with all timings.")
p.add_argument("--num_heads", type=int, required=True)
p.add_argument("--context_length", type=int, required=True)
p.add_argument("--d_model", type=int, required=True)
p.add_argument("--d_ff", type=int, required=True)
p.add_argument("--vocab_size", type=int, required=True)
p.add_argument("--rope_theta", type=float, required=True)
p.add_argument("--num_warmup", type=int, default=5)
p.add_argument("--num_benchmark", type=int, default=10)
p.add_argument("--only_forward", action='store_true')
p.add_argument("--batch_size", type=int, default=4)
p.add_argument("--nvtx", action='store_true')
p.add_argument("--bfloat16", action='store_true')
p.add_argument("--memory", action='store_true')
p.add_argument("--compile", action='store_true')

args = p.parse_args()

# Initialize the model
model = models.BasicsTransformerLM(
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        d_model = args.d_model,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        rope_theta = args.rope_theta,
        nvtx = args.nvtx,
        )
opt = optimizer.AdamW(model.parameters())
dtype = torch.bfloat16 if args.bfloat16 else torch.float32

# Generate random data
batch_size = 4
if torch.cuda.is_available():
    device = "cuda"
random_input = torch.randint(low = 0, high = args.vocab_size, size = (batch_size, args.context_length), device=device)
random_target = torch.randint(low = 0, high = args.vocab_size, size = (batch_size, args.context_length), device=device)
model.to(device)
if args.compile:
    model = torch.compile(model)

# Global profiler for memory mode
profiler = None


def maybe_record(name):
    """Context manager for memory profiling annotations."""
    if args.memory and profiler is not None:
        return record_function(name)
    return nullcontext()


def compute_forward_and_loss():
    """Compute forward pass and loss with autocast and NVTX profiling."""
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=args.bfloat16):
        with maybe_range("model_eval", args.nvtx):
            with maybe_record("## forward ##"):
                logits = model(random_input)
        with maybe_range("loss", args.nvtx):
            with maybe_record("## loss ##"):
                loss = nn_utils.cross_entropy(logits, random_target)
    return loss


def run_forward_only():
    """Run only the forward pass without gradients."""
    with torch.no_grad():
        loss = compute_forward_and_loss()
        # Force computation by accessing the loss value
        # This prevents potential optimization/elimination of the computation
        _ = loss.item()


def run_forward_and_backward():
    """Run forward pass and backward pass with gradients."""
    model.zero_grad()
    loss = compute_forward_and_loss()
    with maybe_range("backward", args.nvtx):
        with maybe_record("## backward ##"):
            loss.backward()


def get_benchmark_function(is_forward_only):
    """Return the appropriate benchmark function based on mode."""
    return run_forward_only if is_forward_only else run_forward_and_backward


def time_iterations(fn, num_iter):
    """Time multiple iterations of a function, handling OOM errors."""
    timings = torch.zeros(num_iter)
    try:
        for i in range(num_iter):
            print(f"Running iteration {i+1} of {num_iter}")
            start = timeit.default_timer()
            fn()
            if device == "cuda":
                torch.cuda.synchronize()
            timings[i] = timeit.default_timer() - start
            # Step the profiler if memory profiling is active
            if args.memory and profiler is not None:
                profiler.step()
        return timings, False  # No OOM          
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            print("Out of memory")
            if device == "cuda":
                torch.cuda.empty_cache()
            return timings, True  # OOM occurred      
        raise                           


def run_warmup(benchmark_fn, num_warmup_iters):
    """Run warmup iterations without NVTX profiling."""
    print("Running warm-up...")
    # Temporarily disable NVTX during warmup
    original_nvtx = args.nvtx
    args.nvtx = False
    _, warmup_oom = time_iterations(benchmark_fn, num_warmup_iters)
    args.nvtx = original_nvtx
    return warmup_oom


def run_benchmarks():
    """Main benchmarking logic with clearer flow."""
    global profiler
    
    # Configure for NVTX mode
    if args.nvtx:
        args.num_warmup = 5
        args.num_benchmark = 1
    
    # Warmup phase
    warmup_fn = get_benchmark_function(args.only_forward)
    with maybe_range("warmup", args.nvtx):
        warmup_oom = run_warmup(warmup_fn, args.num_warmup)
    
    # Memory profiling setup
    if args.memory:
        print("Running memory test...")
        # Set up the profiler for memory timeline
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0, 
                warmup=0, 
                active=args.num_benchmark + (args.num_benchmark if not args.only_forward else 0),
                repeat=1
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        profiler.__enter__()
        # Also record memory history for snapshot
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    # Benchmark phase
    bench_times = None
    bench_oom = False
    
    # Forward pass benchmarking
    if args.nvtx:
        # NVTX always does forward-only first
        with maybe_range("forward_pass", args.nvtx):
            print("Running forward pass...")
            bench_times, bench_oom = time_iterations(run_forward_only, args.num_benchmark)
    elif not args.memory or args.only_forward:
        # Regular benchmarking or memory profiling with forward-only
        print("Running forward pass...")
        benchmark_fn = get_benchmark_function(args.only_forward)
        bench_times, bench_oom = time_iterations(benchmark_fn, args.num_benchmark)
    else:
        print("Skipping forward pass")
    
    # Training step benchmarking (for NVTX or memory profiling with training)
    if args.nvtx or (args.memory and not args.only_forward):
        print("Running train step...")
        with maybe_range("train_step", args.nvtx):
            # Run forward and backward passes
            bench_times, bench_oom = time_iterations(run_forward_and_backward, args.num_benchmark)
            # Optimizer step (only once after all iterations)
            with maybe_range("optimizer_step", args.nvtx):
                with maybe_record("## optimizer ##"):
                    opt.step()
                    opt.zero_grad(set_to_none=True)
    else:
        print("Skipping train step")
    
    # Memory snapshot and profiler cleanup
    if args.memory:
        if profiler is not None:
            profiler.__exit__(None, None, None)
            # Export memory timeline
            export_memory_timeline()
        dump_memory_snapshot()
    
    return bench_times, warmup_oom or bench_oom


def export_memory_timeline():
    """Export memory timeline visualization from profiler."""
    timeline_name = (f"memory_timeline_num_layers_{args.num_layers}_"
                    f"num_heads_{args.num_heads}_d_model_{args.d_model}_"
                    f"d_ff_{args.d_ff}_context_length_{args.context_length}_"
                    f"batch_size_{args.batch_size}_only_forward_{args.only_forward}_"
                    f"bfloat16_{args.bfloat16}")
    
    print(f"Exporting memory timeline for run {timeline_name}")
    os.makedirs("../outputs/memory", exist_ok=True)
    profiler.export_memory_timeline(f"../outputs/memory/{timeline_name}.html", device="cuda:0")


def dump_memory_snapshot():
    """Save memory profiling snapshot."""
    snapshot_name = (f"memory_snapshot_num_layers_{args.num_layers}_"
                    f"num_heads_{args.num_heads}_d_model_{args.d_model}_"
                    f"d_ff_{args.d_ff}_context_length_{args.context_length}_"
                    f"batch_size_{args.batch_size}_only_forward_{args.only_forward}_"
                    f"bfloat16_{args.bfloat16}")
    
    print(f"Dumping memory snapshot for run {snapshot_name}")
    os.makedirs("../outputs/memory", exist_ok=True)
    torch.cuda.memory._dump_snapshot(f"../outputs/memory/{snapshot_name}.pickle")
    torch.cuda.memory._record_memory_history(enabled=False)


def save_results(bench_times, oom):
    """Save benchmark results to CSV if requested."""
    row = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "num_layers": args.num_layers,
        "num_heads":  args.num_heads,
        "d_model":    args.d_model,
        "d_ff":       args.d_ff,
        "context_length":    args.context_length,
        "batch_size":      args.batch_size,
        "only_forward": args.only_forward,
        "bfloat16": args.bfloat16,
        "mean_s": None if oom else float(bench_times.mean()),
        "std_s":  None if oom else float(bench_times.std()),
        "oom": oom,
        "compile": args.compile,
    }
    
    if args.output_csv:
        hdr = list(row.keys())
        need_header = not os.path.exists(args.output_csv)
        with open(args.output_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            if need_header:
                w.writeheader()
            w.writerow(row)
    
    return row


# Main execution
if __name__ == "__main__":
    bench_times, oom = run_benchmarks()
    save_results(bench_times, oom)
    sys.exit(0)