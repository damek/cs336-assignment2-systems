import argparse, csv, datetime as _dt, os, sys, torch, timeit
import cs336_basics.model as models
import cs336_basics.nn_utils as nn_utils
import cs336_basics.optimizer as optimizer
from cs336_basics.utils import maybe_range
import torch 
import timeit


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
optimizer = optimizer(model.parameters())

# Generate random data
# Int[Tensor, " ... sequence_length"]
batch_size = 4
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
random_input = torch.randint(low = 0, high = args.vocab_size, size = (batch_size, args.context_length), device=device)
random_target = torch.randint(low = 0, high = args.vocab_size, size = (batch_size, args.context_length), device=device)
model.to(device)


def run_section(fn, num_iter):
    timings = torch.zeros(num_iter)
    try:
        for i in range(num_iter):
            print(f"Running iteration {i+1} of {num_iter}")
            start = timeit.default_timer()
            fn()
            if device == "cuda":
                torch.cuda.synchronize()
            timings[i] = timeit.default_timer() - start
        return timings, False          
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            print("Out of memory")
            if device == "cuda":
                torch.cuda.empty_cache()
            return timings, True       
        raise                           

def loss_fn():
    with maybe_range("model_eval", args.nvtx):
        logits = model(random_input)
    with maybe_range("loss", args.nvtx):
        loss = nn_utils.cross_entropy(logits)
    return loss

def forward_pass():
    if args.only_forward:
        with torch.no_grad():
            loss_fn()
    else:
        model.zero_grad()
        loss = loss_fn()
        with maybe_range("backward", args.nvtx):
            loss.backward()

# If nvtx we hard code some stuff.
if args.nvtx:
    args.only_forward = True
    args.num_warmpup = 1
    args.num_benchmark = 1

# warm-up
with maybe_range("warmup", args.nvtx):
    print("Running warm-up...")
    _, warmup_oom = run_section(forward_pass, args.num_warmup)

# benchmark
with maybe_range("forward_pass", args.nvtx):
    print("Running forward pass...")
    bench_times, bench_oom = run_section(forward_pass, args.num_benchmark)

if args.nvtx:
    with maybe_range("train_step", args.nvtx):
        bench_times, bench_oom = run_section(forward_pass, args.num_benchmark)
        with maybe_range("optimizer_step", args.nvtx):
            optimizer.step()


oom = warmup_oom or bench_oom
row = {
    "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    "num_layers": args.num_layers,
    "num_heads":  args.num_heads,
    "d_model":    args.d_model,
    "d_ff":       args.d_ff,
    "context_length":    args.context_length,
    "batch_size":      args.batch_size,
    "only_forward": args.only_forward,
    "mean_s": None if oom else float(bench_times.mean()),
    "std_s":  None if oom else float(bench_times.std()),
    "oom": oom,
}

if args.output_csv:
    hdr = list(row.keys())
    need_header = not os.path.exists(args.output_csv)
    with open(args.output_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        if need_header:
            w.writeheader()
        w.writerow(row)

sys.exit(0)