import argparse
import cs336_basics.model as models
import cs336_basics.nn_utils as nn_utils
import torch 
import timeit


p = argparse.ArgumentParser()
p.add_argument("--num_layers", type=int, required=True)
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



args = p.parse_args()

# Initialize the model
model = models.BasicsTransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta
        )


# Generate random data
# Int[Tensor, " ... sequence_length"]
batch_size = 4
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
random_input = torch.randint(low = 0, high = args.vocab_size, size = (batch_size, args.context_length), device=device)
random_target = torch.randint(low = 0, high = args.vocab_size, size = (batch_size, args.context_length), device=device)
model.to(device)

def loss_fn():
    return nn_utils.cross_entropy( model(random_input), random_target )
# Do warmup steps.
avg_time_warmup = 0
print("Starting Warmup")
for i in range(args.num_warmup):
    print(f"Iteration {i}/{args.num_warmup}")
    start_time = timeit.default_timer()
    model.zero_grad()
    loss = loss_fn()
    loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = timeit.default_timer()
    avg_time_warmup += end_time - start_time

print("Avg time of warmup forward + backward", avg_time_warmup/args.num_warmup)

forward_pass_timings = torch.zeros(args.num_benchmark)
for i in range(args.num_benchmark):
    print(f"Iteration {i}/{args.num_benchmark}")
    start_time = timeit.default_timer()
    if args.only_forward: 
        with torch.no_grad(): 
            loss = loss_fn()
    else:
        model.zero_grad()
        loss = loss_fn()
        loss.backward()
    end_time = timeit.default_timer()
    if device == "cuda":
        torch.cuda.synchronize()
    forward_pass_timings[i] += end_time - start_time

print("Avg time of benchmark", forward_pass_timings.mean(), "Std time of benchmark", forward_pass_timings.std())