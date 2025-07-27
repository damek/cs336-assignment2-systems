import argparse
import cs336_basics.model as models
import torch 


p = argparse.ArgumentParser()
p.add_argument("--num_layers", type=float, required=True)
p.add_argument("--num_heads", type=float, required=True)
p.add_argument("--context_length", type=float, required=True)
p.add_argument("--d_model", type=float, required=True)
p.add_argument("--d_ff", type=float, required=True)
p.add_argument("--vocab_size", type=float, required=True)
p.add_argument("--rope_theta", type=float, required=True)

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
batch_size = 32
random_data = torch.randint(low = 0, high = 10000, size = args.sequence_length)
