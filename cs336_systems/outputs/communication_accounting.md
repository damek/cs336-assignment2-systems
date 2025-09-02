# Problem (communication_accounting): 10 points
Consider a new model config, XXL, with $d_{model}=16384$, $d_{ff}=53248$, and $num_{blocks}=126$. Be
cause for very large models, the vast majority of FLOPs are in the feedforward networks, we make
some simplifying assumptions. First, we omit attention, input embeddings, and output linear layers.
Then, we assume that each FFN is simply two linear layers (ignoring the activation function), where
the first has input size $d_{model}$ and output size $d_{ff}$, and the second has input size $d_{ff}$ and output size $d_{model}$. Your model consists of $num_{blocks}$ blocks of these two linear layers. Don’t do any activation checkpointing, and keep your activations and gradient communications in BF16, while your
accumulated gradients, master weights and optimizer state should be in FP32.

## Question (a)
> How much memory would it take to store the master model weights, accumulated gradients and optimizer states in FP32 on a single device? How much memory is saved for backward (these will be in BF16)? How many H100 80GB GPUs worth of memory is this? Deliverable: Your calculations and a one-sentence response.

## Question (b)
> Now assume your master weights, optimizer state, gradients and half of your activations (in practice every second layer) are sharded across NFSDP devices. Write an expression for how much memory this would take per device. What value does NFSDP need to be for the total memory cost to be less than 1 v5p TPU (95GB per device)? Deliverable: Your calculationsand a one-sentence response.

## Question (c)
>  Consider only the forward pass. Use the communication bandwidth of Wici = 2 · 9 · 1010 and FLOPS/s of C = 4.6 · 1014 for TPU v5p as given in the TPU Scaling Book. Following the notation of the Scaling Book, use MX = 2, MY = 1 (a 3D mesh), with X = 16 being your FSDP dimension, and Y = 4 being your TP dimension. At what per-device batch size is this model compute bound? What is the overall batch size in this setting? Deliverable: Your calculations and a one-sentence response.

## Question (d)
> In practice, we want the overall batch size to be as small as possible, and we also always use our compute effectively (in other words we want to never be communication bound). What other tricks can we employ to reduce the batch size of our model but retain high throughput? Deliverable: A one-paragraph response. Back up your claims with references and/or equations.
