# Problem (communication_accounting): 10 points
Consider a new model config, XXL, with $d_{model}=16384$, $d_{ff}=53248$, and $num_{blocks}=126$. Be
cause for very large models, the vast majority of FLOPs are in the feedforward networks, we make
some simplifying assumptions. First, we omit attention, input embeddings, and output linear layers.
Then, we assume that each FFN is simply two linear layers (ignoring the activation function), where
the first has input size $d_{model}$ and output size $d_{ff}$, and the second has input size $d_{ff}$ and output size $d_{model}$. Your model consists of $num_{blocks}$ blocks of these two linear layers. Don’t do any activation checkpointing, and keep your activations and gradient communications in BF16, while your
accumulated gradients, master weights and optimizer state should be in FP32.

## Question (a)
> How much memory would it take to store the master model weights, accumulated gradients and optimizer states in FP32 on a single device? How much memory is saved for backward (these will be in BF16)? How many H100 80GB GPUs worth of memory is this? Deliverable: Your calculations and a one-sentence response.
### Master weights 
They are stored in FP32. 
- Total number of weights per block: $2*d_{model} \cdot d_{ff}$. 
- Number of blocks: $126$. 
- Thus, master weights take 

$$
126\cdot 2 \cdot d_{model} \cdot d_{ff} \text{ bytes} = 4*126*2*53248*16384/1024^3 GB = 819 GB
$$

### Accumulated gradient
They are the same size as the master weights. 
### Optimizer states 
If we're using Adam, there are two states, which are also same size as master weights.

### Total storage 

$$
819*4 GB = 3276 GB
$$

### Memory saved for backward 

Back to activation accounting. So to compute the forward pass, let $X_l$ be the input for the $l$th layer:

$$
X_{l+1} = W_{out, l}  W_{in, l} X_l
$$

So to compute backward, you'll need both $X_l$ and $W_{in, l}X_l$

So let's assume that $X_0$ is just a length $d_{model} x b$ matrix, where $b$ is the batch size (number of tokens). Then we'll need to save: 

$$
b*n_l(n_l*d_{ff} + d_{model})/2 \text{bytes} = b*0.01631399244 GBs
$$

### How many H100s 

We need

$$
(3276 GB + b*0.01631399244 GB)/ 80GB ~ (41 + 0.0002039249055*b) H100s.
$$

## Question (b)
> Now assume your master weights, optimizer state, gradients and half of your activations (in practice every second layer) are sharded across $N_{FSDP}$ devices. Write an expression for how much memory this would take per device. What value does $N_{FSDP}$ need to be for the total memory cost to be less than 1 v5p TPU (95GB per device)? Deliverable: Your calculations and a one-sentence response.

So each device is going to hold half of the activations, which is 

$$
(b/2)*0.01631399244 GB
$$

But then they're going to split the remaining activation across the $N_{FSDP}$ devices, so in total, the activation memory per device is 

$$
(b/2 + b/2N_{FSPD})*0.01631399244 GB.
$$

Therefore, since the parameters, grads, and optimizer states are fully sharded, we have 

$$
(3276/N_{FSDP} + (b/2 + b/2N_{FSPD})*0.01631399244) GBs
$$

per device. Setting this expression less than $95$ and solving for $N_{FSDP}$, we have

$$
N_{FSDP} \geq \frac{3276 + 0.01631399244 (b/2)}{95 - 0.01631399244(b/2)}
$$


## Question (c)
>  Consider only the forward pass. Use the communication bandwidth of $W_{ici} = 2 \cdot 9 \cdot 10^{10}$ and FLOPS/s of $C = 4.6 \cdot 10^{14}$ for TPU v5p as given in the TPU Scaling Book. Following the notation of the Scaling Book, use $M_X = 2$, $M_Y = 1$ (a 3D mesh), with $X = 16$ being your FSDP dimension, and $Y = 4$ being your TP dimension. At what per-device batch size is this model compute bound? What is the overall batch size in this setting? Deliverable: Your calculations and a one-sentence response.

Let's do a bit of accounting on a single layer with tensor parallelism and FSDP. I kind of like the notation from the TPU book, so we'll stick with that. 

If you combine FSDP with TP, the update equation looks like 

$$
\text{In}[B_X, D_Y] \cdot W_{in}[D_X, F_Y] \cdot W_{out}[F_Y, D_X] \rightarrow \text{Out}[B_X, D_Y]
$$

What this means is we shard the data across $X$ GPUs. We also shard the weight matrices across $XY$ GPUs. Then to perform the forward step, we 
1. Independently for each batch of data, 
2. All gather $\text{In}[B_X, D_Y]$ along the $Y$ dimension to get $\text{In}[B_X, D]$.
3. All gather $W_{in}[D_X, F_Y]$ along the $X$ dimension to get $W_{in}[D, F_Y]$ (prefetchable).
4. Multiply out $\\text{Tmp}[B\_X, F\_Y] = \\text{In}[B_X, D]\\cdot W\_{in}[D, F\_Y]$.
5. All gather $W_{out}[F_Y, D_X]$ along $X$ to get $W_{out}[F_Y, D]$ (prefetchable).
6. Multiply out $\text{Out}[B\_X, D]{U\_Y} = \text{Tmp}\_1[B\_X, F\_Y] \cdot W\_{out}[F\_Y, D]$ (NOT REDUCED YET).[^0]
7. Reduce scatter $\text{Out}[B\_X, D]{U\_Y}$ along the $Y$ to get $\text{Out}[B\_X, D\_Y]$.

(So the notation step 6 is really nice. You add a ${U_Y}$ along any direction that is waiting to be reduced. For example, you can think of step 6 as part of a multiplication of larger matrices waiting to be all reduced, so the final matrix is simply:
$$
\sum_{y \in Y} \text{Tmp}[B_X, F_Y] \cdot W_{out}[F_Y, D],
$$
which we then reduce scater over the $Y$ dimension. BTW if you want to think about )



## Question (d)
> In practice, we want the overall batch size to be as small as possible, and we also always use our compute effectively (in other words we want to never be communication bound). What other tricks can we employ to reduce the batch size of our model but retain high throughput? Deliverable: A one-paragraph response. Back up your claims with references and/or equations.

[^0]: Here is a footnote