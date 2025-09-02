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
2*b*n_l(d_{ff} + d_{model}) \text{bytes} = b*0.01631399244 GBs
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
6. Multiply out $\text{Out}[B\_X, D]{U\_Y} = \text{Tmp}\_1[B\_X, F\_Y] \cdot W\_{out}[F\_Y, D]$ (NOT REDUCED YET).[^nice_notation]
7. Reduce scatter $\text{Out}[B\_X, D]{U\_Y}$ along the $Y$ to get $\text{Out}[B\_X, D\_Y]$.

We can calculate the total computation time as two matrix multiplies.

$$
T_{math} = \frac{2*2*BDF}{CXY}
$$

On the other hand, the cost of an all gather / all reduce is $(\text{total bytes})/W_{ici}$. Thus, we have 

$$
T_{comm} = \frac{2\cdot(B/X)D/M_Y + 2\cdot2\cdot(D/M_X)(F/Y) + 2\cdot(B/X)D/M_Y}{W_{ici}} = 4D\frac{B/(XM_Y) +F/(YM_X)}{W_{ici}} 
$$

where the leading 2 comes from the fact that the weights are in FP16.

### When are we compute bound?

I.e., when is $T_{math} > T_{comm}$. This occurs when 

$$
\frac{4BDF}{CXY} > 4D\frac{B/(XM_Y) +F/(YM_X)}{W_{ici}}
$$

The only free parameter here is $B$. So let's solve for $B$. 
We need 

$$
B\left(\frac{F}{CXY} - \frac{1}{XM_YW_{ici}}\right) > \frac{F}{W_{ici}YM_X}
$$

Multiplying both sides by $CXYW\_{ici}M\_Y$, we have 

$$
B\left(FM_YW_{ici} - CY\right) > \frac{FCXM_Y}{M_X}
$$

Therefore, we need 

$$
B \geq \frac{FCXM_Y}{M_X\left(FM_YW_{ici} - CY\right)}
$$

Plugging in the values of $F = 53248, D = 16384$, $M_X = 2, M_Y = 1, X = 16, Y = 4, C = 4.6 \cdot 10^{14}$, and $W_{ici} = 2 \cdot 9 \cdot 10^{10}$, we require that

$$
B/X \geq 1581 \qquad \text{ and } B_{\min} \geq 25302.
$$

## Question (d)
> In practice, we want the overall batch size to be as small as possible, and we also always use our compute effectively (in other words we want to never be communication bound). What other tricks can we employ to reduce the batch size of our model but retain high throughput? Deliverable: A one-paragraph response. Back up your claims with references and/or equations.

We're going to mess with the grid and the allocation of GPUs between X and Y. The book does this also, but I thought about it before reading, lol. I do it a different way I guess. So how do we do it? 

Recall we have $N := 64 = XY$ and $M_XM_Y=2$, so we can replace $X$ by $N/Y$ to get 

$$
B \geq \frac{CX}{M_X\left(W_{ici} - (CY/FM_Y)\right)} = \frac{CN}{YM_X\left(W_{ici} - (CY/FM_Y)\right)}
$$

So we just want to maximize the denominator which is: 

$$
g(Y) = YM_X\left(W_{ici} - (CY/FM_Y)\right)
$$

Notice that the max occurs at $g'(Y) = 0$, meaning $Y = FM_Y W_{ici}/2C$. Plugging this in gives
 
$$
\frac{FM_YM_X W_{ici}}{4C} = \frac{F W_{ici}^2}{2C}
$$

Now, we need $64 \geq Y = FM_Y W_{ici}/2C = 53248* (2 \cdot 9 \cdot 10^{10}/2*(4.6 \cdot 10^{14})) M_Y \sim 10.4180869565 *M_Y$, so this is an OK $Y$, since $M_Y \leq 2$. Ok let's plug this in to get the final batch size formula: 

$$
B \geq \frac{CN}{\frac{F W_{ici}^2}{2C}} = \frac{2C^2N}{F W_{ici}^2} \approx 15699.
$$

I like this answer. But you can also do a grid search over integer $Y$ to find the optimal such that $Y$ that's an integer. In this case, the setting $Y = X = 8$ and the min batch size is $\geq 16593$

## Appendix

### Nice notation
So the notation step 6 is really nice. You add a ${U_Y}$ along any direction that is waiting to be reduced. For example, you can think of step 6 as part of a multiplication of larger matrices waiting to be all reduced, so the final matrix is simply:
$$
\sum_{y \in Y} \text{Tmp}[B_X, F_Y] \cdot W_{out}[F_Y, D],
$$
which we then reduce scater over the $Y$ dimension. BTW if you want to think about TP, it's just this: split the input matrix into two columns and the output matrix into two grows. Then notice that 
$$
\text{In} [A_1, A_2] \begin{bmatrix} B_1 \\ B_2\end{bmatrix} = \text{In}A_1 B_1 + \text{In}A_2 B_2.
$$
You can easily see how to parallelize this by first multiplying $\text{In}A_i$ separately and then multiplying by the $B_1$, then reduce scattering.