# Learning project: sparse attention

In this learning project, I'm probably reinventing a very inefficient wheel. My goal was to learn more about:

- the `gather/scatter/index_select` operators in PyTorch
- how to use the PyTorch profiler
- how to create a simple C++ PyTorch extension

and anything I may discover on the way.

To this end, I decided to try and implement different ways of computing attention for cases where the *"attention mask"* is very sparse, i.e. cases where it could be useful not to compute the whole $QK^T$ matrix.

## Context

### Vanilla Attention

The attention mechanism computes new representations $Y \in \mathcal{R}^{N_q \times d_v}$ as weighted averages of value vectors $V \in \mathcal{R}^{N_v \times d_v}$:

$$ y_i = \sum_{j=1}^{N_v} a_{ij} v_j $$

for $i \in \{1 ... N_q\}$.

The weights of the averages are computed using the dot-product similarity of query and key vectors $Q \in \mathcal{R}^{N_q \times d_k}$ and $K \in \mathcal{R}^{N_v \times d_k}$:

$$ a_{ij} = \frac{e^{q_i^T k_j}}{\sum_{m} e^{q_i^T k_m}} $$

(in practice the dot product is scaled with a factor $1/\sqrt{d_k}$).

The computation of attention leverages the efficient matrix multiplications:

$$ Y = AV = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$


### Attention masks

The previous section describes a full attention, where all values are attended to for all outputs. In many situations, each output $i$ only considers a limited context $\mathcal{C}(i) \subset \{1 ... N_v\}$. 

For example, in training batches with different cardinality of inputs, some of the values correspond to padding: $\mathcal{C}(i) = \{1 ... L\}$, with $\{L+1 ... N_v\}$ being the applied padding.

Another example is causal attention for sequential inputs, where $\mathcal{C}(t) = \{1 ... t\}$, with $\{t+1 ... N_v\}$ being future items that we don't want to consider at $t$.

In these situations, we want to compute $Y$ as:

$$ y_i = \sum_{j \in \mathcal{C}(i)} a_{ij} v_j$$

with:

$$a_{ij} = \frac{e^{q_i^T k_j}}{\sum_{m \in \mathcal{C}(i)} e^{q_i^T k_m}}$$

In practice, the computation in a vanilla implementation is achieved by multiplying the elements of the similarity matrix with an *attention mask* $\mathbf{M}$:

$$ A = softmax\left(\frac{QK^T}{\sqrt{d_k}} \odot \mathbf{M} \right)$$

with $m_{ij} = -\infty$ if $j \notin \mathcal{C}(i)$ and $1$ otherwise.

This amounts to computing all the dot products $q_i^T k_j$ and replace those for which $j \notin \mathcal{C}(i)$ so that $a_{ij} = 0$.

### Sparse attention

While the efficiency of matrix multiplication kernels outweight the waste of computation for the discarded $q_i^T k_j$, we could imagine that a more efficient process could exist for very sparse attention, i.e. when $|\mathcal{C}(i)| \ll N_v$.

The goal of this small learning project is to explore how I could start with a vanilla PyTorch implementation of masked attention and modify it to only compute the relevant $q_i^T k_j$ for very sparse attention.

## Preliminaries

### Baseline Vanilla implementation

```python
def masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    # q, k, v have shape [H, T, D]
    # attn_mask has shape [T, T]
    #   and is a boolean matrix where 
    #   True values indicate C(i)

    # QK / sqrt(D)
    head_dim = q.size(2)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
    # attn_weights has shape [H, T, T]

    # QK . M
    attn_mask = attn_mask.unsqueeze(0)  # shape [1, T, T]
    attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))

    # A = softmax(QK . M)
    attn_weights = torch.softmax(attn_weights, dim=-1)

    # Y = AV
    return torch.matmul(attn_weights, v)
```

### Hardware

All experiments were run on my laptop: Apple MacBook Pro M4, 16GB. 

The compute environment not being completely isolated and running other applications, the profiling results may variate a bit from one run to another...

### Resources

I consulted the following resources while iterating on this small project:

...

## Step 1 - Pure PyTorch implementation

...