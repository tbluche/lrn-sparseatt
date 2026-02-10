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

### Idea

As a first step, I tried a pure PyTorch implementation. The idea is to only compute the $q_i^T k_j$ for $j \in \mathcal{C}(i)$ instead of a full $QK^T$ multiplication, i.e. replace

```python
torch.matmul(q, k.transpose(-2, -1))
```

with 

```python
(qs * ks).sum(dim=-1)
```

where `qs` and `ks` have shape `(M, D)` and `M` is the number of $(i, j)$ pairs satisfying $j \in \mathcal{C}(i)$. Said differently, these are the `Q[i]` and `K[j]` for all indices `[i, j]` where the attention mask is `True`.

This amounts to computing the indices of non-zero elements of the boolean mask and gathering vectors from `Q` and `K`:

```python
indices = torch.nonzero(mask, as_tuple=False)
qs = q.index_select(1, indices[: , 0])
ks = k.index_select(1, indices[: , 1])
```

### Exploration

This option is explored in [1_PyTorch Implementation.ipynb](./notebooks/1_PyTorch%20Implementation.ipynb).

#### Dot products

It looks like using `index_select` for this particular case is more efficient than using the `gather` operator.

In the computation of the dot products, the direct implementation with elementwise multiplication and sum is sometimes faster than doing it with `torch.einsum`. 

#### Softmax

For the softmax normalization of the weights, instead of recomputing the whole $QK^T$ from the relevant dot products, I also limited computation to the attended locations.

First, we compute the shifted exponential of all the computed $q_i^T k_j$:

```python
num = (attn_weights - attn_weights.max()).exp()
```

For the denominator of the softmax, we need to sum all the `num` values for each $i$. I used `index_add`:

```python
den = torch.index_add(
    # One denominator per head and per query vector
    torch.zeros((n_heads, seq_len)), 
    # dim = 1 -> sequence length
    1, 
    # the 'i' part of the (i, j) list
    q_indices, 
    # num contains the [H, M] numerator values (exp)
    num
)
```

This does:

```python
i = q_indices[k]
den[:, i] += num[:, k]
```

To finalize the computation, all numerators need to be divided by the corresponding denominator in `den`, so we copy `den` to all positions in `num`, again with `index_select`:

```python
attn_weights = num / den.index_select(1, q_indices)
```

Looking at the profiles, it looks like:
- most of the time is spent on the indices manipulations
- `index_add` is using `scatter_add`

```
----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                  Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
         ProfilerStep*        14.68%     266.957us       100.00%       1.819ms      60.631us            30  
       aten::index_add         1.29%      23.461us        55.71%       1.013ms      33.775us            30  
    aten::scatter_add_        53.62%     975.244us        53.65%     975.871us      32.529us            30  
             aten::exp        12.95%     235.464us        12.95%     235.464us       7.849us            30  
    aten::index_select         8.08%     147.012us         8.23%     149.677us       4.989us            30  
             aten::max         2.99%      54.411us         3.41%      62.044us       2.068us            30  
             aten::div         2.19%      39.877us         2.19%      39.877us       1.329us            30  
             aten::sub         1.75%      31.877us         1.75%      31.877us       1.063us            30  
           aten::zeros         0.49%       8.919us         0.87%      15.749us       0.525us            30  
           aten::empty         0.81%      14.707us         0.81%      14.707us       0.163us            90  
           aten::copy_         0.58%      10.584us         0.58%      10.584us       0.353us            30  
         aten::flatten         0.22%       4.041us         0.22%       4.041us       0.067us            60  
      aten::as_strided         0.18%       3.335us         0.18%       3.335us       0.111us            30  
           aten::zero_         0.07%       1.291us         0.07%       1.291us       0.043us            30  
           aten::fill_         0.06%       1.130us         0.06%       1.130us       0.038us            30  
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
```

#### Weighted sum

Again, we use:

- `index_select` to gather all value vectors with a non-zero attention weight (one we computed)
- `index_add` to accumulate the weighted sum of values

```python
# Select values
vs_indsel = v.index_select(1, kv_indices)
vs_indsel = vs_indsel.view(n_heads, -1, head_dim)
# Weigh values
weighted_vs = attn_weights.unsqueeze(-1) * vs_indsel
# Sum of weighted values
out = torch.zeros((n_heads, seq_len, head_dim))
out.index_add_(1, q_indices, weighted_vs)
```

#### Putting it all together

Looking at the profiles of the implementation, we see that the `index_select` and `index_add` dominate the computation times:

```


-----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
          ProfilerStep*         9.65%       1.397ms       100.00%      14.482ms     482.736us            30  
       aten::index_add_        31.51%       4.563ms        31.67%       4.586ms     152.867us            30  
     aten::index_select        24.86%       3.600ms        24.93%       3.611ms      30.088us           120  
              aten::mul        14.77%       2.138ms        14.77%       2.138ms      35.639us            60  
        aten::index_add         0.18%      25.835us         7.17%       1.039ms      34.635us            30  
     aten::scatter_add_         6.87%     994.261us         6.87%     995.385us      33.180us            30  
              aten::sum         6.20%     898.131us         6.38%     924.220us      30.807us            30  
              aten::exp         2.70%     390.920us         2.70%     390.920us      13.031us            30  
              aten::div         0.72%     104.168us         1.00%     144.421us       2.407us            60  
              aten::max         0.45%      65.502us         0.51%      73.214us       2.440us            30  
            aten::zeros         0.21%      30.045us         0.40%      57.291us       0.955us            60  
           aten::select         0.27%      39.090us         0.35%      51.213us       0.427us           120  
               aten::to         0.05%       7.297us         0.29%      41.377us       0.690us            60  
              aten::sub         0.28%      39.958us         0.28%      39.958us       1.332us            30  
             aten::view         0.24%      34.668us         0.24%      34.668us       0.385us            90  
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------ 
```

### Conclusions

The full profiling [logs](./profiles/1_profile.log) and [results](./profiles/1_profile_results.md) are available in the `profiles/` folder.

In the end, this implementation works but is not competitive. It is slightly better for very high levels of spasity:

|   seq_len |   d_model |   n_heads | sparsity   | dense_time   | sparse_time   |
|-----------|-----------|-----------|------------|--------------|---------------|
|       512 |        64 |         1 | 99%        | 468.13 µs    | 334.99 µs     |
|       512 |        64 |         4 | 99%        | 1.70 ms      | 417.64 µs     |
|       512 |        64 |         8 | 99%        | 2.78 ms      | 570.55 µs     |
|       512 |       256 |         1 | 99%        | 543.71 µs    | 906.73 µs     |
|       512 |       256 |         4 | 99%        | 1.41 ms      | 1.05 ms       |
|       512 |       256 |         8 | 99%        | 3.15 ms      | 1.06 ms       |
|      1024 |        64 |         1 | 99%        | 1.47 ms      | 1.28 ms       |
|      1024 |        64 |         4 | 99%        | 4.31 ms      | 1.50 ms       |
|      1024 |        64 |         8 | 99%        | 10.21 ms     | 2.11 ms       |
|      1024 |       256 |         1 | 99%        | 1.92 ms      | 3.41 ms       |
|      1024 |       256 |         4 | 99%        | 5.17 ms      | 3.51 ms       |
|      1024 |       256 |         8 | 99%        | 10.01 ms     | 3.56 ms       |

But even with 95% sparsity, the alternative implementation doesn't beat the masked baseline except with many small heads:

|   seq_len |   d_model |   n_heads | sparsity   | dense_time   | sparse_time   |
|-----------|-----------|-----------|------------|--------------|---------------|
|       512 |        64 |         1 | 95%        | 435.82 µs    | 1.58 ms       |
|       512 |        64 |         4 | 95%        | 1.79 ms      | 2.07 ms       |
|       512 |        64 |         8 | 95%        | 2.85 ms      | 2.48 ms       |
|       512 |       256 |         1 | 95%        | 539.87 µs    | 3.91 ms       |
|       512 |       256 |         4 | 95%        | 1.47 ms      | 4.20 ms       |
|       512 |       256 |         8 | 95%        | 3.05 ms      | 4.40 ms       |
|      1024 |        64 |         1 | 95%        | 1.54 ms      | 5.06 ms       |
|      1024 |        64 |         4 | 95%        | 4.19 ms      | 6.64 ms       |
|      1024 |        64 |         8 | 95%        | 10.91 ms     | 9.53 ms       |
|      1024 |       256 |         1 | 95%        | 1.95 ms      | 18.36 ms      |
|      1024 |       256 |         4 | 95%        | 5.68 ms      | 19.49 ms      |
|      1024 |       256 |         8 | 95%        | 10.64 ms     | 20.65 ms      |

Focusing on one head, this implementation is at most barely competitive:

|   seq_len |   d_model | sparsity   | dense_time   | sparse_time   |
|-----------|-----------|------------|--------------|---------------|
|       512 |        64 | 95%        | 435.82 µs    | 1.58 ms       |
|       512 |        64 | 99%        | 468.13 µs    | 334.99 µs     |
|       512 |       256 | 95%        | 539.87 µs    | 3.91 ms       |
|       512 |       256 | 99%        | 543.71 µs    | 906.73 µs     |
|      1024 |        64 | 95%        | 1.54 ms      | 5.06 ms       |
|      1024 |        64 | 99%        | 1.47 ms      | 1.28 ms       |
|      1024 |       256 | 95%        | 1.95 ms      | 18.36 ms      |
|      1024 |       256 | 99%        | 1.92 ms      | 3.41 ms       |

## Step 2 - Nested tensors

...

## Step 3 - Sparse MatMul

As we saw in the profiling results, the PyTorch implementation suffers from the `index_select` operations. In the end, we don't really need to copy the vectors from the input matrices: we could access them directly in the original `Q` and `K` tensors to compute the relevant dot products `q_i^T k_j`.

I implemented a `sparse_matmul` C++ operator for PyTorch to replace

```python
qs = q.index_select(1, indices[: , 0])
ks = k.index_select(1, indices[: , 1])
weights = (qs * ks).sum(dim=-1)
```

with 

```python
weights = sparse_matmul(q, k, indices)
```

I limited the operation to `T x D` tensors for `q` and `k` (i.e. no heads) for simplicity, and to this part of the computation because in this end it will look like a naive matrix multiplication.

In [3_CPP Sparse MM](./notebooks/3_CPP%20Sparse%20MM.ipynb) I explore this option. Comparing the PyTorch version and the C++ operator, there is not a huge change, going from

```
----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                  Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
----------------------  ------------  ------------  ------------  ------------  ------------  ------------  
         ProfilerStep*        14.04%     324.161us       100.00%       2.309ms      76.971us            30  
    aten::index_select        44.79%       1.034ms        45.10%       1.041ms      17.355us            60  
             aten::mul        22.32%     515.473us        22.32%     515.473us      17.182us            30  
             aten::sum        16.83%     388.585us        17.45%     402.876us      13.429us            30  
            aten::view         0.91%      21.006us         0.91%      21.006us       0.350us            60  
           aten::fill_         0.50%      11.582us         0.50%      11.582us       0.386us            30  
           aten::empty         0.30%       7.042us         0.30%       7.042us       0.117us            60  
         aten::flatten         0.19%       4.289us         0.19%       4.289us       0.071us            60  
      aten::as_strided         0.12%       2.709us         0.12%       2.709us       0.090us            30  
----------------------  ------------  ------------  ------------  ------------  ------------  ------------
```

to 

```
--------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                            Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
--------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                   ProfilerStep*         7.46%     171.589us       100.00%       2.299ms      76.646us            30  
    extension_cpp::sparse_matmul        67.30%       1.547ms        91.32%       2.100ms      69.993us            30  
                aten::contiguous         0.28%       6.508us        23.77%     546.641us       6.074us            90  
                     aten::clone         0.38%       8.624us        23.49%     540.133us      18.004us            30  
                     aten::copy_        22.50%     517.305us        22.70%     521.885us      17.396us            30  
                   aten::squeeze         1.03%      23.789us         1.22%      28.001us       0.467us            60  
                     aten::empty         0.69%      15.787us         0.69%      15.787us       0.175us            90  
                aten::empty_like         0.18%       4.127us         0.42%       9.624us       0.321us            30  
                aten::as_strided         0.18%       4.212us         0.18%       4.212us       0.070us            60  
--------------------------------  ------------  ------------  ------------  ------------  ------------  ------------ 
```

A lot of time is now freed from the `index_select` but spent in the operator, which makes sense since the dot products are not vectorized in this implementation (while they probably are in the `aten::mul`).

Taking advantage of what I learned with the nested tensors, in particular the jagged layout, I implemented an alternative computation replacing the indices pairs `indices` with two tensors:

 - `values` containing the `j` indices in $K$ (corresponding to `indices[:, 1]`)
 - `offsets` telling where the values for a given query `i` in $Q$ end.

It looks like this alternative implementation, `sparse_matmul_vo` is slightly more efficient:

```
-----------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                               Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-----------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                      ProfilerStep*         9.13%     156.958us       100.00%       1.719ms      57.310us            30  
    extension_cpp::sparse_matmul_vo        88.74%       1.526ms        89.25%       1.534ms      51.148us            30  
                      aten::squeeze         1.37%      23.624us         1.62%      27.920us       0.465us            60  
                        aten::empty         0.39%       6.788us         0.39%       6.788us       0.226us            30  
                   aten::as_strided         0.25%       4.296us         0.25%       4.296us       0.072us            60  
                   aten::contiguous         0.11%       1.877us         0.11%       1.877us       0.031us            60  
-----------------------------------  ------------  ------------  ------------  ------------  ------------  ------------ 
```

The full profiling [logs](./profiles/3_profile_bis.log) and [results](./profiles/1_profile_results_bis.md) are available in the `profiles/` folder:

- `dense` is the masked attention baseline
- `sparse` is the pure PyTorch implementation
- `sparse_1` uses the `sparse_matmul` C++ op
- `sparse_2` uses the `sparse_matmul_vo` C++ op

We see nice improvements over the pure PyTorch implementation but the C++ ops are still not competitive at even 95% sparsity:

|   seq_len |   d_model | sparsity   | dense_time   | sparse_time   | sparse_1_time   | sparse_2_time   |
|-----------|-----------|------------|--------------|---------------|-----------------|-----------------|
|       512 |        32 | 95%        | 329.05 µs    | 856.26 µs     | 582.22 µs       | 562.29 µs       |
|       512 |        32 | 99%        | 309.37 µs    | 185.47 µs     | 164.03 µs       | 147.41 µs       |
|       512 |        64 | 95%        | 356.32 µs    | 1.42 ms       | 847.18 µs       | 891.73 µs       |
|       512 |        64 | 99%        | 342.89 µs    | 293.63 µs     | 194.96 µs       | 184.10 µs       |
|       512 |       128 | 95%        | 416.40 µs    | 2.41 ms       | 1.47 ms         | 1.72 ms         |
|       512 |       128 | 99%        | 407.72 µs    | 416.32 µs     | 335.98 µs       | 319.59 µs       |
|      1024 |        32 | 95%        | 1.44 ms      | 3.56 ms       | 2.48 ms         | 2.28 ms         |
|      1024 |        32 | 99%        | 1.40 ms      | 636.13 µs     | 479.02 µs       | 426.17 µs       |
|      1024 |        64 | 95%        | 1.63 ms      | 4.95 ms       | 3.27 ms         | 3.04 ms         |
|      1024 |        64 | 99%        | 1.39 ms      | 1.21 ms       | 724.29 µs       | 705.73 µs       |
|      1024 |       128 | 95%        | 1.69 ms      | 9.68 ms       | 6.10 ms         | 5.66 ms         |
|      1024 |       128 | 99%        | 1.62 ms      | 1.95 ms       | 1.26 ms         | 1.19 ms         |
|      2048 |        32 | 95%        | 5.26 ms      | 13.99 ms      | 9.91 ms         | 9.30 ms         |
|      2048 |        32 | 99%        | 5.15 ms      | 2.79 ms       | 2.02 ms         | 1.99 ms         |
|      2048 |        64 | 95%        | 5.69 ms      | 22.27 ms      | 14.76 ms        | 13.79 ms        |
|      2048 |        64 | 99%        | 5.49 ms      | 3.95 ms       | 3.05 ms         | 2.52 ms         |
|      2048 |       128 | 95%        | 6.25 ms      | 38.80 ms      | 24.72 ms        | 23.69 ms        |
|      2048 |       128 | 99%        | 6.14 ms      | 7.69 ms       | 5.05 ms         | 4.58 ms         |


## Step 4 - Full C++ implementation

The previous implementation mostly improves because of the reduced `index_select` operations, at the cost on a non-vectorized dot product computation. As we can see in the [profiling logs](./profiles/3_profile_results_bis.md), the remaining index manipulations and data copies still take a lot of time:

```
-----------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                               Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
-----------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                      ProfilerStep*         5.61%       7.704ms       100.00%     137.322ms       4.577ms            30  
    extension_cpp::sparse_matmul_vo        25.79%      35.421ms        25.81%      35.441ms       1.181ms            30  
                 aten::index_select        24.87%      34.149ms        24.87%      34.157ms     569.287us            60  
                   aten::index_add_        23.28%      31.975ms        23.31%      32.008ms       1.067ms            30  
                          aten::mul        15.15%      20.799ms        15.15%      20.799ms     693.316us            30  
                    aten::index_add         0.02%      32.296us         2.93%       4.029ms     134.310us            30  
                 aten::scatter_add_         2.89%       3.970ms         2.89%       3.972ms     132.394us            30  
                        aten::zeros         0.02%      31.454us         0.68%     936.499us      15.608us            60  
                        aten::zero_         0.01%      20.292us         0.64%     882.376us      14.706us            60  
                        aten::fill_         0.63%     864.582us         0.63%     864.582us      14.410us            60  
                          aten::div         0.57%     776.497us         0.59%     816.580us      13.610us            60  
                          aten::exp         0.38%     526.582us         0.38%     526.582us      17.553us            30  
                          aten::max         0.29%     404.421us         0.30%     416.251us      13.875us            30  
                          aten::sub         0.25%     340.667us         0.25%     340.667us      11.356us            30  
                       aten::select         0.04%      57.040us         0.05%      69.077us       0.576us           120  
-----------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
```

In this step, the rest of the attention computation (i.e. softmax on weights and averaging of values) is added to the C++ operation, resulting in a `sparse_attn` C++ op. 

During the computation of dot products, the maximum value is tracked to stabilize the subsequent softmax.

For the softmax computation, the exponential of the dot products is computed and accumulated in the corresponding softmax denominators:

```cpp
// Compute the denominator for the softmax normalization
torch::stable::Tensor denominator = torch::stable::empty({num_q});
denominator = torch::stable::zero_(denominator);
float* denominator_ptr = denominator.mutable_data_ptr<float>();

q_index = 0;
current_offset = 0;
for (int64_t i = 0; i < num_outputs; i++) {
    double val = std::exp(dot_product[i] - max);
    // result_ptr will hold the softmax numerators
    result_ptr[i] = val;
    // Handles the i indices for queries based on the offsets
    if (q_index < num_q && i >= offsets_ptr[q_index]) {
        q_index++;
        current_offset = offsets_ptr[q_index];
    }
    denominator_ptr[q_index] += val;
}
```

Finally the weighted sum of values is done in one pass as `y[i] += num[k] * V[k] / den[i]`:

```cpp
    q_index = 0;
    current_offset = 0;
    for (int64_t i = 0; i < num_outputs; i++) {
      // Move to the next q_index if we've passed the current offset
      if (q_index < num_q && i >= offsets_ptr[q_index]) {
        q_index++;
        current_offset = offsets_ptr[q_index];
      }
      int64_t out_index = indices_ptr[i];

      float dot_product = 0.0f;
      for (int64_t j = 0; j < v_dim; j++) {
        output_ptr[q_index * v_dim + j] += result_ptr[i] * v_ptr[out_index * v_dim + j] / denominator_ptr[q_index];
      }
    }
```

The full profiling [logs](./profiles/4_profile.log) and [results](./profiles/4_profile_results.md) are available in the `profiles/` folder:

|   seq_len |   d_model | sparsity   | dense_time   | sparse_2_time   | sparse_3_time   |
|-----------|-----------|------------|--------------|-----------------|-----------------|
|       512 |        32 | 95%        | 330.37 µs    | 559.26 µs       | 160.47 µs       |
|       512 |        32 | 99%        | 329.25 µs    | 148.94 µs       | 37.59 µs        |
|       512 |        64 | 95%        | 418.32 µs    | 784.11 µs       | 260.66 µs       |
|       512 |        64 | 99%        | 334.64 µs    | 189.98 µs       | 59.25 µs        |
|       512 |       128 | 95%        | 431.20 µs    | 1.41 ms         | 579.73 µs       |
|       512 |       128 | 99%        | 424.45 µs    | 269.57 µs       | 142.39 µs       |
|      1024 |        32 | 95%        | 1.39 ms      | 2.22 ms         | 668.28 µs       |
|      1024 |        32 | 99%        | 1.35 ms      | 413.51 µs       | 134.94 µs       |
|      1024 |        64 | 95%        | 1.45 ms      | 3.00 ms         | 1.09 ms         |
|      1024 |        64 | 99%        | 1.46 ms      | 713.60 µs       | 243.67 µs       |
|      1024 |       128 | 95%        | 1.81 ms      | 5.51 ms         | 2.26 ms         |
|      1024 |       128 | 99%        | 1.59 ms      | 1.17 ms         | 474.15 µs       |
|      2048 |        32 | 95%        | 5.25 ms      | 9.05 ms         | 2.65 ms         |
|      2048 |        32 | 99%        | 5.22 ms      | 1.91 ms         | 539.95 µs       |
|      2048 |        64 | 95%        | 5.60 ms      | 13.76 ms        | 4.61 ms         |
|      2048 |        64 | 99%        | 5.39 ms      | 2.60 ms         | 936.98 µs       |
|      2048 |       128 | 95%        | 6.54 ms      | 23.77 ms        | 9.46 ms         |
|      2048 |       128 | 99%        | 6.48 ms      | 4.53 ms         | 1.83 ms         |

This new implementation is 3-4x faster than the previous one, and 3 to 5 times faster than the masked attention baseline for very high levels of sparsity (>99%).

For small `d_model`, it is even faster than the masked attention at 95% sparsity:

|   seq_len |   d_model | sparsity   | dense_time   | sparse_3_time   |
|-----------|-----------|------------|--------------|-----------------|
|       512 |        32 | 95%        | 330.37 µs    | 160.47 µs       |
|      1024 |        32 | 95%        | 1.39 ms      | 668.28 µs       |
|      2048 |        32 | 95%        | 5.25 ms      | 2.65 ms         |
|       512 |        64 | 95%        | 418.32 µs    | 260.66 µs       |
|      1024 |        64 | 95%        | 1.45 ms      | 1.09 ms         |
|      2048 |        64 | 95%        | 5.60 ms      | 4.61 ms         |
|       512 |       128 | 95%        | 431.20 µs    | 579.73 µs       |
|      1024 |       128 | 95%        | 1.81 ms      | 2.26 ms         |
|      2048 |       128 | 95%        | 6.54 ms      | 9.46 ms         |

And for the smallest `d_model` it is competitive at 90% sparsity:

|   seq_len |   d_model | sparsity   | dense_time   | sparse_3_time   |
|-----------|-----------|------------|--------------|-----------------|
|       512 |        32 | 90%        | 324.73 µs    | 323.61 µs       |
|      1024 |        32 | 90%        | 1.45 ms      | 1.33 ms         |
|      2048 |        32 | 90%        | 5.49 ms      | 5.56 ms         |
|       512 |        64 | 90%        | 354.19 µs    | 528.68 µs       |
|      1024 |        64 | 90%        | 1.51 ms      | 2.14 ms         |
|      2048 |        64 | 90%        | 6.18 ms      | 9.41 ms         |

The good performance for small `d_model` could indicate that this implementation could be quite competitive with vectorized operations.

## Step 5 - Back to pure PyTorch with Sparse tensors

...

## Going further

### Vectorized operations

Could we modify the `sparse_matmul` or even `sparse_attn` to vectorize the operations (in particular the dot products computations)?

### Sparse attention weights

Could we modify the `sparse_matmul` so that it outputs a sparse tensor representation of $QK^T \odot \mathbf{M}$ and use PyTorch implementations for the rest?

### Multihead support

### Gradient implementation

### CUDA kernels