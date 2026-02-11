import torch
from torch.nested import nested_tensor
from .ops import sparse_matmul, sparse_matmul_vo, sparse_attn


def full_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Compute full attention without any masking.

    Args:
        q: Query tensor of shape [H, T, D]
        k: Key tensor of shape [H, T, D]
        v: Value tensor of shape [H, T, D]

    Returns:
        Output tensor of shape [H, T, D]
    """
    head_dim = q.size(2)
    attn_weights: torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)


def masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked attention (baseline implementation).

    Args:
        q: Query tensor of shape [H, T, D]
        k: Key tensor of shape [H, T, D]
        v: Value tensor of shape [H, T, D]
        attn_mask: Boolean tensor of shape [T, T] where True indicates positions that can attend to each other.

    Returns:
        Output tensor of shape [H, T, D]
    """
    # q, k, v have shape [H, T, D]
    # attn_mask has shape [T, T]
    head_dim = q.size(2)

    attn_weights: torch.Tensor = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
    # attn_weights has shape [H, T, T]

    # attn_mask shape should be broadcastable to attn_weights shape
    attn_mask = attn_mask.unsqueeze(0)  # shape [1, T, T]
    attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch implementation of sparse attention using index selection.

    The idea is to compute attention only for the pairs of positions specified in `indices`,
    which has shape [M, 2] where M is the number of True values in the original mask.
    Each row of `indices` contains a pair of (query_index, key_index) that should attend to each other.

    Args:
        q: Query tensor of shape [H, T, D]
        k: Key tensor of shape [H, T, D]
        v: Value tensor of shape [H, T, D]
        indices: Long tensor of shape [M, 2] where each row is (query_index, key_index) indicating
          which positions should attend to each other.

    Returns:
        Output tensor of shape [H, T, D]
    """
    # q, k, v have shape [H, T, D]
    # attn_mask has shape [M, 2] where M is the number of True values in the original mask
    n_heads, seq_len, head_dim = q.size()

    q_indices = indices[:, 0].flatten()
    kv_indices = indices[:, 1].flatten()

    # Compute the attention weights for the specified pairs of positions using index selection.
    # Use index_select to gather the relevant query and key vectors for the pairs of positions in indices,
    # and then compute their dot product.
    qs_indsel = q.index_select(1, q_indices).view(n_heads, -1, head_dim)
    ks_indsel = k.index_select(1, kv_indices).view(n_heads, -1, head_dim)
    attn_weights = (qs_indsel * ks_indsel).sum(dim=-1) / (head_dim**0.5)

    # Compute the softmax normalization for the attention weights using index_add
    # to sum the exponentials for each query position.
    num = (attn_weights - attn_weights.max()).exp()
    den = torch.index_add(torch.zeros((n_heads, seq_len)), 1, q_indices, num)
    den = den.index_select(1, q_indices)
    attn_weights = num / den

    # Now compute the weighted sum of the value vectors for each query position using index_add again.
    vs_indsel = v.index_select(1, kv_indices).view(n_heads, -1, head_dim)
    weighted_vs = attn_weights.unsqueeze(-1) * vs_indsel
    out = torch.zeros((n_heads, seq_len, head_dim))
    out.index_add_(1, q_indices, weighted_vs)

    return out


def sparse_attention_nested(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: nested_tensor,
) -> torch.Tensor:
    """Pure PyTorch implementation of sparse attention using nested tensors for variable-length attention patterns.

    The `indices` argument is a nested tensor where each sublist corresponds to the key indices that each query
    position attends to.

    Args:
        q: Query tensor of shape [H, T, D]
        k: Key tensor of shape [H, T, D]
        v: Value tensor of shape [H, T, D]
        indices: Nested tensor where the outer list has length T and each inner list contains the column indices
           of the True values in the original mask for the corresponding query position.

    Returns:
        Output tensor of shape [H, T, D]
    """
    n_heads, _, head_dim = q.size()

    return torch.cat(
        [
            full_attention(
                q[:, qi : qi + 1, :],
                k.index_select(1, comp).view(n_heads, -1, head_dim),
                v.index_select(1, comp).view(n_heads, -1, head_dim),
            )
            for qi, comp in enumerate(indices)
        ],
        dim=1,
    )


def sparse_attention_masked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked attention using torch.sparse for the query-key multiplication.

    Args:
        q: Query tensor of shape [H, T, D]
        k: Key tensor of shape [H, T, D]
        v: Value tensor of shape [H, T, D]
        attn_mask: Boolean tensor of shape [T, T] where True indicates positions that can attend to each other.

    Returns:
        Output tensor of shape [H, T, D]
    """
    # q, k, v have shape [H, T, D]
    # attn_mask has shape [T, T]
    head_dim = q.size(2)
    sp_mask = attn_mask.to_sparse_csr().to(torch.float) * 0

    sp_attn_weights = torch.sparse.sampled_addmm(
        sp_mask, q, k.transpose(-2, -1), beta=0.0, alpha=1.0 / (head_dim**0.5)
    )
    attn_weights = sp_attn_weights.to_dense()
    # attn_weights has shape [H, T, T]

    # attn_mask shape should be broadcastable to attn_weights shape
    attn_mask = attn_mask.unsqueeze(0)  # shape [1, T, T]
    attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)


def sparse_attention_1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Implementation of sparse attention using the custom sparse_matmul operator for computing attention weights.

    The sparse_matmul operator computes the attention weights only for the pairs of positions specified in `indices`,
    which has shape [M, 2] where M is the number of True values in the original mask.
    Each row of `indices` contains a pair of (query_index, key_index) that should attend to each other.

    Args:
        q: Query tensor of shape [1, T, D]
        k: Key tensor of shape [1, T, D]
        v: Value tensor of shape [1, T, D]
        indices: Long tensor of shape [M, 2] where each row is (query_index, key_index) indicating
          which positions should attend to each other.

    Returns:
        Output tensor of shape [1, T, D].
    """
    # q, k, v have shape [1, T, D]
    # indices has shape [M, 2] where M is the number of True values in the original mask
    n_heads, seq_len, head_dim = q.size()
    assert n_heads == 1, "This implementation only supports n_heads=1 for simplicity"
    q = q.squeeze(0)  # shape [T, D]
    k = k.squeeze(0)  # shape [T, D]

    # Compute the attention weights for the specified pairs of positions using the custom
    # sparse_matmul operator.
    attn_weights = sparse_matmul(q, k, indices) / (head_dim**0.5)
    attn_weights = attn_weights.unsqueeze(0)

    # Compute the softmax normalization for the attention weights using index_add
    # to sum the exponentials for each query position.
    q_indices = indices[:, 0].flatten()
    kv_indices = indices[:, 1].flatten()
    num = (attn_weights - attn_weights.max()).exp()
    den = torch.index_add(torch.zeros((n_heads, seq_len)), 1, q_indices, num)
    den = den.index_select(1, q_indices)
    attn_weights = num / den

    # Now compute the weighted sum of the value vectors for each query position using index_add again.
    vs_indsel = v.index_select(1, kv_indices).view(n_heads, -1, head_dim)
    weighted_vs = attn_weights.unsqueeze(-1) * vs_indsel
    out = torch.zeros((n_heads, seq_len, head_dim))
    out.index_add_(1, q_indices, weighted_vs)

    return out


def sparse_attention_2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    k_indices: torch.Tensor,
    q_offsets: torch.Tensor,
) -> torch.Tensor:
    """Implementation of sparse attention using the custom sparse_matmul_vo operator for computing attention weights.

    The sparse_matmul_vo operator computes the attention weights only for the pairs of positions specified by
    k_indices and q_offsets, which are derived from the original boolean mask. The k_indices tensor contains the
    key indices for all True values in the mask, and the q_offsets tensor contains the offsets for each query
    position in k_indices.

    Args:
        q: Query tensor of shape [1, T, D]
        k: Key tensor of shape [1, T, D]
        v: Value tensor of shape [1, T, D]
        indices: Long tensor of shape [M, 2] where each row is (query_index, key_index) indicating which positions
            should attend to each other.
        k_indices: Long tensor of shape [M] containing the key indices for all True values in the mask.
        q_offsets: Long tensor of shape [T] containing the ending offsets for each query position in k_indices.

    Returns:
        Output tensor of shape [1, T, D].
    """
    # q, k, v have shape [H, T, D]
    # attn_mask has shape [M, 2] where M is the number of True values in the original mask
    n_heads, seq_len, head_dim = q.size()
    assert n_heads == 1, "This implementation only supports n_heads=1 for simplicity"
    q = q.squeeze(0)  # shape [T, D]
    k = k.squeeze(0)  # shape [T, D]

    # Compute the attention weights for the specified pairs of positions using the custom
    # sparse_matmul_vo operator.
    attn_weights = sparse_matmul_vo(q, k, k_indices, q_offsets) / (head_dim**0.5)
    attn_weights = attn_weights.unsqueeze(0)

    # Compute the softmax normalization for the attention weights using index_add
    # to sum the exponentials for each query position.
    q_indices = indices[:, 0].flatten()
    kv_indices = indices[:, 1].flatten()
    num = (attn_weights - attn_weights.max()).exp()
    den = torch.index_add(torch.zeros((n_heads, seq_len)), 1, q_indices, num)
    den = den.index_select(1, q_indices)
    attn_weights = num / den

    # Now compute the weighted sum of the value vectors for each query position using index_add again.
    vs_indsel = v.index_select(1, kv_indices).view(n_heads, -1, head_dim)
    weighted_vs = attn_weights.unsqueeze(-1) * vs_indsel
    out = torch.zeros((n_heads, seq_len, head_dim))
    out.index_add_(1, q_indices, weighted_vs)

    return out


def sparse_attention_3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_indices: torch.Tensor,
    q_offsets: torch.Tensor,
) -> torch.Tensor:
    """Full C++ implementation of sparse attention using the custom sparse_attn operator that computes the
    entire attention output in a single fused kernel.

    The sparse_attn operator computes the attention output for the query, key, and value tensors at the locations specified
    by k_indices and q_offsets, and applies the scaling factor to the attention weights.

    Args:
        q: Query tensor of shape [1, T, D]
        k: Key tensor of shape [1, T, D]
        v: Value tensor of shape [1, T, D]
        k_indices: Long tensor of shape [M] containing the key indices for all True values in the mask.
        q_offsets: Long tensor of shape [T] containing the ending offsets for each query position in k_indices.

    Returns:
        Output tensor of shape [1, T, D].
    """
    n_heads, _, head_dim = q.size()
    assert n_heads == 1, "This implementation only supports n_heads=1 for simplicity"
    q = q.squeeze(0)  # shape [T, D]
    k = k.squeeze(0)  # shape [T, D]
    v = v.squeeze(0)  # shape [T, D]

    out = sparse_attn(q, k, v, k_indices, q_offsets, head_dim**0.5)
    return out.unsqueeze(0)
