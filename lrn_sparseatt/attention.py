import torch
from torch.nested import nested_tensor
from .ops import sparse_matmul


def full_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
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
    # q, k, v have shape [H, T, D]
    # attn_mask has shape [T, T] or [T, T]
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
    # q, k, v have shape [H, T, D]
    # attn_mask has shape [M, 2] where M is the number of True values in the original mask
    n_heads, seq_len, head_dim = q.size()

    q_indices = indices[:, 0].flatten()
    kv_indices = indices[:, 1].flatten()

    qs_indsel = q.index_select(1, q_indices).view(n_heads, -1, head_dim)
    ks_indsel = k.index_select(1, kv_indices).view(n_heads, -1, head_dim)

    attn_weights = (qs_indsel * ks_indsel).sum(dim=-1) / (head_dim**0.5)

    num = (attn_weights - attn_weights.max()).exp()
    den = torch.index_add(torch.zeros((n_heads, seq_len)), 1, q_indices, num)
    den = den.index_select(1, q_indices)
    attn_weights = num / den

    vs_indsel = v.index_select(1, kv_indices).view(n_heads, -1, head_dim)
    weighted_vs = attn_weights.unsqueeze(-1) * vs_indsel
    out = torch.zeros((n_heads, seq_len, head_dim))
    out.index_add_(1, q_indices, weighted_vs)

    return out


def sparse_attention_1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    # q, k, v have shape [H, T, D]
    # attn_mask has shape [M, 2] where M is the number of True values in the original mask
    n_heads, seq_len, head_dim = q.size()
    assert n_heads == 1, "This implementation only supports n_heads=1 for simplicity"
    q = q.squeeze(0)  # shape [T, D]
    k = k.squeeze(0)  # shape [T, D]

    attn_weights = sparse_matmul(q, k, indices) / (head_dim**0.5)
    attn_weights = attn_weights.unsqueeze(0)

    q_indices = indices[:, 0].flatten()
    kv_indices = indices[:, 1].flatten()
    num = (attn_weights - attn_weights.max()).exp()
    den = torch.index_add(torch.zeros((n_heads, seq_len)), 1, q_indices, num)
    den = den.index_select(1, q_indices)
    attn_weights = num / den

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
