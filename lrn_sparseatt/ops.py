import torch
from torch import Tensor

__all__ = ["sparse_matmul"]


def sparse_matmul(q: Tensor, k: Tensor, indices: Tensor) -> Tensor:
    """Performs QK^T where the output is sparse and only computed at the locations specified by indices.

    Args:
        q: Query tensor of shape [T, D]
        k: Key tensor of shape [T, D]
        indices: Tensor of shape [M, 2] containing the indices of the True values in the attention mask.
                 Each index pair (i, j) corresponds to a position in the output where the attention weight should be computed.

    Returns:
        Tensor of shape [M] containing the computed attention weights at the specified indices.
    """
    return torch.ops.extension_cpp.sparse_matmul.default(q, k, indices)


def sparse_matmul_vo(
    q: Tensor,
    k: Tensor,
    k_indices: Tensor,
    q_offsets: Tensor,
) -> Tensor:
    """Performs QK^T where the output is sparse and only computed at the locations specified by k_indices and q_offsets.

    This is a variant of sparse_matmul that uses a different format for the indices, where k_indices contains
    the key indices for all True values in the mask, and q_offsets contains the offsets for each query position
    in k_indices.

    Args:
        q: Query tensor of shape [T, D]
        k: Key tensor of shape [T, D]
        k_indices: Tensor of shape [M] containing the key indices for all True values in the mask.
        q_offsets: Tensor of shape [T] containing the ending offsets for each query position in k_indices.

    Returns:
        Tensor of shape [M] containing the computed attention weights at the specified indices.
    """
    return torch.ops.extension_cpp.sparse_matmul_vo.default(q, k, k_indices, q_offsets)


def sparse_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    k_indices: Tensor,
    q_offsets: Tensor,
    factor: float = 1.0,
) -> Tensor:
    """Performs the full sparse attention computation in a single fused kernel.

    This function computes the attention output for the query, key, and value tensors at the locations specified
    by k_indices and q_offsets, and applies the scaling factor to the attention weights.

    Args:
        q: Query tensor of shape [T, D]
        k: Key tensor of shape [T, D]
        v: Value tensor of shape [T, D]
        k_indices: Tensor of shape [M] containing the key indices for all True values in the mask.
        q_offsets: Tensor of shape [T] containing the ending offsets for each query position in k_indices.
        factor: Scaling factor to apply to the attention weights (default is 1.0).

    Returns:
        Tensor of shape [T, D] containing the computed attention output at the specified indices.
    """
    assert factor > 0.0, "Expected factor to be positive, got {}".format(factor)
    return torch.ops.extension_cpp.sparse_attn.default(
        q, k, v, k_indices, q_offsets, factor
    )


@torch.library.register_fake("extension_cpp::sparse_matmul")
def _(q, k, indices):
    torch._check(q.shape[1] == k.shape[1])
    torch._check(q.dtype == torch.float)
    torch._check(k.dtype == torch.float)
    torch._check(q.device == k.device)
    torch._check(indices.dtype == torch.int64)
    torch._check(indices.device == q.device)
    return torch.empty(indices.shape[0], device=q.device, dtype=q.dtype)


@torch.library.register_fake("extension_cpp::sparse_matmul_vo")
def _(q, k, k_indices, q_offsets):
    torch._check(q.shape[1] == k.shape[1])
    torch._check(q.dtype == torch.float)
    torch._check(k.dtype == torch.float)
    torch._check(q.device == k.device)
    torch._check(k_indices.dtype == torch.int64)
    torch._check(k_indices.device == q.device)
    torch._check(q_offsets.dtype == torch.int64)
    torch._check(q_offsets.device == q.device)
    return torch.empty(k_indices.shape[0], device=q.device, dtype=q.dtype)


@torch.library.register_fake("extension_cpp::sparse_attn")
def _(q, k, v, k_indices, q_offsets, factor):
    torch._check(q.shape[1] == k.shape[1])
    torch._check(q.shape[1] == v.shape[1])
    torch._check(q.dtype == torch.float)
    torch._check(k.dtype == torch.float)
    torch._check(v.dtype == torch.float)
    torch._check(q.device == k.device)
    torch._check(q.device == v.device)
    torch._check(k_indices.dtype == torch.int64)
    torch._check(k_indices.device == q.device)
    torch._check(q_offsets.dtype == torch.int64)
    torch._check(q_offsets.device == q.device)
    torch._check(isinstance(factor, float))
    torch._check(factor > 0.0)
    return torch.empty(q.shape[0], v.shape[1], device=q.device, dtype=q.dtype)


# TODO: implement backward functions for the above ops to enable autograd support
