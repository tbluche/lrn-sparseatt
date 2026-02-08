import torch
from torch import Tensor

__all__ = ["sparse_matmul"]


def sparse_matmul(q: Tensor, k: Tensor, indices: Tensor) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.extension_cpp.sparse_matmul.default(q, k, indices)


def sparse_matmul_vo(
    q: Tensor,
    k: Tensor,
    k_indices: Tensor,
    q_offsets: Tensor,
) -> Tensor:
    """Performs a * b + c in an efficient fused kernel, using separate k_indices and q_offsets"""
    return torch.ops.extension_cpp.sparse_matmul_vo.default(q, k, k_indices, q_offsets)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
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


# def _backward(ctx, grad):
#     a, b = ctx.saved_tensors
#     grad_a, grad_b = None, None
#     if ctx.needs_input_grad[0]:
#         grad_a = torch.ops.extension_cpp.mymul.default(grad, b)
#     if ctx.needs_input_grad[1]:
#         grad_b = torch.ops.extension_cpp.mymul.default(grad, a)
#     return grad_a, grad_b, None


# def _setup_context(ctx, inputs, output):
#     a, b, c = inputs
#     saved_a, saved_b = None, None
#     if ctx.needs_input_grad[0]:
#         saved_b = b
#     if ctx.needs_input_grad[1]:
#         saved_a = a
#     ctx.save_for_backward(saved_a, saved_b)


# # This adds training support for the operator. You must provide us
# # the backward formula for the operator and a `setup_context` function
# # to save values to be used in the backward.
# torch.library.register_autograd(
#     "extension_cpp::mymuladd", _backward, setup_context=_setup_context
# )
