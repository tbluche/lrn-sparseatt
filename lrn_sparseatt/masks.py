from einops import rearrange
import torch
from torch.nested import nested_tensor


def boolean_mask_to_indices(mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a boolean mask of shape (T, T) to indices of shape (M, 2) where M is the number of True values in the mask.
    Each index pair (i, j) in the output corresponds to a True value in the input mask at position (i, j).

    Args:
        mask (torch.Tensor): A boolean tensor of shape (T, T).
    Returns:
        torch.Tensor: An integer tensor of shape (M, 2) containing the indices of the True values in the input mask.
    """
    indices = torch.nonzero(mask, as_tuple=False)
    # shape (N, 2) where N is the number of True values
    # indices columns are ( i, j)
    return indices


def boolean_mask_to_nested_indices(mask: torch.Tensor) -> nested_tensor:
    """
    Convert a boolean mask of shape (T, T) to a nested list of indices where each sublist corresponds
    to the True values in each row of the mask.

    Args:
        mask (torch.Tensor): A boolean tensor of shape (T, T).
    Returns:
        nested_tensor: A nested tensor where the outer list has length T and each inner list contains
          the column indices of the True values in the corresponding row of the input mask.
    """
    nested_indices = []
    for i in range(mask.size(0)):
        true_indices = torch.nonzero(mask[i], as_tuple=False).flatten()
        nested_indices.append(true_indices)
    nested_indices = nested_tensor(nested_indices, layout=torch.jagged)
    return nested_indices


def boolean_mask_to_jagged_indices(
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a boolean mask of shape (T, T) to jagged indices represented by a pair of tensors: (values, offsets).
    The 'values' tensor contains the column indices of the True values in the input mask, and the 'offsets'
    tensor contains the ending index of each row in the 'values' tensor

    Args:
        mask (torch.Tensor): A boolean tensor of shape (T, T).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A pair of tensors (values, offsets) where 'values' is a
          1D tensor containing the column indices of the True values in the input mask, and 'offsets'
          is a 1D tensor containing the ending index of each row in the 'values' tensor.
    """
    values = []
    offsets = []
    last_offset = 0
    for i in range(mask.size(0)):
        true_indices = torch.nonzero(mask[i], as_tuple=False).flatten()
        values.append(true_indices)
        offsets.append(last_offset + len(true_indices))
        last_offset = offsets[-1]
    values = torch.cat(values)
    offsets = torch.tensor(offsets, device=mask.device)
    return values, offsets


class AttentionMask:
    """Abstraction of an attention mask.

    The attention mask can be represented in different formats (boolean tensor, indices, etc.)
    and can be converted between them.
    """

    def as_tensor(self, seq_len: int) -> torch.Tensor:
        raise NotImplementedError

    def to_indices(self) -> torch.Tensor:
        raise NotImplementedError


class BooleanMask(AttentionMask):
    """Implementation of a boolean attention mask.

    The mask is represented as a boolean tensor of shape (T, T) where True indicates
    positions that can attend to each other.
    """

    def __init__(self, mask: torch.Tensor):
        super().__init__()
        assert mask.dtype == torch.bool, "mask must be a boolean tensor"
        self.mask = mask

    def to_indices(self) -> torch.Tensor:
        return boolean_mask_to_indices(self.mask)

    def as_tensor(self, seq_len: int) -> torch.Tensor:
        return self.mask

    @staticmethod
    def causal(
        seq_len: int, device: torch.device = torch.device("cpu")
    ) -> "BooleanMask":
        """Create a causal attention mask of shape (T, T) where positions can only attend
        to previous positions (including themselves)."""
        mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1)
        mask = mask.bool()
        return BooleanMask(mask)

    @staticmethod
    def blockwise(
        seq_len: int, block_size: int, device: torch.device = torch.device("cpu")
    ) -> "BooleanMask":
        """Create a blockwise attention mask of shape (T, T) where positions can only attend
        to positions within the same block of size block_size."""
        mask = torch.ones((seq_len, seq_len), device=device).bool()
        for i in range(0, seq_len, block_size):
            mask[i : i + block_size, : i + block_size] = False
        return BooleanMask(mask)

    @staticmethod
    def random(
        seq_len: int, sparsity: float, device: torch.device = torch.device("cpu")
    ) -> "BooleanMask":
        """Create a random attention mask of shape (T, T) where each position attends
        to a random subset of other positions."""
        # The generated mask has shape [T, T] and will have between 1 and T True values per row, randomly distributed.
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
        for i in range(seq_len):
            num_true = max(1, int((1.0 - sparsity) * seq_len))
            true_indices = torch.randperm(seq_len)[:num_true]
            mask[i, true_indices] = True
        return BooleanMask(mask)

    def sparsity(self) -> float:
        """Compute the sparsity of the mask, defined as the fraction of False values in the mask."""
        total_elements = self.mask.numel()
        true_elements = self.mask.sum().item()
        return 1.0 - (true_elements / total_elements)


class SparseMask(AttentionMask):

    def __init__(self, indices: torch.Tensor):
        super().__init__()
        assert (
            indices.dim() == 2 and indices.size(-1) == 2
        ), "indices must have shape (M, 2)"
        self.indices = indices

    def to_boolean_mask(self, seq_len: int) -> BooleanMask:
        mask = torch.zeros(
            (seq_len, seq_len),
            dtype=torch.bool,
            device=self.indices.device,
        )
        mask = rearrange(mask, "i j -> (i j)", i=seq_len, j=seq_len)
        i = self.indices[:, 0]
        j = self.indices[:, 1]
        mask.scatter_(0, i * seq_len + j, True)
        mask = rearrange(mask, "(i j) -> i j", i=seq_len, j=seq_len)
        return BooleanMask(mask)

    def as_tensor(self, seq_len: int) -> torch.Tensor:
        return self.to_boolean_mask(seq_len).as_tensor(seq_len)

    def to_indices(self) -> torch.Tensor:
        return self.indices
