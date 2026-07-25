import torch


def cumsum_exclude_current(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    return torch.cumsum(tensor, dim=dim) - tensor
