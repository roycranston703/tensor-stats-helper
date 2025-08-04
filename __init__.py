import numpy as np

def batch_mean(tensor, axis=0):
    """Mean over `axis`; works for numpy or torch tensors."""
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            return tensor.mean(dim=axis)
    except ModuleNotFoundError:
        pass
    return np.mean(tensor, axis=axis)
