"""
Weight initialization.
Taken from https://github.com/GraphPKU/DisGNN.
"""
import torch
from torch import Tensor


def _standardize(weight: Tensor) -> Tensor:
    """
    Normalize weight.
     Makes sure that Var(W) = 1 and E[W] = 0
    Args:
        weight: The weight.

    Returns: Normalized weight.

    """
    eps = 1e-6

    if len(weight.shape) == 3:
        axis = [0, 1]  # last dimension is output dimension
    else:
        axis = 1

    var, mean = torch.var_mean(weight, dim=axis, keepdim=True)
    kernel = (weight - mean) / (var + eps) ** 0.5
    return kernel


def he_orthogonal_init(weight: Tensor, seed: int) -> Tensor:
    """
    Generate a weight matrix with variance according to his initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are de-correlated.
    (stated by e.geometric_information. "Reducing overfitting in deep networks by de-correlating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    Args:
        weight: The weight.
        seed: The seed.

    Returns: The new weight.

    """
    torch.manual_seed(seed=seed)
    tensor = torch.nn.init.orthogonal_(weight)

    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor
