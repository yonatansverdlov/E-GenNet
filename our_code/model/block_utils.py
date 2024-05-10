"""
The block kchain_utils.
"""
import torch.nn as nn


def return_all_activation_funcs(activation_key: str) -> nn.Module:
    """
    Returns the activation function.
    Args:
        activation_key: The activation function.

    Returns: The activation module.

    """
    activation_options = {'SiLU': nn.SiLU(), 'TanH': nn.Tanh(), 'ReLU': nn.ReLU(),
                          'LeakyReLU': nn.LeakyReLU(), 'Identity': nn.Identity()}
    return activation_options[activation_key]
