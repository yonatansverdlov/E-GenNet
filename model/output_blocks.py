"""
Output MLP. After the geometric layers are applied, and we have rotation,
permutation invariant feature, we learn the task from it.
Taken from https://github.com/GraphPKU/DisGNN.
"""
import torch.nn as nn
from torch import Tensor

from .blocks import Dense, Residual

from easydict import EasyDict


class OutPutBlock(nn.Module):
    def __init__(self, config: EasyDict):
        """
        Output MLP.
        Args:
            config: The config dictionary.
        """
        super().__init__()
        # Task.
        self.task = config.task
        # Type.
        self.type = config.type
        # The output dim.
        self.output_dim = config.type_config.task_specific[self.task].s_n_output_dim
        self.num_prediction_dict = {'k_chain': 2, 'Drugs': 1, 'Kraken': 1, 'BDE': 1,'Hard':2}
        # The prediction dimension.
        self.num_pred = self.num_prediction_dict[self.type]
        # The seed.
        self.seed = config.type_config.common_to_all_tasks.seed
        # The activation function.
        self.activation_fun =  getattr(nn, config.general_config.mlp_activation)()
        # The mlp output net is composed of three residual layers and one Dense layer.
        self.output_mlp = nn.Sequential(
            Residual(mlp_num=2, hidden_dim=self.output_dim, activation_fn=self.activation_fun, seed=self.seed),
            Residual(mlp_num=2, hidden_dim=self.output_dim, activation_fn=self.activation_fun, seed=self.seed),
            Residual(mlp_num=2, hidden_dim=self.output_dim, activation_fn=self.activation_fun, add_end_activation=False,
                     seed=self.seed),
            Dense(in_features=self.output_dim, out_features=self.num_pred, bias=False, seed=self.seed))

    def forward(self, final_feature: Tensor) -> Tensor:
        """

        Args:
            final_feature: The final feature.

        Returns: The final feature for regression/classification.

        """
        return self.output_mlp(final_feature)
