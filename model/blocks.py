"""
Body blocks of the model.
Dense and residual are taken from https://github.com/GraphPKU/DisGNN.
"""
import math

import torch
import torch.nn as nn
from easydict import EasyDict
from torch import Tensor
from .init_utils import he_orthogonal_init


class GenericBlock(nn.Module):
    """
    Generic block.
    """

    def __init__(self, config: EasyDict, dims: list, block_id: int):
        """
        Single equivariant block.
        Args:
            config: The config file.
            dims: The dims.
            block_id: The block id.
        """
        super().__init__()
        # The dims.
        self.dims = dims
        # Save config.
        self.config = config
        # The task.
        self.task = config.task
        # block id.
        self.block_id = block_id
        # Save the seed.
        self.seed = config.type_config.common_to_all_tasks.seed
        # Fix the seed.
        torch.manual_seed(self.seed)
        # Input channel.
        self.input_channels = self.dims[self.block_id]
        # Output channels.
        self.output_channels = self.dims[self.block_id + 1]
        # The theta parameters.
        self.theta = nn.Parameter(torch.empty(self.input_channels + 1, self.output_channels))
        # The alpha beta weight.
        self.alpha_beta_weight = nn.Linear(1, self.output_channels)
        # The weight for norm transform.
        self.norm_transformation = nn.Linear(self.input_channels + 1, 1)
        # Init params.
        self.init_parameters()
        # The alpha parameter.
        self.alpha = nn.Parameter(self.alpha_beta_weight.weight.view(1, 1, -1))
        # The beta.
        self.beta = nn.Parameter(self.alpha_beta_weight.bias.view(1, 1, -1))
        # The activation function.
        self.sigma = self.sigma = getattr(nn, config.general_config.activation_fun)()

        start, end = self.config.type_config.task_specific[self.task].block_ids
        self.use_all_norms = (self.config.type_config.task_specific[self.task].use_all_norms and
                              self.block_id in range(start, end))

    def init_parameters(self) -> None:
        """
        Initializes theta with xavier distribution.
        """
        # Fix seed.
        torch.manual_seed(self.seed)
        # Init theta.
        torch.nn.init.xavier_uniform_(self.theta.data)
        # Init alpha beta and norm transform.
        if self.config.type_config.task_specific[self.task].alpha_beta_init == 'xavier':
            torch.nn.init.xavier_uniform_(self.alpha_beta_weight.weight)
        if self.config.type_config.task_specific[self.task].norm_weight_init == 'xavier':
            torch.nn.init.xavier_uniform_(self.norm_transformation.weight)

    def compute_p_alpha_beta(self, norm_and_feature: Tensor) -> Tensor:
        """
        Computes the P_alpha_beta matrix.
        Args:
            norm_and_feature: The relative norm. Tensor of shape [batch_size,n,n]

        Returns: The P_alpha_beta matrix. Tensor of shape [batch_size,output_channels,n,n]

        """
        # Compute P_alpha_beta.
        p_alpha_beta = self.alpha * norm_and_feature
        # Add bias and activate sigma.
        p_alpha_beta = self.sigma(p_alpha_beta + self.beta)
        # return the output.
        return p_alpha_beta

    def forward(self, geometric_information: Tensor, adjacency_matrix: Tensor, relative_point_cloud: Tensor,
                embed_pc_and_features: Tensor) -> Tensor:
        """
        Forward of the block. From previous geometric information to new.
        Args:
            geometric_information: Current geometric information. Tensor of shape [batch,n,channels_in,dim].
            adjacency_matrix: The adjacency data. Tensor of shape [batch,n,n].
            relative_point_cloud: The relative point cloud. tensor of shape [batch,n,n,3].
            embed_pc_and_features: The point cloud norm_and_feature. Tensor of shape [batch,n,n,emb_size].

        Returns: The new geometric information. Tensor of shape [data_obj,n,channels_out,dim].

        """

        # Compute geometric_information * theta2.
        geometric_information_double = geometric_information.unsqueeze(1).expand(-1, geometric_information.size(1), -1,
                                                                                 -1, -1)
        # Cat geometric_information anf x_i - x_j
        embed_x_and_g = torch.cat([geometric_information_double, relative_point_cloud], dim=-1)
        # Compute norm transform.
        if self.block_id != 0 and self.use_all_norms:
            embed_pc_and_features = self.norm_transformation(torch.norm(embed_x_and_g, dim=-2)) + embed_pc_and_features
        # Compute P_alpha_beta.
        p_alpha_beta = self.compute_p_alpha_beta(norm_and_feature=embed_pc_and_features)
        # Multiply by theta.
        embed_x_and_g_with_adjacency = embed_x_and_g @ self.theta
        # Compute P_alpha_beta * geometric_information * theta.
        new_geometric_information = (
                (embed_x_and_g_with_adjacency * p_alpha_beta.unsqueeze(-2)) * adjacency_matrix).sum(2)
        return new_geometric_information


class IHash(nn.Module):
    """
    I-Hash.
    """

    def __init__(self, config: EasyDict):
        """
        O(3) orbit injective layer.
        Args:
            config: The config dictionary.
        """
        super().__init__()
        self.config = config
        self.seed = config.type_config.common_to_all_tasks.seed
        torch.manual_seed(self.seed)
        self.task = config.task
        # dims.
        self.num_blocks = config.type_config.task_specific[self.task].num_blocks
        self.num_different_blocks = len(self.num_blocks)
        self.intermediate_dim = config.type_config.task_specific[self.task].intermediate_dim
        self.dims = config.type_config.task_specific[self.task].intermediate_dim
        self.dims = sum(
            [[self.dims[block_type] for _ in range(self.num_blocks[block_type])] for
             block_type in range(len(self.num_blocks))], [])
        self.input_dim = self.dims[-1]
        self.output_dim = config.type_config.task_specific[self.task].o_3_output_dim
        self.orthogonal_weight = nn.Linear(self.output_dim, self.input_dim).weight
        self.init_parameters()

    def init_parameters(self) -> None:
        """
        Inits params.
        """
        # Fix seed.
        torch.manual_seed(self.seed)
        # Init.
        torch.nn.init.xavier_uniform_(self.orthogonal_weight)

    def forward(self, equivariant_feature: Tensor) -> Tensor:
        """
        Forward The invariant hash function.
        Args:
            equivariant_feature: The equivariant feature. Tensor of shape [data_obj,n,channels_in,3]

        Returns: Tensor of shape [data_obj,n,channels_out]

        """
        # Compute the geometric_information * O.
        output = equivariant_feature @ self.orthogonal_weight
        # Compute norm.
        output = torch.norm(output, dim=-2)
        # Compute the norm_and_feature of the output.
        return output


class PermutationIHash(nn.Module):
    """
    Permutation Hash.
    """

    def __init__(self, config: EasyDict):
        """
        Permutation invariant hash.
        Args:
            config: The config dictionary.
        """
        super().__init__()
        # The task.
        self.task = config.task
        # The seed.
        self.seed = config.type_config.common_to_all_tasks.seed
        # Fix seed.
        torch.manual_seed(self.seed)
        # The input dim.
        self.input_dim = config.type_config.task_specific[self.task].o_3_output_dim
        # The output dim.
        self.output_dim = config.type_config.task_specific[self.task].s_n_output_dim
        # The alpha.
        self.alpha = nn.Parameter(torch.rand(self.output_dim, self.input_dim))
        # The bias.
        self.beta = nn.Parameter(torch.rand(self.output_dim))
        # The activation function.
        self.sigma = getattr(nn, config.general_config.activation_fun, nn.ReLU)()
        # Init params.
        self.init_parameters()

    def init_parameters(self):
        """
        Inits params.
        """
        torch.manual_seed(self.seed)
        torch.nn.init.xavier_uniform_(self.alpha)
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.alpha)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.beta, -bound, bound)

    def forward(self, c_i: Tensor) -> Tensor:
        """
        Sn invariant hash forward.
        Args:
            c_i: All invariant c_i features. Tensor of shape [data_obj,n,c_in]

        Returns: Sn invariant feature. Tensor of shape [data_obj,c_out]

        """
        # Compute sum(sigma(A * x + b))
        output = self.sigma(c_i @ self.alpha + self.beta).sum(dim=1)
        return output


class Dense(nn.Module):
    """
    Dense layer.
    """

    def __init__(
            self,
            seed: int,
            in_features: int,
            out_features: int,
            bias: bool = True,
            activation_fn: torch.nn.Module = nn.Identity(),
    ):
        """
        The dense block.
        Args:
            seed: The seed for initialization.
            in_features: The in feature.
            out_features: The out feature.
            bias: Whether to use bias.
            activation_fn: The activation function.
        """
        super().__init__()
        assert activation_fn is not None
        self.seed = seed
        torch.manual_seed(self.seed)
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.reset_parameters()
        self.weight = self.linear.weight
        self.bias = self.linear.bias
        self._activation = activation_fn

    def reset_parameters(self):
        """
        Resets the parameters.
        """
        if not self.in_features == 1:
            he_orthogonal_init(self.linear.weight, seed=self.seed)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the Dense forward.
        Args:
            x: The input

        Returns: The block output.

        """
        x = self.linear(x)
        x = self._activation(x)
        return x


class Residual(nn.Module):
    """
    Residual block.
    """

    def __init__(
            self, seed: int,
            mlp_num: int,
            hidden_dim: int,
            activation_fn: torch.nn.Module = None,
            bias: bool = True,
            add_end_activation: bool = True,
    ):
        """

        Args:
            seed: The seed for initialization.
            mlp_num: The number of mlp blocks.
            activation_fn: The activation function.
            bias: Whether to use bias.
            add_end_activation: Whether to add activation at last.
        """
        super().__init__()
        assert mlp_num > 0
        self.seed = seed
        torch.manual_seed(self.seed)
        end_activation_fn = activation_fn if add_end_activation else nn.Identity()

        self.mlp = nn.Sequential(
            *[
                Dense(seed=seed, in_features=hidden_dim, out_features=hidden_dim, bias=bias,
                      activation_fn=activation_fn)
                for _ in range(mlp_num - 1)
            ],
            Dense(in_features=hidden_dim, out_features=hidden_dim, seed=seed, bias=bias,
                  activation_fn=end_activation_fn)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes residual forward.
        Args:
            x: The input.

        Returns: The block forward.

        """
        return self.mlp(x) + x
