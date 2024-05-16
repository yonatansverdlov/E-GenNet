"""
Our model.
"""
import torch
import torch.nn as nn
from torch import Tensor

from .blocks import GenericBlock, IHash, PermutationIHash
from .init_blocks import InitialEmbedding
from .output_blocks import OutPutBlock
from easydict import EasyDict
from torch_geometric.data import Data
from .Abstarct_model import InvariantAbstractNet


class EquivariantGenericNet(nn.Module):
    def __init__(self, config: EasyDict, dims: list):
        """
        The full equivariant model.
        Args:
            config: The config file.
            dims: The list of dims.
        """
        super().__init__()
        # The task.
        self.task = config.task
        # The dimensions.
        self.num_dims = sum(config.type_config.task_specific[self.task].num_blocks)
        # The activation function.
        self.activation_fun = config.general_config.activation_fun
        # Initial block.
        self.initial_block = InitialEmbedding(
            config=config)
        self.dims = dims
        # All intermediate blocks.
        self.blocks = nn.ModuleList([GenericBlock(config=config, block_id=i, dims=self.dims) for i in
                                     range(self.num_dims)])

    def forward(self, data_obj: Data) -> Tensor:
        """
        Args:
            data_obj: The data object.
        Returns: The geometric information. Tensor of shape [data_obj,n,last_dim,dim]

        """
        # The feature embedding and the difference vectors.
        embed_pc_and_features, relative_point_cloud = self.initial_block(data_obj=data_obj)
        # The point cloud, adjacency matrix.
        point_cloud, adjacency_matrix = data_obj.pos, data_obj.adjacency_matrix
        # Initialize with zero geometric information.
        geometric_information = torch.zeros_like(point_cloud, device=point_cloud.device, dtype=torch.float).unsqueeze(
            -1)
        # Iterate over all blocks.
        for block in self.blocks:
            # Update the geometric information.
            geometric_information = block(geometric_information=geometric_information,
                                          adjacency_matrix=adjacency_matrix,
                                          relative_point_cloud=relative_point_cloud,
                                          embed_pc_and_features=embed_pc_and_features,
                                          )
        return geometric_information


class InvariantGenericNet(InvariantAbstractNet):
    """
    Invariant Generic Net.
    """
    def __init__(self, config: EasyDict):
        """
        The full invariant injective over S_n * O(3) model.
        Args:
            config: The config file.
        """
        super().__init__()
        # Dictionary.
        self.config = config
        # The task.
        self.task = config.task
        # The O_3 output dim.
        self.o_3_output_dim = config.type_config.task_specific[self.task].o_3_output_dim
        # The Sn dim.
        self.s_n_output_dim = config.type_config.task_specific[self.task].s_n_output_dim
        dims = config.type_config.task_specific[self.task].intermediate_dim
        # Num blocks.
        num_blocks = config.type_config.task_specific[self.task].num_blocks
        # The dims.
        self.dims = [1] + sum(
            [[dims[block_type] for _ in range(num_blocks[block_type])] for
             block_type in range(len(num_blocks))], [])
        # The equivariant model.
        self.equivariant_model = EquivariantGenericNet(config=config, dims=self.dims)
        # The orthogonal hash.
        self.orthogonal_invariant_hash = IHash(config=config)
        # The permutation invariant hash.
        self.permutation_invariant_hash = PermutationIHash(config=config)

    def forward(self, data_obj: Data) -> Tensor:
        """
         Args:
            data_obj: The data object.

        Returns: The model embedding.

        """
        # The equivariant feature.
        geometric_information = self.equivariant_model(data_obj=data_obj)
        # The S_n equivariant feature, O(3) invariant.                              
        geometric_information = self.orthogonal_invariant_hash(geometric_information)
        # The S_n,O(3) invariant.
        geometric_information = self.permutation_invariant_hash(geometric_information)
        # Return last embedding.
        return geometric_information


class GenericNet(nn.Module):
    def __init__(self, config: EasyDict):
        """
        The invariant embedding with MLP for classification tasks.
        Args:
            config: The config dictionary.
        """
        super().__init__()
        # The invariant embedding.
        self.feature_extractor = InvariantGenericNet(config=config)
        # The mlp network.
        self.mlp = OutPutBlock(config=config)
        # Global statistics for normalization.
        self.global_y_std = config.type_config.task_specific.std
        self.global_y_mean = config.type_config.task_specific.mean

    def forward(self, data_obj: Data) -> Tensor:
        """
        Compute forward.
        Args:
            data_obj: The data_obj

        Returns: The final classification dimension.

        """
        # The inputs.
        feature_output = self.feature_extractor(data_obj=data_obj)
        # Pass throw MLP.
        scores = self.mlp(feature_output)
        # Normalize.
        scores = (scores * self.global_y_std + self.global_y_mean).squeeze(-1)
        # Squeeze.
        return scores
