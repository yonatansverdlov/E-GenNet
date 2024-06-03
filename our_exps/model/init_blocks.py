"""
Initialization blocks.
"""
import torch
import torch.nn as nn
from .blocks import Dense, Residual
from torch import Tensor
import math
from torch_geometric.data import Data
from easydict import EasyDict
from typing import Tuple
from .block_utils import return_all_activation_funcs


class SincRadialBasis(nn.Module):
    # Sinc radial basis.
    def __init__(self, seed: int, num_rbf: int, rbound_upper: int, rbf_trainable: bool = False):
        """

        Args:
            seed: The random seed.
            num_rbf: The num of rbf.
            rbound_upper: The upper bound.
            rbf_trainable: if trainable.
        """
        super().__init__()
        torch.manual_seed(seed)
        if rbf_trainable:
            self.register_parameter("n", nn.parameter.Parameter(torch.arange(
                1, num_rbf + 1, dtype=torch.float).unsqueeze(0) / rbound_upper))
        else:
            self.register_buffer("n", torch.arange(
                1, num_rbf + 1, dtype=torch.float).unsqueeze(0) / rbound_upper)

    def forward(self, norm_pos_difference: Tensor) -> Tensor:
        """
        Computes the embedded distance matrix.
        Args:
            norm_pos_difference: The distance matrix.

        Returns: The embedded distance matrix/

        """
        n = self.n
        output = math.pi * n * torch.sinc(norm_pos_difference * n)
        return output


class NormEmbedding(nn.Module):
    def __init__(self, config: EasyDict):
        """
        Embed the Norm information.
        Args:
            config: The config dictionary.

        """
        super().__init__()
        self.seed = config.type_config.common_to_all_tasks.seed
        torch.manual_seed(self.seed)
        self.norm_embedding = SincRadialBasis(
            seed=self.seed,
            num_rbf=config.type_config.common_to_all_tasks.node_emb_dim,
            rbound_upper=config.general_config.rbound_upper)

    def forward(self, pos: Tensor, adjacency_matrix: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes the norm embedding.
        Args:
            pos: The position in 3D. Tensor of shape [batch,n,n,3]
            adjacency_matrix: The adjacency matrix. Tensor of shape [batch,n,n]

        Returns: The embedding and vector difference.

        """
        # Norm difference.
        pos_difference = pos.unsqueeze(1) - pos.unsqueeze(2)  # (B, N, N, 3)
        # The norm.
        relative_norm = torch.norm(pos_difference, dim=-1, keepdim=True)  # (B, N, N, 1)
        # Embed the norm.
        relative_norm_embedding = self.norm_embedding(relative_norm)  # (B, N, N, norm_dim)
        # Multiply by adjacency to see only the visible entries.
        relative_norm_embedding *= adjacency_matrix.squeeze(-1)
        # Unsqueeze.
        pos_difference = pos_difference.unsqueeze(-1)
        return relative_norm_embedding, pos_difference, relative_norm.unsqueeze(-1)


class EdgeFeatureEmbedding(nn.Module):
    def __init__(self, emb_dim: int, seed: int, num_edge_features: int):
        """
        Edge embedding.
        Args:
            emb_dim: The embedding dim.
            seed: The seed.
            num_edge_features: Number of edge attributes.
        """
        super(EdgeFeatureEmbedding, self).__init__()
        self.seed = seed
        self.num_edge_features = num_edge_features

        self.edge_embedding_list = torch.nn.ModuleList(
            [nn.Embedding(num_embeddings=10, embedding_dim=emb_dim, padding_idx=0) for _ in
             range(self.num_edge_features)])
        self.init_parameters()

    def init_parameters(self) -> None:
        """
        Inits params.
        """
        torch.manual_seed(self.seed)
        for i in range(self.num_edge_features):
            emb = self.edge_embedding_list[i]
            torch.nn.init.xavier_uniform_(emb.weight.data)
            emb._fill_padding_idx_with_zero()

    def forward(self, edge_feature: Tensor) -> Tensor:
        """
        Computes multi-feature forward.
        Args:
            edge_feature: The features.

        Returns: The feature embedding.

        """

        x_embedding: Tensor = 0
        for i in range(self.num_edge_features):
            x_embedding += self.edge_embedding_list[i](edge_feature[:, :, :, i])
        return x_embedding


class NodeFeatureEmbedding(torch.nn.Module):
    def __init__(self, emb_dim: int, seed: int, num_atoms_type: int, max_features: int):
        """
        Feature encoder.
        Args:
            emb_dim: The embedding dim.
            seed: The seed.
            num_atoms_type: Number of atom types.
            max_features: The maximal feature.
        """
        super(NodeFeatureEmbedding, self).__init__()
        self.seed = seed
        self.num_atoms_type = num_atoms_type
        self.atom_embedding_list = torch.nn.ModuleList(
            [nn.Embedding(max_features, emb_dim) for _ in range(self.num_atoms_type)])
        self.init_parameters()

    def init_parameters(self):
        """
        Inits params.
        """
        torch.manual_seed(self.seed)
        for i in range(self.num_atoms_type):
            emb = self.atom_embedding_list[i]
            torch.nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, node_feature: Tensor) -> Tensor:
        """
        Computes multi-feature forward.
        Args:
            node_feature: The features.

        Returns: The feature embedding.

        """

        x_embedding: Tensor = 0
        for i in range(self.num_atoms_type):
            x_embedding += self.atom_embedding_list[i](node_feature[:, :, i])
        return x_embedding


class EmbedDistanceMatrixAndFeatures(nn.Module):
    """
    Embeds all features and distance information.
    """

    def __init__(self, config: EasyDict):
        """

        Args:
            config: The config dictionary.
        """
        super().__init__()
        edge_dim_emb = config.type_config.common_to_all_tasks.edge_emb_dim
        node_embedding_dim = config.type_config.common_to_all_tasks.node_emb_dim
        activation_fn = return_all_activation_funcs(config.general_config.init_activation)
        seed = config.type_config.common_to_all_tasks.seed
        torch.manual_seed(seed)
        self.node_feature_embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    Dense(
                        in_features=node_embedding_dim,
                        out_features=1,
                        activation_fn=activation_fn,
                        seed=seed
                    ),
                ) for _ in range(2)
            ]
        )
        self.norm_feature_squeeze = Dense(in_features=node_embedding_dim, out_features=1, bias=False,
                                          seed=seed)
        self.edge_feature_squeeze = Dense(in_features=edge_dim_emb, out_features=1, bias=False, seed=seed)

        self.pattern_embedding = nn.Embedding(
            num_embeddings=3,
            embedding_dim=1,
            padding_idx=0
        )

        self.mix_lin = Residual(
            seed=seed,
            hidden_dim=1,
            activation_fn=activation_fn,
            mlp_num=2
        )

    def forward(self, node_features: Tensor, norm_embedding: Tensor, edge_features: Tensor) -> Tensor:
        """

        Args:
            node_features: The node features.
            norm_embedding: The embedding of the norm.
            edge_features: The edge feature.

        Returns: The initialization forward.

        """
        node0, node1 = [self.node_feature_embeddings[i](node_features) for i in range(2)]

        node_embedding = self.norm_feature_squeeze(norm_embedding)

        z_mixed = node0[:, None, :, :] * node1[:, :, None, :]

        B = z_mixed.shape[0]
        N = z_mixed.shape[1]
        idx = torch.arange(N)
        tuple_pattern = torch.ones(size=(B, N, N), dtype=torch.int, device=z_mixed.device)
        tuple_pattern[:, idx, idx] = 2
        tuple_pattern = self.pattern_embedding(tuple_pattern)
        edge_embedding = self.edge_feature_squeeze(edge_features)
        combined_embeddings = z_mixed * node_embedding * tuple_pattern * edge_embedding

        combined_embeddings = self.mix_lin(combined_embeddings)
        return combined_embeddings


class InitialEmbedding(nn.Module):
    def __init__(self, config: EasyDict):
        """
        Initial embedding with node feature.
        Args:
            config: The config file.
        """
        super().__init__()
        self.seed = config.type_config.common_to_all_tasks.seed
        self.task = config.type
        self.node_feature_embedding = NodeFeatureEmbedding(emb_dim=config.type_config.common_to_all_tasks.node_emb_dim,
                                                           seed=self.seed,
                                                           num_atoms_type=config.type_config.common_to_all_tasks.
                                                           num_atoms_type,
                                                           max_features=config.type_config.common_to_all_tasks.
                                                           max_features)

        self.embed_point_cloud = NormEmbedding(config=config)

        self.combine_distance_node_and_edge_features = EmbedDistanceMatrixAndFeatures(config=config)

        self.edge_feature_embedding = EdgeFeatureEmbedding(emb_dim=config.type_config.common_to_all_tasks.edge_emb_dim,
                                                           seed=self.seed,
                                                           num_edge_features=config.type_config.common_to_all_tasks.num_edge_features)

    def forward(self, data_obj: Data) -> Tuple[Tensor, Tensor]:
        """
        Computes the initial forward, and rhe relative point cloud.
        Args:
            data_obj: The data object.

        Returns: The initial embedding.

        """
        # The point cloud, adjacency, node feature, edge feature.
        point_cloud, adjacency_matrix, node_features, edge_features = (data_obj.pos, data_obj.adjacency_matrix,
                                                                       data_obj.node_features, data_obj.edge_attr)
        # The norm embedding, relative point cloud.
        norm_embedding, relative_point_cloud, relative_norm = self.embed_point_cloud(pos=point_cloud,
                                                                                     adjacency_matrix=adjacency_matrix)
        # The node feature embedding.
        node_feature_embedding = self.node_feature_embedding(node_features)
        # The edge embedding.
        edge_feature_embedding = self.edge_feature_embedding(edge_features)
        # Combined embedding.
        all_embedding = self.combine_distance_node_and_edge_features(node_features=node_feature_embedding,
                                                                     norm_embedding=norm_embedding,
                                                                     edge_features=edge_feature_embedding)
        return all_embedding, relative_point_cloud
