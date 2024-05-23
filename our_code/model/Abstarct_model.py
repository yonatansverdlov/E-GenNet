"""
Abstract Model.
With Sanity checks.
"""
import torch.nn as nn
from torch_geometric.data import Data
import torch
from torch import Tensor


class InvariantAbstractNet(nn.Module):

    def return_random_obj(self):
        num_mols = 10
        pos = torch.rand((num_mols,num_mols,3))
        adjacency_matrix = torch.ones((num_mols,num_mols,num_mols,1,1))
        obj = Data(pos=pos,
             adjacency_matrix=adjacency_matrix,
             node_features=torch.zeros((num_mols, 1, 1), dtype=torch.int),
             edge_attr=torch.zeros((num_mols, 1, 1, 1), dtype=torch.int))
        return obj

    def o_3_invariant_forward(self, data_obj: Data) -> Tensor:
        """
        Forward the equivariant and then the invariant feature, to have Sn equivariant model and O(3) orbit injective.
        Args:
            data_obj: The data object.

        Returns: The O_3 invariant and Sn equivariant feature.

        """
        geometric_information = self.equivariant_model(data_obj)
        invariant_geometric_information = self.orthogonal_invariant_hash(geometric_information)
        return invariant_geometric_information

    def check_invariant_rotation(self, data_obj: Data) -> torch.float:
        """
        Check the model includes the equivariant stack and the O(3) invariant layer, is indeed invariant to rotations.
        Args:
            data_obj: The data object.

        Returns: The norm_and_feature of the difference between the features.

        """
        tolerance = 1e-3
        svd = torch.linalg.svd(torch.randn(3, 3))
        orthogonal_transform = svd[0] @ svd[2]
        before_rotation = self.forward(data_obj=data_obj)
        data_obj.pos = data_obj.pos @ orthogonal_transform
        first_rotation = self.forward(data_obj=data_obj)
        norm = torch.norm(first_rotation - before_rotation)
        if norm < tolerance:
            print(f"The difference is {norm} which is less than {tolerance}, so our model is indeed invariant.")
        else:
            print(f"The difference is {norm} which is more than {tolerance}, so our model is not invariant, check "
                  f"again your model!.")
        return torch.norm(first_rotation - before_rotation)

    def check_equivariant_rotation(self, data_obj: Data) -> torch.float:
        """
        Check the model includes the equivariant stack, is indeed equivariant to rotations.
        Args:
            data_obj: The data object.

        Returns: The norm_and_feature of the difference between the features.

        """
        tolerance = 1e-3
        svd = torch.linalg.svd(torch.randn(3, 3))
        orthogonal_transform = svd[0] @ svd[2]
        before_rotation = self.equivariant_model(data_obj).transpose(-1, -2) @ orthogonal_transform
        data_obj.pos = data_obj.pos @ orthogonal_transform
        first_rotation = self.equivariant_model(data_obj).transpose(-1, -2)
        norm = torch.norm(first_rotation - before_rotation)
        if norm < tolerance:
            print(f"The difference is {norm} which is less than {tolerance}, so our model is indeed equivariant.")
        else:
            print(f"The difference is {norm} which is more than {tolerance}, so our model is not equivariant, check "
                  f"again your model!.")
        return torch.norm(first_rotation - before_rotation)

    def check_invariant_permutation(self, data_obj: Data) -> torch.float:
        """
        Check our model is invariant to the permutation of the point cloud.
        Args:
            data_obj: The data object.

        Returns: The norm_and_feature difference between the features.

        """
        tolerance = 1e-3
        perm = torch.randperm(data_obj.pos.size()[1])
        before_perm = self.forward(data_obj)
        point_cloud_after_perm = data_obj.pos[:, perm]
        data_obj.pos = point_cloud_after_perm
        first_permutation = self.forward(data_obj)
        norm = torch.norm(first_permutation - before_perm) / torch.norm(first_permutation)
        if norm < tolerance:
            print(f"The difference is {norm} which is less than {tolerance}, so our model is indeed invariant.")
        else:
            print(f"The difference is {norm} which is more than {tolerance}, so our model is not invariant, check "
                  f"again your model!.")

        return torch.norm(first_permutation - before_perm) / torch.norm(first_permutation)

    def check_equivariant_permutation(self, data_obj: Data) -> torch.float:
        """
        Check the equivariant model is permutation equivariant.
        Args:
           data_obj: The data object.

        Returns: The norm_and_feature difference.

        """
        tolerance = 1e-3
        perm = torch.randperm(data_obj.pos.size()[1])
        compute_then_permutation = self.o_3_invariant_forward(data_obj).transpose(1, 0)[perm]
        data_obj.pos = data_obj.pos[:, perm]
        first_permutation = self.o_3_invariant_forward(data_obj).transpose(1, 0)
        norm = torch.norm(compute_then_permutation - first_permutation)
        if norm < tolerance:
            print(f"The difference is {norm} which is less than {tolerance}, so our model is indeed equivariant.")
        else:
            print(f"The difference is {norm} which is more than {tolerance}, so our model is not equivariant, check "
                  f"again your model!.")

        return norm

    def compute_feature_difference(self, data_obj1: Data, data_obj2: Data) -> Tensor:
        """
        Return the feature difference.
        Args:
            data_obj1: first object.
            data_obj2: The second data object.

        Returns: The difference feature.

        """
        #
        out1 = self.forward(data_obj=data_obj1)
        out2 = self.forward(data_obj=data_obj2)
        return out1 - out2
