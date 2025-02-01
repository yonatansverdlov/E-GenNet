import torch.nn as nn
from torch_geometric.data import Data
import torch
from torch import Tensor

class InvariantAbstractNet(nn.Module):
    def return_random_obj(self):
        num_mols = 10
        obj = Data(
            pos=torch.rand((num_mols, num_mols, 3)),
            adjacency_matrix=torch.ones((num_mols, num_mols, num_mols, 1, 1)),
            node_features=torch.zeros((num_mols, 1, 1), dtype=torch.int),
            edge_attr=torch.zeros((num_mols, 1, 1, 1), dtype=torch.int)
        )
        return obj

    def o_3_invariant_forward(self, data_obj: Data) -> Tensor:
        return self.orthogonal_invariant_hash(self.equivariant_model(data_obj))

    def _generate_orthogonal_transform(self):
        svd = torch.linalg.svd(torch.randn(3, 3))
        return svd[0] @ svd[2]

    def check_invariant_rotation(self, data_obj: Data) -> torch.float:
        tolerance = 1e-3
        transform = self._generate_orthogonal_transform()
        before_rotation = self.forward(data_obj)
        data_obj.pos = data_obj.pos @ transform
        after_rotation = self.forward(data_obj)
        norm = torch.norm(after_rotation - before_rotation)
        print(f"Rotation Invariance Check: Difference = {norm}, Model is {'' if norm < tolerance else 'not '}rotation invariant.")
        return norm

    def check_equivariant_rotation(self, data_obj: Data) -> torch.float:
        tolerance = 1e-3
        transform = self._generate_orthogonal_transform()
        before_rotation = self.equivariant_model(data_obj).transpose(-1, -2) @ transform
        data_obj.pos = data_obj.pos @ transform
        after_rotation = self.equivariant_model(data_obj).transpose(-1, -2)
        norm = torch.norm(after_rotation - before_rotation)
        print(f"Rotation Equivariance Check: Difference = {norm}, Model is {'' if norm < tolerance else 'not '}rotation equivariant.")
        return norm

    def check_invariant_permutation(self, data_obj: Data) -> torch.float:
        tolerance = 1e-3
        perm = torch.randperm(data_obj.pos.size(1))
        before_perm = self.forward(data_obj)
        data_obj.pos = data_obj.pos[:, perm]
        after_perm = self.forward(data_obj)
        norm = torch.norm(after_perm - before_perm) / torch.norm(after_perm)
        print(f"Permutation Invariance Check: Difference = {norm}, Model is {'' if norm < tolerance else 'not '}permutation invariant.")
        return norm

    def check_equivariant_permutation(self, data_obj: Data) -> torch.float:
        tolerance = 1e-3
        perm = torch.randperm(data_obj.pos.size(1))
        before_perm = self.o_3_invariant_forward(data_obj).transpose(1, 0)[perm]
        data_obj.pos = data_obj.pos[:, perm]
        after_perm = self.o_3_invariant_forward(data_obj).transpose(1, 0)
        norm = torch.norm(before_perm - after_perm)
        print(f"Permutation Equivariance Check: Difference = {norm}, Model is {'' if norm < tolerance else 'not '}permutation equivariant.")
        return norm

    def compute_feature_difference(self, data_obj1: Data, data_obj2: Data) -> Tensor:
        return self.forward(data_obj1) - self.forward(data_obj2)