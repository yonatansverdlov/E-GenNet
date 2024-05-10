"""
Ensemble dataset.
"""

import torch
from torch_geometric.data import InMemoryDataset


class EnsembleDataset(InMemoryDataset):
    def __init__(self, root: str, num_tasks: int = 1, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.scales = [1.0 for _ in range(num_tasks)]
        out = torch.load(self.processed_paths[0])
        self.data, self.slices, self.y = out

    def get_mean(self, target: str) -> float:
        """
        The mean.
        Args:
            target: The id.

        Returns: The mean label.

        """
        y = torch.cat([self.get(i).y for i in range(len(self))])
        target_id = self.descriptors.index(target)
        return float(y[:, target_id].mean())

    def get_std(self, target: str) -> float:
        """
        The std.
        Args:
            target: The id.

        Returns: The std label.

        """
        y = torch.cat([self.get(i).y for i in range(len(self))])
        target_id = self.descriptors.index(target)
        return float(y[:, target_id].std())

    def __repr__(self):
        return f'{self.__class__.__name__}: ' \
               f'{self.num_molecules} molecules, {self.num_conformers} conformers'
