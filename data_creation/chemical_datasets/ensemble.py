"""
Ensemble dataset.
Taken https://github.com/SXKDZ/MARCEL.
"""

import torch
from torch_geometric.data import InMemoryDataset, Data  # Import Data explicitly

class EnsembleDataset(InMemoryDataset):
    def __init__(self, root: str):
        """
        Inits Ensemble dataset.
        Args:
            root: The root.
        """
        super().__init__(root)
        
        # Allowlist Data to enable loading
        torch.serialization.add_safe_globals([Data])

        # Load the file with full deserialization
        out = torch.load(self.processed_paths[0], weights_only=False)
        self.data, self.slices, self.y = out

    def get_mean(self, target: str) -> float:
        """
        The mean.
        Args:
            target: The task id.

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

    def __repr__(self) -> str:
        """
        Repr of the class.
        """
        return f'{self.__class__.__name__}: ' \
               f'{self.num_molecules} molecules, {self.num_conformers} conformers'
