"""
Drugs dataset.
"""
import random

import easydict
from torch.utils.data import Dataset as torchDataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset

from .data_utils import group_same_size, batch_same_size, create_k_chains, create_pair_A, create_pair_B


class BatchDataSet(Dataset):

    def __init__(self, dataset: Dataset, task: str, batch_size: int, descriptors: list):
        """

        Args:
            dataset: The dataset.
            task: The task.
            batch_size: The batch size.
            descriptors: The descriptor.
        """
        super().__init__()
        self.descriptors = descriptors
        assert task in self.descriptors, f"task must be one of {self.descriptors[1:]},you gave {task}"
        self.batch_size = batch_size
        self.non_sorted_set = dataset
        self.task = self.descriptors.index(task)
        self.grouped_data = group_same_size(self.non_sorted_set)
        self.batched_data = batch_same_size(grouped_dataset=self.grouped_data,
                                            batch_size=self.batch_size,
                                            task=self.task)
        self.size = len(self.batched_data)  # batched version, a datapoint is a data_obj.

    def get(self, index) -> Data:
        # to get the whole batched data.
        return self.batched_data[index]

    def len(self) -> int:
        # Returns length.
        return self.size

    def __repr__(self) -> str:
        return f"Batched_Drugs(batch_size={self.batch_size}, size={self.size})"

    def reshuffle_grouped_dataset(self, seed: int = 0) -> None:
        """
        Reshuffles the dataset.
        """
        random.seed(seed)
        for i, group in self.grouped_data:
            random.shuffle(group)

        self.batched_data = batch_same_size(grouped_dataset=self.grouped_data, batch_size=self.batch_size,
                                            task=self.task)


class k_chain_dataset(torchDataset):
    """
    All k_chain experiments dataset.
    """
    def __init__(self, config: easydict):
        """
        Creates dataset object, given a tuple type.
        Args:
            config: The config file.
        """
        self.k = config.type_config.task_specific.classify_original.k
        self.task = config.task
        if self.task == 'classify_original':
            self.data_point = create_k_chains(k=self.k)
        if self.task == 'classify_pair_A':
            self.data_point = create_pair_A(k=self.k)
        if self.task == 'classify_pair_B':
            self.data_point = create_pair_B(k=self.k)

    def __len__(self):
        return 1

    def __getitem__(self, item: int):
        """
        Args:
            item: The item id = 0.

        Returns: The point position, adjacency matrix, label.

        """
        return self.data_point
