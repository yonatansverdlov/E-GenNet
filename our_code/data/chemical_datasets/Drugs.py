"""
The Drugs dataset.
"""
from collections import defaultdict
from typing import Optional

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import extract_zip
from tqdm import tqdm

from our_code.data.data_utils import mol_to_data_obj
from .ensemble import EnsembleDataset


class Drugs(EnsembleDataset):
    descriptors = ['ip', 'ea', 'chi']

    def __init__(self, root: str):
        """
        Stores the root.
        Args:
            root: The root to the data.
        """
        super().__init__(root=root)

    @property
    def processed_file_names(self):
        return 'DrugsEnsemble_processed.pt' if self.max_num_conformers is None \
            else f'DrugsEnsemble_processed_{self.max_num_conformers}.pt'

    @property
    def raw_file_names(self) -> str:
        return 'Drugs.zip'

    @property
    def num_molecules(self) -> int:
        """
        Returns: The number of molecules.
        """
        return self.y.shape[0]

    @property
    def num_conformers(self) -> int:
        """
        Returns: The number of conformers.
        """
        return len(self)

    def process(self)->None:
        data_list = []
        quantities = self.descriptors

        molecules = defaultdict(list)
        raw_file = self.raw_paths[0]
        extract_zip(raw_file, self.raw_dir)
        raw_file = raw_file.replace('.zip', '.sdf')
        with Chem.SDMolSupplier(raw_file, removeHs=False) as suppl:
            for idx, mol in enumerate(tqdm(suppl)):
                id_ = mol.GetProp('ID')
                name = mol.GetProp('_Name')
                smiles = mol.GetProp('smiles')

                data = mol_to_data_obj(mol)
                data.name = name
                data.id = id_

                data.smiles = smiles
                data.y = []
                for quantity in quantities:
                    data.y.append(float(mol.GetProp(quantity)))
                data.y = torch.Tensor(data.y).unsqueeze(0)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                molecules[name].append(data)

        cursor = 0
        y = []

        label_file = raw_file.replace('.sdf', '.csv')
        labels = pd.read_csv(label_file)

        for name, mol_list in tqdm(molecules.items()):
            row = labels[labels['name'] == name]
            y.append(torch.Tensor([row[quantity].item() for quantity in quantities]))

            for mol in mol_list:
                mol.molecule_idx = cursor
                data_list.append(mol)
            cursor += 1

        y = torch.stack(y)
        data, slices = self.collate(data_list)
        torch.save((data, slices, y), self.processed_paths[0])
