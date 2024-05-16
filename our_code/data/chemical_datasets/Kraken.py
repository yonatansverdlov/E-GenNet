"""
Kraken dataset.
"""

import pickle

import torch
from rdkit import Chem
from torch_geometric.data import extract_zip
from tqdm import tqdm

from our_code.data.data_utils import mol_to_data_obj
from .ensemble import EnsembleDataset


class Kraken(EnsembleDataset):
    descriptors = ['B5', 'L', 'burB5', 'burL']

    def __init__(self, root: str):
        """
        Inits the Kraken set.
        Args:
            root: The root to the data.
        """
        super().__init__(root=root)
        out = torch.load(self.processed_paths[0])
        self.data, self.slices, self.y = out

    @property
    def processed_file_names(self) -> str:
        """
        The names of processed files.
        """
        return 'Kraken_processed.pt'

    @property
    def raw_file_names(self) -> str:
        """
        Raw Name.
        """
        return 'Kraken.zip'

    @property
    def num_molecules(self) -> int:
        """
        Returns: Num molecules.
        """
        return self.y.shape[0]

    @property
    def num_conformers(self) -> int:
        """
        Returns: Num conformers.
        """
        return len(self)

    def process(self) -> None:
        """
        Processes the raw data.
        """
        data_list = []
        descriptors = self.descriptors

        raw_file = self.raw_paths[0]
        extract_zip(raw_file, self.raw_dir)
        raw_file = raw_file.replace('.zip', '.pickle')
        with open(raw_file, 'rb') as f:
            kraken = pickle.load(f)

        ligand_ids = list(kraken.keys())
        cursor = 0
        y = []
        for ligand_id in tqdm(ligand_ids):
            smiles, boltzman_avg_properties, conformer_dict = kraken[ligand_id]
            conformer_ids = list(conformer_dict.keys())

            for conformer_id in conformer_ids:
                mol_sdf, boltz_weight, conformer_properties = conformer_dict[conformer_id]
                mol = Chem.MolFromMolBlock(mol_sdf, removeHs=False)

                data = mol_to_data_obj(mol)
                data.name = f'mol{int(ligand_id)}'
                data.id = f'{data.name}_{conformer_id}'
                data.smiles = smiles
                data.y = torch.Tensor(
                    [conformer_properties['sterimol_' + descriptor] for descriptor in descriptors]).unsqueeze(0)
                data.molecule_idx = cursor

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
            cursor += 1

            y.append(torch.Tensor([boltzman_avg_properties['sterimol_' + descriptor] for descriptor in descriptors]))
        y = torch.stack(y)

        data, slices = self.collate(data_list)
        torch.save((data, slices, y), self.processed_paths[0])
