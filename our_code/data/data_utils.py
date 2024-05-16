"""
Data utils.
"""

from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.utils import to_dense_adj as ToAdj
from torch_geometric.utils import to_undirected
from torch_geometric.utils._to_dense_adj import to_dense_adj as to_dense


# allowable multiple choice node and edge features
def return_allowed_features():
    """
    Returns: Allowed features.
    """
    allowable_features = {
        'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
        'possible_chirality_list': [
            'CHI_UNSPECIFIED',
            'CHI_TETRAHEDRAL_CW',
            'CHI_TETRAHEDRAL_CCW',
            'CHI_SQUAREPLANAR',
            'CHI_TRIGONALBIPYRAMIDAL',
            'CHI_OCTAHEDRAL',
            'CHI_OTHER'
        ],
        'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
        'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
        'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
        'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
        'possible_hybridization_list': [
            'SP',
            'SP2',
            'SP3',
            'SP3D',
            'SP3D2',
            'misc'
        ],
        'possible_is_aromatic_list': [False, True],
        'possible_is_in_ring_list': [False, True],
        'possible_bond_type_list': [
            'SINGLE',
            'DOUBLE',
            'TRIPLE',
            'AROMATIC',
            'misc'
        ],
        'possible_bond_stereo_list': [
            'STEREONONE',
            'STEREOZ',
            'STEREOE',
            'STEREOCIS',
            'STEREOTRANS',
            'STEREOANY',
        ],
        'possible_is_conjugated_list': [False, True],
    }
    return allowable_features


def safe_index(a_list: List, e: str) -> int:
    """

    Args:
        a_list:
        e: The index.

    Returns: index of element e in list l. If e is not present, return the last index

    """

    try:
        return a_list.index(e)
    except ValueError:
        return len(a_list) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param atom: rdkit atom object
    :return: list
    """

    allowable_features = return_allowed_features()
    atom_feature = [
        safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
        allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
        safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
        safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
        safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
        safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
        safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
        allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
        allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
    ]
    return atom_feature


def get_atom_feature_dims():
    """
    Get Atom feature dims.
    Returns:

    """
    allowable_features = return_allowed_features()
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
    ]))


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param bond: rdkit bond object
    :return: list
    """
    allowable_features = return_allowed_features()
    bond_feature = [
        safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
        allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
        allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
    ]
    return bond_feature


def mol_to_data_obj(mol) -> Data:
    """
    Mol to data object.
    Args:
        mol: The mol.

    Returns: The data object.

    """
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = torch.tensor(np.asarray(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_feature = bond_to_feature_vector(bond)
            edge_attr.append(edge_feature)
            edge_attr.append(edge_feature)
        edge_index = torch.tensor(np.asarray(edge_index), dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.asarray(edge_attr), dtype=torch.long)

    # coordinates
    pos = mol.GetConformer().GetPositions()
    pos = torch.from_numpy(pos).float()
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

    return data


def batch_same_size(grouped_dataset: List[Tuple[int, List[Data]]], task: int, batch_size: int) \
        -> List[Data]:
    """
    Batch the dataset into same size batches.
    Args:
        grouped_dataset: The chemical_datasets.
        task: The task.
        batch_size: The batch size.

    Returns: The batched data-loader.

    """
    # batched dataset, according to the data_obj size.
    batched_dataset = []
    for size, group in grouped_dataset:
        batch_num_in_group = (len(group) // batch_size) + 1 if len(group) % batch_size != 0 else len(
            group) // batch_size
        for idx, i in enumerate(range(batch_num_in_group)):
            lower_bound = i * batch_size
            upper_bound = min((i + 1) * batch_size, len(group))
            batch = group[lower_bound:upper_bound]
            # Node features.
            node_feature = torch.cat([batch[i].x.unsqueeze(0) for i in range(len(batch))]).long()
            # Pos.
            pos = torch.cat([batch[i].pos.unsqueeze(0) for i in range(len(batch))])
            # Normalize.
            pos = (pos - pos.mean(1).unsqueeze(1))
            # Adjacency.
            adjacency_matrix = torch.cat([ToAdj(batch[j].edge_index) for j in range(len(batch))])[:, :, :, None, None]
            # Edge feature.
            edge_feature = torch.cat([ToAdj(edge_index=batch[j].edge_index, edge_attr=batch[j].edge_attr + 1) for j in
                                      range(len(batch))]).long()
            # The label.
            label = torch.cat([batch[i].y for i in range(len(batch))])[:, task].float()
            # The data object.
            data_point = Data(node_features=node_feature, pos=pos, batch_size=node_feature.shape[0],
                              graph_size=node_feature.shape[1], adjacency_matrix=adjacency_matrix,
                              label=label, edge_attr=edge_feature, id=idx)
            batched_dataset.append(data_point)

    return batched_dataset


def group_same_size(dataset: Dataset) -> List[Tuple[int, List[Data]]]:
    """
    Group dataset into point clouds of same size.
    Args:
        dataset: The dataset of all PCS.

    Returns: The list of chemical_datasets.

    """
    group = []
    data_list: List[Data] = list(dataset)
    data_list.sort(key=lambda sample: sample.x.shape[0])
    # grouped dataset by size
    grouped_dataset = []
    start = 0
    end = len(data_list)
    for i in range(start, end):
        curr_data = data_list[i]
        if i == start:
            group = [curr_data]
        else:
            last_data = data_list[i - 1]
            if curr_data.x.size(0) == last_data.x.size(0):
                group.append(curr_data)
            else:
                grouped_dataset.append((last_data.x.shape[0], group))
                group = [curr_data]

        if i == end - 1:
            # last batch must be added.
            grouped_dataset.append((curr_data.x.shape[0], group))

    return grouped_dataset


def create_k_chains(k: int) -> Data:
    """

    Args:
        k: The number of points in the chain.

    Returns: The two data-points.

    """
    assert k >= 2

    # Graph 0
    edge_index = torch.LongTensor([[i for i in range((k + 2) - 1)], [i for i in range(1, k + 2)]])
    edge_index = to_dense(to_undirected(edge_index)).squeeze(0).unsqueeze(-1).unsqueeze(-1)
    pos1 = torch.FloatTensor(
        [[-4, -3, 0]] +
        [[0, 5 * i, 0] for i in range(k)] +
        [[4, 5 * (k - 1) + 3, 0]]
    )

    # Graph 1
    pos2 = torch.FloatTensor(
        [[4, -3, 0]] +
        [[0, 5 * i, 0] for i in range(k)] +
        [[4, 5 * (k - 1) + 3, 0]]
    )
    points = Data(pos=torch.stack([pos1, pos2]), adjacency_matrix=torch.stack([edge_index, edge_index]),
                  label=torch.tensor([0, 1], dtype=torch.long).view(2, ),
                  node_features=torch.zeros((2, 1, 1), dtype=torch.int),
                  edge_attr=torch.zeros((2, 1, 1, 1), dtype=torch.int))

    return points


def create_pair_A(k: int) -> Data:
    """
    Creating pair A.
    Args:
        k: The graph length.

    Returns: The two graphs.

    """
    torch.manual_seed(0)
    pc = []
    first_vector = 0
    dim = 3
    scale = 1.0
    for i in range(k):
        directon = torch.rand(dim) * torch.tensor([1, 1, 0]) + torch.tensor([0.3, -0.4, 0.0])
        directon = directon * scale
        first_vector = first_vector + directon
        pc.append(first_vector)
    # Last point coordinates.
    x, y, z = first_vector[0], first_vector[1], first_vector[2]
    # Before the last.
    before_last = torch.tensor([x + 0.75, y, z]) * scale
    pc.append(before_last)
    second_pc = pc.copy()
    # The last in pc1.
    last = torch.tensor([x + 1.0, y + 0.3, z]) * scale
    pc.append(last)
    # The last in pc2.
    angle = torch.randint(low=180, high=270, size=(1,))
    theta = (angle / 180.) * torch.pi
    a, b, c, d = torch.cos(theta), -torch.sin(theta), torch.sin(theta), torch.cos(theta)
    rotMatrix = torch.tensor([[a, b, 0], [c, d, 0], [0, 0, 0]])
    second_pc.append(rotMatrix @ (last - before_last) + before_last)
    # Stack all points.
    pos1 = torch.stack(pc)
    pos2 = torch.stack(second_pc)
    # The edges.
    edge_index = create_k_chains(k=k).adjacency_matrix
    points = Data(pos=torch.stack([pos1, pos2]),
                  adjacency_matrix=edge_index,
                  label=torch.tensor([0, 1], dtype=torch.long).view(2, ),
                  node_features=torch.zeros((2, 1, 1), dtype=torch.int),
                  edge_attr=torch.zeros((2, 1, 1, 1), dtype=torch.int))
    return points


def create_pair_B(k: int) -> Data:
    """
    Creating pair B.
    Args:
        k: The k.

    Returns: The data pair.

    """
    torch.manual_seed(1)
    pc = []
    first_vector = 0
    dim = 3
    scale = .5
    for i in range(k):
        directon = torch.rand(dim) * torch.tensor([1, 1, 0]) + torch.tensor([0.3, -0.4, 0.0])
        directon = directon * scale
        first_vector = first_vector + directon
        pc.append(first_vector)
    # Finished the common points.
    # Last point coordinates.
    x, y, z = first_vector[0], first_vector[1], first_vector[2]
    # Before the last.
    before_last = torch.tensor([x + 0.75, y, z]) * scale
    pc.append(before_last)
    second_pc = pc.copy()
    # The last in pc1.
    last = torch.tensor([x + 1.0, y + 0.3, z]) * scale
    pc.append(last)
    # The last in pc2.
    second_pc.append(torch.tensor([x + 1.0, y - 0.3, z]) * scale)
    # Stack all points.
    pos1 = torch.stack(pc)
    pos2 = torch.stack(second_pc)
    # The edges.
    edge_index = create_k_chains(k=k).adjacency_matrix
    points = Data(pos=torch.stack([pos1, pos2]),
                  adjacency_matrix=edge_index,
                  label=torch.tensor([0, 1], dtype=torch.long).view(2, ),
                  node_features=torch.zeros((2, 1, 1), dtype=torch.int),
                  edge_attr=torch.zeros((2, 1, 1, 1), dtype=torch.int))
    return points
