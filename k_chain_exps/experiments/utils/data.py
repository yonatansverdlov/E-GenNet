import torch
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.utils import to_dense_adj as ToAdj
from torch_geometric.utils import to_undirected
from torch_geometric.utils._to_dense_adj import to_dense_adj as to_dense
from torch_geometric.utils import dense_to_sparse


def create_pairB(k,power):
    torch.manual_seed(1)
    pc = []
    first_vector = 0
    dim = 3
    scale = 1.0
    atoms = torch.LongTensor([0] + [0] + [0]*(k-1) + [0] )
    for i in range(k):
        directon = torch.rand(dim) * torch.tensor([1, 1, 0]) + torch.tensor([0.3, -0.4, 0.0])
        directon = directon * scale
        first_vector = first_vector + directon
        pc.append(first_vector)
    # Finished the common points.
    # Last point coordiantes.
    x, y, z = first_vector[0], first_vector[1], first_vector[2]
    # Before the last.
    before_last = torch.tensor([x + 0.75, y, z]) * scale
    pc.append(before_last)
    second_pc = pc.copy()
    # The last in pc1.
    last = torch.tensor([x + 1.0, y + 0.3, z]) * scale
    pc.append(last)
    # The last in pc2.
    second_pc.append(torch.tensor([x + 1.0, y - 0.3, z])*scale)
    # Stack all points.
    pos1 = torch.stack(pc) - torch.stack(pc).mean(0)
    pos2 = torch.stack(second_pc) - torch.stack(second_pc).mean(0)
    # The edges.
    edge_index = torch.LongTensor( [ [i for i in range((k+2) - 1)], [i for i in range(1, k+2)] ] )
    edge_index = to_undirected(edge_index)
    # Compute the power graph.
    edge_index = create_path_power_graph(edge_index,power)
    # Graph 1.
    data1 = Data(atoms = atoms, pos=pos1, edge_index=edge_index,
                  y=torch.tensor([0], dtype=torch.long))
    # Graph 2.
    data2 = Data(pos=pos2, edge_index=edge_index,
                  y=torch.tensor([1], dtype=torch.long),atoms = atoms)
    points = [data1, data2]
    return points

def create_pairA(k,power):
    torch.manual_seed(1)
    pc = []
    first_vector = 0
    dim = 3
    scale = 1.0
    atoms = torch.LongTensor([0] + [0] + [0]*(k-1) + [0] )
    for i in range(k):
        directon = torch.rand(dim) * torch.tensor([1, 1, 0]) + torch.tensor([0.3, -0.4, 0.0])
        directon = directon * scale
        first_vector = first_vector + directon
        pc.append(first_vector)
    # Finished the common points.
    # Last point coordiantes.
    x, y, z = first_vector[0], first_vector[1], first_vector[2]
    # Before the last.
    before_last = torch.tensor([x + 0.75, y, z]) * scale
    pc.append(before_last)
    second_pc = pc.copy()
    # The last in pc1.
    last = torch.tensor([x + 1.0, y + 0.3, z]) * scale
    pc.append(last)
    # The last in pc2.
    angle = torch.tensor([180.0])
    theta = (angle/180.) * torch.pi
    a,b,c,d = torch.cos(theta), torch.sin(theta), -torch.sin(theta),  torch.cos(theta)
    rotMatrix = torch.tensor([[a,b,0],[c,d,0],[0,0,0]])

    second_pc.append(rotMatrix @ (last-before_last)+before_last)
    # Stack all points.
    pos1 = torch.stack(pc) - torch.stack(pc).mean(0)
    pos2 = torch.stack(second_pc) - torch.stack(second_pc).mean(0)
    # The edges.
    edge_index = torch.LongTensor( [ [i for i in range((k+2) - 1)], [i for i in range(1, k+2)] ] )
    edge_index = to_undirected(edge_index)
    # Compute the power graph.
    edge_index = create_path_power_graph(edge_index,power)
    # Graph 1.
    data1 = Data(atoms = atoms, pos=pos1, edge_index=edge_index,
                  y=torch.tensor([0], dtype=torch.long))
    # Graph 2.
    data2 = Data(pos=pos2, edge_index=edge_index,
                  y=torch.tensor([1], dtype=torch.long),atoms = atoms)
    points = [data1, data2]
    return points


def create_kchains(k):
    assert k >= 2

    dataset = []

    # Graph 0
    atoms = torch.LongTensor( [0] + [0] + [0]*(k-1) + [0] )
    edge_index = torch.LongTensor( [ [i for i in range((k+2) - 1)], [i for i in range(1, k+2)] ] )
    pos = torch.FloatTensor(
        [[-4, -3, 0]] +
        [[0, 5*i , 0] for i in range(k)] +
        [[4, 5*(k-1) + 3, 0]]
    )
    center_of_mass = torch.mean(pos, dim=0)
    pos = pos - center_of_mass
    y = torch.LongTensor([0])  # Label 0
    data1 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data1.edge_index = to_undirected(data1.edge_index)
    dataset.append(data1)

    # Graph 1
    atoms = torch.LongTensor( [0] + [0] + [0]*(k-1) + [0] )
    edge_index = torch.LongTensor( [ [i for i in range((k+2) - 1)], [i for i in range(1, k+2)] ] )
    pos = torch.FloatTensor(
        [[4, -3, 0]] +
        [[0, 5*i , 0] for i in range(k)] +
        [[4, 5*(k-1) + 3, 0]]
    )
    center_of_mass = torch.mean(pos, dim=0)
    pos = pos - center_of_mass
    y = torch.LongTensor([1])  # Label 1
    data2 = Data(atoms=atoms, edge_index=edge_index, pos=pos, y=y)
    data2.edge_index = to_undirected(data2.edge_index)
    dataset.append(data2)

    return dataset

def compute_power_graph(graph, times:int):
    # Compute power graph.
    # Return the graph.
    if times == 1:
        return graph
    if times == 2:
        # Compute the second power of G.
        n = graph.size(1)
        double = (torch.eye(n) + graph) @ (torch.eye(n) + graph)
        double[double > 1] = 1
        double = double - torch.eye(n, n)
        return double
    else:
        # Recursevily compute the power graph times - 1 and multiply by G.
        power_times_minus_1 = compute_power_graph(graph, times - 1)
        return compute_power_graph(power_times_minus_1, 2)

def create_path_power_graph(edge_index,times):
  # Converting the power graph to edge index tuples.
  # Makes dense.
  adj_matrix = to_dense(edge_index=edge_index)
  # Computes power graph.
  power_graph = compute_power_graph(graph = adj_matrix, times = times)
  # Back to sparse.
  power_graph = dense_to_sparse(power_graph)[0]
  return power_graph