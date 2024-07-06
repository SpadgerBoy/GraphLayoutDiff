import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
import pdb
# from utils.test_t import *

transform = T.AddRandomWalkPE(walk_length=2)

edge_index = torch.tensor([[0, 1, 1, 3, 0, 2, 2, 3],
                           [1, 0, 3, 1, 2, 0, 3, 2]], dtype=torch.long)
data = Data(edge_index=edge_index)
data = transform(data)

edge_index_1 = torch.tensor([[0, 1, 1, 3, 0, 2, 2],
                           [1, 0, 3, 1, 2, 0, 3]], dtype=torch.long)
                           
data_1 = Data(edge_index=edge_index_1)
data_1 = transform(data_1)

# print(data.size())
# print(data_1.size())
print(data.random_walk_pe)
print(data_1.random_walk_pe)






