import os
import pickle
import copy
import json
from collections import defaultdict

import numpy as np
import random

import torch
from torch_geometric.data import Data, Dataset, Batch

from torch_geometric.utils import to_networkx
from torch_scatter import scatter

class GraphLayoutDataset(Dataset):

    def __init__(self, path=None, transform=None, path2=None, path3=None):
        super().__init__()
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
            self.transform = transform
            for item in self.data:
                self.transform(item) 

        if path2 is not None:   # path_graph 100
            with open(path2, 'rb') as f2:
                tmp = pickle.load(f2)
                for item in tmp:
                    self.transform(item) 
                for _ in range(80):
                    self.data.extend(tmp)

        if path3 is not None:   # tree_graph 800
            with open(path3, 'rb') as f3:
                tmp = pickle.load(f3)
                for item in tmp:
                    self.transform(item) 
                for _ in range(10):
                    self.data.extend(tmp)



    def __getitem__(self, idx):
        data = self.data[idx].clone()    
        return data

    def __len__(self):
        return len(self.data)

class PackedGraphLayoutDataset(GraphLayoutDataset):

    def __init__(self, path, transform=None):
        super().__init__(path, transform)
        self._pack_data_by_topology()
    
    def _pack_data_by_topology(self):
        self._pack_data = defaultdict(list)
        for i in range(len(self.data)):
            self._pack_data[self.data[i].graph_name].append(self.data[i])
        
        print('[Packed] %d graph, %d Layouts.' % (len(self._pack_data), len(self.data)))

        new_data = []
        cnt = 0
        for k, v in self._pack_data.items():
            data = copy.deepcopy(v[0])
            all_pos = []
            for i in range(len(v)):
                all_pos.append(v[i].pos)
            data.pos_ref = torch.cat(all_pos, 0)
            data.num_pos_ref = torch.tensor([len(all_pos)], dtype=torch.long)
            
            new_data.append(data)
        self.new_data = new_data
    
    def __getitem__(self, idx):
        data = self.new_data[idx].clone()
        return data
    
    def __len__(self):
        return len(self.new_data)

class DemoGraphLayoutDataset(Dataset):

    def __init__(self, data=None, transform=None):
        super().__init__()
        self.data = data
        self.transform = transform
        for item in self.data:
            self.transform(item) 

    def __getitem__(self, idx):
        data = self.data[idx].clone()    
        return data

    def __len__(self):
        return len(self.data)


        
