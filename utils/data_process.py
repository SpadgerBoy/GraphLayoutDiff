import json
import torch
from torch_geometric.data import Data
import os
import pickle

data_list = []

count = 0

folder_path = '../raw_data/rome/r_spring_layout'
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r') as file:
            data_dict = json.load(file)
        
        pos = torch.tensor(data_dict['pos'], dtype=torch.float32)
        edge_index = torch.tensor(data_dict['edge_index'], dtype=torch.long)
        edge_index = torch.transpose(edge_index, 0, 1)

        data = Data(pos=pos, edge_index=edge_index)

        data_list.append(data)

        count += 1
        if count == 100:
            break

save_path = '../data/rome/r_spring_layout.pkl'
with open(save_path, 'wb') as file:
    pickle.dump(data_list, file)