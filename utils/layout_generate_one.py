import networkx as nx
import os
import numpy as np
import json
import torch
from torch_geometric.data import Data
import pickle

def layout_compute(g, layout, graph_name):

    for i in range(1):

        # 节点整数索引
        H = nx.convert_node_labels_to_integers(g, label_attribute='original_label')

        # 计算pos
        center = [0, 0]
        
        if layout == "spring_layout":
            pos = nx.spring_layout(H, center = center)
        elif layout == "kamada_kawai_layout":
            pos = nx.kamada_kawai_layout(H, center = center)

        # 获得pos_list
        pos_values_sorted = [pos[key] for key in sorted(pos.keys())]
        pos_list = np.array(pos_values_sorted).tolist()

        # 获得edge_list
        edge_list = list(H.edges())
        edge_list = np.array(edge_list).tolist()

        torch_pos = torch.tensor(pos_list, dtype=torch.float32)
        torch_edge_index = torch.tensor(edge_list, dtype=torch.long)
        torch_edge_index = torch.transpose(torch_edge_index, 0, 1)

        data = Data(pos=torch_pos, edge_index=torch_edge_index, graph_name=graph_name)

        return data

folder_path = "../graph_data/control_7/"

file_paths = os.listdir(folder_path)

data_list = []

for i, file_path in enumerate(file_paths):
    print("i: ", i)

    file_name = os.path.basename(file_path)
    try:
        G = nx.read_graphml(folder_path+file_name)
    except:
        print(file_name)

    # 输出非连通图
    is_connected = nx.is_connected(G)
    if not is_connected:
        print("no_connected: ", i)
        print("file_name: ", file_name)
    
    data = layout_compute(G, "kamada_kawai_layout", file_name)
    data_list.append(data)

save_path = '../data/control/intelligent_graph_7.pkl'
with open(save_path, 'wb') as file:
    pickle.dump(data_list, file)   