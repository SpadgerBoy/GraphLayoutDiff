import networkx as nx
import os
import numpy as np
import json
import torch
from torch_geometric.data import Data
import pickle
import random
import pdb


def layout_compute(g, layout, graph_name):

    for i in range(1):

        # 节点整数索引
        H = nx.convert_node_labels_to_integers(g, label_attribute='original_label')

        center = [0, 0]
        
        if layout == "spring_layout":
            pos = nx.spring_layout(H, center = center)
        elif layout == "kamada_kawai_layout":
            pos = nx.kamada_kawai_layout(H, center = center)
        # pos = nx.kamada_kawai_layout(H, center = center)
        # pos = nx.spectral_layout(g)
        # pos = nx.circular_layout(g)

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

np.random.seed(2023)
random.seed(2023)

folder_path = "../graph_data/rome/"
folder_names = [name for name in os.listdir(folder_path)]
random.shuffle(folder_names)

node_num = {}   # 记录节点数相同的图的个数

all_data_list = []
train_data_list = []
val_data_list = []
test_data_list = []

for i, file_name in enumerate(folder_names):
    print("i: ", i)

    file_path = os.path.join(folder_path, file_name)
    try:
        G = nx.read_graphml(folder_path+file_name)
    except:
        print(file_name)
    
    # 输出非连通图
    is_connected = nx.is_connected(G)
    if not is_connected:
        print("no_connected: ", i)
        print("file_name: ", file_name)
    if G.number_of_nodes() >= 50:
        # 统计不同graph的node_num
        if G.number_of_nodes() not in node_num:
            node_num[G.number_of_nodes()] = 1
        else:
            node_num[G.number_of_nodes()] += 1
        all_data_list.append(layout_compute(G, "kamada_kawai_layout", file_name))

random.shuffle(all_data_list)
train_data_list = all_data_list[0:5300]    # 11200
val_data_list = all_data_list[5300:5400]  # 100
test_data_list = all_data_list[5400: ]     # 51

print("train_data_list: ", len(train_data_list))
print("val_data_list: ", len(val_data_list))
print("test_data_list: ", len(test_data_list))

train_save_path = '../data/rome_kamada/50_train_layout.pkl'
with open(train_save_path, 'wb') as file:
    pickle.dump(train_data_list, file)    

val_save_path = '../data/rome_kamada/50_val_layout.pkl'
with open(val_save_path, 'wb') as file:
    pickle.dump(val_data_list, file)    

test_save_path = '../data/rome_kamada/50_test_layout.pkl'
with open(test_save_path, 'wb') as file:
    pickle.dump(test_data_list, file)  