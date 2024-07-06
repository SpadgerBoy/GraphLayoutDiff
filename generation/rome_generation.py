import networkx as nx
import os
import numpy as np
import json
import torch
from torch_geometric.data import Data
import pickle
import random
import pdb


def layout_compute(g, graph_name):

    # 节点整数索引
    H = nx.convert_node_labels_to_integers(g, label_attribute='original_label')

    center = [0, 0]
    
    pos = nx.kamada_kawai_layout(H, center = center)
    # pos = nx.kamada_kawai_layout(H, center = center)
    # pos = nx.spectral_layout(g)
    # pos = nx.circular_layout(g)
    # pos = nx.spring_layout(g,  pos=fixed_nodes, fixed=fixed_nodes.keys())

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

example_data_list = []

for i, file_name in enumerate(folder_names):
    print("i: ", i)
    if file_name=="grafo3024.10.graphml" or file_name=="grafo940.20.graphml" or file_name=="grafo1002.20.graphml" \
        or file_name=="grafo1702.30.graphml" or file_name=="grafo1246.30.graphml" or file_name=="grafo490.40.graphml" \
        or file_name=="grafo3684.50.graphml" or file_name=="grafo6034.60.graphml" or file_name=="grafo5120.60.graphml" \
        or file_name=="grafo7801.70.graphml" or file_name=="grafo8091.70.graphml" or file_name=="grafo9352.80.graphml"\
        or file_name=="grafo8067.80.graphml" or file_name=="grafo11440.90.graphml" or file_name=="grafo10372.100.graphml"\
        or file_name=="grafo10820.100.graphml" or file_name=="grafo10525.100.graphml":
        pass
    else:
        continue

    file_path = os.path.join(folder_path, file_name)
    try:
        G = nx.read_graphml(folder_path+file_name)
    except:
        print(file_name)
    
    # 统计不同graph的node_num
    if G.number_of_nodes() not in node_num:
        node_num[G.number_of_nodes()] = 1
    else:
        node_num[G.number_of_nodes()] += 1

    # 输出非连通图
    is_connected = nx.is_connected(G)
    if not is_connected:
        print("no_connected: ", i)
        print("file_name: ", file_name)
    
    # all_data_list.append(layout_compute(G, file_name))

    example_data_list.append(layout_compute(G, file_name))

# random.shuffle(all_data_list)
# # train_data_list = all_data_list[0:11200]    # 11200
# train_data_list = all_data_list[0: ]    # 11200
# val_data_list = all_data_list[11200:11400]  # 200
# test_data_list = all_data_list[11400: ]     # 131

# print("train_data_list: ", len(train_data_list))
# print("val_data_list: ", len(val_data_list))
# print("test_data_list: ", len(test_data_list))

# train_save_path = '../final_data/rome/train_layout.pkl'
# with open(train_save_path, 'wb') as file:
#     pickle.dump(train_data_list, file)    

# val_save_path = '../final_data/rome/val_layout.pkl'
# with open(val_save_path, 'wb') as file:
#     pickle.dump(val_data_list, file)    

# test_save_path = '../final_data/rome/test_layout.pkl'
# with open(test_save_path, 'wb') as file:
#     pickle.dump(test_data_list, file)  

print("example_data_list: ", len(example_data_list))
example_save_path = '../final_data/rome/1_example_layout.pkl'
with open(example_save_path, 'wb') as file:
    pickle.dump(example_data_list, file)    

# node_num = dict(sorted(node_num.items(), key=lambda x: x[0]))

# # 输出节点数相同的图的个数
# for key, value in node_num.items():
#     print(key, " : ", value)


# grafo3024.10.graphml
# grafo940.20.graphml grafo1002.20.graphml
# grafo1702.30.graphml grafo1246.30.graphml
# grafo490.40.graphml
# grafo3684.50.graphml
# grafo6034.60.graphml grafo5120.60.graphml
# grafo7801.70.graphml grafo8091.70.graphml
# grafo9352.80.graphml grafo8067.80.graphml
# grafo11440.90.graphml
# grafo10372.100.graphml grafo10820.100.graphml grafo10525.100.graphml
