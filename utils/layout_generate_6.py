# 生成tree
import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
import random
import pdb

data_list = []
for i in range(5, 100):
    print("i: {}".format(i))
    
    G = nx.path_graph(i)
    pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
    pos = np.array(list(pos.values()))
    pos = nx.rescale_layout(pos, scale=1)

    # 获得pos_list
    # pos_values_sorted = [pos[key] for key in sorted(pos.keys())]
    pos_list = np.array(pos).tolist()

    # 获得edge_list
    edge_list = list(G.edges())
    edge_list = np.array(edge_list).tolist()

    torch_pos = torch.tensor(pos_list, dtype=torch.float32)
    torch_edge_index = torch.tensor(edge_list, dtype=torch.long)
    torch_edge_index = torch.transpose(torch_edge_index, 0, 1)

    data = Data(pos=torch_pos, edge_index=torch_edge_index, graph_name="path_" + str(i) +".graphml")
    data_list.append(data)

    folder_path = "../graph_data/path_graph/"
    # 保存图为GraphML格式
    nx.write_graphml(G, folder_path + "path_" + str(i) +".graphml")


print("data_list: ", len(data_list))
data_save_path = '../data/path_graph/path_graph.pkl'
with open(data_save_path, 'wb') as file:
    pickle.dump(data_list, file)    