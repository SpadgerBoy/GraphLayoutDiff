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
for i in range(2, 10):
    for j in range(5, 100):
        print("i: {}, j: {} ".format(i, j))
        
        G = nx.full_rary_tree(i, j)

        # pos = nx.kamada_kawai_layout(G)
        # pos_values_sorted = [pos[key] for key in sorted(pos.keys())]
        # pos_list = np.array(pos_values_sorted).tolist()

        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        pos = np.array(list(pos.values()))
        pos = nx.rescale_layout(pos, scale=1)

        # # 找到 x 和 y 的最大值和最小值
        # x_values = [pos[node][0] for node in G.nodes()]
        # y_values = [pos[node][1] for node in G.nodes()]
        # x_min, x_max = min(x_values), max(x_values)
        # y_min, y_max = min(y_values), max(y_values)

        # # 缩放坐标值到 [-0.5, 0.5] 范围内
        # for node in G.nodes():
        #     pos[node] = [((pos[node][0] - x_min) / (x_max - x_min) - 0.5) * 2, ((pos[node][1] - y_min) / (y_max - y_min) - 0.5) * 2]

        # 获得pos_list
        # pos_values_sorted = [pos[key] for key in sorted(pos.keys())]
        pos_list = np.array(pos).tolist()

        # 获得edge_list
        edge_list = list(G.edges())
        edge_list = np.array(edge_list).tolist()

        torch_pos = torch.tensor(pos_list, dtype=torch.float32)
        torch_edge_index = torch.tensor(edge_list, dtype=torch.long)
        torch_edge_index = torch.transpose(torch_edge_index, 0, 1)

        data = Data(pos=torch_pos, edge_index=torch_edge_index, graph_name="tree_" + str(i) + "_" + str(j) + ".graphml")
        data_list.append(data)

        folder_path = "../graph_data/tree_1_graph/"
        # 保存图为GraphML格式
        nx.write_graphml(G, folder_path + "tree_" + str(i) + "_" + str(j) + ".graphml")


print("data_list: ", len(data_list))
data_save_path = '../data/tree_1_graph/tree_graph.pkl'
with open(data_save_path, 'wb') as file:
    pickle.dump(data_list, file)    