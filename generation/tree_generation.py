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
example_list = []
for i in range(2, 3):
    for j in range(10, 100):
        print("i: {}, j: {} ".format(i, j))


        # if (i==2 and j==20) or (i==2 and j==40) or (i==2 and j==60) or (i==2 and j==80):
        #     pass
        # else:
        #     continue
        
        G = nx.full_rary_tree(i, j)

        # --------------------------- Spring ---------------------------
        # pos = nx.planar_layout(G)
        # pos = nx.spring_layout(G,pos=pos) 
        # pos = np.array(list(pos.values()))
        # pos = nx.rescale_layout(pos, scale=1)

        # # 获得pos_list
        # pos_list = np.array(pos).tolist()

        # # 获得edge_list
        # edge_list = list(G.edges())
        # edge_list = np.array(edge_list).tolist()

        # torch_pos = torch.tensor(pos_list, dtype=torch.float32)
        # torch_edge_index = torch.tensor(edge_list, dtype=torch.long)
        # torch_edge_index = torch.transpose(torch_edge_index, 0, 1)

        # data = Data(pos=torch_pos, edge_index=torch_edge_index, graph_name="spring_tree_" + str(i) + "_" + str(j) + ".graphml")
        # data_list.append(data)

        # folder_path = "../graph_data/hierarchy_tree/"
        # # 保存图为GraphML格式
        # nx.write_graphml(G, folder_path + "spring_tree_" + str(i) + "_" + str(j) + ".graphml")

        # --------------------------- Radial ---------------------------
        pos = nx.kamada_kawai_layout(G)
        pos = np.array(list(pos.values()))
        pos = nx.rescale_layout(pos, scale=1)

        # 获得pos_list
        pos_list = np.array(pos).tolist()

        # 获得edge_list
        edge_list = list(G.edges())
        edge_list = np.array(edge_list).tolist()

        torch_pos = torch.tensor(pos_list, dtype=torch.float32)
        torch_edge_index = torch.tensor(edge_list, dtype=torch.long)
        torch_edge_index = torch.transpose(torch_edge_index, 0, 1)

        data = Data(pos=torch_pos, edge_index=torch_edge_index, graph_name="radial_tree_" + str(i) + "_" + str(j) + ".graphml")
        data_list.append(data)

        folder_path = "../graph_data/hierarchy_tree/"
        # 保存图为GraphML格式
        nx.write_graphml(G, folder_path + "radial_tree_" + str(i) + "_" + str(j) + ".graphml")

        # --------------------------- Hierarchy ---------------------------
        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        pos = np.array(list(pos.values()))
        pos = nx.rescale_layout(pos, scale=1)

        # 找到 x 和 y 的最大值和最小值
        x_values = [pos[node][0] for node in G.nodes()]
        y_values = [pos[node][1] for node in G.nodes()]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        # 缩放坐标值到 [-0.5, 0.5] 范围内
        for node in G.nodes():
            pos[node] = [((pos[node][0] - x_min) / (x_max - x_min) - 0.5) * 2, ((pos[node][1] - y_min) / (y_max - y_min) - 0.5) * 2]

        # 获得pos_list
        pos_list = np.array(pos).tolist()

        # 获得edge_list
        edge_list = list(G.edges())
        edge_list = np.array(edge_list).tolist()

        torch_pos = torch.tensor(pos_list, dtype=torch.float32)
        torch_edge_index = torch.tensor(edge_list, dtype=torch.long)
        torch_edge_index = torch.transpose(torch_edge_index, 0, 1)

        data = Data(pos=torch_pos, edge_index=torch_edge_index, graph_name="hierarchy_tree_" + str(i) + "_" + str(j) + ".graphml")
        data_list.append(data)

        folder_path = "../graph_data/hierarchy_tree/"
        # 保存图为GraphML格式
        nx.write_graphml(G, folder_path + "hierarchy_tree_" + str(i) + "_" + str(j) + ".graphml")
        # example_list.append(data)


# print("example_list: ", len(example_list))
# data_save_path = '../final_data/hierarchy_tree/example_layout.pkl'
# with open(data_save_path, 'wb') as file:
#     pickle.dump(example_list, file)  


print("data_list: ", len(data_list))
data_save_path = '../final_data/hierarchy_tree/1_control_train_layout.pkl'
with open(data_save_path, 'wb') as file:
    pickle.dump(data_list, file)    