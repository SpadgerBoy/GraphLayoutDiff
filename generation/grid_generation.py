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
for i in range(2, 15):
    for j in range(2, 15):
        n = i*j
        if n<10 or n>100:
            continue

        if (i==5 and j==4) or (i==8 and j==5) or (i==10 and j==6) or (i==10 and j==8):
            pass
        else:
            continue

        print("i: {}".format(i))
        print("j: {}".format(j))
        


        # --------------------------- Spring ---------------------------
        # G = nx.grid_2d_graph(i, j)
        # H = nx.convert_node_labels_to_integers(G, label_attribute='original_label')
        # pos = nx.planar_layout(G)
        # pos = nx.spring_layout(G,pos=pos) 
        # pos = np.array(list(pos.values()))
        # pos = nx.rescale_layout(pos, scale=1)

        # # 获得pos_list
        # pos_list = np.array(pos).tolist()

        # # 获得edge_list
        # edge_list = list(H.edges())
        # edge_list = np.array(edge_list).tolist()

        # torch_pos = torch.tensor(pos_list, dtype=torch.float32)
        # torch_edge_index = torch.tensor(edge_list, dtype=torch.long)
        # torch_edge_index = torch.transpose(torch_edge_index, 0, 1)

        # data = Data(pos=torch_pos, edge_index=torch_edge_index, graph_name="spring_grid_" + str(i) + "_" + str(j) + ".graphml")
        # data_list.append(data)

        # folder_path = "../graph_data/grid/"
        # # 保存图为GraphML格式
        # nx.write_graphml(G, folder_path + "spring_grid_" + str(i) + "_" + str(j) + ".graphml")

        # --------------------------- Lean ---------------------------
        G = nx.grid_2d_graph(i, j)
        H = nx.convert_node_labels_to_integers(G, label_attribute='original_label')
        pos = {}
        tmp_n = 0
        for x, y in G.nodes():
            pos[tmp_n] = np.array([float(y + 0.6 * x), float(x)])
            tmp_n += 1

        tmp_pos = np.array(list(pos.values()))
        tmp_pos = np.array(tmp_pos).tolist()

        # # 找到 x 和 y 的最大值和最小值

        x_values = [tmp_pos[i][0] for i in range(len(tmp_pos))]
        y_values = [tmp_pos[i][1] for i in range(len(tmp_pos))]
        # x_values = [pos[node][0] for node in G.nodes()]
        # y_values = [pos[node][1] for node in G.nodes()]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        # 缩放坐标值到 [-1, 1] 范围内
        # for node in G.nodes():
        for value in pos.values():
            value[0] = ((value[0] - x_min) / (x_max - x_min) - 0.5) * 2
            value[1] = ((value[1]  - y_min) / (y_max - y_min) - 0.5) * 2

        pos = np.array(list(pos.values()))
        pos = nx.rescale_layout(pos, scale=1)

        # 获得pos_list
        # pos_values_sorted = [pos[key] for key in sorted(pos.keys())]
        pos_list = np.array(pos).tolist()

        # 获得edge_list
        edge_list = list(H.edges())
        edge_list = np.array(edge_list).tolist()

        torch_pos = torch.tensor(pos_list, dtype=torch.float32)
        torch_edge_index = torch.tensor(edge_list, dtype=torch.long)
        torch_edge_index = torch.transpose(torch_edge_index, 0, 1)

        data = Data(pos=torch_pos, edge_index=torch_edge_index, graph_name="lean_grid_" + str(i) + "_" + str(j) + ".graphml")
        data_list.append(data)


        folder_path = "../graph_data/grid/"
        # 保存图为GraphML格式
        # nx.write_graphml(G, folder_path + "lean_grid_" + str(i) + "_" + str(j) + ".graphml")


        # --------------------------- Grid ---------------------------
        G = nx.grid_2d_graph(i, j)
        H = nx.convert_node_labels_to_integers(G, label_attribute='original_label')
        pos = {}
        tmp_n = 0
        for x, y in G.nodes():
            pos[tmp_n] = np.array([float(y), float(x)])
            tmp_n += 1
        # pos = {(x, y): np.array([float(y), float(-x)]) for x, y in G.nodes()} 

        tmp_pos = np.array(list(pos.values()))
        tmp_pos = np.array(tmp_pos).tolist()
        # # pos = np.array(list(pos.values()))
        # print("pos: ", pos)

        # # 找到 x 和 y 的最大值和最小值

        x_values = [tmp_pos[i][0] for i in range(len(tmp_pos))]
        y_values = [tmp_pos[i][1] for i in range(len(tmp_pos))]
        # x_values = [pos[node][0] for node in G.nodes()]
        # y_values = [pos[node][1] for node in G.nodes()]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        # 缩放坐标值到 [-1, 1] 范围内
        # for node in G.nodes():
        for value in pos.values():
            value[0] = ((value[0] - x_min) / (x_max - x_min) - 0.5) * 2
            value[1] = ((value[1]  - y_min) / (y_max - y_min) - 0.5) * 2

        pos = np.array(list(pos.values()))
        pos = nx.rescale_layout(pos, scale=1)

        # 获得pos_list
        # pos_values_sorted = [pos[key] for key in sorted(pos.keys())]
        pos_list = np.array(pos).tolist()

        # 获得edge_list
        edge_list = list(H.edges())
        edge_list = np.array(edge_list).tolist()

        torch_pos = torch.tensor(pos_list, dtype=torch.float32)
        torch_edge_index = torch.tensor(edge_list, dtype=torch.long)
        torch_edge_index = torch.transpose(torch_edge_index, 0, 1)

        data = Data(pos=torch_pos, edge_index=torch_edge_index, graph_name="grid_" + str(i) + "_" + str(j) + ".graphml")
        data_list.append(data)


        folder_path = "../graph_data/grid/"
        # 保存图为GraphML格式
        # nx.write_graphml(G, folder_path + "grid_" + str(i) + "_" + str(j) + ".graphml")

        example_list.append(data)

print("example_list: ", len(example_list))
data_save_path = '../final_data/grid/example_layout.pkl'
with open(data_save_path, 'wb') as file:
    pickle.dump(example_list, file)  

# print("data_list: ", len(data_list))
# data_save_path = '../final_data/grid/1_control_train_layout.pkl'
# with open(data_save_path, 'wb') as file:
#     pickle.dump(data_list, file)   

# 4*5
# 5*8
# 6*10
# 8*10 