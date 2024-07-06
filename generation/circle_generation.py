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
for i in range(10, 101):
    print("i: {}".format(i))

    # if i == 20 or i==40 or i==60 or i==80:
    #     pass
    # else:
    #     continue
    

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

    # data = Data(pos=torch_pos, edge_index=torch_edge_index, graph_name="spring_cycle_" + str(i) + ".graphml")
    # data_list.append(data)

    # folder_path = "../graph_data/circle/"
    # # 保存图为GraphML格式
    # nx.write_graphml(G, folder_path + "spring_cycle_" + str(i)  + ".graphml")

    # --------------------------- Square ---------------------------
    if i%4 == 0:
        G = nx.cycle_graph(i) 
        # 获取节点数量
        num_nodes = len(G.nodes())

        # 计算节点在四条边上的坐标
        pos = {}
        for j, node in enumerate(G.nodes()):
            if j < num_nodes / 4:
                # 左边
                x = -1
                y = j / (num_nodes / 4) * 2 - 1
            elif j < num_nodes / 2:
                # 上边
                x = (j - num_nodes / 4) / (num_nodes / 4) * 2 - 1
                y = 1
            elif j < num_nodes * 3 / 4:
                # 右边
                x = 1
                y = 1 - (j - num_nodes / 2) / (num_nodes / 4) * 2
            else:
                # 下边
                x = 1 - (j - num_nodes * 3 / 4) / (num_nodes / 4) * 2
                y = -1

            pos[node] = [x, y]
        
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

        data = Data(pos=torch_pos, edge_index=torch_edge_index, graph_name="square_circle_" + str(i) +".graphml")
        data_list.append(data)

        folder_path = "../graph_data/circle/"
        # 保存图为GraphML格式
        nx.write_graphml(G, folder_path + "square_circle_" + str(i) +".graphml")


    # --------------------------- Circular ---------------------------
    G = nx.cycle_graph(i) 
    pos = nx.circular_layout(G)   

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

    data = Data(pos=torch_pos, edge_index=torch_edge_index, graph_name="circle_" + str(i) +".graphml")
    data_list.append(data)

    folder_path = "../graph_data/circle/"
    # 保存图为GraphML格式
    nx.write_graphml(G, folder_path + "circle_" + str(i) +".graphml")

    # example_list.append(data)


# print("example_list: ", len(example_list))
# data_save_path = '../final_data/circle/example_layout.pkl'
# with open(data_save_path, 'wb') as file:
#     pickle.dump(example_list, file)  

print("data_list: ", len(data_list))
data_save_path = '../final_data/circle/1_control_train_layout.pkl'
with open(data_save_path, 'wb') as file:
    pickle.dump(data_list, file)    