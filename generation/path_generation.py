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

for i in range(10, 100):
    print("i: {}".format(i))

    # if i == 20 or i==40 or i==60 or i==80:
    #     pass
    # else:
    #     continue
    
    G = nx.path_graph(i)
    
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

    # data = Data(pos=torch_pos, edge_index=torch_edge_index, graph_name="spring_path_" + str(i) + ".graphml")
    # data_list.append(data)

    # folder_path = "../graph_data/spiral_path/"
    # # 保存图为GraphML格式
    # nx.write_graphml(G, folder_path + "spring_path_" + str(i) +  ".graphml")

    # --------------------------- Straight ---------------------------
    pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
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

    data = Data(pos=torch_pos, edge_index=torch_edge_index, graph_name="straight_path_" + str(i) +  ".graphml")
    data_list.append(data)

    folder_path = "../graph_data/spiral_path/"
    # 保存图为GraphML格式
    nx.write_graphml(G, folder_path + "straight_path_" + str(i) + ".graphml")

    # --------------------------- Sprial ---------------------------
    pos = nx.spiral_layout(G)   # 螺旋
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

    data = Data(pos=torch_pos, edge_index=torch_edge_index, graph_name="spiral_path_" + str(i) +".graphml")
    data_list.append(data)

    folder_path = "../graph_data/spiral_path/"
    # 保存图为GraphML格式
    nx.write_graphml(G, folder_path + "spiral_path_" + str(i) +".graphml")

    # example_list.append(data)



# print("example_list: ", len(example_list))
# data_save_path = '../final_data/spiral_path/example_layout.pkl'
# with open(data_save_path, 'wb') as file:
#     pickle.dump(data_list, file)  


print("data_list: ", len(data_list))
data_save_path = '../final_data/spiral_path/1_control_train_layout.pkl'
with open(data_save_path, 'wb') as file:
    pickle.dump(data_list, file)    