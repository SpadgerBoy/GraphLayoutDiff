import networkx as nx
import os
import numpy as np
import json
import torch
from torch_geometric.data import Data
import pickle

Max_x = -10000
Min_x = 10000
Max_y = -10000
Min_y = 10000

train_data_list = []
val_data_list = []
test_data_list = []

train_10_data_list = []
test_10_data_list = []

node_50_data_list = []

test_data_list_discrete = []

count = 0
import pdb

def layout_compute(g, layout, type, graph_name):
    global Max_x
    global Min_x
    global Max_y
    global Min_y

    for i in range(1):

        # 固定部分节点
        # fixed_nodes = {'n0': (0, 0), 'n22': (0, 2), 'n20':(1,0)}

        # 节点整数索引
        H = nx.convert_node_labels_to_integers(g, label_attribute='original_label')
        # print(H.nodes.data())

        # 计算pos
        center = [0, 0]
        
        if layout == "spring_layout":
            pos = nx.spring_layout(H, center = center)
        elif layout == "kamada_kawai_layout":
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

        # if type == 'train_5':
        #     train_10_data_list.append(data)
        # elif type=='test_5':
        #     test_10_data_list.append(data)
        # if type == 'train':
        #     train_data_list.append(data)
        # elif type == 'val':
        #     val_data_list.append(data)
        # else:
        #     test_data_list.append(data)

        # if type == 'node_50':
        #     node_50_data_list.append(data)

        
        # 获取 pos 中 x 和 y 坐标的最大值
        # max_x = max(pos.values(), key=lambda x: x[0])[0]
        # max_y = max(pos.values(), key=lambda x: x[1])[1]
        # min_x = min(pos.values(), key=lambda x: x[0])[0]
        # min_y = min(pos.values(), key=lambda x: x[1])[1]

folder_path = "../graph_data/rome/"

file_paths = os.listdir(folder_path)

node_num = {}   # 记录节点数相同的图的个数

f = [0,0,0,0,0,0,0,0,0,0,0]

for i, file_path in enumerate(file_paths):
    print("i: ", i)

    file_name = os.path.basename(file_path)
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
    
    # if i<5:
    #     layout_compute(G, "spring_layout", 'train_5', file_name)
    # elif 10131 <=i < 10136:
    #     layout_compute(G, "spring_layout", 'test_5', file_name)

    # if G.number_of_nodes() == 50:
    #     layout_compute(G, "spring_layout", 'node_50', file_name)
    #     break

    if i > 10131:
        if G.number_of_nodes() == 10 and f[0]==0:
            data = layout_compute(G, "spring_layout", 'node_discrete', file_name)
            test_data_list_discrete.append(data)
            f[0] = 1
        if G.number_of_nodes() == 20 and f[1]==0:
            data = layout_compute(G, "spring_layout", 'node_discrete', file_name)
            test_data_list_discrete.append(data)
            f[1] = 1
        if G.number_of_nodes() == 30 and f[2]==0:
            data = layout_compute(G, "spring_layout", 'node_discrete', file_name)
            test_data_list_discrete.append(data)
            f[2] = 1
        if G.number_of_nodes() == 40 and f[3]==0:
            data = layout_compute(G, "spring_layout", 'node_discrete', file_name)
            test_data_list_discrete.append(data)
            f[3] = 1
        if G.number_of_nodes() == 50 and f[4]==0:
            data = layout_compute(G, "spring_layout", 'node_discrete', file_name)
            test_data_list_discrete.append(data)
            f[4] = 1
        if G.number_of_nodes() == 60 and f[5]==0:
            data = layout_compute(G, "spring_layout", 'node_discrete', file_name)
            test_data_list_discrete.append(data)
            f[5] = 1
        if G.number_of_nodes() == 70 and f[6]==0:
            data = layout_compute(G, "spring_layout", 'node_discrete', file_name)
            test_data_list_discrete.append(data)
            f[6] = 1
        if G.number_of_nodes() == 80 and f[7]==0:
            data = layout_compute(G, "spring_layout", 'node_discrete', file_name)
            test_data_list_discrete.append(data)
            f[7] = 1
        if G.number_of_nodes() == 90 and f[8]==0:
            data = layout_compute(G, "spring_layout", 'node_discrete', file_name)
            test_data_list_discrete.append(data)
            f[8] = 1
        if G.number_of_nodes() == 100 and f[9]==0:
            data = layout_compute(G, "spring_layout", 'node_discrete', file_name)
            test_data_list_discrete.append(data)
            f[9] = 1


    # if 20<=G.number_of_nodes()<=30:
    
    #     if i < 10000:
    #         layout_compute(G, "kamada_kawai_layout", 'train', file_name)
    #     elif i < 10131:
    #         layout_compute(G, "kamada_kawai_layout", 'val', file_name)
    #     else:
    #         layout_compute(G, "kamada_kawai_layout", 'test', file_name)
    # exit()

# print("train_data_list: ", len(train_data_list))
# print("val_data_list: ", len(val_data_list))
# print("test_data_list: ", len(test_data_list))
# print("node_50_data_list: ", len(node_50_data_list))

test_data_list_discrete = sorted(test_data_list_discrete, key=lambda x: len(x['pos']))
print("test_data_list_discrete: ", len(test_data_list_discrete))

# train_save_path = '../data/rome_kamada/node_20_train_layout.pkl'
# with open(train_save_path, 'wb') as file:
#     pickle.dump(train_data_list, file)    

# val_save_path = '../data/rome_kamada/node_20_val_layout.pkl'
# with open(val_save_path, 'wb') as file:
#     pickle.dump(val_data_list, file)    

# test_save_path = '../data/rome_kamada/node_20_test_layout.pkl'
# with open(test_save_path, 'wb') as file:
#     pickle.dump(test_data_list, file)  

# train_10_save_path = '../data/rome/train_10_spring_layout.pkl'
# with open(train_10_save_path, 'wb') as file:
#     pickle.dump(train_10_data_list, file)  

# test_10_save_path = '../data/rome/test_10_spring_layout.pkl'
# with open(test_10_save_path, 'wb') as file:
#     pickle.dump(test_10_data_list, file)  

test_save_path = '../data/rome/test_data_discrete.pkl'
with open(test_save_path, 'wb') as file:
    pickle.dump(test_data_list_discrete, file)   

# node_num = dict(sorted(node_num.items(), key=lambda x: x[0]))

# # 输出节点数相同的图的个数
# for key, value in node_num.items():
#     print(key, " : ", value)