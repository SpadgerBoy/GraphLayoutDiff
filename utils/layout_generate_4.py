import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from torch_geometric.data import Data
import pickle
import random
import pdb

def generate_polygon_graph(n):
    # 创建一个空的图对象
    G = nx.Graph()

    # 添加节点
    for i in range(n):
        theta = i * 2 * math.pi / n
        x = math.cos(theta)
        y = math.sin(theta)
        G.add_node(i, pos=(x, y))  # 保存节点位置信息

    # 添加边
    for i in range(n):
        G.add_edge(i, (i + 1) % n)  # 添加首尾相接的边

    return G

data_list = []
for i in range(5, 90):
    print("i: ", i)
    # 生成首尾相接的正多边形图，边数为6
    n = i
    polygon_graph = generate_polygon_graph(n)

    # 绘制图形
    pos = nx.get_node_attributes(polygon_graph, 'pos')

    # 获得pos_list
    pos_values_sorted = [pos[key] for key in sorted(pos.keys())]
    pos_list = np.array(pos_values_sorted).tolist()

    # 获得edge_list
    edge_list = list(polygon_graph.edges())
    edge_list = np.array(edge_list).tolist()

    torch_pos = torch.tensor(pos_list, dtype=torch.float32)
    torch_edge_index = torch.tensor(edge_list, dtype=torch.long)
    torch_edge_index = torch.transpose(torch_edge_index, 0, 1)

    data = Data(pos=torch_pos, edge_index=torch_edge_index, graph_name="regular_" + str(i) +".graphml")
    data_list.append(data)

    # folder_path = "../graph_data/regular_graph/"
    # # 保存图为GraphML格式
    # nx.write_graphml(polygon_graph, folder_path + "regular_" + str(i) +".graphml")

print("data_list: ", len(data_list))

data_save_path = '../data/regular_graph/regular_graph.pkl'
with open(data_save_path, 'wb') as file:
    pickle.dump(data_list, file)    









