import networkx as nx
import os
import numpy as np
import json
import torch
from torch_geometric.data import Data
import pickle
import random
import pdb


def data_process(graph_data):

    # 创建一个无向图
    graph = nx.Graph()

    # 添加节点
    for node in graph_data['nodes']:
        graph.add_node(node['id'])

    # 添加边
    for link in graph_data['links']:
        graph.add_edge(link['source'], link['target'])
    
    H = nx.convert_node_labels_to_integers(graph, label_attribute='original_label')

    fragment_mask = np.array(graph_data['mask']).tolist()
    torch_fragment_mask = torch.tensor(fragment_mask, dtype=torch.long).unsqueeze(-1)
    torch_linker_mask = 1-torch_fragment_mask
    # linker_mask = [0 if element == 1 else 1 for element in fragment_mask]

    # 从节点数据中提取 x 和 y 坐标
    x_coords = [node["x"] for node in graph_data['nodes']]
    y_coords = [node["y"] for node in graph_data['nodes']]

    # 将 x 和 y 坐标合并为一个大小为 n*2 的 tensor
    pos_ref = np.array(np.vstack((x_coords, y_coords)).T).tolist()
    torch_pos_ref = torch.tensor(pos_ref, dtype=torch.float)

    edge_list = list(H.edges())
    edge_list = np.array(edge_list).tolist()

    torch_edge_index = torch.tensor(edge_list, dtype=torch.long)
    torch_edge_index = torch.transpose(torch_edge_index, 0, 1)

    control_flag = graph_data['control_flag']

    data = Data(pos_ref=torch_pos_ref, edge_index=torch_edge_index, fragment_mask=torch_fragment_mask, linker_mask=torch_linker_mask, control_flag = control_flag)

    return data


folder_path = "../final_data/control_json/"
folder_names = [name for name in os.listdir(folder_path)]

data_list = []
for i, file_name in enumerate(folder_names):
    if file_name == "query_hierarchy_tree_no_center.json":
        continue
    if file_name == "query_radial_tree.json":
        continue
    if file_name == "query_spiral_path.json":
        continue
    if file_name == "query_spiral_path_new.json":
        continue
    # print("file_name: ", file_name)
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r") as f:
        json_data = json.load(f)
    data = data_process(json_data)
    data_list.append(data)


print("data_list: ", len(data_list))
example_save_path = '../final_data/control_final/example_layout_4.pkl'
with open(example_save_path, 'wb') as file:
    pickle.dump(data_list, file)    


