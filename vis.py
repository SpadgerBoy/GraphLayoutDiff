import pickle
import numpy as np
import os
import networkx as nx
import pdb

logs_name = "rome_default_2023_09_05__15_39_56"
sample_name = "sample_2023_09_06__15_32_30"

path = "logs_final/" + logs_name + "/" + sample_name + "/samples_all.pkl"
# path = "logs/sample_2023_06_25__10_50_10/samples_all.pkl"

# -----------------------------
start_idx = 0
end_idx = 4
for i in range(start_idx, end_idx):
    idx = i     # 第几个图
    with open(path, "rb") as f:
        data = pickle.load(f)[idx]


    num_nodes = int(data['node_emb'].shape[0])
    graph_name = data['graph_name']

    print("data_graph_name: ", graph_name)

    # if i ==0:
    #     graph_path = './graph_data/hierarchy_tree/hierarchy_tree2_20.graphml'
    # if i ==1:
    #     graph_path = './graph_data/hierarchy_tree/hierarchy_tree2_40.graphml'
    # if i ==2:
    #     graph_path = './graph_data/hierarchy_tree/hierarchy_tree2_60.graphml'
    # if i ==3:
    #     graph_path = './graph_data/hierarchy_tree/hierarchy_tree2_80.graphml'

    folder_path = "./graph_data/grid"
    graph_path = os.path.join(folder_path, graph_name)

    G = nx.read_graphml(graph_path)
    H = nx.convert_node_labels_to_integers(G, label_attribute='original_label')

    # -----------------------------
    pos_idx = 0     # 第几种pos
    positions = data.pos_gen[pos_idx*num_nodes: (pos_idx+1)*num_nodes, :].tolist()
    # positions = data.pos_ref[pos_idx*num_nodes: (pos_idx+1)*num_nodes, :].tolist()


    pos_dict = {node: (positions[node][0], positions[node][1]) for node in range(0, num_nodes)}
    nx.set_node_attributes(H, pos_dict, 'pos')

    path_generate = "logs_final/" + logs_name + "/" + sample_name + "/generate/" + str(pos_idx) + "_test" + str(idx) +".pkl"
    with open(path_generate, 'wb') as file:
        pickle.dump(H, file)


