import networkx as nx
import os
import numpy as np
import json
import torch
from torch_geometric.data import Data
import pickle
import random
import pdb

final_rome = "../final_data/rome/train_layout.pkl"

final_hierarchy_tree = "../final_data/hierarchy_tree/1_control_train_layout.pkl"

final_spiral_path = "../final_data/spiral_path/1_control_train_layout.pkl"

final_circle = "../final_data/circle/1_control_train_layout.pkl"

final_grid = "../final_data/grid/1_control_train_layout.pkl"

final_triangular = "../final_data/triangular/1_control_train_layout.pkl"

data_list = []

# with open(final_rome, 'rb') as f:
#     rome = pickle.load(f)

with open(final_hierarchy_tree, 'rb') as f:
    tree = pickle.load(f)

with open(final_spiral_path, 'rb') as f:
    path = pickle.load(f)

with open(final_circle, 'rb') as f:
    circle = pickle.load(f)

with open(final_grid, 'rb') as f:
    grid = pickle.load(f)

with open(final_triangular, 'rb') as f:
    triangular = pickle.load(f)

# data_list.extend(rome)
data_list.extend(tree)
data_list.extend(path)
data_list.extend(circle)
data_list.extend(grid)
data_list.extend(triangular)

print("data_list: ", len(data_list))
data_save_path = '../final_data/compose_no_rome/1_control_train_layout.pkl'
with open(data_save_path, 'wb') as file:
    pickle.dump(data_list, file)   
