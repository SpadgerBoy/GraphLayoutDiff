import networkx as nx
import os
import numpy as np
import json
import torch
from torch_geometric.data import Data
import pickle
import random
import pdb

f_path = "../data/rome_kamada/train_data_discrete.pkl"

with open(f_path, 'rb') as f:
    dataset = pickle.load(f)

node_num = {}

f = [0,0,0,0,0,0,0,0,0,0,0]
test_data_list_discrete = []

for g in dataset:
    if int(g.num_nodes) not in node_num:
        node_num[int(g.num_nodes)] = 1
    else:
        node_num[int(g.num_nodes)] += 1

#     if 0 < int(g.num_nodes) < 10 and f[0]==0:
#         test_data_list_discrete.append(g)
#         f[0] = 1
#     if 10 <= int(g.num_nodes) <= 20  and f[1]==0:
#         test_data_list_discrete.append(g)
#         f[1] = 1
#     if 20 < int(g.num_nodes) <= 30  and f[2]==0:
#         test_data_list_discrete.append(g)
#         f[2] = 1
#     if 30 < int(g.num_nodes) <= 40  and f[3]==0:
#         test_data_list_discrete.append(g)
#         f[3] = 1
#     if 40 < int(g.num_nodes) <= 50 and f[4]==0:
#         test_data_list_discrete.append(g)
#         f[4] = 1
#     if 50 < int(g.num_nodes) <= 60 and f[5]==0:
#         test_data_list_discrete.append(g)
#         f[5] = 1
#     if 60 < int(g.num_nodes) <= 70  and f[6]==0:
#         test_data_list_discrete.append(g)
#         f[6] = 1
#     if 70 < int(g.num_nodes) <= 80  and f[7]==0:
#         test_data_list_discrete.append(g)
#         f[7] = 1
#     if 80 < int(g.num_nodes) <= 90 and f[8]==0:
#         test_data_list_discrete.append(g)
#         f[8] = 1
#     if 90 < int(g.num_nodes) <= 100  and f[9]==0:
#         test_data_list_discrete.append(g)
#         f[9] = 1
#     if 100 < int(g.num_nodes)  and f[10]==0:
#         test_data_list_discrete.append(g)
#         f[10] = 1

# test_data_list_discrete = sorted(test_data_list_discrete, key=lambda x: len(x['pos']))

# test_save_path = '../data/rome_kamada/train_data_discrete.pkl'
# with open(test_save_path, 'wb') as file:
#     pickle.dump(test_data_list_discrete, file)  


node_num = dict(sorted(node_num.items(), key=lambda x: x[0]))

integrate_node_num = [0,0,0,0,0,0,0,0,0,0,0]

for key, value in node_num.items():
    if 0 < int(key) < 10:
        integrate_node_num[0] += int(value)
    if 10 <= int(key) <= 20:
        integrate_node_num[1] += int(value)
    if 20 < int(key) <= 30:
        integrate_node_num[2] += int(value)
    if 30 < int(key) <= 40:
        integrate_node_num[3] += int(value)
    if 40 < int(key) <= 50:
        integrate_node_num[4] += int(value)
    if 50 < int(key) <= 60:
        integrate_node_num[5] += int(value)
    if 60 < int(key) <= 70:
        integrate_node_num[6] += int(value)
    if 70 < int(key) <= 80:
        integrate_node_num[7] += int(value)
    if 80 < int(key) <= 90:
        integrate_node_num[8] += int(value)
    if 90 < int(key) <= 100:
        integrate_node_num[9] += int(value)
    if 100 < int(key):
        integrate_node_num[10] += int(value)
for i, value in enumerate(integrate_node_num):
    print(i, " : ", value)    