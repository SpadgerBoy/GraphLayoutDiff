import pickle
import pdb

data = []

path_1 = "../data/rome_kamada/train_layout.pkl"
data_1 = []
with open(path_1, 'rb') as f:
    data_1 = tmp = pickle.load(f)
    data.extend(data_1)

path_2 = "../data/path_graph/path_graph.pkl"
data_2 = []
with open(path_2, 'rb') as f:
    tmp = pickle.load(f)
    for _ in range(80):
        data_2.extend(tmp)
    data.extend(data_2)

path_3 = "../data/tree_graph/tree_graph.pkl"
data_3 = []
with open(path_3, 'rb') as f:
    tmp = pickle.load(f)
    for _ in range(10):
        data_3.extend(tmp)
    
    data.extend(data_3)

data_save_path = '../data/rome_tree_path_graph/rome_tree_path_graph.pkl'
with open(data_save_path, 'wb') as file:
    pickle.dump(data, file)    