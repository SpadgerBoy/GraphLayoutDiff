import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear, Embedding
from torch_geometric.nn import MessagePassing, radius_graph
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from math import pi as PI

from ..common import MultiLayerPerceptron

import pdb

class MLPNodeEncoder(Module):

    def __init__(self, node_dim, hidden_dim=100, activation='relu'):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.type_emb = Embedding(120, embedding_dim=self.hidden_dim)
        self.mlp = MultiLayerPerceptron(self.node_dim, [self.hidden_dim, self.hidden_dim], activation=activation)

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, node_emb, node_type):
        x1 = self.mlp(node_emb) # (num_node, hidden_dim)
        x2 = self.type_emb(node_type) # (num_node, hidden_dim)
        return x1 * x2 # (num_node, hidden)

class MLPDegreeEncoder(Module):

    def __init__(self, hidden_dim=100, activation='relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.degree_emb = Embedding(20, embedding_dim=self.hidden_dim)

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, node_degree):
        x = self.degree_emb(node_degree) # (num_node, hidden_dim)
        return x # (num_node, hidden)

class MLPNodeDegreeEncoder(Module):

    def __init__(self, node_dim, hidden_dim=100, activation='relu'):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.type_emb = Embedding(20, embedding_dim=self.hidden_dim)
        self.mlp = MultiLayerPerceptron(self.node_dim, [self.hidden_dim, self.hidden_dim], activation=activation)
        self.lin = Linear(self.hidden_dim, self.hidden_dim)

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, node_emb, node_degree):
        x1 = self.mlp(node_emb) # (num_node, hidden_dim)
        x2 = self.type_emb(node_degree) # (num_node, hidden_dim)
        return self.lin(x1 * x2) # (num_node, hidden)

def get_node_encoder(config):
    return MLPNodeEncoder(config.laplacian_eigenvector, config.hidden_dim, config.mlp_act)

def get_degree_encoder(config):
    return MLPDegreeEncoder(hidden_dim = config.hidden_dim, activation = config.mlp_act)

def get_node_degree_encoder(config):
    return MLPNodeDegreeEncoder(config.laplacian_eigenvector, config.hidden_dim, config.mlp_act)
