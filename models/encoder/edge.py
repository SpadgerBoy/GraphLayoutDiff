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

class GaussianSmearingEdgeEncoder(Module):

    def __init__(self, num_gaussians=64, cutoff=10.0):
        super().__init__()
        #self.NUM_BOND_TYPES = 22
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.rbf = GaussianSmearing(start=0.0, stop=cutoff * 2, num_gaussians=num_gaussians)    # Larger `stop` to encode more cases
        self.bond_emb = Embedding(100, embedding_dim=num_gaussians)

    @property
    def out_channels(self):
        return self.num_gaussians * 2

    def forward(self, edge_length, edge_type):
        """
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * num_gaussians)
        """
        edge_attr = torch.cat([self.rbf(edge_length), self.bond_emb(edge_type)], dim=1)
        return edge_attr


class MLPEdgeEncoder(Module):

    def __init__(self, hidden_dim=100, activation='relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_emb = Embedding(10, embedding_dim=self.hidden_dim)
        self.mlp = MultiLayerPerceptron(1, [self.hidden_dim, self.hidden_dim], activation=activation)

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, edge_length, edge_type):
        d_emb = self.mlp(edge_length) # (num_edge, hidden_dim)
        edge_attr = self.bond_emb(edge_type) # (num_edge, hidden_dim)
        return d_emb * edge_attr # (num_edge, hidden)

class MLPEdgeEncoder_1(Module):

    def __init__(self, hidden_dim=100, activation='relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_emb = Embedding(10, embedding_dim=self.hidden_dim)
        self.mlp = MultiLayerPerceptron(2, [self.hidden_dim, self.hidden_dim], activation=activation)

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, dd_dr, edge_type):
        """
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * hidden_dim)
        """
        d_emb = self.mlp(dd_dr) # (num_edge, hidden_dim)
        edge_attr = self.bond_emb(edge_type) # (num_edge, hidden_dim)
        return d_emb * edge_attr # (num_edge, hidden)

class MLPEdgeEncoder_2(Module):

    def __init__(self, hidden_dim=100, activation='relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_emb = Embedding(10, embedding_dim=self.hidden_dim)
        self.mlp = MultiLayerPerceptron(1, [self.hidden_dim, self.hidden_dim], activation=activation)
        self.mlp2 = MultiLayerPerceptron(hidden_dim*2, [self.hidden_dim, self.hidden_dim], activation=activation)

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, edge_length, edge_type):
        """
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * hidden_dim)
        """
        d_emb = self.mlp(edge_length) # (num_edge, hidden_dim)
        edge_attr = self.bond_emb(edge_type) # (num_edge, hidden_dim)
        out = self.mlp2(torch.cat([d_emb, edge_attr], dim=-1))
        return out # (num_edge, hidden)

class MLPEdgeEncoder_3(Module):

    def __init__(self, hidden_dim=100, activation='relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_emb = MultiLayerPerceptron(1, [self.hidden_dim, self.hidden_dim], activation=activation)
        self.mlp = MultiLayerPerceptron(1, [self.hidden_dim, self.hidden_dim], activation=activation)

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, edge_length, edge_type):
        d_emb = self.mlp(edge_length) # (num_edge, hidden_dim)
        edge_type = edge_type.unsqueeze(-1).float()
        edge_attr = self.bond_emb(edge_type) # (num_edge, hidden_dim)
        return d_emb * edge_attr # (num_edge, hidden)

def get_edge_encoder(config):
    if config.edge_encoder == 'mlp':
        return MLPEdgeEncoder(config.hidden_dim, config.mlp_act)
    elif config.edge_encoder == 'gaussian':
        return GaussianSmearingEdgeEncoder(config.hidden_dim // 2, cutoff=config.cutoff)
    else:
        raise NotImplementedError('Unknown edge encoder: %s' % config.edge_encoder)
    
def get_edge_encoder_1(config):
    if config.edge_encoder == 'mlp':
        return MLPEdgeEncoder_1(config.hidden_dim, config.mlp_act)
    elif config.edge_encoder == 'gaussian':
        return GaussianSmearingEdgeEncoder(config.hidden_dim // 2, cutoff=config.cutoff)
    else:
        raise NotImplementedError('Unknown edge encoder: %s' % config.edge_encoder)
    
def get_edge_encoder_2(config):
    if config.edge_encoder == 'mlp':
        return MLPEdgeEncoder_2(config.hidden_dim, config.mlp_act)
    elif config.edge_encoder == 'gaussian':
        return GaussianSmearingEdgeEncoder(config.hidden_dim // 2, cutoff=config.cutoff)
    else:
        raise NotImplementedError('Unknown edge encoder: %s' % config.edge_encoder)
    
def get_edge_encoder_3(config):
    if config.edge_encoder == 'mlp':
        return MLPEdgeEncoder_3(config.hidden_dim, config.mlp_act)
    elif config.edge_encoder == 'gaussian':
        return GaussianSmearingEdgeEncoder(config.hidden_dim // 2, cutoff=config.cutoff)
    else:
        raise NotImplementedError('Unknown edge encoder: %s' % config.edge_encoder)