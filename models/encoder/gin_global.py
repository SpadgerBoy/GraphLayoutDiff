from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Module, Sequential, ModuleList, Linear, Embedding

from ..common import MultiLayerPerceptron

import pdb

class GINConv_Global(MessagePassing):

    def __init__(self, nn_1:Callable, nn_2:Callable, eps: float=0., train_eps: bool=False,
                 activation="softplus",cutoff=10.0, **kwargs):
        super(GINConv_Global, self).__init__(aggr='add', **kwargs)
        self.nn_1 = nn_1
        self.nn_2 = nn_2
        self.initial_eps = eps
        self.cutoff = cutoff

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor=None, edge_length: Tensor=None, size: Size=None, ) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        
        # Node和edge的attribute的维度需要统一
        if isinstance(edge_index, Tensor):
            assert edge_attr is not None
            assert x[0].size(-1) == edge_attr.size(-1)
        elif isinstance(edge_index, SparseTensor):
            assert x[0].size(-1) == edge_index.size(-1)
        
        C = (edge_length <= self.cutoff).float() 
        W = W = self.nn_2(edge_attr) * C.view(-1, 1) 

        out = self.propagate(edge_index, x=x, W=W, size=size)

        x_r = x[1]

        if x_r is not None:
            out += (1 + self.eps) * x_r
        
        return self.nn_1(out)
    
    def message(self, x_j: Tensor, W: Tensor) -> Tensor:

        if self.activation:
            return self.activation(x_j + W)
        else:
            return x_j + W
    
    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

class GINEncoder_Global(torch.nn.Module):

    def __init__(self, hidden_dim, num_convs=3, activation='relu', short_cut=True, concat_hidden=False, node_channels=5, cutoff=10.0,):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_convs = num_convs
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.cutoff = cutoff
        # self.node_emb = nn.Embedding(100, hidden_dim)
        self.node_emb = MultiLayerPerceptron(node_channels, [self.hidden_dim, self.hidden_dim], activation=activation)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        
        mlp_1 = MultiLayerPerceptron(hidden_dim, [hidden_dim, hidden_dim], activation=activation)
        mlp_2 = MultiLayerPerceptron(hidden_dim, [hidden_dim, hidden_dim], activation=activation)

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            self.convs.append(GINConv_Global(nn_1=mlp_1, nn_2=mlp_2, activation=activation, cutoff=self.cutoff))
        
    def forward(self, z, edge_index, edge_attr, edge_length):

        node_attr = self.node_emb(z)

        hiddens = []
        conv_input = node_attr

        for conv_id, conv in enumerate(self.convs):
            hidden = conv(conv_input, edge_index, edge_attr, edge_length = edge_length)
            if conv_id < len(self.convs)-1 and self.activation is not None:
                hidden = self.activation(hidden)
            assert hidden.shape == conv_input.shape
            if self.short_cut and hidden.shape == conv_input.shape: # res连接
                hidden += conv_input
            
            hiddens.append(hidden)
            conv_input = hidden
        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        
        return node_feature
            
        