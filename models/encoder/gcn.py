from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing

from ..common import MultiLayerPerceptron

import pdb

from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.nn.inits import glorot, zeros


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass



class GCNConv(MessagePassing):


    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = False,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor=None) -> Tensor:
        """"""


        x = x @ self.weight

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        if self.bias is not None:
            out += self.bias

        return out


    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return F.relu((x_j + edge_attr))


    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class GCNEncoder(torch.nn.Module):

    def __init__(self, hidden_dim, num_convs=3, activation='relu', short_cut=True, concat_hidden=False, node_channels=5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_convs = num_convs
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        # self.node_emb = nn.Embedding(100, hidden_dim)
        self.node_emb = MultiLayerPerceptron(node_channels, [self.hidden_dim, self.hidden_dim], activation=activation)

        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        
        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            self.convs.append(GCNConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, edge_dim=self.hidden_dim))
        
    def forward(self, z, edge_index, edge_attr, embed_node=False):
        if embed_node:
            node_attr = self.node_emb(z)
        else:
            node_attr = z

        hiddens = []
        conv_input = node_attr

        for conv_id, conv in enumerate(self.convs):
            hidden = conv(conv_input, edge_index, edge_attr)
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
            
        