from typing import Callable, Union
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing

from ..common import MultiLayerPerceptron


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result

class GCLConv(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf,  activation, normalization_factor=100, aggregation_method='sum',
                 edges_in_d=0, attention=False, normalization=None):
        super(GCLConv, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            activation,
            nn.Linear(hidden_nf, hidden_nf),
            activation)
        
        # self.edge_mlp = MultiLayerPerceptron(
        #     input_edge + edges_in_d,
        #     [hidden_nf, hidden_nf, hidden_nf],
        #     activation = activation
        # )

        if normalization is None:
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf, hidden_nf),
                activation,
                nn.Linear(hidden_nf, output_nf)
            )

            # self.node_mlp = MultiLayerPerceptron(
            #     hidden_nf + input_nf,
            #     [hidden_nf, hidden_nf, output_nf],
            #     activation = activation
            # )

        elif normalization == 'batch_norm':
            self.node_mlp = nn.Sequential(
                nn.Linear(hidden_nf + input_nf, hidden_nf),
                nn.BatchNorm1d(hidden_nf),
                activation,
                nn.Linear(hidden_nf, output_nf),
                nn.BatchNorm1d(output_nf),
            )
        else:
            raise NotImplementedError

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        return out

    def node_model(self, h, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        
        agg = torch.cat([h, agg], dim=1)
        out = h + self.node_mlp(agg)
        return out

    def forward(self, h, edge_index, edge_attr=None):
        row, col = edge_index
        edge_feat = self.edge_model(h[row], h[col], edge_attr)
        h = self.node_model(h, edge_index, edge_feat)
        return h

class GCLEncoder(torch.nn.Module):

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
            # self.convs.append(GCLConv(self.hidden_dim, self.hidden_dim, self.hidden_dim, edges_in_d=hidden_dim,
            #                                   activation=nn.ReLU()))
            
            self.convs.append(GCLConv(self.hidden_dim, self.hidden_dim, self.hidden_dim, edges_in_d=hidden_dim,
                                    activation=nn.SiLU()))
    
    def forward(self, z, edge_index, edge_attr, embed_node=False):
        if embed_node:
            node_attr = self.node_emb(z)
        else:
            node_attr = z

        hiddens = []
        conv_input = node_attr

        for conv_id, conv in enumerate(self.convs):
            hidden = conv(conv_input, edge_index, edge_attr)
            # if conv_id < len(self.convs)-1 and self.activation is not None:
            #     hidden = self.activation(hidden)
            # assert hidden.shape == conv_input.shape
            # if self.short_cut and hidden.shape == conv_input.shape: # res连接
            #     hidden += conv_input
            
            hiddens.append(hidden)
            conv_input = hidden
        if self.concat_hidden:
            node_feature = torch.cat(hiddens, dim=-1)
        else:
            node_feature = hiddens[-1]
        
        return node_feature
