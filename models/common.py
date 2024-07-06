import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, radius
from torch_scatter import scatter_mean, scatter_add, scatter_max
from torch_sparse import coalesce
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import numpy as np
import pdb

class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer Perceptron.
    Note there is no activation or dropout in the last layer.
    Parameters:
        input_dim (int): input dimension
        hidden_dim (list of int): hidden dimensions
        activation (str or function, optional): activation function
        dropout (float, optional): dropout rate
    """

    def __init__(self, input_dim, hidden_dims, activation="relu", dropout=0):
        super(MultiLayerPerceptron, self).__init__()

        self.dims = [input_dim] + hidden_dims
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i + 1]))

    def forward(self, input):
        """"""
        x = input
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x

def assemble_node_pair_feature(node_attr, edge_index, edge_attr):
    h_row, h_col = node_attr[edge_index[0]], node_attr[edge_index[1]]
    h_pair = torch.cat([h_row*h_col, edge_attr], dim=-1)    # (E, 2H)
    return h_pair

def assemble_node_pair_feature_1(node_attr, edge_index, edge_attr):
    h_row, h_col = node_attr[edge_index[0]], node_attr[edge_index[1]]
    h_pair = torch.cat([h_row, h_col, edge_attr], dim=-1)
    return h_pair
   

def _extend_graph_order(num_nodes, edge_index, edge_type, order=3):

    # 将矩阵二值化
    def binarize(x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    # 根据order的值，添加n阶边，返回order_mat，order_mat[i][j]代表i到j的阶数
    def get_higher_order_adj_matrix(adj, order):
        # 1:单位矩阵；2:添加自环边，并将邻接矩阵二值化
        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                    binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]
        
        # 根据order的值，添加n阶边
        for i in range(2, order+1):
            adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        # 根据边的阶数，得到order_mat
        for i in range(1, order+1):
            order_mat += (adj_mats[i] - adj_mats[i-1]) * i

        return order_mat

    # num_types = len(BOND_TYPES)
    num_types = 1

    N = num_nodes
    adj = to_dense_adj(edge_index).squeeze(0)
    adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

    type_mat = to_dense_adj(edge_index, edge_attr=edge_type).squeeze(0)   # (N, N)
    type_highorder = torch.where(adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order)) # 给新加进来的边赋予新的type=阶数+len(type)
    assert (type_mat * type_highorder == 0).all()
    type_new = type_mat + type_highorder  # 整合连边

    new_edge_index, new_edge_type = dense_to_sparse(type_new)
    _, edge_order = dense_to_sparse(adj_order)

    # data.bond_edge_index = data.edge_index  # Save original edges
    new_edge_index, new_edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N) # modify data
    
    # [Note] This is not necessary
    # data.is_bond = (data.edge_type < num_types)

    # [Note] In earlier versions, `edge_order` attribute will be added. 
    #         However, it doesn't seem to be necessary anymore so I removed it.
    # edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N) # modify data
    # assert (data.edge_index == edge_index_1).all()

    return new_edge_index, new_edge_type

# 根据节点的几何距离对graph进行拓展，其中cutoff表示截断距离
# 新加入的边的type为0
def extend_to_radius_graph(pos, node_type, edge_index, edge_type, cutoff, batch, unspecified_type_number=0):

    assert edge_type.dim() == 1
    N = pos.size(0)

    bgraph_adj = torch.sparse.LongTensor(
        edge_index, 
        edge_type, 
        torch.Size([N, N])
    )

    # # Computes graph edges to all points within a given distance.
    rgraph_edge_index = radius_graph(pos, r=cutoff, batch=batch)    # (2, E_r)

    # less_than_25_indices = torch.nonzero(node_type <= 25).squeeze()
    # greater_than_25_indices = torch.nonzero(node_type > 25).squeeze()

    # # Computes graph edges to all points within a given distance.
    # rgraph_edge_index_1 = radius_graph(pos, r=3.0, batch=batch)    # (2, E_r)

    # element = rgraph_edge_index_1[0].unsqueeze(1)
    # test_elements = less_than_25_indices.unsqueeze(0)
    # eq = element == test_elements
    # result = eq.any(dim=1)

    # tmp_edges_1 = rgraph_edge_index_1[0][result].unsqueeze(0)
    # tmp_edges_2 = rgraph_edge_index_1[1][result].unsqueeze(0)

    # selected_edges_1 = torch.cat((tmp_edges_1, tmp_edges_2), dim=0)


    # rgraph_edge_index_2 = radius_graph(pos, r=0.1, batch=batch)    # (2, E_r)

    # element = rgraph_edge_index_2[0].unsqueeze(1)
    # test_elements = greater_than_25_indices.unsqueeze(0)
    # eq = element == test_elements
    # result = eq.any(dim=1)

    # tmp_edges_1 = rgraph_edge_index_2[0][result].unsqueeze(0)
    # tmp_edges_2 = rgraph_edge_index_2[1][result].unsqueeze(0)

    # selected_edges_2 = torch.cat((tmp_edges_1, tmp_edges_2), dim=0)

    # rgraph_edge_index = torch.cat((selected_edges_1, selected_edges_2), dim=1)


    rgraph_adj = torch.sparse.LongTensor(
        rgraph_edge_index, 
        torch.ones(rgraph_edge_index.size(1)).long().to(pos.device) * unspecified_type_number,
        torch.Size([N, N])
    )

    composed_adj = (bgraph_adj + rgraph_adj).coalesce()  # Sparse (N, N, T)
    # edge_index = composed_adj.indices()
    # dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

    new_edge_index = composed_adj.indices()
    new_edge_type = composed_adj.values().long()
    
    return new_edge_index, new_edge_type

# 根据节点的几何距离和节点的图距离，分别对graph进行拓展
def extend_graph_order_radius(num_nodes, node_type, pos, edge_index, edge_type, batch, order=3, cutoff=10.0, 
                              extend_order=True, extend_radius=True):
    
    if extend_order:
        edge_index, edge_type = _extend_graph_order(
            num_nodes=num_nodes, 
            edge_index=edge_index, 
            edge_type=edge_type, order=order
        )
        # edge_index_order = edge_index
        # edge_type_order = edge_type

    if extend_radius:
        edge_index, edge_type = extend_to_radius_graph(
            pos=pos, 
            node_type=node_type,
            edge_index=edge_index, 
            edge_type=edge_type, 
            cutoff=cutoff, 
            batch=batch
        )
    
    return edge_index, edge_type