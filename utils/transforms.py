import copy
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import Compose
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_sparse import coalesce
import numpy as np
from typing import Any, Optional
from scipy.sparse.linalg import eigs, eigsh

from torch import Tensor
from torch_geometric.typing import OptTensor

from torch_geometric.typing import SparseTensor
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
)
from torch_geometric.utils import to_undirected
from torch_geometric.utils import degree

import pdb

class CountNodesPerGraph(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        data.num_nodes_per_graph = torch.LongTensor([data.num_nodes])
        return data

class AddNodeType(object):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, data):
        data.node_type = torch.full((data.num_nodes, ), data.num_nodes, dtype=torch.long)
        return data

class AddNodeDegree(object):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, data):
        data.degrees = degree(data.edge_index[0], num_nodes=data.num_nodes_per_graph).int()
        return data

class AddEdgeType(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data):
        data.edge_type = torch.ones((data.edge_index.size(1), ), dtype=torch.long)
        return data

class AddUndiectedEdge(object):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, data):
        data.initial_edge_index = data.edge_index
        data.edge_index = to_undirected(data.edge_index)
        return data

class AddNodeMask(object):
    def __init__(self, node_mask) -> None:
        super().__init__()
        self.node_mask = node_mask
    
    def __call__(self, data):
        num_ones = int(data.num_nodes_per_graph * self.node_mask)
        tmp = torch.zeros(int(data.num_nodes_per_graph), 1)
        tmp[:num_ones] = 1
        data.fragment_mask = tmp[torch.randperm(int(data.num_nodes_per_graph))]
        data.linker_mask = 1-data.fragment_mask

        return data

class AddFragmentEdge(object):
    def __init__(self, fragment_edge_type) -> None:
        super().__init__()
        self.fragment_edge_type = fragment_edge_type
    
    def __call__(self, data):

        node_mask = data.fragment_mask.squeeze(-1)

        # node_mask为1的节点 连到 node_mask为0的节点，即fragment_mask，将fragment的信息传递出去。 
        fragment_edge_index = torch.cartesian_prod(node_mask.nonzero(as_tuple=False).view(-1),
                                        (1 - node_mask).nonzero(as_tuple=False).view(-1)).t()
        m = fragment_edge_index.size()[1]
        if m != 0:
            fragment_edge_type = torch.full((m, ), self.fragment_edge_type)

            fragment_mat = to_dense_adj(fragment_edge_index, edge_attr=fragment_edge_type).squeeze(0) 

            initial_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).squeeze(0) 

            compose_mat = torch.where(initial_mat==0, fragment_mat, initial_mat)

            new_edge_index, new_edge_type = dense_to_sparse(compose_mat)
            
            N = data.num_nodes
            data.edge_index, data.edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N) # modify data

        return data
    

def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    # TODO Move to `BaseTransform`.
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data

class AddLaplacianEigenvectorPE(object):
    r"""Adds the Laplacian eigenvector positional encoding from the
    `"Benchmarking Graph Neural Networks" <https://arxiv.org/abs/2003.00982>`_
    paper to the given graph
    (functional name: :obj:`add_laplacian_eigenvector_pe`).

    Args:
        k (int): The number of non-trivial eigenvectors to consider.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"laplacian_eigenvector_pe"`)
        is_undirected (bool, optional): If set to :obj:`True`, this transform
            expects undirected graphs as input, and can hence speed up the
            computation of eigenvectors. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :meth:`scipy.sparse.linalg.eigs` (when :attr:`is_undirected` is
            :obj:`False`) or :meth:`scipy.sparse.linalg.eigsh` (when
            :attr:`is_undirected` is :obj:`True`).
    """
    def __init__(self, k: int):
        self.k = k
        self.attr_name = 'node_emb'
        self.is_undirected = True

    def __call__(self, data: Data):
        eig_fn = eigs if not self.is_undirected else eigsh
        # eig_fn = eigsh

        num_nodes = data.num_nodes
        edge_index, edge_weight = get_laplacian(
            data.edge_index,
            normalization='sym',
            num_nodes=num_nodes,
        )
        

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        eig_vals, eig_vecs = eig_fn(
            L,
            k=self.k + 1,
            which='SR' if not self.is_undirected else 'SA',
            return_eigenvectors=True,
        )

        eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
        pe = torch.from_numpy(eig_vecs[:, 1:self.k + 1])
        sign = -1 + 2 * torch.randint(0, 2, (self.k, ))
        pe *= sign

        data = add_node_attr(data, pe, attr_name=self.attr_name)

        return data

class AddHigherOrderEdges(object):

    def __init__(self, order, num_types=1):
        super().__init__()
        self.order = order
        self.num_types = num_types

    def binarize(self, x):
        return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

    def get_higher_order_adj_matrix(self, adj, order):
        """
        Args:
            adj:        (N, N)
            type_mat:   (N, N)
        """
        adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                    self.binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

        for i in range(2, order+1):
            adj_mats.append(self.binarize(adj_mats[i-1] @ adj_mats[1]))
        order_mat = torch.zeros_like(adj)

        for i in range(1, order+1):
            order_mat += (adj_mats[i] - adj_mats[i-1]) * i

        return order_mat

    def __call__(self, data: Data):
        # if data.num_nodes_per_graph <= 25:
        #     self.order = 3
        # elif data.num_nodes_per_graph > 25:
        #     self.order = 2

        N = data.num_nodes
        adj = to_dense_adj(data.edge_index).squeeze(0)
        adj_order = self.get_higher_order_adj_matrix(adj, self.order)  # (N, N)

        type_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).squeeze(0)   # (N, N)
        type_highorder = torch.where(adj_order > 1, self.num_types + adj_order - 1, torch.zeros_like(adj_order))
        assert (type_mat * type_highorder == 0).all()
        type_new = type_mat + type_highorder

        new_edge_index, new_edge_type = dense_to_sparse(type_new)
        _, edge_order = dense_to_sparse(adj_order)

        data.bond_edge_index = data.edge_index  # Save original edges
        data.edge_index, data.edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N) # modify data
        edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N) # modify data
        data.is_bond = (data.edge_type < self.num_types)
        assert (data.edge_index == edge_index_1).all()

        return data

class AddRandomWalkPE(object):
    r"""Adds the random walk positional encoding from the `"Graph Neural
    Networks with Learnable Structural and Positional Representations"
    <https://arxiv.org/abs/2110.07875>`_ paper to the given graph
    (functional name: :obj:`add_random_walk_pe`).

    Args:
        walk_length (int): The number of random walk steps.
        attr_name (str, optional): The attribute name of the data object to add
            positional encodings to. If set to :obj:`None`, will be
            concatenated to :obj:`data.x`.
            (default: :obj:`"random_walk_pe"`)
    """
    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = 'node_emb',
    ):
        self.walk_length = walk_length
        self.attr_name = attr_name

    def __call__(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        
        data.edge_weight = None
        edge_index, edge_weight = data.edge_index, data.edge_weight

        adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                           sparse_sizes=(num_nodes, num_nodes))

        # Compute D^{-1} A:
        deg_inv = 1.0 / adj.sum(dim=1)
        deg_inv[deg_inv == float('inf')] = 0
        adj = adj * deg_inv.view(-1, 1)

        out = adj
        row, col, value = out.coo()
        pe_list = [get_self_loop_attr((row, col), value, num_nodes)]
        for _ in range(self.walk_length - 1):
            out = out @ adj
            row, col, value = out.coo()
            pe_list.append(get_self_loop_attr((row, col), value, num_nodes))
        pe = torch.stack(pe_list, dim=-1)

        data = add_node_attr(data, pe, attr_name=self.attr_name)
        # pdb.set_trace()
        return data



def get_self_loop_attr(edge_index: Tensor, edge_attr: OptTensor = None,
                       num_nodes: Optional[int] = None) -> Tensor:
    r"""Returns the edge features or weights of self-loops
    :math:`(i, i)` of every node :math:`i \in \mathcal{V}` in the
    graph given by :attr:`edge_index`. Edge features of missing self-loops not
    present in :attr:`edge_index` will be filled with zeros. If
    :attr:`edge_attr` is not given, it will be the vector of ones.

    .. note::
        This operation is analogous to getting the diagonal elements of the
        dense adjacency matrix.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
            features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`

    Examples:

        >>> edge_index = torch.tensor([[0, 1, 0],
        ...                            [1, 0, 0]])
        >>> edge_weight = torch.tensor([0.2, 0.3, 0.5])
        >>> get_self_loop_attr(edge_index, edge_weight)
        tensor([0.5000, 0.0000])

        >>> get_self_loop_attr(edge_index, edge_weight, num_nodes=4)
        tensor([0.5000, 0.0000, 0.0000, 0.0000])
    """
    loop_mask = edge_index[0] == edge_index[1]
    loop_index = edge_index[0][loop_mask]

    if edge_attr is not None:
        loop_attr = edge_attr[loop_mask]
    else:  # A vector of ones:
        loop_attr = torch.ones_like(loop_index, dtype=torch.float)

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    full_loop_attr = loop_attr.new_zeros((num_nodes, ) + loop_attr.size()[1:])
    full_loop_attr[loop_index] = loop_attr

    return full_loop_attr


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        if is_torch_sparse_tensor(edge_index):
            return max(edge_index.size(0), edge_index.size(1))
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))

def is_torch_sparse_tensor(src: Any) -> bool:
    r"""Returns :obj:`True` if the input :obj:`src` is a
    :class:`torch.sparse.Tensor` (in any sparse layout).

    Args:
        src (Any): The input object to be checked.
    """
    if isinstance(src, Tensor):
        if src.layout == torch.sparse_coo:
            return True
        # if src.layout == torch.sparse_csr:
        #     return True
        # if src.layout == torch.sparse_csc:
        #     return True
    return False
