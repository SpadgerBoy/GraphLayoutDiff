import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
import pdb
from utils.transforms import *
from torch_scatter import scatter_add, scatter_mean

# transform = T.AddRandomWalkPE(walk_length=20)
# transform = AddHigherOrderEdges(order=2)
# transform = Compose([
#     AddEdgeType(),
#     AddHigherOrderEdges(order=2), # Offline edge augmentation
# ])

# edge_index = torch.tensor([[0, 1, 1, 3, 0, 2, 2, 3],
#                            [1, 0, 3, 1, 2, 0, 3, 2]], dtype=torch.long)
# data = Data(edge_index=edge_index)
# data = transform(data)

# edge_index_1 = torch.tensor([[0, 1, 1, 3, 0, 2, 2, 3, 4, 3],
#                            [1, 0, 3, 1, 2, 0, 3, 2, 3, 4]], dtype=torch.long)

# data_1 = Data(edge_index=edge_index_1)
# data_1 = transform(data_1)

# edge_index_2 = torch.tensor([[ 0,  1,  1,  2,  2,  2,  2,  3,  4,  4,  5,  6,  6,  6,  9, 11, 13, 13, 14, 15, 16, 18, 8, 11, 10, 10, 12, 15,  8, 22, 17, 18, 21,  7, 15, 19, 19, 18, 18, 22, 17, 21, 20, 20],
#                              [ 8, 11, 10, 10, 12, 15,  8, 22, 17, 18, 21,  7, 15, 19, 19, 18, 18, 22, 17, 21, 20, 20, 0,  1,  1,  2,  2,  2,  2,  3,  4,  4,  5,  6,  6,  6,  9, 11, 13, 13, 14, 15, 16, 18]], dtype=torch.long)
# data_2 = Data(edge_index=edge_index_2)
# data_2 = transform(data_2)

# edge_index_3 = torch.tensor([[0, 1, 3, 2],
#                            [1,  3, 2, 0]], dtype=torch.long)
# data_3 = Data(edge_index=edge_index_3)
# data_3 = transform(data_3)

# edge_index_4 = torch.tensor([[ 0,  1,  1,  2,  2,  2,  2,  3,  4,  4,  5,  6,  6,  6,  9, 11, 13, 13, 14, 15, 16, 18],
#                              [ 8, 11, 10, 10, 12, 15,  8, 22, 17, 18, 21,  7, 15, 19, 19, 18, 18, 22, 17, 21, 20, 20]], dtype=torch.long)
# data_4 = Data(edge_index=edge_index_4)
# data_4 = transform(data_4)

# pdb.set_trace()

# print(data.size())
# print(data_1.size())
# print(data_2.random_walk_pe)
# print(data_1.node_emb)
# print(data_2.node_emb)

from torch_scatter import scatter_add
import pdb


# def get_distance(pos, edge_index):
#     return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)

# def eq_transform(score_d, pos, edge_index, edge_length):
#     N = pos.size(0)
#     dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])   # (E, 3)
#     score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0, dim_size=N) \
#         + scatter_add(- dd_dr * score_d, edge_index[1], dim=0, dim_size=N) # (N, 3)
    
#     return score_pos

# edge_index = torch.tensor([[0, 1, 1, 3, 0, 2, 2, 3],
#                            [1, 0, 3, 1, 2, 0, 3, 2]])
# pos = torch.tensor([[0, 0], [1,1], [0,1], [1, 0]], dtype=torch.float64)

# pos_noise = torch.zeros(size=pos.size())
# pos_noise.normal_() # 标准正态分布，pos_noise中的元素被随机生成的值替换

# tmp = 1.0
# pos_perturbed = pos + pos_noise * tmp

# d_gt = get_distance(pos, edge_index).unsqueeze(-1) # (E, 1)，原始的edge_length
# d_perturbed = get_distance(pos_perturbed, edge_index).unsqueeze(-1)   # 扰动后的edge_length

# target_d_global = (d_gt - d_perturbed) / tmp
# target_pos_global = eq_transform(target_d_global, pos_perturbed, edge_index, d_perturbed)

# print("pos: ", pos)
# print("pos_noise:", pos_noise)
# print("pos_perturbed: ", pos_perturbed)
# print("target_pos_global: ", target_pos_global)

# values = torch.tensor([True, False, True, False, True])

# edge_index = torch.tensor([[1, 2, 3, 4, 5],
#                            [1, 2, 3, 4, 5]])
# tmp = torch.cat((edge_index, edge_index), dim=1)
# print(tmp)
# exit()

# selected_edges_1 = edge_index[0][values].unsqueeze(0)
# selected_edges_2 = edge_index[1][values].unsqueeze(0)

# selected_edges = torch.cat((selected_edges_1, selected_edges_2), dim=0)

# print(selected_edges)

# import torch
# from torch_geometric.data import Data
# from torch_geometric.utils import degree

# # 假设你已经有了一个Graph数据，包含边的信息 edge_index 和节点的总数 num_nodes
# edge_index = torch.tensor([[0, 1], [2, 1]], dtype=torch.long)
# num_nodes = 3

# # 创建一个Data对象
# data = Data(edge_index=edge_index, num_nodes=num_nodes)

# # 获取每个节点的度数
# degrees = degree(data.edge_index[0], num_nodes=num_nodes).int()

# print(degrees)  # 输出每个节点的度数


import torch
import torch.nn.functional as F

# def rotate_graph(batch, pos):
#     # batch: LongTensor of shape (num_nodes,) containing graph indices for each node
#     # pos: FloatTensor of shape (num_nodes, 2) containing node positions

#     # Step 1: Separate nodes into individual graphs
#     unique_batches = torch.unique(batch)
#     rotated_pos = []

#     # Step 2 and Step 3: Randomly rotate nodes in each graph
#     for b in unique_batches:
#         mask = (batch == b)
#         graph_pos = pos[mask]
#         num_nodes = graph_pos.size(0)

#         # Generate a random rotation angle between 0 and 2*pi
#         theta = torch.rand(1) * 2 * torch.tensor([3.141592653589793])  # 2 * pi
#         # theta = torch.tensor([0.25]) * 2 * torch.tensor([3.141592653589793])  # 2 * pi

#         # Create rotation matrix
#         rotation_matrix = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
#                                        [torch.sin(theta), torch.cos(theta)]])

#         # Rotate node positions using the rotation matrix
#         rotated_graph_pos = torch.matmul(graph_pos, rotation_matrix)

#         rotated_pos.append(rotated_graph_pos)

#     # Concatenate rotated positions for all graphs
#     rotated_pos = torch.cat(rotated_pos, dim=0)

#     return rotated_pos

# # Example usage
# batch = torch.tensor([0, 0, 1, 1, 2, 2])  # Example graph indices for each node
# pos = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0], [6.0, 7.0]])  # Example node positions

# rotated_pos = rotate_graph(batch, pos)
# print(rotated_pos)


# num_ones = int(10 * 0.2)
# tmp = torch.zeros(int(10), 1)
# tmp[:num_ones] = 1
# fragment_mask = tmp[torch.randperm(int(10))]
# linker_mask = 1-fragment_mask
# pdb.set_trace()

# pos = torch.tensor([[-1, -1], [1, -2], [0,1], [1, 0], [0, -1], [1, -2], [0,1]], dtype=torch.float64)
# batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
# fragment_mask = torch.tensor([1, 1, 1, 0, 1, 1, 0]).unsqueeze(-1)
# # pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]


# pos_masked = pos * fragment_mask

# N = scatter_add(fragment_mask, batch, dim=0)[batch]
# pos_add = scatter_add(pos_masked, batch, dim=0)[batch]

# mean = pos_add / N

# pos_center = pos - mean

# pdb.set_trace()

# torch.manual_seed(2021)
# np.random.seed(2021)
# # random.seed(seed)
# pos = torch.tensor([[-1, -1], [1, -2], [0,1], [1, 0], [0, -1], [1, -2], [0,1]], dtype=torch.float64)
# time_step = torch.randn(10, 2)

# pos_noise = torch.zeros(size=pos.size())
# pos_noise.normal_()
# print(pos_noise)

# rand_tensor = torch.rand(10, 2)
# rand_tensor = 2 * rand_tensor - 1
# print(rand_tensor)

# tmp = pos + time_step
# print(time_step)


# import torch
# from torch_geometric.utils import add_self_loops
# from torch_geometric.data import Data

# # 假设你有一个节点掩码 node_mask，表示哪些节点要连接（0）或被连接（1）
# node_mask = torch.tensor([0, 0, 1, 1, 0, 1])

# initial_edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 3, 3],
#                            [0, 1, 2, 3, 4, 5, 0, 1]])
# initial_edge_type = torch.tensor([2, 2, 2, 2, 2, 2, 1, 6])

# # 创建节点特征（示例中使用空特征）
# x = torch.randn(node_mask.size(0), 16)

# # 创建边索引
# # 这里创建的边索引会让所有 node_mask 为 0 的节点与所有 node_mask 为 1 的节点都相互连接
# edge_index = torch.cartesian_prod(node_mask.nonzero(as_tuple=False).view(-1),
#                                   (1 - node_mask).nonzero(as_tuple=False).view(-1)).t()

# m = edge_index.size()[1]
# fragment_edge_type = torch.full((m, ), 4)

# fragment_mat = to_dense_adj(edge_index, edge_attr=fragment_edge_type).squeeze(0) 

# initial_mat = to_dense_adj(initial_edge_index, edge_attr=initial_edge_type).squeeze(0) 

# compose_mat = torch.where(initial_mat==0, fragment_mat, initial_mat)

# new_edge_index, new_edge_type = dense_to_sparse(compose_mat)

# # 创建图数据
# data = Data(x=x, edge_index=edge_index)

# # 添加自环
# # edge_index, _ = add_self_loops(data.edge_index, num_nodes=node_mask.size(0))
# # data.edge_index = edge_index

# # 打印图数据的信息
# print("Node Features:", data.x)
# print("Edge Index:", data.edge_index)
# print("fragment_edge_type: ", fragment_edge_type)
# print("fragment_mat: ", fragment_mat)
# print("initial_mat: ", initial_mat)
# print("compose_mat: ", compose_mat)
# print("new_edge_index: ", new_edge_index)
# print("new_edge_type: ", new_edge_type)

# node_mask = torch.tensor([0, 0, 1, 1, 0, 1])

# initial_edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 3, 3],
#                            [0, 1, 2, 3, 4, 5, 0, 1]])
# initial_edge_type = torch.tensor([2, 2, 2, 2, 2, 2, 1, 6])
# bgraph_adj = torch.sparse.LongTensor(
#         initial_edge_index, 
#         initial_edge_type, 
#         torch.Size([6, 6])
#     )

# edge_index = torch.cartesian_prod(node_mask.nonzero(as_tuple=False).view(-1),
#                                   (1 - node_mask).nonzero(as_tuple=False).view(-1)).t()

# m = edge_index.size()[1]
# rgraph_adj = torch.sparse.LongTensor(
#     edge_index, 
#     torch.ones(edge_index.size(1)).long() * 1,
#     torch.Size([6, 6])
# )

# composed_adj = (bgraph_adj + rgraph_adj).coalesce()  # Sparse (N, N, T)

# pdb.set_trace()

# pos = torch.tensor([[-1, -1], [1, -2], [0,1], [1, 0], [0, -1], [1, -2], [0,1]], dtype=torch.float64)
# batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
# num_graphs = batch.max() + 1
# rand_tensor = torch.rand(num_graphs, 2)
# rand_tensor = 2 * rand_tensor - 1
# pos = pos + rand_tensor[batch]

# pdb.set_trace()

# def get_repaint_schedule(resamplings, jump_length, timesteps):
#     """ Each integer in the schedule list describes how many denoising steps
#     need to be applied before jumping back """
#     repaint_schedule = []
#     curr_t = 0
#     while curr_t < timesteps:
#         if curr_t + jump_length < timesteps:
#             if len(repaint_schedule) > 0:
#                 repaint_schedule[-1] += jump_length
#                 repaint_schedule.extend([jump_length] * (resamplings - 1))
#             else:
#                 repaint_schedule.extend([jump_length] * resamplings)
#             curr_t += jump_length
#         else:
#             residual = (timesteps - curr_t)
#             if len(repaint_schedule) > 0:
#                 repaint_schedule[-1] += residual
#             else:
#                 repaint_schedule.append(residual)
#             curr_t += residual

#     return list(reversed(repaint_schedule))

# resamplings = 5
# jump_length = 1
# timesteps = 10
# schedule = get_repaint_schedule(resamplings, jump_length, timesteps)

# sum = 0
# s = timesteps - 1
# for i, n_denoise_steps in enumerate(schedule):
#     sum += n_denoise_steps

# s = timesteps - 1
# for i, n_denoise_steps in enumerate(schedule):
#     if i == 1:
#         print("s11:", s)
#     for j in range(n_denoise_steps):
#         print("s: ", s)
#         # Noise combined representation
#         if j == n_denoise_steps - 1 and i < len(schedule) - 1:
#             # Go back jump_length steps
#             # t = s + jump_length
#             # s = t
#             # print("ss", s)
#             s = s+1
#         s -= 1
        

# print(schedule)
# print(len(schedule))
# print(sum)

# pdb.set_trace()

# def get_schedule_jump(t_T, n_sample, jump_length, jump_n_sample,
#                       jump2_length=1, jump2_n_sample=1,
#                       jump3_length=1, jump3_n_sample=1,
#                       start_resampling=100000000):

#     jumps = {}
#     for j in range(0, t_T - jump_length, jump_length):
#         jumps[j] = jump_n_sample - 1

#     jumps2 = {}
#     for j in range(0, t_T - jump2_length, jump2_length):
#         jumps2[j] = jump2_n_sample - 1

#     jumps3 = {}
#     for j in range(0, t_T - jump3_length, jump3_length):
#         jumps3[j] = jump3_n_sample - 1

#     t = t_T
#     ts = []

#     while t >= 1:
#         t = t-1
#         ts.append(t)

#         if (
#             t + 1 < t_T - 1 and
#             t <= start_resampling
#         ):
#             for _ in range(n_sample - 1):
#                 t = t + 1
#                 ts.append(t)

#                 if t >= 0:
#                     t = t - 1
#                     ts.append(t)

#         if (
#             jumps3.get(t, 0) > 0 and
#             t <= start_resampling - jump3_length
#         ):
#             jumps3[t] = jumps3[t] - 1
#             for _ in range(jump3_length):
#                 t = t + 1
#                 ts.append(t)

#         if (
#             jumps2.get(t, 0) > 0 and
#             t <= start_resampling - jump2_length
#         ):
#             jumps2[t] = jumps2[t] - 1
#             for _ in range(jump2_length):
#                 t = t + 1
#                 ts.append(t)
#             jumps3 = {}
#             for j in range(0, t_T - jump3_length, jump3_length):
#                 jumps3[j] = jump3_n_sample - 1

#         if (
#             jumps.get(t, 0) > 0 and
#             t <= start_resampling - jump_length
#         ):
#             jumps[t] = jumps[t] - 1
#             for _ in range(jump_length):
#                 t = t + 1
#                 ts.append(t)
#             jumps2 = {}
#             for j in range(0, t_T - jump2_length, jump2_length):
#                 jumps2[j] = jump2_n_sample - 1

#             jumps3 = {}
#             for j in range(0, t_T - jump3_length, jump3_length):
#                 jumps3[j] = jump3_n_sample - 1

#     ts.append(-1)

#     # _check_times(ts, -1, t_T)

#     return ts

# t_T = 100
# n_sample = 2
# jump_length = 1
# jump_n_sample = 1

# tmp = get_schedule_jump(t_T, n_sample, jump_length, jump_n_sample)

# sum = 0
# for i,t in enumerate(tmp):
#     sum += t

# print("sum: ", sum)
# print("tmp: ", tmp)
# pdb.set_trace()

# pos = torch.tensor([[-1, -1], [1, -2], [0,1], [1, 0], [0, -1], [1, -2], [0,1]], dtype=torch.float64)
# batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
# fragment_mask = torch.tensor([1, 1, 1, 0, 1, 1, 0]).unsqueeze(-1)

# com_noised = scatter_mean(pos[fragment_mask.bool().view(-1)], batch[fragment_mask.bool().view(-1)], dim=0)

# print(fragment_mask.bool().view(-1))
# print(pos[fragment_mask.bool().view(-1)])
# print(batch[fragment_mask.bool().view(-1)])
# print(com_noised)

# pdb.set_trace()

# import torch
# from torch_geometric.data import Data

# # 创建一个简单的PyG的Data对象
# edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
# pos = torch.tensor([[-1, -1], [1, -2], [0,1], [1, 0]], dtype=torch.float64)
# data = Data(pos=pos, edge_index=edge_index)

# # 提取节点和边的信息
# nodes = [{'id': f'{i+1}', 'x': data.pos[i, 0].item(), 'y': data.pos[i, 1].item()} for i in range(data.num_nodes)]
# edges = [{'source': f'{data.edge_index[0, i].item() + 1}', 'target': f'{data.edge_index[1, i].item() + 1}'} for i in range(data.num_edges)]

# # 构造JavaScript格式的图数据
# graph_data = {
#     'nodes': nodes,
#     'links': edges
# }
# print("graph_data: ", graph_data)

# nodes = [
#     { "id": "node1", "x": 0.5, "y": 0.5 },
#     { "id": "node2", "x": -0.5, "y": 0.5 },
#     { "id": "node3", "x": -0.5, "y": -0.5 },
#     { "id": "node4", "x": 0.5, "y": -0.5 },
# ]

# # 从节点数据中提取 x 和 y 坐标
# x_coords = [node["x"] for node in nodes]
# y_coords = [node["y"] for node in nodes]

# # 将 x 和 y 坐标合并为一个大小为 n*2 的 tensor
# pos = np.vstack((x_coords, y_coords)).T


# mask = [1,1,0,0]
# fragment_mask = np.array(mask).tolist()
# torch_fragment_mask = torch.tensor(fragment_mask, dtype=torch.long)
# torch_linker_mask = 1-torch_fragment_mask

# print("fragment_mask: ", torch_fragment_mask)
# print("linker_mask: ", torch_linker_mask)

# print("pos:", pos)

import pickle
import pdb
demo_tree = "./data/tree_graph/demo_tree_graph.pkl"
with open(demo_tree, 'rb') as f:
    tree = pickle.load(f)
pdb.set_trace()




