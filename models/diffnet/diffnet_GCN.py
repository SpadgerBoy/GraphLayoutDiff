import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
import numpy as np
from numpy import pi as PI
from tqdm.auto import tqdm

from ..common import MultiLayerPerceptron, assemble_node_pair_feature, extend_graph_order_radius, assemble_node_pair_feature_1
from ..geometry import get_distance, eq_transform, get_distance_2
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder, GINEncoder_Global, get_node_encoder, get_degree_encoder, GCNEncoder

import pdb

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)
    
    if beta_schedule == 'quad':
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64
            )
        )
    elif beta_schedule == 'linear':
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd': # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == 'sigmoid':
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps, )
    return betas

class EDM_Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.edge_encoder_local = get_edge_encoder(config)

        self.encoder_local = GCNEncoder(
            hidden_dim = config.hidden_dim,
            num_convs = config.num_convs_local,
            node_channels = config.laplacian_eigenvector,
            activation = config.mlp_act
        )

        self.grad_local_dist_mlp = MultiLayerPerceptron(
            config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 2],
            activation = config.mlp_act
        )
    
    def forward(self, node_emb, node_type, node_degree, pos, edge_index, edge_type, edge_length, batch, time_step):
        
        edge_attr_local = self.edge_encoder_local(
            edge_length = edge_length,
            edge_type = edge_type
        )

        # 结合node_emb和edge_attr，得到node_attr, Encoder是GIN
        node_attr_local = self.encoder_local(
            z = node_emb,
            edge_index = edge_index,
            edge_attr = edge_attr_local,
            embed_node = True
        )

        node_pos = self.grad_local_dist_mlp(node_attr_local)

        pos = pos + node_pos

        return pos
            

class GraphDiffusionNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.convs = nn.ModuleList()
        for _ in range(self.config.num_edm_block):
            block = EDM_Block(self.config)
            self.convs.append(block)
        
        self.model_local = self.convs

        self.encoder_global = nn.Linear(10, 20)
        self.model_global = self.encoder_global 

        self.model_type = config.type
        if self.model_type == 'diffusion':
            betas = get_beta_schedule(  # betas代表每个时间步t的噪音水平
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps
            )   
            betas = torch.from_numpy(betas).float()
            self.betas = nn.Parameter(betas, requires_grad=False)
            alphas = (1. - betas).cumprod(dim=0)    # 1-betas，再累积
            self.alphas = nn.Parameter(alphas, requires_grad=False)
            self.num_timesteps = self.betas.size(0)

    def forward(self, node_emb, node_type, node_degree, pos, edge_index, edge_type, batch, time_step,
                 return_edges=False,extend_order=True, extend_radius=True):
        
        N = node_emb.size(0)
        edge_index, edge_type = extend_graph_order_radius(
            num_nodes=N,
            node_type=node_type,
            pos=pos,
            edge_index=edge_index,
            edge_type=edge_type,
            batch=batch,
            order=self.config.edge_order,
            cutoff=self.config.cutoff,
            extend_order=extend_order,
            extend_radius=extend_radius
        )

        pos_final = pos
        edge_length = get_distance(pos_final, edge_index).unsqueeze(-1)
        for block in self.convs:
            # edge_length = get_distance(pos_final, edge_index).unsqueeze(-1)
            pos_final = block(
                node_emb = node_emb,
                node_type = node_type,
                node_degree = node_degree,
                pos = pos_final,
                edge_index = edge_index,
                edge_length = edge_length,
                edge_type = edge_type,
                batch = batch,
                time_step = time_step
            )
        pos_noise = pos_final - pos

        return pos_noise
    

    def get_loss(self, node_emb, node_type, node_degree, pos, edge_index, edge_type, batch, num_nodes_per_graph, num_graphs,
                anneal_power=2.0, return_unreduced_loss=False, return_unreduced_edge_loss=False, extend_order=True, extend_radius=True, is_sidechain=None):
        if self.model_type == 'diffusion':
            return self.get_loss_diffusion(node_emb,node_type, node_degree, pos, edge_index, edge_type, batch, num_nodes_per_graph, num_graphs,
                anneal_power, return_unreduced_loss, return_unreduced_edge_loss, extend_order, extend_radius)
    

    def get_loss_diffusion(self, node_emb, node_type, node_degree, pos, edge_index, edge_type, batch, num_nodes_per_graph, num_graphs,
                anneal_power=2.0, return_unreduced_loss=False, return_unreduced_edge_loss=False, extend_order=True, extend_radius=True, is_sidechain=None):
        N = node_emb.size(0)
        node2graph = batch # 哪个node属于哪个graph

        # Four elements for DDPM: original_data(pos), gaussian_noise(pos_noise), beta(sigma), time_step

        # ----------------------- Sample Noise Level -----------------------

        pos = center_pos(pos, batch)

        # 随机初始化时间t
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs//2+1, ), device=pos.device
        )
        time_step = torch.cat([time_step, self.num_timesteps-time_step-1], dim=0)[:num_graphs]

        # 得到扰动率alphas
        a = self.alphas.index_select(0, time_step)  # 0代表维度，time_step代表索引。 size: (G, )
        a_pos = a.index_select(0, node2graph).unsqueeze(-1) # (N,1)

        # 得到pos上的噪声，以及扰动后的pos_perturbed
        pos_noise = torch.zeros(size=pos.size(), device=pos.device)
        pos_noise.normal_() # 标准正态分布，pos_noise中的元素被随机生成的值替换

        pos_perturbed = pos * a_pos.sqrt() + pos_noise * (1 - a_pos).sqrt()

        pos_perturbed = center_pos(pos_perturbed, batch)

        # Update invariant edge features, as shown in equation 5-7
        pos_noise_predict = self(
            node_emb = node_emb,
            node_type = node_type,
            node_degree = node_degree,
            pos = pos_perturbed,
            edge_index = edge_index,
            edge_type = edge_type,
            batch = batch,
            time_step = time_step,
            return_edges = True,
            extend_order = extend_order,
            extend_radius = extend_radius
        )   # edge的features都是由节点间相对位置算出来的，最后得到的edge_features物理意义为edge_length上的噪音
            # edge_length是扰动后的长度

        loss_local = (pos_noise_predict  - pos_noise) ** 2
        loss_local = torch.sum(loss_local, dim=-1, keepdim=True)

        loss_global = torch.zeros(size=loss_local.size(), device=pos.device)

        loss = loss_global + loss_local

        if return_unreduced_edge_loss:
            pass
        elif return_unreduced_loss:
            return loss, loss_global, loss_local
        else:
            return loss


    def langevin_dynamics_sample(self, node_emb, node_type, node_degree, pos_ref, fragment_mask, linker_mask, pos_init, edge_index, edge_type, batch, num_graphs, extend_order, extend_radius=True,
                                 n_steps=100, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None, min_sigma=0,
                                 global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, **kwargs):
        
        if self.model_type == 'diffusion':
            return self.langevin_dynamics_sample_diffusion(node_emb, node_type, node_degree, pos_ref, fragment_mask, linker_mask, pos_init, edge_index, edge_type, batch, num_graphs, extend_order, extend_radius,
                        n_steps, step_lr, clip, clip_local, clip_pos, min_sigma,
                        global_start_sigma, w_global, w_reg,
                        sampling_type=kwargs.get("sampling_type", 'ddpm_noisy'), eta=kwargs.get("eta", 1.))
    

    def langevin_dynamics_sample_diffusion(self, node_emb, node_type, node_degree, pos_ref, fragment_mask, linker_mask, pos_init, edge_index, edge_type, batch, num_graphs, extend_order, extend_radius=True,
                                 n_steps=100, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None, min_sigma=0,
                                 global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, **kwargs):
        
        alpha = (1.0-self.betas).sqrt()
        betas = self.betas
        sigmas = (1.0-self.alphas).sqrt()
        sigmas_1 = (1.0-self.alphas)
        alpha_cumprod = self.alphas

        pos_traj = []
        with torch.no_grad():
            seq = range(self.num_timesteps-n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])

            pos = pos_init

            pos = center_pos(pos_init, batch) 
            
            resamplings = 1
            for i,j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='sample'):
                for u in range(resamplings):
                    t = torch.full(size=(num_graphs, ), fill_value=i, dtype=torch.long, device=pos.device)

                    pos_noise_predict = self(
                        node_emb = node_emb,
                        node_type = node_type,
                        node_degree = node_degree,
                        pos = pos,
                        edge_index = edge_index,
                        edge_type = edge_type,
                        batch = batch,
                        time_step = t,
                        return_edges = True,
                        extend_order = extend_order,
                        extend_radius = extend_radius
                    )   
                    
                    if i>0:
                        eps_linker = torch.randn_like(pos)
                        pos_next = 1/alpha[i] * (pos - betas[i]/sigmas[i] * pos_noise_predict) + eps_linker * sigmas_1[i-1]/sigmas_1[i] * betas[i]
                    else:
                        pos_next = 1/alpha[i] * (pos - betas[i]/sigmas[i] * pos_noise_predict)
                    

                    pos = pos_next
                    pos = center_pos(pos, batch)


                    if u < resamplings - 1:
                        pos = (1-betas[i]).sqrt() * pos + betas[i].sqrt() * torch.randn_like(pos)
                        pos = center_pos(pos, batch)


                    if torch.isnan(pos).any():
                        print('NaN detected. Please restart.')
                        raise FloatingPointError()
                    if clip_pos is not None:
                        pos = torch.clamp(pos, min=-clip_pos, max=clip_pos)
                    pos_traj.append(pos.clone().cpu())
        
        return pos, pos_traj


        

def is_train_edge(edge_index, is_sidechain):
    if is_sidechain is None:
        return torch.ones(edge_index.size(1), device=edge_index.device).bool()
    else:
        is_sidechain = is_sidechain.bool()
        return torch.logical_or(is_sidechain[edge_index[0]], is_sidechain[edge_index[1]])

def is_local_edge(edge_type):
    return edge_type > 0

def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center

def random_center_pos(pos, batch):
    num_graphs = batch.max() + 1
    rand_tensor = torch.rand(num_graphs, 2).to(pos.device)
    rand_tensor = 2 * rand_tensor - 1
    pos = pos + rand_tensor[batch]
    
    return pos

def center_pos_fragment(pos, fragment_mask, batch):
    pos_masked = pos * fragment_mask
    N = scatter_add(fragment_mask, batch, dim=0)[batch]
    pos_add = scatter_add(pos_masked, batch, dim=0)[batch]
    mean = pos_add / N
    pos_center = pos - mean
    return pos_center

def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom

        
