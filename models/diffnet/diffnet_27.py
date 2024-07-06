import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
import numpy as np
from numpy import pi as PI
from tqdm.auto import tqdm

from ..common import MultiLayerPerceptron, assemble_node_pair_feature, extend_graph_order_radius, assemble_node_pair_feature_1
from ..geometry import get_distance, eq_transform
from ..encoder import SchNetEncoder, GINEncoder, get_edge_encoder, GINEncoder_Global, get_node_encoder, get_degree_encoder, get_node_degree_encoder

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

        self.degree_encoder = get_degree_encoder(config)

        self.edge_encoder_local = get_edge_encoder(config)

        self.encoder_local = GINEncoder(
            hidden_dim = config.hidden_dim,
            num_convs = config.num_convs_local,
            node_channels = config.laplacian_eigenvector,
            activation = config.mlp_act
        )

        self.grad_local_dist_mlp = MultiLayerPerceptron(
            3 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation = config.mlp_act
        )
    
    def forward(self, node_emb, node_type, node_degree, pos, edge_index, edge_type, edge_length, batch, time_step,
                 return_edges=False,extend_order=True, extend_radius=True):
        
        edge_attr_local = self.edge_encoder_local(
            edge_length = edge_length,
            edge_type = edge_type
        )

        degree_emb = self.degree_encoder(
            node_degree = node_degree,
        )

        # 结合node_emb和edge_attr，得到node_attr, Encoder是GIN
        node_attr_local = self.encoder_local(
            z = degree_emb,
            edge_index = edge_index,
            edge_attr = edge_attr_local,
            embed_node = False
        )

        h_pair_local = assemble_node_pair_feature_1(
            node_attr = node_attr_local,
            edge_index=edge_index,
            edge_attr=edge_attr_local,
        )

        edge_inv_local = self.grad_local_dist_mlp(h_pair_local)

        node_eq_local = eq_transform(edge_inv_local, pos, edge_index, edge_length)

        pos = pos + node_eq_local

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

    def forward(self, node_emb, node_type, node_degree, pos, linker_mask, fragment_mask, edge_index, edge_type, batch, time_step,
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
        edge_length = get_distance(pos, edge_index).unsqueeze(-1)

        pos_final = pos
        for block in self.convs:
            pos_final = block(
                node_emb = node_emb,
                node_type = node_type,
                node_degree = node_degree,
                pos = pos_final,
                edge_index = edge_index,
                edge_length = edge_length,
                edge_type = edge_type,
                batch = batch,
                time_step = time_step,
                return_edges = True,
                extend_order = extend_order,
                extend_radius = extend_radius
            )
            pos_final = pos_final * linker_mask + pos * fragment_mask
        pos_noise = pos_final - pos

        return pos_noise
    

    def get_loss(self, node_emb, node_type, node_degree, fragment_mask, linker_mask, pos, edge_index, edge_type, batch, num_nodes_per_graph, num_graphs,
                anneal_power=2.0, return_unreduced_loss=False, return_unreduced_edge_loss=False, extend_order=True, extend_radius=True, is_sidechain=None):
        if self.model_type == 'diffusion':
            return self.get_loss_diffusion(node_emb,node_type, node_degree, fragment_mask, linker_mask, pos, edge_index, edge_type, batch, num_nodes_per_graph, num_graphs,
                anneal_power, return_unreduced_loss, return_unreduced_edge_loss, extend_order, extend_radius)
    

    def get_loss_diffusion(self, node_emb, node_type, node_degree, fragment_mask, linker_mask, pos, edge_index, edge_type, batch, num_nodes_per_graph, num_graphs,
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

        pos_perturbed = pos_perturbed * linker_mask + pos * fragment_mask

        # Update invariant edge features, as shown in equation 5-7
        pos_noise_predict = self(
            node_emb = node_emb,
            node_type = node_type,
            node_degree = node_degree,
            pos = pos_perturbed,
            linker_mask = linker_mask,
            fragment_mask = fragment_mask,
            edge_index = edge_index,
            edge_type = edge_type,
            batch = batch,
            time_step = time_step,
            return_edges = True,
            extend_order = extend_order,
            extend_radius = extend_radius
        )   # edge的features都是由节点间相对位置算出来的，最后得到的edge_features物理意义为edge_length上的噪音
            # edge_length是扰动后的长度

        loss_local = (pos_noise_predict * linker_mask  - pos_noise * linker_mask ) ** 2
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
        
        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # .view(-1, 1, 1, 1)
            return a
        
        alpha = (1.0-self.betas).sqrt()
        betas = self.betas
        sigmas = (1.0-self.alphas).sqrt()
        sigmas_1 = (1.0-self.alphas)

        pos_traj = []
        with torch.no_grad():
            seq = range(self.num_timesteps-n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])

            # pos = pos_init
            pos = center_pos(pos_init, batch)

            pos = pos * linker_mask + pos_ref * fragment_mask

            for i,j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='sample'):
                t = torch.full(size=(num_graphs, ), fill_value=i, dtype=torch.long, device=pos.device)

                pos_noise_predict = self(
                    node_emb = node_emb,
                    node_type = node_type,
                    node_degree = node_degree,
                    pos = pos,
                    linker_mask = linker_mask,
                    fragment_mask = fragment_mask,
                    edge_index = edge_index,
                    edge_type = edge_type,
                    batch = batch,
                    time_step = t,
                    return_edges = True,
                    extend_order = extend_order,
                    extend_radius = extend_radius
                )   

                # Local
                node_eq_local = pos_noise_predict
                if clip_local is not None:
                    node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                
                # Sum
                eps_pos = node_eq_local

                # Update
                sampling_type = kwargs.get("sampling_type", "ddpm_noisy")   # types: generalized, ddpm_noisy, ld

                noise = torch.randn_like(pos)
                if sampling_type == 'generalized' or sampling_type == 'ddpm_noisy':
                    b = self.betas
                    t = t[0]
                    next_t = (torch.ones(1) * j).to(pos.device)
                    at = compute_alpha(b, t.long())
                    at_next = compute_alpha(b, next_t.long())

                    if sampling_type == 'generalized':
                        eta = kwargs.get("eta", 1.)
                        et = -eps_pos
                        ## original
                        # pos0_t = (pos - et * (1 - at).sqrt()) / at.sqrt()
                        ## reweighted
                        # pos0_t = pos - et * (1 - at).sqrt() / at.sqrt()
                        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                        c2 = ((1 - at_next) - c1 ** 2).sqrt()
                        # pos_next = at_next.sqrt() * pos0_t + c1 * noise + c2 * et
                        # pos_next = pos0_t + c1 * noise / at_next.sqrt() + c2 * et / at_next.sqrt()

                        # pos_next = pos + et * (c2 / at_next.sqrt() - (1 - at).sqrt() / at.sqrt()) + noise * c1 / at_next.sqrt()
                        step_size_pos_ld = step_lr * (sigmas[i] / 0.01) ** 2 / sigmas[i]
                        step_size_pos_generalized = 5 * ((1 - at).sqrt() / at.sqrt() - c2 / at_next.sqrt())
                        step_size_pos = step_size_pos_ld if step_size_pos_ld<step_size_pos_generalized else step_size_pos_generalized

                        step_size_noise_ld = torch.sqrt((step_lr * (sigmas[i] / 0.01) ** 2) * 2)
                        step_size_noise_generalized = 3 * (c1 / at_next.sqrt())
                        step_size_noise = step_size_noise_ld if step_size_noise_ld<step_size_noise_generalized else step_size_noise_generalized

                        pos_next = pos - et * step_size_pos +  noise * step_size_noise

                    elif sampling_type == 'ddpm_noisy':
                        atm1 = at_next
                        beta_t = 1 - at / atm1
                        e = -eps_pos
                        pos0_from_e = (1.0 / at).sqrt() * pos - (1.0 / at - 1).sqrt() * e
                        mean_eps = (
                            (atm1.sqrt() * beta_t) * pos0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * pos
                        ) / (1.0 - at)
                        mean = mean_eps
                        mask = 1 - (t == 0).float()
                        logvar = beta_t.log()
                        pos_next = mean + mask * torch.exp(0.5 * logvar) * noise

                elif sampling_type == 'ld':
                    if i>0:
                        pos_next = 1/alpha[i] * (pos - betas[i]/sigmas[i] * eps_pos) + noise * sigmas_1[i-1]/sigmas_1[i] * betas[i]
                    else:
                        pos_next = 1/alpha[i] * (pos - betas[i]/sigmas[i] * eps_pos)
                
                pos = pos_next * linker_mask + pos_ref * fragment_mask

                if torch.isnan(pos).any():
                    print('NaN detected. Please restart.')
                    raise FloatingPointError()
                pos = center_pos(pos, batch)
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

def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom

        
