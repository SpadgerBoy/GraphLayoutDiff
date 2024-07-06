import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
import numpy as np
from numpy import pi as PI
from tqdm.auto import tqdm
import torch.nn.functional as F

from ..common import MultiLayerPerceptron, assemble_node_pair_feature, extend_graph_order_radius, assemble_node_pair_feature_1
from ..geometry import get_distance, eq_transform
from ..encoder import SchNetEncoder, GINEncoder, GCLEncoder,get_edge_encoder, GINEncoder_Global, get_node_encoder, get_degree_encoder

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

def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = (alphas2[1:] / alphas2[:-1])

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2

def polynomial_schedule(timesteps: int, s=1e-4, power=3.):
    """
    A noise schedule based on a simple polynomial equation: 1 - x^power.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas2 = (1 - np.power(x / steps, power)) ** 2

    alphas2 = clip_noise_schedule(alphas2, clip_value=0.001)

    precision = 1 - 2 * s

    alphas2 = precision * alphas2 + s

    return alphas2

class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps, precision):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        splits = noise_schedule.split('_')
        assert len(splits) == 2
        power = float(splits[1])
        alphas2 = polynomial_schedule(timesteps, s=precision, power=power)

        # print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        # print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]

def GetNoiseSchedule(noise_schedule, timesteps, precision):
    timesteps = timesteps

    splits = noise_schedule.split('_')
    assert len(splits) == 2
    power = float(splits[1])
    alphas2 = polynomial_schedule(timesteps, s=precision, power=power)

    # print('alphas2', alphas2)

    sigmas2 = 1 - alphas2

    log_alphas2 = np.log(alphas2)
    log_sigmas2 = np.log(sigmas2)

    log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

    # print('gamma', -log_alphas2_to_sigmas2)

    gamma = torch.nn.Parameter(
        torch.from_numpy(-log_alphas2_to_sigmas2).float(),
        requires_grad=False)
    
    return gamma


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

class EDM_Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.edge_encoder_local = get_edge_encoder(config)

        self.encoder_local = GCLEncoder(
            hidden_dim = config.hidden_dim,
            num_convs = config.num_convs_local,
            node_channels = config.laplacian_eigenvector
        )

        self.grad_local_dist_mlp = MultiLayerPerceptron(
            3 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1],
            activation = 'relu'
        )
    
    def forward(self, node_emb, node_type, node_degree, pos, edge_index, edge_type, edge_length, batch, time_step,
                 return_edges=False,extend_order=True, extend_radius=True):
        
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

        h_pair_local = assemble_node_pair_feature_1(
            node_attr = node_attr_local,
            edge_index=edge_index,
            edge_attr=edge_attr_local,
        )

        edge_inv_local = self.grad_local_dist_mlp(h_pair_local)

        # node_eq_local = eq_transform(edge_inv_local, pos, edge_index, edge_length)
        # pos = pos + node_eq_local

        row, col = edge_index
        dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])   # (E, 3)
        trans = dd_dr * edge_inv_local
        agg = unsorted_segment_sum(trans, row, num_segments=pos.size(0),
                            normalization_factor=100,
                            aggregation_method='sum')

        pos = pos + agg

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
            # betas = get_beta_schedule(  # betas代表每个时间步t的噪音水平
            #     beta_schedule=config.beta_schedule,
            #     beta_start=config.beta_start,
            #     beta_end=config.beta_end,
            #     num_diffusion_timesteps=config.num_diffusion_timesteps
            # )   
            # betas = torch.from_numpy(betas).float()
            # self.betas = nn.Parameter(betas, requires_grad=False)
            # alphas = (1. - betas).cumprod(dim=0)    # 1-betas，再累积
            # self.alphas = nn.Parameter(alphas, requires_grad=False)
            # self.num_timesteps = self.betas.size(0)

            gamma = GetNoiseSchedule("polynomial_2", timesteps=config.num_diffusion_timesteps, precision=1e-5)
            alpha = torch.sqrt(torch.sigmoid(-gamma))
            sigma = torch.sqrt(torch.sigmoid(gamma))
            self.gamma = nn.Parameter(gamma, requires_grad=False)
            self.alpha = nn.Parameter(alpha, requires_grad=False)
            self.sigma = nn.Parameter(sigma, requires_grad=False)
            self.num_timesteps = config.num_diffusion_timesteps

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
        pos_noise = pos_final - pos

        return pos_noise
    

    def get_loss(self, node_emb, node_type, node_degree, pos, edge_index, edge_type, batch, num_nodes_per_graph, num_graphs,
                anneal_power=2.0, return_unreduced_loss=False, return_unreduced_edge_loss=False, extend_order=True, extend_radius=True, is_sidechain=None):
        if self.model_type == 'diffusion':
            return self.get_loss_diffusion(node_emb,node_type,  node_degree, pos, edge_index, edge_type, batch, num_nodes_per_graph, num_graphs,
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
        a = self.alpha.index_select(0, time_step)  # 0代表维度，time_step代表索引。 size: (G, )
        a_pos = a.index_select(0, node2graph).unsqueeze(-1) # (N,1)

        s = self.sigma.index_select(0, time_step)
        s_pos = s.index_select(0, node2graph).unsqueeze(-1)

        # 得到pos上的噪声，以及扰动后的pos_perturbed
        pos_noise = torch.zeros(size=pos.size(), device=pos.device)
        pos_noise.normal_() # 标准正态分布，pos_noise中的元素被随机生成的值替换

        pos_perturbed = pos * a_pos + pos_noise * s_pos

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

        loss_local = (pos_noise_predict - pos_noise) ** 2
        loss_local = torch.sum(loss_local, dim=-1, keepdim=True)

        loss_global = torch.zeros(size=loss_local.size(), device=pos.device)

        loss = loss_global + loss_local

        if return_unreduced_edge_loss:
            pass
        elif return_unreduced_loss:
            return loss, loss_global, loss_local
        else:
            return loss


    def langevin_dynamics_sample(self, node_emb, node_type, node_degree, pos_init, edge_index, edge_type, batch, num_graphs, extend_order, extend_radius=True,
                                 n_steps=100, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None, min_sigma=0,
                                 global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, **kwargs):
        
        if self.model_type == 'diffusion':
            return self.langevin_dynamics_sample_diffusion(node_emb, node_type, node_degree, pos_init, edge_index, edge_type, batch, num_graphs, extend_order, extend_radius,
                        n_steps, step_lr, clip, clip_local, clip_pos, min_sigma,
                        global_start_sigma, w_global, w_reg,
                        sampling_type=kwargs.get("sampling_type", 'ddpm_noisy'), eta=kwargs.get("eta", 1.))
    

    def langevin_dynamics_sample_diffusion(self, node_emb, node_type, node_degree, pos_init, edge_index, edge_type, batch, num_graphs, extend_order, extend_radius=True,
                                 n_steps=100, step_lr=0.0000010, clip=1000, clip_local=None, clip_pos=None, min_sigma=0,
                                 global_start_sigma=float('inf'), w_global=0.2, w_reg=1.0, **kwargs):
        
        def compute_alpha(beta, t):
            beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
            a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)  # .view(-1, 1, 1, 1)
            return a
        
        # alpha = (1.0-self.betas).sqrt()
        # betas = self.betas
        # sigmas = (1.0-self.alphas).sqrt()
        # sigmas_1 = (1.0-self.alphas)

        pos_traj = []
        with torch.no_grad():
            seq = range(self.num_timesteps-n_steps, self.num_timesteps)
            seq_next = [-1] + list(seq[:-1])

            pos = pos_init

            for i,j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='sample'):
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

                # Local
                node_eq_local = pos_noise_predict
                if clip_local is not None:
                    node_eq_local = clip_norm(node_eq_local, limit=clip_local)
                
                # Sum
                eps_pos = node_eq_local

                # Update
                sampling_type = kwargs.get("sampling_type", "ddpm_noisy")   # types: generalized, ddpm_noisy, ld

                noise = torch.randn_like(pos)

                if sampling_type == 'ld':
                    if i > 0:
                        gamma_i = self.gamma[i]
                        gamma_j = self.gamma[j]

                        sigma_i = self.sigma[i]
                        sigma_j = self.sigma[j]

                        sigma2_i_given_j, sigma_i_given_j, alpha_i_given_j = self.sigma_and_alpha_t_given_s(gamma_i, gamma_j)

                        mu = pos / alpha_i_given_j - (sigma2_i_given_j / alpha_i_given_j / sigma_i) * eps_pos

                        sigma = sigma_i_given_j * sigma_j / sigma_i

                        pos_next = mu + noise * sigma
                    else:
                        gamma_i = self.gamma[i]
                        sigma_i = self.sigma[i]
                        alpha_i = self.alpha[i]

                        mu = 1. / alpha_i * (pos - sigma_i * eps_pos)
                        sigma = torch.exp(-(-0.5 * gamma_i))

                        pos_next = mu + sigma * noise
                
                pos = pos_next

                if torch.isnan(pos).any():
                    print('NaN detected. Please restart.')
                    raise FloatingPointError()
                pos = center_pos(pos, batch)
                if clip_pos is not None:
                    pos = torch.clamp(pos, min=-clip_pos, max=clip_pos)
                pos_traj.append(pos.clone().cpu())
        
        return pos, pos_traj
    
    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t))

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s


        

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

        
