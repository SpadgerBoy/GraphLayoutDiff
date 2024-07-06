import os
import argparse
import pickle
import yaml
import torch
from glob import glob
from tqdm.auto import tqdm
from easydict import EasyDict

from torch_geometric.data import Data
from utils.datasets import *
from utils.transforms import *
from utils.misc import *
from models.diffnet.sample import GraphDiffusionNetwork
import networkx as nx
from networkx.readwrite import json_graph
import json
import pdb

def num_confs(num:str):
    if num.endswith('x'):
        return lambda x:x*int(num[:-1])
    elif int(num) > 0: 
        return lambda x:int(num)
    else:
        raise ValueError()

def data_process(graph_data):

    # 创建一个无向图
    graph = nx.Graph()

    # 添加节点
    for node in graph_data['nodes']:
        graph.add_node(node['id'])

    # 添加边
    for link in graph_data['links']:
        graph.add_edge(link['source'], link['target'])
    
    H = nx.convert_node_labels_to_integers(graph, label_attribute='original_label')

    fragment_mask = np.array(graph_data['mask']).tolist()
    torch_fragment_mask = torch.tensor(fragment_mask, dtype=torch.long).unsqueeze(-1)
    torch_linker_mask = 1-torch_fragment_mask
    # linker_mask = [0 if element == 1 else 1 for element in fragment_mask]

    # 从节点数据中提取 x 和 y 坐标
    x_coords = [node["x"] for node in graph_data['nodes']]
    y_coords = [node["y"] for node in graph_data['nodes']]

    # 将 x 和 y 坐标合并为一个大小为 n*2 的 tensor
    pos_ref = np.array(np.vstack((x_coords, y_coords)).T).tolist()
    torch_pos_ref = torch.tensor(pos_ref, dtype=torch.float)

    edge_list = list(H.edges())
    edge_list = np.array(edge_list).tolist()

    torch_edge_index = torch.tensor(edge_list, dtype=torch.long)
    torch_edge_index = torch.transpose(torch_edge_index, 0, 1)

    control_flag = graph_data['control_flag']

    data = [Data(pos_ref=torch_pos_ref, edge_index=torch_edge_index, fragment_mask=torch_fragment_mask, linker_mask=torch_linker_mask, control_flag = control_flag)]

    return data

def layout_generate(graph_data, cuda_idx):

    device = "cuda:"+str(cuda_idx)
    parser = argparse.ArgumentParser()
    # parser.add_argument('--ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('--device', type=str, default='cuda:6')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--num_confs', type=num_confs, default=num_confs('2x'))
    # parser.add_argument('--test_set', type=str, default='./data/rome_kamada/test_data_discrete.pkl')
    parser.add_argument('--test_set', type=str, default='./data/tree_graph/tree_graph.pkl')
    # parser.add_argument('--test_set', type=str, default='./data/path_graph/path_graph.pkl')
    # parser.add_argument('--test_set', type=str, default='./data/regular_graph/regular_graph.pkl')
    # parser.add_argument('--test_set', type=str, default='./data/control/intelligent_graph_4.pkl')
    # parser.add_argument('--test_set', type=str, default='./data/rome_circular/test_data_discrete.pkl')
    # parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=800)
    parser.add_argument('--end_idx', type=int, default=1000)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--n_steps', type=int, default=5000,
                    help='sampling num steps; for DSM framework, this means num steps for each noise scale')
    parser.add_argument('--w_global', type=float, default=1.0,
                    help='weight for global gradients')
    parser.add_argument('--global_start_sigma', type=float, default=0.5,
                    help='enable global gradients only when noise is low')
    parser.add_argument('--sampling_type', type=str, default='ld',
                    help='generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics')
    parser.add_argument('--eta', type=float, default=1.0,
                    help='weight for DDIM and DDPM: 0->DDIM, 1->DDPM')
    parser.add_argument('--clip', type=float, default=1000)
    parser.add_argument('--save_traj', action='store_true', default=False,
                    help='whether store the whole trajectory for sampling')
    args = parser.parse_args()

    # args.device = device

    # Load checkpoint
    ckpt_type = "compose_control_final"
    ckpt_path = "./logs_final/" + ckpt_type+ "/checkpoints/400000.pt"
    ckpt = torch.load(ckpt_path)
    config_path = "./logs_final/" + ckpt_type + "/rome_default.yml"

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    seed_all(config.train.seed)
    # seed_all(42)

    log_path = "./logs/demo"
    log_dir = log_path

    # Logging
    output_dir = get_new_log_dir(log_dir, 'sample', tag=args.tag)
    logger = get_logger('test', output_dir)
    logger.info(args)

    # Dataset and loaders
    logger.info('Loading datasets...')
    if config.model.pe_type == "laplacian":
        transforms = Compose([
            CountNodesPerGraph(),
            AddUndiectedEdge(),
            AddNodeType(), 
            # AddNodeMask(node_mask=0.0),
            AddNodeDegree(),
            AddLaplacianEigenvectorPE(k=int(config.model.laplacian_eigenvector)), # Offline edge augmentation
            # AddRandomWalkPE(walk_length=int(config.model.laplacian_eigenvector)),
            AddEdgeType(),
            AddHigherOrderEdges(order=config.model.edge_order), # Offline edge augmentation
            # AddFragmentEdge(fragment_edge_type=config.model.fragment_edge_type),
        ])
    elif config.model.pe_type == "rdwalk":
        transforms = Compose([
            CountNodesPerGraph(),
            AddUndiectedEdge(),
            AddNodeType(), 
            # AddNodeMask(node_mask=0.0),
            AddNodeDegree(),
            # AddLaplacianEigenvectorPE(k=int(config.model.laplacian_eigenvector)), # Offline edge augmentation
            AddRandomWalkPE(walk_length=int(config.model.laplacian_eigenvector)),
            AddEdgeType(),
            AddHigherOrderEdges(order=config.model.edge_order), # Offline edge augmentation
            # AddFragmentEdge(fragment_edge_type=config.model.fragment_edge_type),
    ])
        
    demo_set = data_process(graph_data)
    test_set_selected = DemoGraphLayoutDataset(demo_set, transform=transforms)

    control_flag = test_set_selected[0].control_flag
    
    # Model
    logger.info('Loading model...')
    model = GraphDiffusionNetwork(config.model).to(args.device)
    model.load_state_dict(ckpt['model'])

    
    results = []

    for i,data in enumerate(tqdm(test_set_selected)):
        num_samples = 1

        data_input = data.clone()
        batch = repeat_data(data_input, num_samples).to(args.device) 

        clip_local = None
        for _ in range(1):
            try:
                pos_init = torch.randn(batch.num_nodes, 2).to(args.device)  # 标准正态分布sample
                # pos_ref = torch.zeros(size=(batch.num_nodes, 2)).to(args.device) 
                if control_flag:
                    pos_gen, pos_gen_traj = model.langevin_dynamics_sample(
                        node_emb = batch.node_emb,
                        node_type = batch.node_type,
                        node_degree = batch.degrees,
                        fragment_mask = batch.fragment_mask,
                        linker_mask = batch.linker_mask,
                        pos_ref = batch.pos_ref,
                        pos_init = pos_init,
                        edge_index = batch.edge_index,
                        edge_type = batch.edge_type,
                        batch = batch.batch,
                        num_graphs = batch.num_graphs,
                        extend_order = False,    # Done in transforms.
                        n_steps = config.model.num_diffusion_timesteps,
                        step_lr = 1e-6,
                        w_global=args.w_global,
                        global_start_sigma=args.global_start_sigma,
                        clip=args.clip,
                        clip_local=clip_local,
                        sampling_type=args.sampling_type,
                        eta=args.eta
                    )
                else:
                    pos_gen, pos_gen_traj = model.langevin_dynamics_sample_diffusion_no_control(
                        node_emb = batch.node_emb,
                        node_type = batch.node_type,
                        node_degree = batch.degrees,
                        fragment_mask = batch.fragment_mask,
                        linker_mask = batch.linker_mask,
                        pos_ref = batch.pos_ref,
                        pos_init = pos_init,
                        edge_index = batch.edge_index,
                        edge_type = batch.edge_type,
                        batch = batch.batch,
                        num_graphs = batch.num_graphs,
                        extend_order = False,    # Done in transforms.
                        n_steps = config.model.num_diffusion_timesteps,
                        step_lr = 1e-6,
                        w_global=args.w_global,
                        global_start_sigma=args.global_start_sigma,
                        clip=args.clip,
                        clip_local=clip_local,
                        sampling_type=args.sampling_type,
                        eta=args.eta
                    )
                pos_gen = pos_gen.cpu()
                if args.save_traj:
                    data.pos_gen = torch.stack(pos_gen_traj)
                else:
                    data.pos_gen = pos_gen
                
                results.append(data)

                save_path = os.path.join(output_dir, 'samples_%d.pkl' % i)
                logger.info('Saving samples to: %s' % save_path)
                with open(save_path, 'wb') as f:
                    pickle.dump(results, f)
                
                break # No errors occured, break the retry loop


            except FloatingPointError:
                clip_local = 20
                logger.warning('Retrying with local clipping.')

    save_path = os.path.join(output_dir, 'samples_all.pkl')
    logger.info('Saving samples to: %s ' % save_path)

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    data = results[0]
    # 提取节点和边的信息
    nodes = [{'id': f'{i}', 'x': data.pos_gen[i, 0].item(), 'y': data.pos_gen[i, 1].item()} for i in range(data.num_nodes)]
    edges = [{'source': f'{data.initial_edge_index[0, i].item()}', 'target': f'{data.initial_edge_index[1, i].item()}'} for i in range(data.initial_edge_index.size()[1])]

    # 构造JavaScript格式的图数据
    return_data = {
        'nodes': nodes,
        'links': edges
    }

    return json.dumps(return_data)




    
