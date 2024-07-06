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
from models.diffnet.sample_control import GraphDiffusionNetwork
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('--device', type=str, default='cuda:6')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--num_confs', type=num_confs, default=num_confs('2x'))
    parser.add_argument('--test_set', type=str, default='./final_data/control_final/example_layout_4.pkl')
    # parser.add_argument('--test_set', type=str, default='./final_data/control_final/example_layout_1.pkl')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=4)
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


    # Load checkpoint
    ckpt = torch.load(args.ckpt)
    config_path = glob(os.path.join(os.path.dirname(os.path.dirname(args.ckpt)), '*.yml'))[0]
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    seed_all(42)

    log_dir = os.path.dirname(os.path.dirname(args.ckpt))

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
    
    test_set = GraphLayoutDataset(args.test_set, transform=transforms)

    test_set_selected = []
    for i, data in enumerate(test_set):
        if not (0 <= i < 10): continue
        test_set_selected.append(data)

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
    
    json_data = []
    for data in results:
        # 提取节点和边的信息
        nodes = [{'id': f'{i}', 'x': data.pos_gen[i, 0].item(), 'y': data.pos_gen[i, 1].item()} for i in range(data.num_nodes)]
        edges = [{'source': f'{data.initial_edge_index[0, i].item()}', 'target': f'{data.initial_edge_index[1, i].item()}'} for i in range(data.initial_edge_index.size()[1])]

        # 构造JavaScript格式的图数据
        tmp_data = {
            'nodes': nodes,
            'links': edges
        }
        json_data.append(tmp_data)

    # json_data = json.dumps(json_data)
    json_path = os.path.join(output_dir, "output.json")
    # 将JSON数据写入文件
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file)
    




    
