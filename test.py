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
from models.diffnet.diffnet_41 import GraphDiffusionNetwork
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
    # parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--num_confs', type=num_confs, default=num_confs('2x'))
    # parser.add_argument('--test_set', type=str, default='./final_data/hierarchy_tree/example_layout.pkl')
    parser.add_argument('--test_set', type=str, default='./data/rome_kamada/val_layout.pkl')
    parser.add_argument('--test_graph', type=str, default=None)
    # parser.add_argument('--test_set', type=str, default='./data/tree_graph/tree_graph.pkl')
    # parser.add_argument('--test_set', type=str, default='./data/path_graph/path_graph.pkl')
    # parser.add_argument('--test_set', type=str, default='./data/regular_graph/regular_graph.pkl')
    # parser.add_argument('--test_set', type=str, default='./data/control/intelligent_graph_4.pkl')
    # parser.add_argument('--test_set', type=str, default='./data/rome_circular/test_data_discrete.pkl')
    # parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=17)
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
    # seed_all(config.train.seed)
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
            AddNodeMask(node_mask=0.0),
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
            AddNodeMask(node_mask=0.0),
            AddNodeDegree(),
            # AddLaplacianEigenvectorPE(k=int(config.model.laplacian_eigenvector)), # Offline edge augmentation
            AddRandomWalkPE(walk_length=int(config.model.laplacian_eigenvector)),
            AddEdgeType(),
            AddHigherOrderEdges(order=config.model.edge_order), # Offline edge augmentation
            # AddFragmentEdge(fragment_edge_type=config.model.fragment_edge_type),
    ])

    test_set = PackedGraphLayoutDataset(args.test_set, transform=transforms)

    test_set_selected = []
    for i, data in enumerate(test_set):
        if not (args.start_idx <= i < args.end_idx): continue
        test_set_selected.append(data)


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
        for _ in range(2):
            try:
                pos_init = torch.randn(batch.num_nodes, 2).to(args.device)  # 标准正态分布sample
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
    

    generate_dir = os.path.join(output_dir, "generate")
    os.makedirs(generate_dir)
    for i in range(0, len(results)):

        data = results[i]
        num_nodes = int(data['node_emb'].shape[0])
        graph_name = data['graph_name']

        print("data_graph_name: ", graph_name)

        #folder_path = "./graph_data/rome"
        folder_path = args.test_graph
        graph_path = os.path.join(folder_path, graph_name)

        G = nx.read_graphml(graph_path)
        H = nx.convert_node_labels_to_integers(G, label_attribute='original_label')

        # -----------------------------
        pos_idx = 0     # 第几种pos
        positions = data.pos_gen[pos_idx*num_nodes: (pos_idx+1)*num_nodes, :].tolist()
        # positions = data.pos_ref[pos_idx*num_nodes: (pos_idx+1)*num_nodes, :].tolist()


        pos_dict = {node: (positions[node][0], positions[node][1]) for node in range(0, num_nodes)}
        nx.set_node_attributes(H, pos_dict, 'pos')

        path_generate = os.path.join(generate_dir, str(pos_idx) + "_test" + str(i) +".pkl" )
        with open(path_generate, 'wb') as file:
            pickle.dump(H, file)





    
