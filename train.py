import os
import shutil
import argparse
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
from glob import glob
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
import pdb

from utils.misc import *
from utils.transforms import *
from utils.datasets import GraphLayoutDataset
from models.diffnet.diffnet_9 import GraphDiffusionNetwork
from utils.common import get_optimizer, get_scheduler
from torch_geometric.utils import to_undirected


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()

    config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]

    seed_all(config.train.seed)


    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    logger = get_logger('train', log_dir)
    logger.info(args)
    logger.info(config)

    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)   

    writer = torch.utils.tensorboard.SummaryWriter(log_dir)

    shutil.copytree('./models', os.path.join(log_dir, 'models'))
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    # Datasets and loaders
    logger.info('Loading datasets...')
    if config.model.pe_type == "laplacian":
        transforms = Compose([
            CountNodesPerGraph(),
            AddUndiectedEdge(),
            AddNodeType(),
            AddNodeMask(),
            AddNodeDegree(),
            AddLaplacianEigenvectorPE(k=int(config.model.laplacian_eigenvector)), # Offline edge augmentation
            # AddRandomWalkPE(walk_length=int(config.model.laplacian_eigenvector)),
            AddEdgeType(),
            AddHigherOrderEdges(order=config.model.edge_order), # Offline edge augmentation
        ])
    elif config.model.pe_type == "rdwalk":
        transforms = Compose([
            CountNodesPerGraph(),
            AddUndiectedEdge(),
            AddNodeType(),  # 每个node上添加上graph的大小
            AddNodeMask(),
            AddNodeDegree(),
            # AddLaplacianEigenvectorPE(k=int(config.model.laplacian_eigenvector)), # Offline edge augmentation
            AddRandomWalkPE(walk_length=int(config.model.laplacian_eigenvector)),
            AddEdgeType(),
            AddHigherOrderEdges(order=config.model.edge_order), # Offline edge augmentation
        ])

    if config.model.regular_data:
        train_set = GraphLayoutDataset(config.dataset.train, transform=transforms, path2=config.dataset.regular)
    else:
        train_set = GraphLayoutDataset(config.dataset.train_circular, transform=transforms)
    val_set = GraphLayoutDataset(config.dataset.val_circular, transform=transforms)
    # test_set = GraphLayoutDataset(config.dataset.test, transform=transforms)  

    # pdb.set_trace()


    
    train_iterator = inf_iterator(DataLoader(train_set, config.train.batch_size, shuffle = True))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=True)

    logger.info('Building model...')
    model = GraphDiffusionNetwork(config.model).to(args.device)

    # Optimizer
    optimizer_global = get_optimizer(config.train.optimizer, model.model_global)
    optimizer_local = get_optimizer(config.train.optimizer, model.model_local)
    scheduler_global = get_scheduler(config.train.scheduler, optimizer_global)
    scheduler_local = get_scheduler(config.train.scheduler, optimizer_local)

    start_iter = 1

    def train(it):
        model.train()
        optimizer_global.zero_grad()
        optimizer_local.zero_grad()
        batch = next(train_iterator).to(args.device)
        loss, loss_global, loss_local = model.get_loss(
            node_emb = batch.node_emb,
            node_type = batch.node_type,
            node_degree = batch.degrees,
            pos = batch.pos,
            edge_index = batch.edge_index,
            edge_type = batch.edge_type,
            batch = batch.batch,
            num_nodes_per_graph = batch.num_nodes_per_graph,
            num_graphs = batch.num_graphs,
            anneal_power = config.train.anneal_power,
            return_unreduced_loss = True,
            extend_order = False,    # Done in transforms.
        )
        loss = loss.mean()
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer_global.step()
        optimizer_local.step()

        logger.info('[Train] Iter %05d | Loss %.2f | Loss(Global) %.2f | Loss(Local) %.2f | Grad %.2f | LR(Global) %.6f | LR(Local) %.6f' % (
            it, loss.item(), loss_global.mean().item(), loss_local.mean().item(), orig_grad_norm, optimizer_global.param_groups[0]['lr'], optimizer_local.param_groups[0]['lr'],
        ))


        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('train/loss_global', loss_global.mean(), it)
        writer.add_scalar('train/loss_local', loss_local.mean(), it)
        writer.add_scalar('train/lr_global', optimizer_global.param_groups[0]['lr'], it)
        writer.add_scalar('train/lr_local', optimizer_local.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad_norm', orig_grad_norm, it)
        writer.flush()
    
    def validata(it):
        sum_loss, sum_n = 0, 0
        sum_loss_global, sum_n_global = 0, 0
        sum_loss_local, sum_n_local = 0, 0
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(val_loader, desc="Validation")):
                batch = batch.to(args.device)
                loss, loss_global, loss_local = model.get_loss(
                    node_emb = batch.node_emb,
                    node_type = batch.node_type,
                    node_degree = batch.degrees,
                    pos = batch.pos,
                    edge_index = batch.edge_index,
                    edge_type = batch.edge_type,
                    batch = batch.batch,
                    num_nodes_per_graph = batch.num_nodes_per_graph,
                    num_graphs = batch.num_graphs,
                    anneal_power = config.train.anneal_power,
                    return_unreduced_loss = True,
                    extend_order = False,    # Done in transforms.
                )
                sum_loss += loss.sum().item()
                sum_n += loss.size(0)
                sum_loss_global += loss_global.sum().item()
                sum_n_global += loss_global.size(0)
                sum_loss_local += loss_local.sum().item()
                sum_n_local += loss_local.size(0)
        avg_loss = sum_loss / sum_n
        avg_loss_global = sum_loss_global / sum_n_global
        avg_loss_local = sum_loss_local / sum_n_local

        if config.train.scheduler.type == 'plateau':
            scheduler_global.step(avg_loss_global)
            scheduler_local.step(avg_loss_local)
        else:
            scheduler_global.step()
            scheduler_local.step()

        logger.info('[Validate] Iter %05d | Loss %.6f | Loss(Global) %.6f | Loss(Local) %.6f' % (
            it, avg_loss, avg_loss_global, avg_loss_local,
        ))
        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_global', avg_loss_global, it)
        writer.add_scalar('val/loss_local', avg_loss_local, it)
        writer.flush()
        return avg_loss
    
    try:
        for it in range(start_iter, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                avg_val_loss = validata(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer_global': optimizer_global.state_dict(),
                    'scheduler_global': scheduler_global.state_dict(),
                    'optimizer_local': optimizer_local.state_dict(),
                    'scheduler_local': scheduler_local.state_dict(),
                    'iteration': it,
                    'avg_val_loss': avg_val_loss,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')








