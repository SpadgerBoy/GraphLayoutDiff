import os
import time
import random
import logging
import torch
import numpy as np
from glob import glob
from logging import Logger
from tqdm.auto import tqdm
from torch_geometric.data import Batch

def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir

def get_logger(name, log_dir=None, log_fn='log.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, log_fn))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def repeat_data(data, num_repeat):
    datas = [data.clone() for i in range(num_repeat)]
    return Batch.from_data_list(datas)