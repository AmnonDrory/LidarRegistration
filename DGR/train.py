# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import open3d as o3d  # prevent loading error

import os
import sys
import json
import logging
import torch
from easydict import EasyDict as edict

from config import get_config

from dataloader.data_loaders import make_data_loader
from dataloader.paths import fcgf_weights_file
from core.trainer import WeightedProcrustesTrainer

import pickle
from datetime import datetime
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])

torch.manual_seed(0)
torch.cuda.manual_seed(0)

logging.basicConfig(level=logging.INFO, format="")


def main(config, resume=False):
  seed = np.random.randint(np.iinfo(np.int32).max)
  world_size = torch.cuda.device_count()  
  config.batch_size = int(np.ceil(config.batch_size/world_size))
  config.val_batch_size = int(np.ceil(config.val_batch_size/world_size))
  print("%d GPUs are available, re-adjusted local batch size to %d (train), %d (validation)" % (world_size, config.batch_size, config.val_batch_size))
  
  if world_size == 1:
    train_parallel(0, world_size, seed, config)
  else:
    mp.spawn(train_parallel, nprocs=world_size, args=(world_size,seed, config))  

def train_parallel(rank, world_size, seed, config):
  # This function is performed in parallel in several processes, one for each available GPU
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '8883'
  dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
  torch.manual_seed(seed)
  np.random.seed(seed)
  torch.cuda.set_device(rank)
  device = 'cuda:%d' % torch.cuda.current_device()
  print("process %d, GPU: %s" % (rank, device))

  train_loader = make_data_loader(
      config,
      config.train_phase,
      config.batch_size,
      num_workers=config.train_num_workers,
      rank=rank, world_size=world_size, seed=seed)

  if config.test_valid:
    val_loader = make_data_loader(
        config,
        config.val_phase,
        config.val_batch_size,
        num_workers=config.val_num_workers,
        rank=rank, world_size=world_size, seed=seed)
  else:
    val_loader = None

  trainer = WeightedProcrustesTrainer(
      config=config,
      data_loader=train_loader,
      val_data_loader=val_loader,
      rank=rank
  )

  trainer.train()

def prep_config():
  with open('train_DGR_kitti_argv.pickle', 'rb') as fid:
    sys.argv = pickle.load(fid)  
    
  logger = logging.getLogger()
  config = get_config()
  config.out_dir = config.out_dir.replace('2021-04-07_18-38-35',datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
  print("out_dir: %s" % config.out_dir)
  with open ('out_dir.txt', 'w') as fid:
    fid.write(config.out_dir + '\n')
  
  config.procrustes_loss_weight = 0.0 # DGR paper says that they don't use this loss [originaly: 1.0]
  config.success_rte_thresh = 0.6 # for consistency with test [originaly: 2.0]

  config.dataset = 'ApolloSouthbayBalancedPairDataset'
  config.weights = fcgf_weights_file

  config.use_random_scale = False

  config.max_epoch = 40

  if not(os.path.isdir(config.out_dir)):
    os.makedirs(config.out_dir)

  dconfig = vars(config)
  if config.resume_dir:
    resume_config = json.load(open(config.resume_dir + '/config.json', 'r'))
    for k in dconfig:
      if k not in ['resume_dir'] and k in resume_config:
        dconfig[k] = resume_config[k]
    dconfig['resume'] = resume_config['out_dir'] + '/checkpoint.pth'

  logging.info('===> Configurations')
  for k in dconfig:
    logging.info('    {}: {}'.format(k, dconfig[k]))

  # Convert to dict
  config = edict(dconfig)
  
  return config

if __name__ == "__main__":
  config = prep_config()
  main(config)
