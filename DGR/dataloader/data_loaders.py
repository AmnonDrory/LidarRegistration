# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019

# This version includes substantial additions by Amnon Drory (amnondrory@mail.tau.ac.il), Tel-Aviv University, 2021
# Please also cite the following paper:
# - Amnon Drory, Shai Avidan, Raja Giryes, Stress Testing LiDAR Registration

from dataloader.threedmatch_loader import *
from dataloader.kitti_loader import *
from dataloader.KITTI_balanced_loader import *
from dataloader.ApolloSouthbay_balanced_loader import *
from dataloader.NuScenes_balanced_loader import *
from dataloader.LyftLEVEL5_balanced_loader import *

ALL_DATASETS = [
    ThreeDMatchPairDataset07, ThreeDMatchPairDataset05, ThreeDMatchPairDataset03,
    ThreeDMatchTrajectoryDataset, KITTIPairDataset, KITTINMPairDataset, 
    KITTIBalancedPairDataset, ApolloSouthbayBalancedPairDataset, LyftLEVEL5BalancedPairDataset,
    NuScenesBostonDataset, NuScenesSingaporeDataset
]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}


def make_data_loader(config, phase, batch_size, rank=0, world_size=1, seed=0, num_workers=0, shuffle=None):
  assert phase in ['train', 'trainval', 'val', 'test']
  if shuffle is None:
    shuffle = phase != 'test'

  if config.dataset not in dataset_str_mapping.keys():
    logging.error(f'Dataset {config.dataset}, does not exists in ' +
                  ', '.join(dataset_str_mapping.keys()))

  Dataset = dataset_str_mapping[config.dataset]

  use_random_scale = False
  use_random_rotation = False
  transforms = []
  if phase in ['train', 'trainval']:
    use_random_rotation = config.use_random_rotation
    use_random_scale = config.use_random_scale
    transforms += [t.Jitter()]

  if phase in ['val', 'test']:
    use_random_rotation = config.test_random_rotation

  dset = Dataset(phase,
                 transform=t.Compose(transforms),
                 random_scale=use_random_scale,
                 random_rotation=use_random_rotation,
                 config=config,
                 rank=rank)

  sampler = torch.utils.data.distributed.DistributedSampler(
    dset,
    num_replicas=world_size,
    rank=rank,
    shuffle=shuffle,
    seed=seed)

  collation_fn = CollationFunctionFactory(concat_correspondences=False,
                                          collation_type='collate_pair')

  loader = torch.utils.data.DataLoader(
      dset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      collate_fn=collation_fn,
      sampler=sampler,
      pin_memory=True,
      drop_last=True)

  return loader
