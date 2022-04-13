# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
from dataloader.threedmatch_loader import *
from dataloader.kitti_loader import *
from dataloader.KITTI_balanced_loader import *
from dataloader.ApolloSouthbay_balanced_loader import *
from dataloader.NuScenes_balanced_loader import *
try:
  from dataloader.LyftLEVEL5_balanced_loader import *
except Exception as E:
  print("Ignoring exception: " + str(E))
  class LyftDummy():
    __name__ = None
  LyftLEVEL5BalancedPairDataset = LyftDummy()

ALL_DATASETS = [
    ThreeDMatchPairDataset07, ThreeDMatchPairDataset05, ThreeDMatchPairDataset03,
    ThreeDMatchTrajectoryDataset, KITTIPairDataset, KITTINMPairDataset, 
    KITTIBalancedPairDataset, ApolloSouthbayBalancedPairDataset, LyftLEVEL5BalancedPairDataset,
    NuScenesBostonDataset, NuScenesSingaporeDataset
]
dataset_str_mapping = {d.__name__: d for d in ALL_DATASETS}

def get_dataset_name(dataset_nickname):
  short_names_LUT = {
    'K': 'KITTI',
    'A': 'ApolloSouthbay',
    'L': 'LyftLEVEL5',
    'B': 'NuScenesBoston',
    'S': 'NuScenesSingapore',
    'KITTI': 'KITTI',
    'ApolloSouthbay': 'ApolloSouthbay',
    'LyftLEVEL5': 'LyftLEVEL5',
    'NuScenesBoston': 'NuScenesBoston',
    'NuScenesSingapore': 'NuScenesSingapore',    
    'KITTIBalancedPairDataset': 'KITTI',
    'ApolloSouthbayBalancedPairDataset': 'ApolloSouthbay',
    'LyftLEVEL5BalancedPairDataset': 'LyftLEVEL5',
    'NuScenesBostonDataset': 'NuScenesBoston',
    'NuScenesSingaporeDataset': 'NuScenesSingapore'    
  }  

  if dataset_nickname not in short_names_LUT.keys():
    if dataset_nickname in dataset_str_mapping.keys():
      return dataset_nickname, dataset_nickname
  assert dataset_nickname in short_names_LUT.keys(), "dataset name should be one of the following:" + ', '.join(short_names_LUT.keys())
  
  full_names_LUT = {
    'KITTI': 'KITTIBalancedPairDataset',
    'ApolloSouthbay': 'ApolloSouthbayBalancedPairDataset',
    'LyftLEVEL5': 'LyftLEVEL5BalancedPairDataset',
    'NuScenesBoston': 'NuScenesBostonDataset',
    'NuScenesSingapore': 'NuScenesSingaporeDataset',
  }   
  
  short_name = short_names_LUT[dataset_nickname]
  full_name = full_names_LUT[short_name]
  return full_name, short_name


def make_data_loader(dataset_nickname, config, phase, batch_size, rank=0, world_size=1, seed=0, num_workers=0, shuffle=None):
  assert phase in ['train', 'trainval', 'val', 'test', 'validation']
  if phase == 'validation':
    phase = 'val'
  if shuffle is None:
    shuffle = not phase in ['test', 'val']

  dataset_name, _ = get_dataset_name(dataset_nickname)
  if dataset_name not in dataset_str_mapping.keys():
    logging.error(f'Dataset {dataset_name}, does not exists in ' +
                  ', '.join(dataset_str_mapping.keys()))

  Dataset = dataset_str_mapping[dataset_name]

  use_random_scale = False
  use_random_rotation = False
  transforms = []
  if phase in ['train', 'trainval']:
    use_random_rotation = config.use_random_rotation
    use_random_scale = config.use_random_scale
    transforms += [t.Jitter()]

  if phase in ['val', 'test']:
    use_random_rotation = False

  dset = Dataset(phase,
                 transform=t.Compose(transforms),
                 random_scale=use_random_scale,
                 random_rotation=use_random_rotation,
                 config=config,
                 rank=rank)

  collation_fn = CollationFunctionFactory(concat_correspondences=False,
                                          collation_type='collate_pair')

  if world_size == 1:

    loader = torch.utils.data.DataLoader(
        dset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collation_fn,
        num_workers=num_workers)

  else:

    sampler = torch.utils.data.distributed.DistributedSampler(
      dset,
      num_replicas=world_size,
      rank=rank,
      shuffle=shuffle,
      seed=seed)

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
