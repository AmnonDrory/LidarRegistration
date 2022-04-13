import torch
from torch.utils.data import DataLoader, dataset
from dataloader.KITTI_balanced_loader import KITTIBalancedPairDataset
from dataloader.ApolloSouthbay_balanced_loader import ApolloSouthbayBalancedPairDataset
from dataloader.LyftLEVEL5_balanced_loader import LyftLEVEL5BalancedPairDataset
from dataloader.NuScenes_balanced_loader import NuScenesBostonDataset, NuScenesSingaporeDataset
from dataloader.generic_balanced_loader import collate_fn
from utils.experiment_utils import print_to_file_and_screen

def get_dataset_name(dataset_code):
    LUT = {
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

    assert dataset_code in LUT.keys(), "dataset name should be one of the following:" + ', '.join(LUT.keys())
    return LUT[dataset_code]
    
def make_dataloader(phase, args, world_size, rank, seed, outfile):
    
    LUT = {
        'KITTI': KITTIBalancedPairDataset,
        'ApolloSouthbay': ApolloSouthbayBalancedPairDataset,
        'LyftLEVEL5': LyftLEVEL5BalancedPairDataset,
        'NuScenesBoston': NuScenesBostonDataset,
        'NuScenesSingapore': NuScenesSingaporeDataset,
    }    

    if phase == 'train':
        do_shuffle = args['SHUFFLE']
    else:
        do_shuffle = False
        
    Dataset = LUT[ get_dataset_name(args['DATASET']) ]
    dataset = Dataset(phase, do_shuffle)

    if rank==0:
        print_to_file_and_screen(outfile, "%d %s samples" % (len(dataset), phase))

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=do_shuffle,
        seed=seed)

    loader = DataLoader(dataset,
                              num_workers=0,
                              batch_size=args['LOCAL_BATCH_SIZE'],
                              shuffle=False,
                              drop_last=False,
                              pin_memory=True,
                              sampler=sampler,
                              collate_fn=collate_fn)
    
    s = "machine %d: Total %s Pairs: %d" % (rank, phase, len(loader.sampler))
    if rank==0:
        print_to_file_and_screen(outfile, s)
    else:
        print(s)

    return loader