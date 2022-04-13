import torch
from torch.utils.data import DataLoader
from dataloader.ApolloSouthbay_balanced_loader import ApolloSouthbayRefinementDataset
from dataloader.generic_refinement_loader import collate_fn
from dataloader.dataloader import get_dataset_name
from utils.experiment_utils import print_to_file_and_screen


   
def make_dataloader(args, world_size, rank, seed, outfile):
    
    LUT = {
        'ApolloSouthbay': ApolloSouthbayRefinementDataset,
    }    

    do_shuffle = False
    phase = 'test'
        
    Dataset = LUT[ get_dataset_name(args['DATASET']) ]
    dataset = Dataset()

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