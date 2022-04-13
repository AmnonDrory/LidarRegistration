import numpy as np
import os
import sys
import torch
import tempfile
from glob import glob
from easydict import EasyDict as edict

from dataloader.dataloader import get_dataset_name
from dataloader.refinement_dataloader import make_dataloader
from general.TicToc import *
from utils.experiment_utils import print_to_file_and_screen, generate_output_dir
from net.refinement_tester import refinement_tester

def analyze_stats(config):    
    
    try:
        outfile = open(config.outfile_name, "a")
    except:
        outfile = open(config.outfile, "a")

    file_base = config.tmp_file_base
    file_names = glob(file_base + '*.stats.npy')
    arrs_list = []
    for filename in file_names:
        stats = np.load(filename)        
        arrs_list.append(stats)
    all_stats = np.vstack(arrs_list)


    np.save(config.outdir + 'raw_stats.npy', all_stats)

    failures_dir = config.outdir + 'failed/'
    os.makedirs(failures_dir)

    stats_mean = all_stats.mean(0)    
    stats_median = np.median(all_stats,axis=0)
    stats_95 = np.quantile(all_stats, 0.95, axis=0)    
    
    num_total = all_stats.shape[0]
    ICP_num_failed = num_total - np.sum(all_stats[:,0])
    BBR_num_failed = num_total - np.sum(all_stats[:,4])
    sym_num_failed = num_total - np.sum(all_stats[:,8])

    pm = u"\u00B1"
    s = "\n"
    offs = 0
    s += f"ICP          | recall: {100*stats_mean[0+offs]:.2f}, #failed/#total: {int(ICP_num_failed)}/{num_total}\n"
    s += f"      (mean) | TE(cm): {100*stats_mean[1+offs]:.3f}, RE(deg): {stats_mean[2+offs]:.3f}, reg time(s): {stats_mean[3+offs]:.3f}\n"
    s += f"    (median) | TE(cm): {100*stats_median[1+offs]:.3f}, RE(deg): {stats_median[2+offs]:.3f}, reg time(s): {stats_median[3+offs]:.3f}\n"
    s += f"       (95%) | TE(cm): {100*stats_95[1+offs]:.3f}, RE(deg): {stats_95[2+offs]:.3f}, reg time(s): {stats_95[3+offs]:.3f}\n\n"

    offs = 4
    s += f"BBR-F        | recall: {100*stats_mean[0+offs]:.2f}, #failed/#total: {int(BBR_num_failed)}/{num_total}\n"
    s += f"      (mean) | TE(cm): {100*stats_mean[1+offs]:.3f}, RE(deg): {stats_mean[2+offs]:.3f}, reg time(s): {stats_mean[3+offs]:.3f}\n"
    s += f"    (median) | TE(cm): {100*stats_median[1+offs]:.3f}, RE(deg): {stats_median[2+offs]:.3f}, reg time(s): {stats_median[3+offs]:.3f}\n"
    s += f"       (95%) | TE(cm): {100*stats_95[1+offs]:.3f}, RE(deg): {stats_95[2+offs]:.3f}, reg time(s): {stats_95[3+offs]:.3f}\n\n"

    offs = 8
    s += f"Symmetric ICP| recall: {100*stats_mean[0+offs]:.2f}, #failed/#total: {int(sym_num_failed)}/{num_total}\n"
    s += f"      (mean) | TE(cm): {100*stats_mean[1+offs]:.3f}, RE(deg): {stats_mean[2+offs]:.3f}, reg time(s): {stats_mean[3+offs]:.3f}\n"
    s += f"    (median) | TE(cm): {100*stats_median[1+offs]:.3f}, RE(deg): {stats_median[2+offs]:.3f}, reg time(s): {stats_median[3+offs]:.3f}\n"
    s += f"       (95%) | TE(cm): {100*stats_95[1+offs]:.3f}, RE(deg): {stats_95[2+offs]:.3f}, reg time(s): {stats_95[3+offs]:.3f}\n\n"

    print_to_file_and_screen(outfile, s)
    print(f"saved to {outfile.name}")
    outfile.close()

def usage():
    print("""
    refinement should be called with the following command-line arguments:
        1. the name of the dataset to test on (see dataloaders/dataloader.py for options)
    """
    )

def get_config(start_time=None):
    config = edict({})    
    config.args = {}

    if sys.argv[1] == 'test_parallel':
        start_time = sys.argv[2]
        config.tmp_file_base = sys.argv[3]
        config.world_size = int(sys.argv[4])
        if sys.argv[5] == 'analysis':
            config.rank = None
            config.do_analysis = True
        else:
            config.rank = int(sys.argv[5])
            config.do_analysis = False
        sys.argv = sys.argv[5:]
    else:
        start_time=None
        config.tmp_file_base = tempfile.gettempdir() + '/refinement_%016d' % int(np.random.rand()*10**16)    
        config.world_size = 1
        config.rank = 0 
        config.do_analysis = True

    if len(sys.argv) < 2:
        usage()
        exit(1)

    config.args['DATASET'] = sys.argv[1]
    config.phase = 'test'

    config.args['LOCAL_BATCH_SIZE'] = 6
    config.args['SHUFFLE'] = False
    config.outdir = generate_output_dir(get_dataset_name(config.args['DATASET']), 'Test.%s' % config.phase, start_time)
    config.outfile = config.outdir + 'log.txt'
    config.trans_err_thresh = 0.6  # m
    config.rot_err_thresh = 5  # deg

    if config.rank == 0:
        with open(config.outfile, 'a+') as fid:
            for k in config.__dict__.keys():
                print_to_file_and_screen(fid, f"config.{k}: {config.__dict__[k]}")

    return config
    
def main():
    config = get_config()
    seed = 0
    
    if config.rank is not None:
        refinement_subset(config.rank, config.world_size, seed, config)

    if config.do_analysis:
	    analyze_stats(config)            
    
	    tmp_files = glob(config.tmp_file_base + '*')
	    for f in tmp_files:
	        os.remove(f)

def refinement_subset(rank, world_size, seed, config):
      # This function is performed in parallel in several processes, one for each available GPU
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda:%d' % torch.cuda.current_device()
    print("process %d, GPU: %s" % (rank, device))

    if rank == 0:
        config.outfile_name = config.outfile
        config.outfile = open(config.outfile, 'a+')

    test_loader = make_dataloader(config.args, world_size, rank, seed, config.outfile)
    tester = refinement_tester(config, rank) 
    results = tester.run_refinement_test(test_loader)
    np.save(f"{config.tmp_file_base}_{world_size}_{rank}.stats.npy", results)    

    if rank == 0:
        config.outfile.close()

if __name__ == '__main__':
    main()
