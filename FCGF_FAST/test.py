import numpy as np
import os
import sys
import torch
import tempfile
from glob import glob
from easydict import EasyDict as edict

from dataloader.dataloader import make_dataloader, get_dataset_name
from general.TicToc import *
from utils.experiment_utils import print_to_file_and_screen, generate_output_dir
from net.RANSAC import FCGF_RANSAC_tester

def analyze_stats(config):    
    
    try:
        outfile = open(config.outfile_name, "a")
    except:
        outfile = open(config.outfile, "a")

    file_base = config.tmp_file_base
    file_names = glob(file_base + '*.stats.npy')
    arrs_list = []
    coarse_mots_list = []
    for filename in file_names:
        stats = np.load(filename)        
        arrs_list.append(stats)
        coarse_mots_filename = filename.replace('.stats.npy','.coarse_mots.npy')
        cur_coarse_mots = np.load(coarse_mots_filename)
        coarse_mots_list.append(cur_coarse_mots)        
    all_stats = np.vstack(arrs_list)
    coarse_mots = np.dstack(coarse_mots_list)

    np.save(config.outdir + 'raw_stats.npy', all_stats)

    failures_dir = config.outdir + 'failed/'
    os.makedirs(failures_dir)

    num_repeats = all_stats.shape[2]
    res_list = [] # R_recall, R_num_failed, R_re, R_te, R_time, I_recall, I_num_failed, I_re, I_te, I_time, GT_inlier_ratio
    for repeat in range(num_repeats):
        stats = all_stats[:,:,repeat]
        m_stats = stats.mean(0)    
        RANSAC_succeeded = (stats[:, 0] > 0)
        ICP_succeeded = (stats[:, 4] > 0)
        m_RANSAC_stats = stats[RANSAC_succeeded,:].mean(0)
        m_ICP_stats = stats[RANSAC_succeeded,:].mean(0)
    
        num_total = stats.shape[0]
        num_failed_RANSAC =  num_total - RANSAC_succeeded.sum()
        num_failed_ICP =  num_total - ICP_succeeded.sum()
        cur_res = [
            m_stats[0], num_failed_RANSAC, m_RANSAC_stats[1], m_RANSAC_stats[2], m_stats[3],
            m_stats[4], num_failed_ICP, m_ICP_stats[5], m_ICP_stats[6], m_stats[7],
            m_stats[-4] ]
        res_list.append(cur_res)

        RANSAC_failures = stats[~RANSAC_succeeded]
        ICP_failures = stats[~ICP_succeeded]

        with open(failures_dir + f"RANSAC.repeat_{repeat}.txt", "w") as fid:
            for line in RANSAC_failures:
                fid.write(f"{int(line[-3])} {int(line[-2])} {int(line[-1])} Failed with RTE: {line[1]}, RRE: {line[2]}\n")

        with open(failures_dir + f"ICP.repeat_{repeat}.txt", "w") as fid:
            for line in ICP_failures:
                fid.write(f"{int(line[-3])} {int(line[-2])} {int(line[-1])} Failed with RTE: {line[5]}, RRE: {line[6]}\n")

    res = np.vstack(res_list)
    np.save(config.outdir + 'per_repeat_summaries.npy', res)
    print_to_file_and_screen(outfile, "per-repeat results: R_recall, R_num_failed, R_re, R_te, R_time, I_recall, I_num_failed, I_re, I_te, I_time")
    print_to_file_and_screen(outfile, res)
    print_to_file_and_screen(outfile, "==============================")
    means = np.mean(res, axis=0)
    stds = np.std(res, axis=0)

    pm = u"\u00B1"
    s = "\n"
    s += f"GT inlier ratio: {means[-1]:.3f}\n"
    s += f"RANSAC     | recall: {100*means[0]:.2f}%{pm}{100*stds[0]:.2f}%, #failed/#total: {int(means[1])}{pm}{int(stds[1])}/{num_total}, TE(cm): {100*means[2]:.3f}{pm}{100*stds[2]:.3f}, RE(deg): {means[3]:.3f}{pm}{stds[3]:.3f}, reg time(s): {means[4]:.3f}{pm}{stds[4]:.3f}\n"
    s += f"RANSAC+ICP | recall: {100*means[5]:.2f}%{pm}{100*stds[5]:.2f}%, #failed/#total: {int(means[6])}{pm}{int(stds[6])}/{num_total}, TE(cm): {100*means[7]:.3f}{pm}{100*stds[7]:.3f}, RE(deg): {means[8]:.3f}{pm}{stds[8]:.3f}, ICP time(s): {means[9]:.3f}{pm}{stds[9]:.3f}\n"
    print_to_file_and_screen(outfile, s)
    print(f"saved to {outfile.name}")
    outfile.close()

    coarse_mots_file = open(config.outdir + 'coarse_motions.txt','w')    
    coarse_mots_file.write('session_ind source_ind target_ind mot0 mot1 mot2 mot3 mot4 mot5 mot6 mot7 mot8 mot9 mot10 mot11 mot12 mot13 mot14 mot15\n')
    o1 = np.argsort(all_stats[:,-2])
    all_stats = all_stats[o1,:]
    coarse_mots = coarse_mots[:,:,o1]
    o0 = np.argsort(all_stats[:,-3], kind='stable')
    all_stats = all_stats[o0,:]
    coarse_mots = coarse_mots[:,:,o0]

    for i in range(all_stats.shape[0]):
        cur_mot = coarse_mots[:,:,i].flatten()
        session_ind = int(all_stats[i,-3])
        src_ind = int(all_stats[i,-2])
        tgt_ind = int(all_stats[i,-1])
        s = "%d %d %d " % (session_ind, src_ind, tgt_ind)
        for i in range(len(cur_mot)-1):
            s += '%.16f ' % cur_mot[i]
        s += '%.16f\n' % cur_mot[-1]
        coarse_mots_file.write(s)

    coarse_mots_file.close()


def usage():
    print("""
    test should be called with the following command-line arguments:
        1. path to a .pth file containing weights for the FCGF network
        2. the name of the dataset to test on (see dataloaders/dataloader.py for options)
        3. (optional) which subset to run the test on: 'test' (default), 'train', 'validation'
        4. (optional) number of repeats (to allow calculation of standard deviation, default is 1)
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
        config.tmp_file_base = tempfile.gettempdir() + '/test_%016d' % int(np.random.rand()*10**16)    
        config.world_size = 1
        config.rank = 0 
        config.do_analysis = True

    if len(sys.argv) < 3:
        usage()
        exit(1)

    config.weights = sys.argv[1]
    assert os.path.isfile(config.weights), "Weights file not found: " + config.weights
    assert os.path.splitext(config.weights)[-1] == '.pth', "Weights file should have extension .pth"     
    config.args['DATASET'] = sys.argv[2]

    if len(sys.argv) > 3:
        config.phase = sys.argv[3]
    else:
        config.phase = 'test'

    config.num_repeats = 1 # to calculate standard deviation, repeat registration this many times
    if len(sys.argv) > 4:
        config.num_repeats = int(sys.argv[4])

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
        test_subset(config.rank, config.world_size, seed, config)

    if config.do_analysis:
	    analyze_stats(config)            
    
	    tmp_files = glob(config.tmp_file_base + '*')
	    for f in tmp_files:
	        os.remove(f)

def test_subset(rank, world_size, seed, config):
      # This function is performed in parallel in several processes, one for each available GPU
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda:%d' % torch.cuda.current_device()
    print("process %d, GPU: %s" % (rank, device))

    if rank == 0:
        config.outfile_name = config.outfile
        config.outfile = open(config.outfile, 'a+')

    test_loader = make_dataloader(config.phase, config.args, world_size, rank, seed, config.outfile)
    tester = FCGF_RANSAC_tester(config, rank) 
    results, coarse_mots = tester.test(test_loader)
    np.save(f"{config.tmp_file_base}_{world_size}_{rank}.stats.npy", results)
    np.save(f"{config.tmp_file_base}_{world_size}_{rank}.coarse_mots.npy", coarse_mots)

    if rank == 0:
        config.outfile.close()

if __name__ == '__main__':
    main()
