# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019

# This version includes substantial additions by Amnon Drory (amnondrory@mail.tau.ac.il), Tel-Aviv University, 2021
# Please also cite the following paper:
# - Amnon Drory, Shai Avidan, Raja Giryes, Stress Testing LiDAR Registration

import os
import sys
import logging
import argparse
import numpy as np
import open3d as o3d

from glob import glob

import torch
import pickle

from config import get_config

from core.deep_global_registration import DeepGlobalRegistration
from dataloader.NuScenes_balanced_loader import NuScenesBostonDataset
from dataloader.base_loader import CollationFunctionFactory
from dataloader.paths import fcgf_weights_file
from util.pointcloud import make_open3d_point_cloud, make_open3d_feature, pointcloud_to_spheres
from util.timer import AverageMeter, Timer

from scripts.test_3dmatch import rte_rre

DATASET = NuScenesBostonDataset

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])

TE_THRESH = 0.6  # m
RE_THRESH = 5  # deg
VISUALIZE = False 


def visualize_pair(xyz0, xyz1, T, voxel_size):
    pcd0 = pointcloud_to_spheres(xyz0,
                                 voxel_size,
                                 np.array([0, 0, 1]),
                                 sphere_size=0.6)
    pcd1 = pointcloud_to_spheres(xyz1,
                                 voxel_size,
                                 np.array([0, 1, 0]),
                                 sphere_size=0.6)
    pcd0.transform(T)
    o3d.visualization.draw_geometries([pcd0, pcd1])


def analyze_all_stats(file_base):
	# collect outputs from all parallel processes and analyze results
    file_names = glob(file_base + '*.pickle')
    phases = []
    for file in file_names:
        flds = file.replace('.pickle','').split('_')        
        phase = flds[-2]
        if phase not in phases:
            phases.append(phase)

    with open(file_names[0], 'rb') as fid:
        stats = pickle.load(fid)     
    titles = stats.keys()
    
    for phase in phases:
        for title in titles:            
            analyze_stats(file_base, phase, title)


def analyze_stats(file_base, phase, title):
    file_names = glob(file_base + f'_{phase}_*')
    arrs_list = []
    for filename in file_names:
        with open(filename, 'rb') as fid:
            stats = pickle.load(fid)        
        arrs_list.append(stats[title])
    stats = np.vstack(arrs_list)
    m_stats = stats.mean(0)
    sel_stats = stats[stats[:, 0] > 0]
    m_sel_stats = sel_stats.mean(0)
    print(f"{title} Total result mean")
    print(m_stats)    
    print(m_sel_stats)
    num_total = stats.shape[0]
    num_failed =  num_total - sel_stats.shape[0]
    print("%s | %s | recall: %f%%, #failed/#total: %d/%d, TE(cm): %f, RE(deg): %f, reg time(s): %f, icp time(s): %f" % (phase, title, 100*m_stats[0], num_failed, num_total, 100*m_sel_stats[1], m_sel_stats[2], m_stats[3], m_stats[5]))

def evaluate(config, data_loader, method, phase, rank):    
    
    if rank==0:
        logging.info("Starting")

    data_timer = Timer()

    stats = {}
    stats['base'] = []  # statistics before ICP refinement
    stats['w_icp'] = []  # after ICP refinement
    fail_file = {} # keep track of inputs for which the test failed (i.e., has errors above threshold)
    for k in stats.keys():
        fail_file[k] = open(f"logs/failed_{phase}_{rank}_{k}.txt",'w')

    if rank==0:
        print(f"len(data_loader) == {len(data_loader)}")

    iter_timer = Timer()
    iter_timer.tic()

    data_timer.tic()
    for data_dict in data_loader:        
        if data_dict is None:
            continue

        torch.cuda.empty_cache()

        drive = data_dict['extra_packages'][0]['drive']
        xyz0, xyz1 = data_dict['pcd0'][0], data_dict['pcd1'][0]
        T_gt = data_dict['T_gt'][0].numpy()
        xyz0np, xyz1np = xyz0.numpy(), xyz1.numpy()

        if phase == 'DGR':
            T_pred = method.register(xyz0np, xyz1np)
        elif phase == 'FCGF':
            T_pred = method.register_FCGF(xyz0np, xyz1np)            
        else:
            assert False

        for k in stats.keys():
            cur_stats = np.zeros((6))  # bool succ, rte, rre, reg_time, drive, icp_time
            cur_stats[:3] = rte_rre(T_pred[k], T_gt, TE_THRESH, RE_THRESH)
            cur_stats[4] = drive
            cur_stats[3] = method.reg_timer.diff
            if k == 'w_icp':                 
                try:
                    cur_stats[5] = method.icp_timer.diff
                except:
                    pass
            stats[k].append(cur_stats)
            if cur_stats[0] == 0:
                fail_file[k].write(f"{data_dict['extra_packages'][0]['drive']} {data_dict['extra_packages'][0]['t0']} {data_dict['extra_packages'][0]['t1']} Failed with RTE: {cur_stats[1]}, RRE: {cur_stats[2]}\n")
                fail_file[k].flush()

        data_timer.tic()
        iter_timer.tic()

    for k in stats.keys():
        stats[k] = np.vstack(stats[k])
        succ_rate, rte, rre, avg_time, _, _ = stats[k].mean(0)

        if rank==0:
            logging.info(
                f"<{k}> Data time: {data_timer.avg}, Feat time: {method.feat_timer.avg}," +
                f" Reg time: {method.reg_timer.avg}, RTE: {rte}," +
                f" RRE: {rre}, Success: {succ_rate * 100} %")

    out_file = config.tmp_file_base + f'_{phase}_{rank}.pickle'
    if rank==0:
        print(f'Saving the stats to {out_file}') # will later be collected and analyzed by analyze_all_stats()
    with open(out_file, 'wb') as fid:
        pickle.dump(stats, fid)        

    for k in stats.keys():
        fail_file[k].close()

def calibrate_clip_weight_thresh(config):
	# The DGR paper uses a "failsafe", wherein DGR results are discarded if a weight is above a threshold.
	# We've found that for some test sets, using a constant threshold causes almost all DGR results to be discarded.
	# Here, instead, we calibarate the treshold so that a requested _ratio_ of cases is discarded (approximately). 
	# We set this ratio to 30%, which is approximately the same as in the example KITTI setting supplied with the 
	# original DGR code. 
    N = 20
    THRESH_FRACTION=0.3
    device = 'cuda:%d' % torch.cuda.current_device()    
    method = DeepGlobalRegistration(config, device=device, rank=0)

    dataset = DATASET

    dset = dataset('val',
                    transform=None,
                    random_rotation=False,
                    random_scale=False,
                    config=config)

    num_samples = len(dset)
    inds_raw = np.linspace(0,num_samples,2+N)
    inds = np.round(inds_raw[1:-1]).astype(int)
    all_weights = []
    for i in inds:
        data_dict = dset.__getitem__(i)

        torch.cuda.empty_cache()

        xyz0, xyz1 = data_dict[0], data_dict[1]        
        xyz0np, xyz1np = xyz0.numpy(), xyz1.numpy()

        cur_weights = method.register(xyz0np, xyz1np, only_calc_weights=True)
        all_weights.append(cur_weights.cpu().detach().numpy().flatten())
    w = np.hstack(all_weights)
    s = np.sort(w)
    thresh_ind = np.round(len(s)*THRESH_FRACTION).astype(int)
    thresh = s[thresh_ind]
    return thresh

def main(cmd_line_args):
    config = get_config()
    config.do_DGR = False
    config.do_RANSAC = True
    
    config.weights = fcgf_weights_file # if testing in DGR mode, the weights file created by training DGR goes here

    config.tmp_file_base = cmd_line_args[1]
    seed = 0
    world_size = int(cmd_line_args[2])
    try:
        rank = int(cmd_line_args[3])
    except Exception as E:
        if not 'invalid literal for int' in str(E):
            raise(E)
        rank = None
    
    if rank == 0:
        print("%d GPUs are available" % world_size)

    if rank is not None: # this is one of the parallel testing processes
        if config.do_DGR:
            config.clip_weight_thresh = calibrate_clip_weight_thresh(config)
            print(f"config.clip_weight_thresh={config.clip_weight_thresh}")

        test_subset(rank, world_size, seed, config)
    else: # this the analysis process, which is run after the testing processes end
        analyze_all_stats(config.tmp_file_base)            
        tmp_files = glob(config.tmp_file_base + '*')
        for f in tmp_files:
            os.remove(f)

def test_subset(rank, world_size, seed, config):
    # This function is performed in parallel in several processes, one for each available GPU
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda:%d' % torch.cuda.current_device()
    print("process %d, GPU: %s" % (rank, device))

    dgr = DeepGlobalRegistration(config, device=device, rank=rank)

    dataset = DATASET

    dset = dataset('test',
                                transform=None,
                                random_rotation=False,
                                random_scale=False,
                                config=config)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=seed)                              

    data_loader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=CollationFunctionFactory(concat_correspondences=False,
                                            collation_type='collate_pair'),
        sampler=sampler,
        pin_memory=False,
        drop_last=False)

    if config.do_DGR: 
        dgr.use_icp = True
        evaluate(config, data_loader, dgr, 'DGR', rank)
    if config.do_RANSAC: 
        dgr.init_timers()
        dgr.use_icp = True
        evaluate(config, data_loader, dgr, 'FCGF', rank)

if __name__ == '__main__':
	# This file should not be run directly, but instead through test_parallel.sh
    cmd_line_args = sys.argv
    with open('train_DGR_kitti_argv.pickle', 'rb') as fid:
        sys.argv = pickle.load(fid)  
    main(cmd_line_args)