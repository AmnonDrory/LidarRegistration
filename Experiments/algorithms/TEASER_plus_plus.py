import open3d as o3d
import numpy as np 
from scipy.spatial import cKDTree
from copy import deepcopy
import torch
import os
import time
import sys
import teaserpp_python

from algorithms.matching import find_2nn, nn_to_mutual, Grid_Prioritized_Filter

VOXEL_SIZE = 0.3
MAX_WAIT = 10 # seconds, when mode==FAIL_TOLERANT

def do_TEASER_inner(A_corr,B_corr, args):

    SLEEP_TIME = 1

    if os.path.isfile(f'{args.tmp_file_base}_T.npy'):
        os.remove(f'{args.tmp_file_base}_T.npy')

    cmd = f"python -m algorithms.TEASER_plus_plus {args.tmp_file_base} & echo $! > {args.tmp_file_base}_latest_pid.txt"
    np.save(f'{args.tmp_file_base}_A_corr.npy', A_corr)
    np.save(f'{args.tmp_file_base}_B_corr.npy', B_corr)
    print(cmd)
    os.system(cmd)
    start_time = time.time()
    time.sleep(SLEEP_TIME)
    elapsed_time = time.time() - start_time
    while not os.path.isfile(f'{args.tmp_file_base}_T.npy') and (elapsed_time < MAX_WAIT):
        time.sleep(SLEEP_TIME)
        elapsed_time = time.time() - start_time

    if os.path.isfile(f'{args.tmp_file_base}_T.npy'):
        os.system(f"echo 1 >> {args.tmp_file_base}_success_or_failure.txt")
        T = np.load(f'{args.tmp_file_base}_T.npy')        
        inner_time = np.load(f'{args.tmp_file_base}_inner_time.npy')
    else:
        os.system(f"echo 0 >> {args.tmp_file_base}_success_or_failure.txt")
        print("Took too long, skipping.")
        with open(f"{args.tmp_file_base}_latest_pid.txt", 'r') as fid:
            pid = fid.read()
        cmd = f"kill -9 {pid}"        
        print(cmd)
        os.system(cmd)        
        T = np.eye(4)
        inner_time = MAX_WAIT

    try:    
        os.remove(f'{args.tmp_file_base}_A_corr.npy')
        os.remove(f'{args.tmp_file_base}_B_corr.npy')
        os.remove(f"{args.tmp_file_base}_latest_pid.txt")
        os.remove(f'{args.tmp_file_base}_T.npy')
        os.remove(f'{args.tmp_file_base}_inner_time.npy')
    except FileNotFoundError:
        pass

    return T, inner_time

def go(A_corr,B_corr):
    print("starting go")
    start_time = time.time()
    NOISE_BOUND = VOXEL_SIZE
    teaser_solver = get_teaser_solver(NOISE_BOUND)
    teaser_solver.solve(A_corr,B_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    T_teaser = Rt2T(R_teaser,t_teaser)
    inner_time = time.time() - start_time

    return T_teaser, inner_time

def pcd2xyz(pcd):
    return np.asarray(pcd.points).T

def get_teaser_solver(noise_bound):
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.inlier_selection_mode = \
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_tim_graph = \
        teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver

def Rt2T(R,t):
    T = np.identity(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T 

def TEASER(A_pcd, B_pcd, A_feats, B_feats, A_tensor, args):
    A_xyz = pcd2xyz(A_pcd) # np array of size 3 by N
    B_xyz = pcd2xyz(B_pcd) # np array of size 3 by M
    # establish correspondences by nearest neighbour search in feature space
    device = 'cuda:%d' % torch.cuda.current_device()
    A_feats = A_feats.to(device)
    B_feats = B_feats.to(device)
    
    corres_idx0, corres_idx1, idx1_2nd, correspondence_time = find_2nn(A_feats, B_feats)
    corres_idx0, corres_idx1, idx1_2nd, _, _, _, _ = Grid_Prioritized_Filter(A_feats, B_feats, corres_idx0, corres_idx1, idx1_2nd, A_tensor, args, BB_first=True)
    
    print(f"num pairs TEASER: {len(corres_idx0)}")
    corrs_A = corres_idx0.cpu().detach().numpy()
    corrs_B = corres_idx1.cpu().detach().numpy()
    A_corr = A_xyz[:,corrs_A] # np array of size 3 by num_corrs
    B_corr = B_xyz[:,corrs_B] # np array of size 3 by num_corrs

    # robust global registration using TEASER++
    if args.mode == "FAIL_TOLERANT": # For some datasets the teaser algorithms becomes stuck. When mode=="FAIL_TOLERANT", the teaser algorithm is run in a separate process, which is killed if it takes too long.
        T_teaser, inner_time = do_TEASER_inner(A_corr, B_corr, args)
    else:
        T_teaser, inner_time = go(A_corr,B_corr)
    
    elapsed_time = correspondence_time + inner_time

    return T_teaser, elapsed_time

if __name__ == "__main__":
    tmp_file_base = sys.argv[1]
    A_corr = np.load(f'{tmp_file_base}_A_corr.npy')
    B_corr = np.load(f'{tmp_file_base}_B_corr.npy')
    T, inner_time = go(A_corr, B_corr)
    np.save(f'{tmp_file_base}_T.npy', T)
    np.save(f'{tmp_file_base}_inner_time.npy', inner_time)
