import json
import sys
import argparse
import logging
import torch
import numpy as np
import importlib
import open3d as o3d
import tempfile
from glob import glob
import datetime
import errno
from tqdm import tqdm
from easydict import EasyDict as edict
from libs.loss import TransformationLoss, ClassificationLoss
from datasets.KITTI import KITTIDataset
from utils.pointcloud import make_point_cloud
from evaluation.benchmark_utils import set_seed, icp_refine
from utils.timer import Timer
from shutil import move

from dataloader.data_loaders import make_data_loader, get_dataset_name
from datasets.LidarFeatureExtractor import LidarFeatureExtractor
from dataloader.base_loader import CollationFunctionFactory
from torch.utils.data import DataLoader
from algorithms.FR import FR
try:
    from algorithms.TEASER_plus_plus import TEASER
except Exception as E:
    print("Ignoring exception: " +str(E))

import torch.multiprocessing as mp
import torch.distributed as dist
import os
import time

from dataloader.paths import kitti_dir, fcgf_weights_file

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d %H:%M:%S',
                    handlers=[ch])

logging.basicConfig(level=logging.INFO, format="")

def analyze_stats(args):    
    
    file_base = args.tmp_file_base
    file_names = glob(file_base + '*')
    res_files = []
    for f in file_names:
        if '_res_' in f:
            res_files.append(f)
        elif 'success_or_failure' in f:
            move(f, args.outdir + 'TEASER_success_or_failure.txt')
    arrs_list = []
    for filename in res_files:
        stats = np.load(filename)        
        arrs_list.append(stats)
    all_stats = np.vstack(arrs_list)

    np.save(args.outdir + 'raw_stats.npy', all_stats)

    allpair_stats = all_stats
    allpair_average = allpair_stats.mean(0)
    correct_pair_average = allpair_stats[allpair_stats[:, 0] == 1].mean(0)
    model_time_99 = np.quantile(allpair_stats[:,9], 0.99)

    logging.info(f"*"*40)
    logging.info(f"All {allpair_stats.shape[0]} pairs, Mean Success Rate={allpair_average[0]*100:.2f}%, Mean Re={correct_pair_average[1]:.2f}, Mean Te={correct_pair_average[2]:.2f}")
    logging.info(f"\tInput:  Mean Inlier Num={allpair_average[3]:.2f}(ratio={allpair_average[4]*100:.2f}%)")
    logging.info(f"\tOutput: Mean Inlier Num={allpair_average[5]:.2f}(precision={allpair_average[6]*100:.2f}%, recall={allpair_average[8]*100:.2f}%, f1={allpair_average[8]*100:.2f}%)")
    logging.info(f"\tMean model time: {allpair_average[9]:.2f}s, 99% model time: {model_time_99:.2f}, Mean icp time: {allpair_average[11]:.2f}s, Mean data time: {allpair_average[10]:.2f}s")

    num_total = allpair_stats.shape[0]
    num_failed_algo = (allpair_stats[:,0] == 0).sum()
    num_failed_icp = (allpair_stats[:,12] == 0).sum()
    
    s = "\n"
    s += f"{allpair_average[15]:.0f} nn pairs ({allpair_average[16]:.3f} inliers), {allpair_average[17]:.0f} filtered pairs ({allpair_average[18]:.3f} inliers)\n"
    s += f"{args.algo}     | recall: {100*allpair_average[0]:.2f}%, #failed/#total: {num_failed_algo}/{num_total}, TE(cm): { correct_pair_average[2]:.3f}, RE(deg): { correct_pair_average[1]:.3f}, mean reg time(s): {allpair_average[9]:.3f}, 99% reg time(s): {model_time_99:.3f}\n"
    s += f"{args.algo}+ICP | recall: {100*allpair_average[12]:.2f}%, #failed/#total: {num_failed_icp}/{num_total}, TE(cm): {correct_pair_average[14]:.3f}, RE(deg): {correct_pair_average[13]:.3f}, ICP time(s): {allpair_average[11]:.3f}, Total time(s) {allpair_average[9]+allpair_average[11]:.3f}\n"
    logging.info(s)

    with open(args.outdir + 'log.txt','w') as fid:
        for k in args.__dict__.keys():
            fid.write(f"{k} = {args.__dict__[k]}\n")        
        fid.write("\n" + s)

def eval_KITTI_per_pair(model, dloader, feature_extractor, config, args, rank):
    """
    Evaluate our model on KITTI testset.
    """
    num_pair = dloader.__len__()
    if args.max_samples is not None:
        num_pair = min(num_pair, args.max_samples)
    # 0.success, 1.RE, 2.TE, 3.input inlier number, 4.input inlier ratio,  5. output inlier number 
    # 6. output inlier precision, 7. output inlier recall, 8. output inlier F1 score 9. model_time, 10. data_time 11. icp_time
    # 12. recall_icp 13. RE_icp 14. TE_icp 15.num_pairs_init 16.inlier_ratio_init 17.num_pairs_filtered 18.inlier_ratio_filtered 19. drive 20.t0 21.t1
    stats = np.zeros([num_pair, 22])
    dloader_iter = dloader.__iter__()
    class_loss = ClassificationLoss()
    evaluate_metric = TransformationLoss(re_thre=config.re_thre, te_thre=config.te_thre)
    data_timer, model_timer = Timer(), Timer()
    icp_timer = Timer()
    with torch.no_grad():
        for i in range(num_pair):
            #################################
            # load data 
            #################################
            data_timer.tic()
            input_dict = dloader_iter.next()
            corr, src_keypts, tgt_keypts, gt_trans, gt_labels, src_features, tgt_features = feature_extractor.process_batch(input_dict)
            corr, src_keypts, tgt_keypts, gt_trans, gt_labels = \
                    corr.cuda(), src_keypts.cuda(), tgt_keypts.cuda(), gt_trans.cuda(), gt_labels.cuda()
            data = {
                'corr_pos': corr,
                'src_keypts': src_keypts,
                'tgt_keypts': tgt_keypts,
                'testing': True,
            }
            data_time = data_timer.toc()

            #################################
            # forward pass 
            #################################
            num_pairs_init, inlier_ratio_init, num_pairs_filtered, inlier_ratio_filtered = 0,0,0,0
            if args.algo == 'PointDSC':                
                model_timer.tic()
                res = model(data)
                pred_trans, pred_labels = res['final_trans'], res['final_labels']

                if args.solver == 'SVD':
                    pass
                
                elif args.solver == 'RANSAC':
                    # our method can be used with RANSAC as a outlier pre-filtering step.
                    src_pcd = make_point_cloud(src_keypts[0].detach().cpu().numpy())
                    tgt_pcd = make_point_cloud(tgt_keypts[0].detach().cpu().numpy())
                    corr = np.array([np.arange(src_keypts.shape[1]), np.arange(src_keypts.shape[1])])
                    pred_inliers = np.where(pred_labels.detach().cpu().numpy() > 0)[1]
                    corr = o3d.utility.Vector2iVector(corr[:, pred_inliers].T)
                    reg_result = o3d.registration.registration_ransac_based_on_correspondence(
                        src_pcd, tgt_pcd, corr,
                        max_correspondence_distance=config.inlier_threshold,
                        estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
                        ransac_n=3,
                        criteria=o3d.registration.RANSACConvergenceCriteria(max_iteration=5000, max_validation=5000)
                    )
                    inliers = np.array(reg_result.correspondence_set)
                    pred_labels = torch.zeros_like(gt_labels)
                    pred_labels[0, inliers[:, 0]] = 1
                    pred_trans = torch.eye(4)[None].to(src_keypts.device)
                    pred_trans[:, :4, :4] = torch.from_numpy(reg_result.transformation)
                
                model_time = model_timer.toc()

                src_pcd = make_point_cloud(src_keypts.detach().cpu().numpy()[0])
                tgt_pcd = make_point_cloud(tgt_keypts.detach().cpu().numpy()[0])
                initial_trans = pred_trans[0].detach().cpu().numpy()

            elif args.algo in ['RANSAC', 'GC']:
                
                initial_trans, model_time, src_pcd, tgt_pcd, num_pairs_init, \
                     inlier_ratio_init, num_pairs_filtered, inlier_ratio_filtered = \
                         FR(input_dict['pcd0'][0], input_dict['pcd1'][0], src_features, tgt_features, args, gt_trans[0,...].detach().cpu().numpy())
                pred_trans = torch.eye(4)[None].to(src_keypts.device)
                pred_trans[:, :4, :4] = torch.from_numpy(initial_trans)
                pred_labels = torch.zeros_like(gt_labels) + np.nan

            elif args.algo == 'TEASER':                
                src_pcd = make_point_cloud(input_dict['pcd0'][0].detach().cpu().numpy())
                tgt_pcd = make_point_cloud(input_dict['pcd1'][0].detach().cpu().numpy())
                initial_trans, model_time = TEASER(src_pcd, tgt_pcd, src_features, tgt_features, input_dict['pcd0'][0], args)
                pred_trans = torch.eye(4)[None].to(src_keypts.device)
                pred_trans[:, :4, :4] = torch.from_numpy(initial_trans)
                pred_labels = torch.zeros_like(gt_labels) + np.nan                
            
            else:
                assert False, "unkown value for args.algo: " + args.algo

            icp_timer.tic()
            # change the convension of transformation because open3d use left multi.
            refined_T = o3d.pipelines.registration.registration_icp(
                src_pcd, tgt_pcd, 0.6, initial_trans,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()).transformation
            icp_time = icp_timer.toc()
            pred_trans_icp = torch.from_numpy(refined_T[None, :, :]).to(pred_trans.device).float()            
            
            class_stats = class_loss(pred_labels, gt_labels)
            loss, recall, Re, Te, rmse = evaluate_metric(pred_trans, gt_trans, src_keypts, tgt_keypts, pred_labels)
            loss, recall_icp, Re_icp, Te_icp, rmse = evaluate_metric(pred_trans_icp, gt_trans, src_keypts, tgt_keypts, pred_labels)
            pred_trans = pred_trans[0]

            # save statistics
            stats[i, 0] = float(recall / 100.0)                      # success
            stats[i, 1] = float(Re)                                  # Re (deg)
            stats[i, 2] = float(Te)                                  # Te (cm)
            stats[i, 3] = int(torch.sum(gt_labels))                  # input inlier number 
            stats[i, 4] = float(torch.mean(gt_labels.float()))       # input inlier ratio
            stats[i, 5] = int(torch.sum(gt_labels[pred_labels > 0])) # output inlier number 
            stats[i, 6] = float(class_stats['precision'])            # output inlier precision 
            stats[i, 7] = float(class_stats['recall'])               # output inlier recall
            stats[i, 8] = float(class_stats['f1'])                   # output inlier f1 score
            stats[i, 9] = model_time
            stats[i, 10] = data_time
            stats[i, 11] = icp_time
            stats[i, 12] = float(recall_icp / 100.0)                      # success
            stats[i, 13] = float(Re_icp)                                  # Re (deg)
            stats[i, 14] = float(Te_icp)                                  # Te (cm)
            stats[i, 15] = num_pairs_init
            stats[i, 16] = inlier_ratio_init 
            stats[i, 17] = num_pairs_filtered
            stats[i, 18] = inlier_ratio_filtered 
            stats[i, 19] = input_dict['extra_packages'][0]['drive']
            stats[i, 20] = input_dict['extra_packages'][0]['t0']
            stats[i, 21] = input_dict['extra_packages'][0]['t1']

            if rank==0:
                print(f"{time.strftime('%m/%d %H:%M:%S')} Finished pair:{i}/{num_pair}", flush=True)
                if recall == 0:
                    from evaluation.benchmark_utils import rot_to_euler
                    R_gt, t_gt = gt_trans[0][:3, :3], gt_trans[0][:3, -1]
                    euler = rot_to_euler(R_gt.detach().cpu().numpy())

                    input_ir = float(torch.mean(gt_labels.float()))
                    input_i = int(torch.sum(gt_labels))
                    output_i = int(torch.sum(gt_labels[pred_labels > 0]))

                    logging.info(f"Pair {i}, GT Rot: {euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}, Trans: {t_gt[0]:.2f}, {t_gt[1]:.2f}, {t_gt[2]:.2f}, RE: {float(Re):.2f}, TE: {float(Te):.2f}")
                    logging.info((f"\tInput Inlier Ratio :{input_ir*100:.2f}%(#={input_i}), Output: IP={float(class_stats['precision'])*100:.2f}%(#={output_i}) IR={float(class_stats['recall'])*100:.2f}%"))

    return stats

def eval_KITTI(model, config, world_size, seed, rank, args):

    DL_config=edict({'voxel_size': 0.3, 
    'positive_pair_search_voxel_size_multiplier': 4, 
    'use_random_rotation': False, 'use_random_scale': False})
    dloader = make_data_loader(args.dataset, DL_config, args.phase, 1, rank, world_size, seed, 0)

    feature_extractor = LidarFeatureExtractor(
            split='test',
            in_dim=config.in_dim,
            inlier_threshold=config.inlier_threshold,
            num_node=12000, 
            use_mutual=config.use_mutual,
            augment_axis=0,
            augment_rotation=0.0,
            augment_translation=0.0,   
            fcgf_weights_file=args.fcgf_weights_file             
            )                                        
    
    stats = eval_KITTI_per_pair(model, dloader, feature_extractor, config, args, rank)

    np.save(f"{args.tmp_file_base}_res_{world_size}_{rank}.npy", stats)

def generate_output_dir(dataset_name, phase, start_time=None):
    if start_time is not None:
        current_time_str = start_time
    else:
        current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
    output_dir =  f"outputs/{dataset_name}.{phase}.{current_time_str}/"
    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e        
    
    return output_dir

def get_args_and_config():

    if sys.argv[1] == 'test_parallel':
        start_time = sys.argv[2]
        tmp_file_base = sys.argv[3]
        world_size = int(sys.argv[4])
        if sys.argv[5] == 'analysis':
            rank = None
            do_analysis = True
        else:
            rank = int(sys.argv[5])
            do_analysis = False
        sys.argv = sys.argv[5:]
    else:
        start_time=None
        tmp_file_base = tempfile.gettempdir() + '/test_%016d' % int(np.random.rand()*10**16)    
        world_size = 1
        rank = 0 
        do_analysis = True

    from config import str2bool
    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='', type=str, help='snapshot dir')
    parser.add_argument('--solver', default='SVD', type=str, choices=['SVD', 'RANSAC'])
    parser.add_argument('--save_npz', default=False, type=str2bool)
    parser.add_argument('--fcgf_weights_file', type=str, default=None, help='file containing FCGF network weights')
    parser.add_argument('--dataset', type=str, default=None, help='name of dataset for testing')
    parser.add_argument('--algo', type=str, default='PointDSC', help='algorithm to use for testing', choices=['PointDSC', 'RANSAC', 'TEASER'])
    parser.add_argument('--codebase', type=str, default='GC', help='codebase for RANSAC', choices=['open3D', 'GC'])
    parser.add_argument('--mode', type=str, default=None, help='algorithm mode')
    parser.add_argument('--max_samples', type=int, default=None, help='maximum nuimber of samples to use in test')    
    parser.add_argument('--iters', type=int, default=None, help='RANSAC iters')
    parser.add_argument('--phase', type=str, default='test', help='which part of the dataset to use: train, test, or validation', choices=['train', 'validation', 'test'])
    parser.add_argument('--spatial_coherence_weight', type=float, default=0.0, help='spatial_coherence_weight for GC_RANSAC')
    parser.add_argument('--fast_rejection', type=str, default='ELC', help='type of fast rejection to perform with GC_RANSAC', choices=['SPRT', 'ELC', 'NONE'])
    parser.add_argument('--prosac', type=str2bool, default=True, help='use prosac for GC_RANSAC')
    parser.add_argument('--GPF_factor', type=float, default=2.0, help='factor for GPF (phi)')
    parser.add_argument('--GPF_grid_wid', type=int, default=10, help='grid_wid for GPF')
    parser.add_argument('--GPF_max_matches', type=int, default=10**9, help='maximum matches for GPF (only used for TEASER++)')    
    parser.add_argument('--GC_conf', type=float, default=0.999, help='confidence for GC_RANSAC')    
    parser.add_argument('--GC_LO', type=str2bool, default=True, help='perform local-optimization in GC_RANSAC')
    
    args = parser.parse_args()

    args.start_time    = start_time      
    args.tmp_file_base = tmp_file_base 
    args.world_size    = world_size 
    args.rank          = rank 
    args.do_analysis   = do_analysis  
    _, dataset_name = get_dataset_name(args.dataset)
    args.outdir = generate_output_dir(dataset_name, 'Test', args.start_time)
    
    if args.algo != 'PointDSC':
        config = edict({
            'in_dim': 6, 
            'inlier_threshold': 0.6, 
            'use_mutual': False, 
            're_thre': 5 , 
            'te_thre': 60 })
    else:
        config_path = f'snapshot/{args.chosen_snapshot}/config.json'
        config = json.load(open(config_path, 'r'))
        config = edict(config)

        ## in case test the generalization ability of model trained on 3DMatch
        config.inlier_threshold = 0.6
        config.sigma_d = 1.2
        config.re_thre = 5
        config.te_thre = 60
        config.descriptor = 'fcgf'

    if args.rank == 0:
        print("args:\n====")
        for k in args.__dict__.keys():
            print(f"\t{k} = {args.__dict__[k]}")

        print("config:\n======")
        for k in config.__dict__.keys():
            print(f"\t{k} = {config.__dict__[k]}")        

    return args, config

def main():
    args, config = get_args_and_config()
    seed = 51

    logging.info("Starting")
    if args.rank is not None:
        test_subset(args.rank, args.world_size, seed, config, args)

    if args.do_analysis:
	    analyze_stats(args)            
    
	    tmp_files = glob(args.tmp_file_base + '_res_*')
	    for f in tmp_files:
	        os.remove(f)

def test_subset(rank, world_size, seed, config, args):
    # This function is performed in parallel in several processes, one for each available GPU

    set_seed(seed)
    device = 'cuda:%d' % torch.cuda.current_device()
    print("process %d, GPU: %s" % (rank, device))

    if args.algo == "PointDSC":
        # load from models/PointDSC.py
        from models.PointDSC import PointDSC
        model = PointDSC(
                in_dim=config.in_dim,
                num_layers=config.num_layers,
                num_channels=config.num_channels,
                num_iterations=config.num_iterations,
                ratio=config.ratio,
                inlier_threshold=config.inlier_threshold,
                sigma_d=config.sigma_d,
                k=config.k,
                nms_radius=config.inlier_threshold,
                )
        device = 'cuda:%d' % torch.cuda.current_device()
        checkpoint = torch.load(f'snapshot/{args.chosen_snapshot}/models/model_best.pkl', map_location=device)
        miss = model.load_state_dict(checkpoint)
        if rank==0:
            print(miss)
        model.eval()
        model = model.cuda()
    else:
        model = None

    # evaluate on the test set
    eval_KITTI(model, config, world_size, seed, rank, args)

if __name__ == '__main__':
    main()
