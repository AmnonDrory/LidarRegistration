import json
import sys
import argparse
import logging
import torch
import numpy as np
import importlib
import open3d as o3d
from tqdm import tqdm
from easydict import EasyDict as edict
from libs.loss import TransformationLoss, ClassificationLoss
from datasets.KITTI import KITTIDataset
from datasets.dataloader import get_dataloader
from utils.pointcloud import make_point_cloud
from evaluation.benchmark_utils import set_seed, icp_refine
from utils.timer import Timer

from dataloader.kitti_loader import KITTINMPairDataset
from dataloader.KITTI_balanced_loader import KITTIBalancedPairDataset
from dataloader.ApolloSouthbay_balanced_loader import ApolloSouthbayBalancedPairDataset
from datasets.LidarFeatureExtractor import LidarFeatureExtractor
from dataloader.base_loader import CollationFunctionFactory
from torch.utils.data import DataLoader

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

def eval_KITTI_per_pair(model, dloader, feature_extractor, config, use_icp, args, rank):
    """
    Evaluate our model on KITTI testset.
    """
    num_pair = dloader.__len__()
    # 0.success, 1.RE, 2.TE, 3.input inlier number, 4.input inlier ratio,  5. output inlier number 
    # 6. output inlier precision, 7. output inlier recall, 8. output inlier F1 score 9. model_time, 10. data_time 11. icp_time
    stats = np.zeros([num_pair, 12])
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
            corr, src_keypts, tgt_keypts, gt_trans, gt_labels, _, _ = feature_extractor.process_batch(input_dict)
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
            icp_timer.tic()
            if use_icp:
                pred_trans = icp_refine(src_keypts, tgt_keypts, pred_trans)
            
            icp_time = icp_timer.toc()
            
            class_stats = class_loss(pred_labels, gt_labels)
            loss, recall, Re, Te, rmse = evaluate_metric(pred_trans, gt_trans, src_keypts, tgt_keypts, pred_labels)
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

def eval_KITTI(model, config, use_icp, world_size, seed, rank, args):

    dloader = make_test_loader(rank, world_size, seed)

    feature_extractor = LidarFeatureExtractor(
            split='test',
            in_dim=config.in_dim,
            inlier_threshold=config.inlier_threshold,
            num_node=12000, 
            use_mutual=config.use_mutual,
            augment_axis=0,
            augment_rotation=0.0,
            augment_translation=0.0,   
            fcgf_weights_file=config.fcgf_weights_file             
            )                                        
    
    stats = eval_KITTI_per_pair(model, dloader, feature_extractor, config, use_icp, args, rank)

    if rank == 0:
        logging.info(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB")

    # pair level average 
    allpair_stats = stats
    allpair_average = allpair_stats.mean(0)
    correct_pair_average = allpair_stats[allpair_stats[:, 0] == 1].mean(0)

    report = torch.tensor([1.0, allpair_stats.shape[0], allpair_average[0], correct_pair_average[1], correct_pair_average[2], allpair_average[3], allpair_average[4], allpair_average[5], allpair_average[6], allpair_average[7], allpair_average[8], allpair_average[9], allpair_average[10], allpair_average[11] ], device=torch.cuda.current_device())
    dist.all_reduce(report, op=dist.ReduceOp.SUM)

    count = report[0].item()
    allpair_stats_shape_0  = report[1].item()
    allpair_average_0      = report[2].item() / count    
    correct_pair_average_1 = report[3].item() / count    
    correct_pair_average_2 = report[4].item() / count    
    allpair_average_3      = report[5].item() / count    
    allpair_average_4      = report[6].item() / count    
    allpair_average_5      = report[7].item() / count    
    allpair_average_6      = report[8].item() / count    
    allpair_average_7      = report[9].item() / count    
    allpair_average_8      = report[10].item() / count    
    allpair_average_9      = report[11].item() / count    
    allpair_average_10     = report[12].item() / count    
    allpair_average_11     = report[13].item() / count    

    if rank == 0:
        logging.info(f"*"*40)
        logging.info(f"All {allpair_stats_shape_0} pairs, Mean Success Rate={allpair_average_0*100:.2f}%, Mean Re={correct_pair_average_1:.2f}, Mean Te={correct_pair_average_2:.2f}")
        logging.info(f"\tInput:  Mean Inlier Num={allpair_average_3:.2f}(ratio={allpair_average_4*100:.2f}%)")
        logging.info(f"\tOutput: Mean Inlier Num={allpair_average_5:.2f}(precision={allpair_average_6*100:.2f}%, recall={allpair_average_7*100:.2f}%, f1={allpair_average_8*100:.2f}%)")
        logging.info(f"\tMean model time: {allpair_average_9:.2f}s, Mean icp time: {allpair_average_11:.2f}s, Mean data time: {allpair_average_10:.2f}s")

    return allpair_stats

def main():
    from config import str2bool
    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='', type=str, help='snapshot dir')
    parser.add_argument('--solver', default='SVD', type=str, choices=['SVD', 'RANSAC'])
    parser.add_argument('--use_icp', default=False, type=str2bool)
    parser.add_argument('--save_npz', default=False, type=str2bool)
    parser.add_argument('--fcgf_weights_file', type=str, default=None, help='file containing FCGF network weights')
    args = parser.parse_args()

    
    config_path = f'snapshot/{args.chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    ## in case test the generalization ability of model trained on 3DMatch
    config.inlier_threshold = 0.6
    config.sigma_d = 1.2
    config.re_thre = 5
    config.te_thre = 60
    config.descriptor = 'fcgf'
    config.fcgf_weights_file = args.fcgf_weights_file

    seed = 51
    world_size = torch.cuda.device_count()  
    print("%d GPUs are available" % world_size)

    logging.info("Starting")
  
    if world_size == 1:
        train_parallel(0, world_size, seed, config, args)
    else:
        mp.spawn(train_parallel, nprocs=world_size, args=(world_size,seed, config, args))      

def make_test_loader(rank, world_size, seed):

    Dataset = ApolloSouthbayBalancedPairDataset # KITTIBalancedPairDataset # KITTINMPairDataset
    num_workers = 1

    config=edict(
            {'kitti_dir': os.path.split(kitti_dir)[0], 
            'icp_cache_path': 'icp',
            'voxel_size': 0.3, 
            'positive_pair_search_voxel_size_multiplier': 1.5,
            })
    
    dset = Dataset('test',
                transform=None, random_rotation=False, random_scale=False,
                manual_seed=False, config=config, rank=rank)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=seed)

    collation_fn = CollationFunctionFactory(concat_correspondences=False,
                                            collation_type='collate_pair')

    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collation_fn,
        sampler=sampler,
        pin_memory=True,
        drop_last=False)

    return loader

def train_parallel(rank, world_size, seed, config, args):
    # This function is performed in parallel in several processes, one for each available GPU
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8880'
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    set_seed(seed)
    torch.cuda.set_device(rank)
    device = 'cuda:%d' % torch.cuda.current_device()
    print("process %d, GPU: %s" % (rank, device))

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

    # evaluate on the test set
    stats = eval_KITTI(model.cuda(), config, args.use_icp, world_size, seed, rank, args)


if __name__ == '__main__':
    main()
