# Parts of this implementation were copied from the Deep Global Registration project, that carries the following copyright notice:
    # Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)

__doc__ = """
Fast RANSAC algorithms 
"""

import numpy as np
import torch
import open3d as o3d
from copy import deepcopy
from time import time
from algorithms.matching import find_2nn, nn_to_mutual, measure_inlier_ratio, Grid_Prioritized_Filter, calc_distance_ratio_in_feature_space
from algorithms.GC_RANSAC import GC_RANSAC

def FR(A,B, A_feat, B_feat, args, T_gt):    

    voxel_size = 0.3
    
    def make_open3d_point_cloud(xyz):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        return pcd
    
    xyz0 = A
    xyz1 = B
    xyz0_np = xyz0.detach().cpu().numpy().astype(np.float64)
    xyz1_np = xyz1.detach().cpu().numpy().astype(np.float64)
    pcd0 = make_open3d_point_cloud(xyz0_np) 
    pcd1 = make_open3d_point_cloud(xyz1_np)

    device = 'cuda:%d' % torch.cuda.current_device()
    fcgf_feats0 = A_feat.to(device)
    fcgf_feats1 = B_feat.to(device)

    with torch.no_grad():

        # 1. Coarse correspondences

        corres_idx0, corres_idx1, idx1_2nd, additional_time_for_finding_2nd_closest = find_2nn(fcgf_feats0, fcgf_feats1)

        num_pairs_init = len(corres_idx0)
        inlier_ratio_init = measure_inlier_ratio(corres_idx0, corres_idx1, pcd0, pcd1, T_gt, voxel_size)

        start_time = time()

        # 2. Filter correspondences:
        if args.mode == "MNN":
            corres_idx0_orig, corres_idx1_orig, idx1_2nd_orig = corres_idx0, corres_idx1, idx1_2nd
            corres_idx0, corres_idx1, idx1_2nd = nn_to_mutual(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, idx1_2nd, force_return_2nd=True)
        elif args.mode == "GPF":
            corres_idx0, corres_idx1, idx1_2nd, corres_idx0_orig, corres_idx1_orig, idx1_2nd_orig, norm_feat_dist = Grid_Prioritized_Filter(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, idx1_2nd, xyz0, args)
        elif args.mode == "no_filter":
            corres_idx0_orig, corres_idx1_orig, idx1_2nd_orig = corres_idx0, corres_idx1, idx1_2nd
        else:
            assert False, "unknown mode"

        filter_time = time() - start_time

        num_pairs_filtered = len(corres_idx0)
        inlier_ratio_filtered = measure_inlier_ratio(corres_idx0, corres_idx1, pcd0, pcd1, T_gt, voxel_size)

    start_time = time()

    ransac_iters = 500*10**3
    if args.iters is not None:
        ransac_iters = args.iters

    # 3. Perform RANSAC
    if args.codebase == "GC":        
        
        A = xyz0_np[corres_idx0,:].astype(np.float32)
        B = xyz1_np[corres_idx1,:].astype(np.float32)
        if args.prosac:
            if args.mode=='GPF': 
                feat_dist = norm_feat_dist.detach().cpu().numpy()
            else:
                feat_dist = calc_distance_ratio_in_feature_space(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, idx1_2nd).detach().cpu().numpy()

            match_quality = -feat_dist
        else:
            match_quality = None

        T, GC_time = GC_RANSAC( A,B, 
                                distance_threshold=2*voxel_size,
                                num_iterations=ransac_iters,
                                args=args, match_quality=match_quality)

        
    elif args.codebase == "open3D":
        T = RANSAC_registration(pcd0,
                                pcd1,
                                corres_idx0,
                                corres_idx1,
                                2 * voxel_size,
                                num_iterations=ransac_iters,
                                args=args)
		
		# estimate motion using all inlier pairs:                            
        corres_idx0_ = corres_idx0_orig.detach().cpu().numpy()
        corres_idx1_ = corres_idx1_orig.detach().cpu().numpy()
        pcd0_trans = deepcopy(pcd0)
        pcd0_trans.transform(T)
        dist2 = np.sum((np.array(pcd0_trans.points)[corres_idx0_,:] - np.array(pcd1.points)[corres_idx1_,:])**2, axis=1)
        is_close = dist2 < (2*voxel_size)**2
        inlier_corres_idx0 = corres_idx0_[is_close]
        inlier_corres_idx1 = corres_idx1_[is_close]
        corres = np.stack((inlier_corres_idx0, inlier_corres_idx1), axis=1)
        corres_ = o3d.utility.Vector2iVector(corres)
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        T = p2p.compute_transformation(pcd0, pcd1, corres_)
                                
    else:
        assert False, "unknown codebase"

    algo_time = time() - start_time
    elapsed_time = filter_time + algo_time + additional_time_for_finding_2nd_closest

    return T, elapsed_time, pcd0, pcd1, num_pairs_init, inlier_ratio_init, num_pairs_filtered, inlier_ratio_filtered


def RANSAC_registration(pcd0, pcd1, idx0, idx1,
                        distance_threshold, num_iterations, args):        

    corres = np.stack((idx0, idx1), axis=1)
    corres = o3d.utility.Vector2iVector(corres)
    
    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        pcd0, 
        pcd1,
        corres, 
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        max_correspondence_distance=distance_threshold, 
        ransac_n=4,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength()],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=num_iterations, confidence=0.9995)
    )

    return result.transformation

