# Parts of this implementation were copied from the Deep Global Registration project, that carries the following copyright notice:
    # Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)

from matplotlib.pyplot import draw
import numpy as np
import torch
import open3d as o3d
from copy import deepcopy
import math

from model.resunet import FCGF_net
from utils.experiment_utils import draw_loss_along_x, print_to_file_and_screen
from general.TicToc import *
from dataloader.generic_balanced_loader import VOXEL_SIZE
from net.BBR_F import BBR_F
from net.symmetric_icp import symmetric_icp
from utils.tools_3d import apply_transformation
from utils.visualization import draw_multiple_clouds

class refinement_tester():
    def __init__(self, config, rank):
        self.config = config
        self.rank = rank
        self.device = 'cuda:%d' % torch.cuda.current_device()
        self.voxel_size = 0.3

        if self.rank == 0:
            print_to_file_and_screen(self.config.outfile, f"o3d.__version__ = {o3d.__version__}")

    def run_refinement_test(self, refinement_test_loader):
        results_list = []    
        start_time = time()        
        for batch_ind, (GT_motion, coarse_motion, arrs) in enumerate(refinement_test_loader):
            elapsed = time() - start_time
            if self.rank == 0:
                print_to_file_and_screen(self.config.outfile, f"{elapsed: 7.2f}: batch {batch_ind} of {len(refinement_test_loader)}")
                self.config.outfile.flush()            
            batch_results = self.refinement_batch(GT_motion, coarse_motion, arrs)
            results_list.append(batch_results)
        results = np.vstack(results_list)

        return results

    def refinement_batch(self, GT_motion, coarse_motion, arrs):
        batch_size = GT_motion.shape[0]
        res_list = []
        to_ = [0, 0]
        from_ = [0, 0]
        for sample_ind in range(batch_size):
            # extract data for sample:
            sample_arrs = [{k: None for k in arrs[j].keys()} for j in [0,1]]
            for i in [0,1]:                
                from_[i] = to_[i]
                to_[i] = from_[i] + arrs[i]['len'][sample_ind]
                for k in sample_arrs[i].keys():
                    if arrs[i][k].ndim == 2: # Minkowski Engine style batching
                        sample_arrs[i][k] = arrs[i][k][from_[i]:to_[i], :]
                    else:
                        sample_arrs[i][k] = arrs[i][k][sample_ind]

            gt_motion = GT_motion[sample_ind,...]
            init_motion = coarse_motion[sample_ind,...]
            sample_res = self.refinement_sample(gt_motion, init_motion, sample_arrs)
            res_list.append(sample_res)

        batch_result = np.vstack(res_list)
        return batch_result

    def downsample(self, X, voxel_size):
        x = self.make_open3d_point_cloud(X)
        x_ = o3d.geometry.PointCloud.voxel_down_sample(x, voxel_size=voxel_size)
        X_ = np.array(x_.points)    
        return X_        

    def refinement_sample(self, gt_motion, init_motion, sample_arrs):
        A = self.downsample(sample_arrs[0]['PC'], self.voxel_size)
        B = self.downsample(sample_arrs[1]['PC'], self.voxel_size)

        A = apply_transformation(init_motion, A)
        gt_motion = gt_motion @ np.linalg.inv(init_motion)

        ICP_M, ICP_time = self.ICP(A, B)
        ICP_recall, ICP_te, ICP_re = self.calc_errors(ICP_M, gt_motion)
        BBR_M, BBR_time = BBR_F(A, B)
        BBR_recall, BBR_te, BBR_re = self.calc_errors(BBR_M, gt_motion)
        sym_M, sym_time = symmetric_icp(A, B)         
        sym_recall, sym_te, sym_re = self.calc_errors(sym_M, gt_motion)   
        res = [[  ICP_recall, ICP_te, ICP_re, ICP_time, 
        BBR_recall, BBR_te, BBR_re, BBR_time, 
        sym_recall, sym_te, sym_re, sym_time, 
        sample_arrs[0]['session_ind'], sample_arrs[0]['cloud_ind'], sample_arrs[1]['cloud_ind']]]

        return res

    def make_open3d_point_cloud(self, xyz):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        return pcd

    def ICP(self, A, B, M_R=None):        
        if M_R is None:
            M_R = np.eye(4)

        xyz0 = A
        xyz1 = B
        xyz0_np = xyz0.astype(np.float64)
        xyz1_np = xyz1.astype(np.float64)
        pcd0 = self.make_open3d_point_cloud(xyz0_np) 
        pcd1 = self.make_open3d_point_cloud(xyz1_np)

        tic()        
        T = o3d.pipelines.registration.registration_icp(
            pcd0,
            pcd1, self.voxel_size * 2, M_R,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()).transformation        
        time_elapsed = toc(silent=True)        
        return T, time_elapsed

    def calc_errors(self, T_pred, T_gt, eps=1e-16):
        rte_thresh = self.config.trans_err_thresh
        rre_thresh = self.config.rot_err_thresh
        
        if T_pred is None:
            return np.array([0, np.inf, np.inf])

        rte = np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3])
        rre = np.arccos(
            np.clip((np.trace(T_pred[:3, :3].T @ T_gt[:3, :3]) - 1) / 2, -1 + eps,
                    1 - eps)) * 180 / math.pi
        return np.array([rte < rte_thresh and rre < rre_thresh, rte, rre])

    def filter_pairs_by_distance_in_feature_space(self, fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1, xyz0):
        F0 = fcgf_feats0[corres_idx0,:]
        F1 = fcgf_feats1[corres_idx1,:]
        feat_dist = torch.sqrt(torch.sum((F0-F1)**2,axis=1))        

        NUM_QUADS = 10
        TOTAL_NUM = 10000 # in practice, about half is selected. 
        NUM_PER_QUAD = int(np.ceil(TOTAL_NUM/NUM_QUADS**2))
        def to_quads(X, NUM_QUADS):
            EPS = 10**-3
            m = torch.min(X)
            M = torch.max(X)
            X_ = (X - m) / (M-m+EPS)
            res = torch.floor(NUM_QUADS*X_)
            return res

        quadrant_i = to_quads(xyz0[:,0], NUM_QUADS)
        quadrant_j = to_quads(xyz0[:,1], NUM_QUADS)
        keep = np.zeros(len(feat_dist), dtype=bool)
        num_remaining_quads = NUM_QUADS**2
        num_remaining_samples = TOTAL_NUM          
        for qi in range(NUM_QUADS):
            for qj in range(NUM_QUADS):
                samples_per_quad = int(np.ceil(num_remaining_samples / num_remaining_quads))
                is_quad_mask = (quadrant_i == qi) & (quadrant_j == qj)  
                is_quad_inds = is_quad_mask.nonzero(as_tuple=True)[0]
                if len(is_quad_inds) > samples_per_quad:
                    ord = torch.argsort(feat_dist[is_quad_mask])
                    is_quad_inds = is_quad_inds[ord[:samples_per_quad]]
                keep[is_quad_inds] = True
                num_remaining_samples -= len(is_quad_inds)
                num_remaining_quads -= 1

        corres_idx0_orig = deepcopy(corres_idx0)
        corres_idx1_orig = deepcopy(corres_idx1)
        corres_idx0 = corres_idx0[keep]
        corres_idx1 = corres_idx1[keep]

        return corres_idx0, corres_idx1, corres_idx0_orig, corres_idx1_orig

    def measure_inlier_ratio(self, corres_idx0, corres_idx1, pcd0, pcd1, T_gt):
        corres_idx0_ = corres_idx0.detach().numpy()
        corres_idx1_ = corres_idx1.detach().numpy()
        pcd0_trans = deepcopy(pcd0)
        pcd0_trans.transform(T_gt)
        
        dist2 = np.sum((np.array(pcd0_trans.points)[corres_idx0_,:] - np.array(pcd1.points)[corres_idx1_,:])**2, axis=1)
        is_close = dist2 < (2*self.voxel_size)**2
        return float(is_close.sum()) / len(is_close)

    def fcgf_feature_matching(self, F0, F1):
        nn_max_n = 250
    
        def knn_dist(f0, f1):
            # Fast implementation with torch.einsum()
            with torch.no_grad():      
                # L2 distance:
                dist2 = torch.sum(f0**2, dim=1).reshape([-1,1]) + torch.sum(f1**2, dim=1).reshape([1,-1]) -2*torch.einsum('ac,bc->ab', f0, f1)
                dist = dist2.clamp_min(1e-30).sqrt_()
                # Cosine distance:
                # dist = 1-torch.einsum('ac,bc->ab', f0, f1)                  
                min_dist, ind = dist.min(dim=1, keepdim=True)      
            return ind
        
        N = len(F0)
        C = int(np.ceil(N / nn_max_n))
        stride = nn_max_n
        inds = []
        for i in range(C):
            with torch.no_grad():
                ind = knn_dist(F0[i * stride:(i + 1) * stride], F1)
                inds.append(ind)
        
        inds = torch.cat(inds)
        assert len(inds) == N

        corres_idx0 = torch.arange(len(inds)).long().squeeze()
        corres_idx1 = inds.long().squeeze().cpu()
        return corres_idx0, corres_idx1

    def RANSAC_registration(self, pcd0, pcd1, idx0, idx1,
                            distance_threshold, num_iterations):        

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
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=num_iterations, confidence=0.9999)
        )

        return result.transformation

    def torch_intersect(self, Na, Nb, i_ab,j_ab,i_ba,j_ba):    
        def make_sparse_mat(i_,j_,sz):
            inds = torch.cat([i_.reshape([1,-1]),
                                j_.reshape([1,-1])],dim=0)
            vals = torch.ones_like(inds[0,:])
            
            M = torch.sparse.FloatTensor(inds,vals,sz)
            return M

        sz = [Na,Nb]
        M_ab = make_sparse_mat(i_ab,j_ab,sz)
        M_ba = make_sparse_mat(i_ba,j_ba,sz)

        M = M_ab.add(M_ba).coalesce()
        i, j = M._indices()
        v = M._values()
        is_both = (v == 2)
        i_final = i[is_both]
        j_final = j[is_both]

        return i_final, j_final

    def nn_to_mutual(self, feats0, feats1, corres_idx0, corres_idx1):
        
        uniq_inds_1=torch.unique(corres_idx1)
        inv_corres_idx1, inv_corres_idx0 = self.fcgf_feature_matching(feats1[uniq_inds_1,:], feats0)
        inv_corres_idx1 = uniq_inds_1

        final_corres_idx0, final_corres_idx1 = self.torch_intersect(
        feats0.shape[0], feats1.shape[0],
        corres_idx0, corres_idx1,
        inv_corres_idx0, inv_corres_idx1)

        return final_corres_idx0, final_corres_idx1


