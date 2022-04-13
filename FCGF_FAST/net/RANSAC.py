# Parts of this implementation were copied from the Deep Global Registration project, that carries the following copyright notice:
    # Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)

import numpy as np
import torch
import open3d as o3d
from copy import deepcopy
import math

from dataloader.generic_balanced_loader import send_arrs_to_device
from model.resunet import FCGF_net
from utils.experiment_utils import print_to_file_and_screen
from general.TicToc import *
from dataloader.generic_balanced_loader import VOXEL_SIZE

class FCGF_RANSAC_tester():
    def __init__(self, config, rank):
        self.config = config
        self.rank = rank
        self.device = 'cuda:%d' % torch.cuda.current_device()
        self.voxel_size = VOXEL_SIZE

        self.FeatureNet = FCGF_net(self.device)   
        checkpoint_file = self.config.weights
        if self.rank == 0:
            print_to_file_and_screen(self.config.outfile, "Loading model from checkpoint " + checkpoint_file)

        d = torch.load(checkpoint_file, map_location=self.device)
        self.FeatureNet.Model.load_state_dict(d['state_dict'], strict=True)     
        self.FeatureNet.eval()
        
        if self.rank == 0:
            print_to_file_and_screen(self.config.outfile, f"o3d.__version__ = {o3d.__version__}")

    def test(self, test_loader):
        results_list = []    
        coarse_mots = []
        start_time = time()        
        for batch_ind, (GT_motion, arrs) in enumerate(test_loader):
            elapsed = time() - start_time
            if self.rank == 0:
                print_to_file_and_screen(self.config.outfile, f"{elapsed: 7.2f}: batch {batch_ind} of {len(test_loader)}")
                self.config.outfile.flush()
            arrs = send_arrs_to_device(arrs, self.device, keep_PC_on_cpu=True)
            batch_results, batch_coarse_mots = self.test_batch(GT_motion, arrs)
            results_list.append(batch_results)
            coarse_mots.append(batch_coarse_mots)

        results = np.vstack(results_list)
        coarse_mots = np.dstack(coarse_mots)
        return results, coarse_mots

    def test_batch(self, GT_motion, arrs):
        for i in range(2):
            arrs[i]['Feature'] = self.FeatureNet(arrs[i]['coords'])

        batch_size = GT_motion.shape[0]
        res_list = []
        coarse_mots = []
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

            gt_motion = GT_motion[sample_ind,...].detach().numpy()
            sample_res, sample_coarse_mots = self.test_sample(gt_motion, sample_arrs)
            res_list.append(sample_res)
            coarse_mots.append(sample_coarse_mots)
        batch_result = np.vstack(res_list)
        coarse_mots = np.dstack(coarse_mots)
        return batch_result, coarse_mots

    def test_sample(self, gt_motion, sample_arrs):
        A = sample_arrs[0]['PC']
        B = sample_arrs[1]['PC']
        A_feat = sample_arrs[0]['Feature']
        B_feat = sample_arrs[1]['Feature']

        res_list = []
        coarse_mots = []
        for repeat in range(self.config.num_repeats):
            M_R, R_time, A_pcd, B_pcd, GT_inlier_ratio = self.RANSAC(A,B, A_feat, B_feat, gt_motion)
            M_I, I_time = self.ICP(A_pcd, B_pcd, M_R)
            R_recall, R_te, R_re = self.calc_errors(M_R, gt_motion)
            I_recall, I_te, I_re = self.calc_errors(M_I, gt_motion)
            repeat_res = [[  R_recall, R_te, R_re, R_time, 
                            I_recall, I_te, I_re, I_time, 
                            GT_inlier_ratio,
                            sample_arrs[0]['session_ind'].item(), sample_arrs[0]['cloud_ind'].item(), sample_arrs[1]['cloud_ind'].item()]]
            res_list.append(repeat_res)
            coarse_mots.append(M_R)
        res = np.stack(res_list,axis=2)
        coarse_mots = np.dstack(coarse_mots)
        return res, coarse_mots

    def ICP(self, pcd0, pcd1, M_R):        
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

    def measure_inlier_ratio(self, corres_idx0, corres_idx1, pcd0, pcd1, T_gt):
        corres_idx0_ = corres_idx0.detach().numpy()
        corres_idx1_ = corres_idx1.detach().numpy()
        pcd0_trans = deepcopy(pcd0)
        pcd0_trans.transform(T_gt)
        
        dist2 = np.sum((np.array(pcd0_trans.points)[corres_idx0_,:] - np.array(pcd1.points)[corres_idx1_,:])**2, axis=1)
        is_close = dist2 < (2*self.voxel_size)**2
        return float(is_close.sum()) / len(is_close)

    def RANSAC(self, A,B, A_feat, B_feat, gt_motion):
        
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

        fcgf_feats0 = A_feat
        fcgf_feats1 = B_feat

        with torch.no_grad():

            tic()
        
            # 1. Coarse correspondences
            corres_idx0, corres_idx1 = self.fcgf_feature_matching(fcgf_feats0, fcgf_feats1)


            GT_inlier_ratio = self.measure_inlier_ratio(corres_idx0, corres_idx1, pcd0, pcd1, gt_motion)

            # 2. Filter by distances in feature space

            corres_idx0_orig, corres_idx1_orig = corres_idx0, corres_idx1
            corres_idx0, corres_idx1 = self.nn_to_mutual(fcgf_feats0, fcgf_feats1, corres_idx0, corres_idx1)
                
                
        # 3. Perform RANSAC
        T = self.RANSAC_registration(pcd0,
                                        pcd1,
                                        corres_idx0,
                                        corres_idx1,
                                        2 * self.voxel_size,
                                        num_iterations=500*10**3) 

        # 4. estimate motion using all inlier pairs:
        corres_idx0_ = corres_idx0_orig.detach().numpy()
        corres_idx1_ = corres_idx1_orig.detach().numpy()
        pcd0_trans = deepcopy(pcd0)
        pcd0_trans.transform(T)
        dist2 = np.sum((np.array(pcd0_trans.points)[corres_idx0_,:] - np.array(pcd1.points)[corres_idx1_,:])**2, axis=1)
        is_close = dist2 < (2*self.voxel_size)**2
        inlier_corres_idx0 = corres_idx0_[is_close]
        inlier_corres_idx1 = corres_idx1_[is_close]
        corres = np.stack((inlier_corres_idx0, inlier_corres_idx1), axis=1)
        corres_ = o3d.utility.Vector2iVector(corres)
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        T = p2p.compute_transformation(pcd0, pcd1, corres_)

        elapsed_time = toc(silent=True)

        return T, elapsed_time, pcd0, pcd1, GT_inlier_ratio

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


