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
import math
import logging
import open3d as o3d
import numpy as np
import time
import torch
import copy
import MinkowskiEngine as ME
from scipy.spatial import cKDTree
from copy import deepcopy

sys.path.append('.')
from model import load_model

from core.registration import GlobalRegistration
from core.knn import find_knn_gpu

from util.timer import Timer
from util.pointcloud import make_open3d_point_cloud
from util.procrustes import procrustes
from train import prep_config

MEASURE_FRACTION_CLIPPED = False # used to estimate the fraction of cases where the failsafe is applied

# Feature-based registrations in Open3D
def registration_ransac_based_on_feature_matching(pcd0, pcd1, feats0, feats1,
                                                  distance_threshold, num_iterations):
  assert feats0.shape[1] == feats1.shape[1]

  source_feat = o3d.registration.Feature()
  source_feat.resize(feats0.shape[1], len(feats0))
  source_feat.data = feats0.astype('d').transpose()

  target_feat = o3d.registration.Feature()
  target_feat.resize(feats1.shape[1], len(feats1))
  target_feat.data = feats1.astype('d').transpose()

  result = o3d.registration.registration_ransac_based_on_feature_matching(
      pcd0, pcd1, source_feat, target_feat, distance_threshold,
      o3d.registration.TransformationEstimationPointToPoint(False), 4,
      [o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
      o3d.registration.RANSACConvergenceCriteria(num_iterations, 1000))

  return result.transformation


def registration_ransac_based_on_correspondence(pcd0, pcd1, idx0, idx1,
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


class DeepGlobalRegistration:
  def __init__(self, config, device, rank):
    # Basic config
    self.rank = rank
    self.total_clipped = 0
    self.total_weights = 0
    self.config = config
    self.clip_weight_thresh = self.config.clip_weight_thresh
    self.device = device

    # Safeguard
    self.safeguard_method = 'correspondence'  # correspondence, feature_matching

    # Final tuning
    self.use_icp = True

    # Misc
    self.init_timers()

    # Model config loading
    if self.rank == 0:
      print("=> loading checkpoint '{}'".format(config.weights))
    assert os.path.exists(config.weights)

    state = torch.load(config.weights, map_location=self.device)
    if 'state_dict_inlier' in state.keys():
      # this is a DGR weights file
      network_config = state['config']
    else:
      # this is not a DGR weights file, so it doesn't include the config that we need here:
      network_config = prep_config()

    self.network_config = network_config
    self.config.inlier_feature_type = network_config.inlier_feature_type
    self.voxel_size = network_config.voxel_size
    if self.rank == 0:
      print(f'=> Setting voxel size to {self.voxel_size}')

    # FCGF network initialization
    num_feats = 1
    try:
      FCGFModel = load_model(network_config['feat_model'])
      self.fcgf_model = FCGFModel(
          num_feats,
          network_config['feat_model_n_out'],
          bn_momentum=network_config['bn_momentum'],
          conv1_kernel_size=network_config['feat_conv1_kernel_size'],
          normalize_feature=network_config['normalize_feature'])

    except KeyError:  # legacy pretrained models
      FCGFModel = load_model(network_config['model'])
      self.fcgf_model = FCGFModel(num_feats,
                                  network_config['model_n_out'],
                                  bn_momentum=network_config['bn_momentum'],
                                  conv1_kernel_size=network_config['conv1_kernel_size'],
                                  normalize_feature=network_config['normalize_feature'])

    self.fcgf_model.load_state_dict(state['state_dict'])
    self.fcgf_model = self.fcgf_model.to(device)
    self.fcgf_model.eval()

    # Inlier network initialization
    num_feats = 6 if network_config.inlier_feature_type == 'coords' else 1
    InlierModel = load_model(network_config['inlier_model'])
    self.inlier_model = InlierModel(
        num_feats,
        1,
        bn_momentum=network_config['bn_momentum'],
        conv1_kernel_size=network_config['inlier_conv1_kernel_size'],
        normalize_feature=False,
        D=6)

    try:
      self.inlier_model.load_state_dict(state['state_dict_inlier'])
      self.inlier_model = self.inlier_model.to(self.device)
      self.inlier_model.eval()
    except KeyError as E:
      if 'state_dict_inlier' not in str(E):
        raise E
      if self.rank == 0:
        print("Inlier model not available. This is not a problem if you're only testing with RANSAC")      

    if self.rank == 0:
      print("=> loading finished")

  def init_timers(self):
    self.feat_timer = Timer()
    self.reg_timer = Timer()
    self.icp_timer = Timer()

  def preprocess(self, pcd):
    '''
    Stage 0: preprocess raw input point cloud
    Input: raw point cloud
    Output: voxelized point cloud with
    - xyz:    unique point cloud with one point per voxel
    - coords: coords after voxelization
    - feats:  dummy feature placeholder for general sparse convolution
    '''
    if isinstance(pcd, o3d.geometry.PointCloud):
      xyz = np.array(pcd.points)
    elif isinstance(pcd, np.ndarray):
      xyz = pcd
    else:
      raise Exception('Unrecognized pcd type')

    # Voxelization:
    # Maintain double type for xyz to improve numerical accuracy in quantization
    _, sel = ME.utils.sparse_quantize(xyz / self.voxel_size, return_index=True)
    npts = len(sel)

    xyz = torch.from_numpy(xyz[sel])

    # ME standard batch coordinates
    coords = ME.utils.batched_coordinates([torch.floor(xyz / self.voxel_size).int()])
    feats = torch.ones(npts, 1)

    return xyz.float(), coords, feats

  def fcgf_feature_extraction(self, feats, coords):
    '''
    Step 1: extract fast and accurate FCGF feature per point
    '''
    sinput = ME.SparseTensor(feats, coordinates=coords, device=self.device)

    return self.fcgf_model(sinput).F

  def fcgf_feature_matching(self, feats0, feats1):
    '''
    Step 2: coarsely match FCGF features to generate initial correspondences
    '''
    nns = find_knn_gpu(feats0,
                       feats1,
                       nn_max_n=self.network_config.nn_max_n,
                       knn=1,
                       return_distance=False)
    corres_idx0 = torch.arange(len(nns)).long().squeeze()
    corres_idx1 = nns.long().squeeze()

    return corres_idx0, corres_idx1

  def torch_intersect(self, Na, Nb, i_ab,j_ab,i_ba,j_ba):   
    # intersection between two sets of coordinates (fast implementation in pytorch)
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
    
  
  def fcgf_BB_feature_matching(self, feats0, feats1, mode='GPU'):
    # find pairs of mutual nearest neighbors (a.k.a Best-Buddies) in feature space

    if mode == 'CPU':
      f0 = feats0.detach().cpu().numpy()
      f1 = feats1.detach().cpu().numpy()

      feat1tree = cKDTree(f1)
      _, j_ab = feat1tree.query(f0, k=1, n_jobs=-1)
      i_ab = np.arange(len(j_ab))


      feat1tree = cKDTree(f0)
      _, i_ba = feat1tree.query(f1, k=1, n_jobs=-1)
      j_ba = np.arange(len(i_ba))

      Z = 50000
      assert f1.shape[1] <= Z, "cannot uniquely hash due to numerical limits. change implementation accordingly"
      assert Z*f0.shape[0] + f1.shape[0] <= np.iinfo(i_ba.dtype).max, "cannot uniquely hash due to numerical limits. change implementation accordingly"
      hash = Z * i_ab +  j_ab
      inv_hash = Z * i_ba +  j_ba
      final_hash = np.intersect1d(hash, inv_hash)
      corres_idx0_np, corres_idx1_np = np.divmod(final_hash, Z)
      final_corres_idx0 = torch.tensor(corres_idx0_np, device=feats0.device)
      final_corres_idx1 = torch.tensor(corres_idx1_np, device=feats1.device)

    elif mode == 'GPU':
      nns = find_knn_gpu(feats0,
                        feats1,
                        nn_max_n=self.network_config.nn_max_n,
                        knn=1,
                        return_distance=False)                       
      corres_idx0 = torch.arange(len(nns), dtype=torch.long, device=nns.device, requires_grad=False)
      corres_idx1 = nns.long().squeeze()
      
      uniq_inds_1=torch.unique(corres_idx1)
      inv_nns = find_knn_gpu(feats1[uniq_inds_1,:],
                        feats0,
                        nn_max_n=self.network_config.nn_max_n,
                        knn=1,
                        return_distance=False)                       
      inv_corres_idx1 = uniq_inds_1
      inv_corres_idx0 = inv_nns.long().squeeze()

      final_corres_idx0, final_corres_idx1 = self.torch_intersect(
        feats0.shape[0],
        feats1.shape[0],
        corres_idx0,
        corres_idx1,
        inv_corres_idx0,
        inv_corres_idx1)

    return final_corres_idx0, final_corres_idx1


  def inlier_feature_generation(self, xyz0, xyz1, coords0, coords1, fcgf_feats0,
                                fcgf_feats1, corres_idx0, corres_idx1):
    '''
    Step 3: generate features for inlier prediction
    '''
    assert len(corres_idx0) == len(corres_idx1)

    feat_type = self.config.inlier_feature_type
    assert feat_type in ['ones', 'feats', 'coords']

    corres_idx0 = corres_idx0.to(self.device)
    corres_idx1 = corres_idx1.to(self.device)

    if feat_type == 'ones':
      feat = torch.ones((len(corres_idx0), 1)).float()
    elif feat_type == 'feats':
      feat = torch.cat((fcgf_feats0[corres_idx0], fcgf_feats1[corres_idx1]), dim=1)
    elif feat_type == 'coords':
      feat = torch.cat((torch.cos(xyz0[corres_idx0]), torch.cos(xyz1[corres_idx1])),
                       dim=1)
    else:  # should never reach here
      raise TypeError('Undefined feature type')

    return feat

  def inlier_prediction(self, inlier_feats, coords):
    '''
    Step 4: predict inlier likelihood
    '''
    sinput = ME.SparseTensor(inlier_feats, coordinates=coords, device=self.device)
    soutput = self.inlier_model(sinput)

    return soutput.F

  def safeguard_registration(self, pcd0, pcd1, idx0, idx1, feats0, feats1,
                             distance_threshold, num_iterations):
    if self.safeguard_method == 'correspondence':
      T = registration_ransac_based_on_correspondence(pcd0,
                                                      pcd1,
                                                      idx0.cpu().numpy(),
                                                      idx1.cpu().numpy(),
                                                      distance_threshold,
                                                      num_iterations=num_iterations)
    elif self.safeguard_method == 'fcgf_feature_matching':
      T = registration_ransac_based_on_fcgf_feature_matching(pcd0, pcd1,
                                                             feats0.cpu().numpy(),
                                                             feats1.cpu().numpy(),
                                                             distance_threshold,
                                                             num_iterations)
    else:
      raise ValueError('Undefined')
    return T

  def register(self, xyz0, xyz1, inlier_thr=0.00, only_calc_weights=False):
    '''
    Main algorithm of DeepGlobalRegistration
    '''
    silent = (self.rank!=0)

    xyz0_np = xyz0.astype(np.float64)
    xyz1_np = xyz1.astype(np.float64)
    pcd0 = make_open3d_point_cloud(xyz0_np)
    pcd1 = make_open3d_point_cloud(xyz1_np)

    with torch.no_grad():
      # Step 0: voxelize and generate sparse input
      xyz0, coords0, feats0 = self.preprocess(xyz0)
      xyz1, coords1, feats1 = self.preprocess(xyz1)

      # Step 1: Feature extraction
      self.feat_timer.tic()
      fcgf_feats0 = self.fcgf_feature_extraction(feats0, coords0)
      fcgf_feats1 = self.fcgf_feature_extraction(feats1, coords1)
      self.feat_timer.toc()

      # Step 2: Coarse correspondences
      corres_idx0, corres_idx1 = self.fcgf_feature_matching(fcgf_feats0, fcgf_feats1)

      self.reg_timer.tic()
      # Step 3: Inlier feature generation
      # coords[corres_idx0]: 1D temporal + 3D spatial coord
      # coords[corres_idx1, 1:]: 3D spatial coord
      # => 1D temporal + 6D spatial coord
      inlier_coords = torch.cat((coords0[corres_idx0], coords1[corres_idx1, 1:]),
                                dim=1).int()
      inlier_feats = self.inlier_feature_generation(xyz0, xyz1, coords0, coords1,
                                                    fcgf_feats0, fcgf_feats1,
                                                    corres_idx0, corres_idx1)

      # Step 4: Inlier likelihood estimation and truncation
      logit = self.inlier_prediction(inlier_feats.contiguous(), coords=inlier_coords)
      weights = logit.sigmoid()
      if only_calc_weights:
        return weights
        
      if self.clip_weight_thresh > 0:
        if MEASURE_FRACTION_CLIPPED:
          self.total_clipped += (weights < self.clip_weight_thresh).sum().item()
          self.total_weights += weights.numel()
          print("fraction clipped = %f (%d/%d)" % (self.total_clipped/self.total_weights, self.total_clipped, self.total_weights))
          if self.total_weights > 0.5*10**6:
            assert False, "Done"
        weights[weights < self.clip_weight_thresh] = 0

      wsum = weights.sum().item()

    # Step 5: Registration. Note: torch's gradient may be required at this stage
    # > Case 0: Weighted Procrustes + Robust Refinement
    wsum_threshold = max(4000, len(weights)) * self.clip_weight_thresh
    sign = '>=' if wsum >= wsum_threshold else '<'
    if not silent:
      print(f'=> Weighted sum {wsum:.2f} {sign} threshold {wsum_threshold}')

    T = np.identity(4)
    if wsum >= wsum_threshold:
      try:
        rot, trans, opt_output = GlobalRegistration(xyz0[corres_idx0],
                                                    xyz1[corres_idx1],
                                                    weights=weights.detach().cpu(),
                                                    break_threshold_ratio=1e-4,
                                                    quantization_size=2 *
                                                    self.voxel_size,
                                                    verbose=False)
        T[0:3, 0:3] = rot.detach().cpu().numpy()
        T[0:3, 3] = trans.detach().cpu().numpy()
        dgr_time = self.reg_timer.toc()
        if not silent:
          print(f'=> DGR takes {dgr_time:.2} s')

      except RuntimeError:
        # Will directly go to Safeguard
        print('###############################################')
        print('# WARNING: SVD failed, weights sum: ', wsum)
        print('# Falling back to Safeguard')
        print('###############################################')

    else:
      # > Case 1: Safeguard RANSAC + (Optional) ICP
      T = self.safeguard_registration(pcd0,
                                      pcd1,
                                      corres_idx0,
                                      corres_idx1,
                                      feats0,
                                      feats1,
                                      2 * self.voxel_size,
                                      num_iterations=80000)
      safeguard_time = self.reg_timer.toc()
      if not silent:
        print(f'=> Safeguard takes {safeguard_time:.2} s')

    res = {'base': T, 'w_icp': T}

    if self.use_icp:
      self.icp_timer.tic()    
      T = o3d.pipelines.registration.registration_icp(
          pcd0,
          pcd1, self.voxel_size * 2, T,
          o3d.pipelines.registration.TransformationEstimationPointToPoint()).transformation
      self.icp_timer.toc()    
      res['w_icp'] = T            
    return res

  def register_FCGF(self, xyz0, xyz1, inlier_thr=0.00):
    '''
    Use RANSAC to register FCGF features
    '''
    silent = (self.rank!=0)    

    xyz0_np = xyz0.astype(np.float64)
    xyz1_np = xyz1.astype(np.float64)
    pcd0 = make_open3d_point_cloud(xyz0_np)
    pcd1 = make_open3d_point_cloud(xyz1_np)

    with torch.no_grad():
      # Step 0: voxelize and generate sparse input

      xyz0, coords0, feats0 = self.preprocess(xyz0)
      xyz1, coords1, feats1 = self.preprocess(xyz1)

      # Step 1: Feature extraction

      self.feat_timer.tic()
      fcgf_feats0 = self.fcgf_feature_extraction(feats0, coords0)
      fcgf_feats1 = self.fcgf_feature_extraction(feats1, coords1)
      self.feat_timer.toc() 

      self.reg_timer.tic()

      corres_idx0, corres_idx1 = self.fcgf_feature_matching(fcgf_feats0, fcgf_feats1)
      
      corres_idx0_raw = deepcopy(corres_idx0)
      corres_idx1_raw = deepcopy(corres_idx1)

      MUTUAL_ONLY = False # use only mutual nearest neighbors 

      if MUTUAL_ONLY:      
        keep0 = torch.randperm(fcgf_feats0.shape[0], device=fcgf_feats0.device)
        keep1 = torch.randperm(fcgf_feats1.shape[0], device=fcgf_feats1.device)

        # sparsify interest points
        NUM_INTEREST = 5000        
        if NUM_INTEREST < fcgf_feats0.shape[0]:
          keep0 = keep0[:NUM_INTEREST]               
        if NUM_INTEREST < fcgf_feats1.shape[0]:
          keep1 = keep1[:NUM_INTEREST]                         
        
        corres_idx0_in_subset, corres_idx1_in_subset = self.fcgf_BB_feature_matching(fcgf_feats0[keep0,:], fcgf_feats1[keep1,:])
        corres_idx0 = keep0[corres_idx0_in_subset]
        corres_idx1 = keep1[corres_idx1_in_subset]


      # > Case 1: Safeguard RANSAC + (Optional) ICP
      T = self.safeguard_registration(pcd0,
                                      pcd1,
                                      corres_idx0,
                                      corres_idx1,
                                      feats0,
                                      feats1,
                                      2 * self.voxel_size,
                                      num_iterations=500*10**3) # XXXXX

      # estimate motion using all inlier pairs:
      corres_idx0_ = corres_idx0_raw.cpu().detach().numpy()
      corres_idx1_ = corres_idx1_raw.cpu().detach().numpy()
      pcd0_trans = deepcopy(pcd0)
      pcd0_trans.transform(T)
      dist2 = np.sum((np.array(pcd0_trans.points)[corres_idx0_,:] - np.array(pcd1.points)[corres_idx1_,:])**2, axis=1)
      is_close = dist2 < (2*self.voxel_size)**2
      inlier_corres_idx0 = corres_idx0_[is_close]
      inlier_corres_idx1 = corres_idx1_[is_close]

      try:
        inlier_xyz0 = xyz0[inlier_corres_idx0,:]
        inlier_xyz1 = xyz1[inlier_corres_idx1,:]
        F0 = fcgf_feats0[corres_idx0,:]
        F1 = fcgf_feats1[corres_idx1,:]
        feat_dist = torch.sqrt(torch.sum((F0-F1)**2,axis=1))                
        w = feat_dist[is_close]**(-1)
        T_torch = procrustes(inlier_xyz0, inlier_xyz1, w) # weighted LS
        T = T_torch.cpu().detach().numpy()
      except Exception as E:
        print("Weighted refinement failed. error was" + str(E))
        print("performing non-weighted refinement.")
        try:
          corres = np.stack((inlier_corres_idx0, inlier_corres_idx1), axis=1)
          corres_ = o3d.utility.Vector2iVector(corres)
          p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
          T = p2p.compute_transformation(pcd0, pcd1, corres_) # non-weighted LS
        except:
          print("Non-weighted refinement failed")

      safeguard_time = self.reg_timer.toc()
      if not silent:
        print(f'=> Safeguard takes {safeguard_time:.3} s')

    res = {'base': T, 'w_icp': T}
    
    if self.use_icp:
      self.icp_timer.tic()    
      T = o3d.pipelines.registration.registration_icp(
          pcd0,
          pcd1, self.voxel_size * 2, T,
          o3d.pipelines.registration.TransformationEstimationPointToPoint()).transformation
      print(f"icp elapsed: {self.icp_timer.toc(average=False)}")
      res['w_icp'] = T

    return res
