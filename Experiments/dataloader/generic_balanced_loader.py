import numpy as np
from easydict import EasyDict as edict
import copy

from dataloader.base_loader import *
from dataloader.transforms import *
from util.pointcloud import get_matching_indices, make_open3d_point_cloud

default_config = edict(
  { 'voxel_size': 0.3, 
    'positive_pair_search_voxel_size_multiplier': 4.0,
  })

class GenericBalancedLoader(PairDataset):

    def __init__(self, 
                    phase,
                    transform=None,
                    random_rotation=True,
                    random_scale=True,
                    manual_seed=False,
                    config=None,
                    rank=None):
        
        if config is None:
            config=default_config                    

        PairDataset.__init__(self, phase, transform, random_rotation, random_scale,
                              manual_seed, config, rank)                          


    def __getitem__(self, idx):

        drive = self.U.pairs[idx, 0]
        t0 = self.U.pairs[idx, 1]
        t1 = self.U.pairs[idx, 2]
        M2, xyz0, xyz1= self.U.get_pair(idx)

        if self.random_rotation:
            T0 = sample_almost_planar_rotation(self.randg)
            T1 = sample_almost_planar_rotation(self.randg)
            trans = T1 @ M2 @ np.linalg.inv(T0)

            xyz0 = self.apply_transform(xyz0, T0)
            xyz1 = self.apply_transform(xyz1, T1)
        else:
            trans = M2

        matching_search_voxel_size = self.matching_search_voxel_size
        if self.random_scale and random.random() < 0.95:
            scale = self.min_scale + \
                (self.max_scale - self.min_scale) * random.random()
            matching_search_voxel_size *= scale
            xyz0 = scale * xyz0
            xyz1 = scale * xyz1
            M2[:3,3] *= scale

        # Voxelization
        xyz0_th = torch.from_numpy(xyz0)
        xyz1_th = torch.from_numpy(xyz1)

        _, sel0 = ME.utils.sparse_quantize(xyz0_th / self.voxel_size, return_index=True)
        _, sel1 = ME.utils.sparse_quantize(xyz1_th / self.voxel_size, return_index=True)

        # Make point clouds using voxelized points
        pcd0 = make_open3d_point_cloud(xyz0[sel0])
        pcd1 = make_open3d_point_cloud(xyz1[sel1])

        # Get matches
        matches = get_matching_indices(pcd0, pcd1, trans, matching_search_voxel_size)

        # Get features
        npts0 = len(sel0)
        npts1 = len(sel1)

        feats_train0, feats_train1 = [], []

        unique_xyz0_th = xyz0_th[sel0]
        unique_xyz1_th = xyz1_th[sel1]

        feats_train0.append(torch.ones((npts0, 1)))
        feats_train1.append(torch.ones((npts1, 1)))

        feats0 = torch.cat(feats_train0, 1)
        feats1 = torch.cat(feats_train1, 1)

        coords0 = torch.floor(unique_xyz0_th / self.voxel_size)
        coords1 = torch.floor(unique_xyz1_th / self.voxel_size)

        if self.transform:
            coords0, feats0 = self.transform(coords0, feats0)
            coords1, feats1 = self.transform(coords1, feats1)

        extra_package = {'drive': drive, 't0': t0, 't1': t1}

        return (unique_xyz0_th.float(),
                unique_xyz1_th.float(), coords0.int(), coords1.int(), feats0.float(),
                feats1.float(), matches, trans, extra_package)


    def __len__(self):
        return len(self.U.pairs)
