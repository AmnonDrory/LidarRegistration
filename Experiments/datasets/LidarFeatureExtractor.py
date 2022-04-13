import os
import torch.utils.data as data
from utils.pointcloud import make_point_cloud, estimate_normal
from utils.SE3 import *
from misc.fcgf import ResUNetBN2C as FCGF
import MinkowskiEngine as ME

from dataloader.kitti_loader import KITTINMPairDataset

def collate_fn(list_data):
    min_num = 1e10
    # clip the pair having more correspondence during training.
    for ind, (corr_pos, src_keypts, tgt_keypts, gt_trans, gt_labels, src_features, tgt_features ) in enumerate(list_data):
        if len(gt_labels) < min_num:
            min_num = min(min_num, len(gt_labels))

    batched_corr_pos = []
    batched_src_keypts = []
    batched_tgt_keypts = []
    batched_gt_trans = []
    batched_gt_labels = []
    batched_src_features = []
    batched_tgt_features = []
    for ind, (corr_pos, src_keypts, tgt_keypts, gt_trans, gt_labels, src_features, tgt_features ) in enumerate(list_data):
        sel_ind = np.random.choice(len(gt_labels), min_num, replace=False)        
        batched_corr_pos.append(corr_pos[sel_ind, :][None,:,:])
        batched_src_keypts.append(src_keypts[sel_ind, :][None,:,:])
        batched_tgt_keypts.append(tgt_keypts[sel_ind, :][None,:,:])
        batched_gt_trans.append(gt_trans[None,:,:])
        batched_gt_labels.append(gt_labels[sel_ind][None, :])
        batched_src_features.append(src_features)
        batched_tgt_features.append(tgt_features)

    batched_corr_pos = torch.from_numpy(np.concatenate(batched_corr_pos, axis=0))
    batched_src_keypts = torch.from_numpy(np.concatenate(batched_src_keypts, axis=0))
    batched_tgt_keypts = torch.from_numpy(np.concatenate(batched_tgt_keypts, axis=0))
    batched_gt_trans = torch.from_numpy(np.concatenate(batched_gt_trans, axis=0))
    batched_gt_labels = torch.from_numpy(np.concatenate(batched_gt_labels, axis=0))
    batched_src_features = torch.from_numpy(np.concatenate(batched_src_features, axis=0))
    batched_tgt_features = torch.from_numpy(np.concatenate(batched_tgt_features, axis=0))

    return batched_corr_pos, batched_src_keypts, batched_tgt_keypts, batched_gt_trans, batched_gt_labels, batched_src_features, batched_tgt_features


class LidarFeatureExtractor():
    def __init__(self,
                split='train',
                in_dim=6,
                inlier_threshold=0.60,
                num_node=5000,
                use_mutual=True,
                augment_axis=0,
                augment_rotation=1.0,
                augment_translation=0.01,
                fcgf_weights_file = None
                ):
        assert fcgf_weights_file is not None, "user must supply a path to the FCGF weights as an argument"
        self.split = split
        self.inlier_threshold = inlier_threshold
        self.in_dim = in_dim
        self.descriptor = 'fcgf'
        self.num_node = num_node
        self.use_mutual = use_mutual
        self.augment_axis = augment_axis
        self.augment_rotation = augment_rotation
        self.augment_translation = augment_translation

        # containers
        self.ids_list = []

        self.feat_model = FCGF(
            1,
            32,
            bn_momentum=0.05,
            conv1_kernel_size=5,
            normalize_feature=True)
        self.feat_model = self.feat_model.cuda()
        self.device = 'cuda:%d' % torch.cuda.current_device()
        checkpoint = torch.load(fcgf_weights_file, map_location=self.device)
        self.feat_model.load_state_dict(checkpoint['state_dict'])
        self.feat_model.eval()               

    def get_pairs(self, src_keypts, tgt_keypts, src_features, tgt_features, orig_trans):
        if self.split == 'train':
            src_keypts += np.random.rand(src_keypts.shape[0], 3) * 0.05
            tgt_keypts += np.random.rand(tgt_keypts.shape[0], 3) * 0.05
        
        aug_R = rotation_matrix(self.augment_axis, self.augment_rotation)
        aug_T = translation_matrix(self.augment_translation)
        aug_trans = integrate_trans(aug_R, aug_T)
        tgt_keypts = transform(tgt_keypts, aug_trans)
        gt_trans = concatenate(aug_trans, orig_trans)

        # select {self.num_node} numbers of keypoints
        N_src = src_features.shape[0]
        N_tgt = tgt_features.shape[0]
        src_sel_ind = np.arange(N_src)
        tgt_sel_ind = np.arange(N_tgt)
        if self.num_node != 'all' and N_src > self.num_node:
            src_sel_ind = np.random.choice(N_src, self.num_node, replace=False)
        if self.num_node != 'all' and N_tgt > self.num_node:
            tgt_sel_ind = np.random.choice(N_tgt, self.num_node, replace=False)

        src_desc = src_features[src_sel_ind, :]
        tgt_desc = tgt_features[tgt_sel_ind, :]
        src_keypts = src_keypts[src_sel_ind, :]
        tgt_keypts = tgt_keypts[tgt_sel_ind, :]

        # construct the correspondence set by mutual nn in feature space.
        distance = np.sqrt(2 - 2 * (src_desc @ tgt_desc.T) + 1e-6)
        source_idx = np.argmin(distance, axis=1)
        if self.use_mutual:
            target_idx = np.argmin(distance, axis=0)
            mutual_nearest = (target_idx[source_idx] == np.arange(source_idx.shape[0]))
            corr = np.concatenate([np.where(mutual_nearest == 1)[0][:,None], source_idx[mutual_nearest][:,None]], axis=-1)
        else:
            corr = np.concatenate([np.arange(source_idx.shape[0])[:, None], source_idx[:, None]], axis=-1)

        # compute the ground truth label
        frag1 = src_keypts[corr[:, 0]]
        frag2 = tgt_keypts[corr[:, 1]]
        frag1_warp = transform(frag1, gt_trans)
        distance = np.sqrt(np.sum(np.power(frag1_warp - frag2, 2), axis=1))
        labels = (distance < self.inlier_threshold).astype(np.int)

        # add random outlier to input data
        if self.split == 'train' and np.mean(labels) > 0.5:
            num_outliers = int(0.0 * len(corr))
            src_outliers = np.random.randn(num_outliers, 3) * np.mean(src_keypts, axis=0)
            tgt_outliers = np.random.randn(num_outliers, 3) * np.mean(tgt_keypts, axis=0)
            input_src_keypts = np.concatenate( [src_keypts[corr[:, 0]], src_outliers], axis=0)
            input_tgt_keypts = np.concatenate( [tgt_keypts[corr[:, 1]], tgt_outliers], axis=0)
            labels = np.concatenate( [labels, np.zeros(num_outliers)], axis=0)
        else:
            # prepare input to the network
            input_src_keypts = src_keypts[corr[:, 0]]
            input_tgt_keypts = tgt_keypts[corr[:, 1]]

        if self.in_dim == 3:
            corr_pos = input_src_keypts - input_tgt_keypts
        elif self.in_dim == 6:
            corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts], axis=-1)
            # move the center of each point cloud to (0,0,0).
            corr_pos = corr_pos - corr_pos.mean(0)
        elif self.in_dim == 9:
            corr_pos = np.concatenate([input_src_keypts, input_tgt_keypts, input_src_keypts-input_tgt_keypts], axis=-1)
        elif self.in_dim == 12:
            src_pcd = make_point_cloud(src_keypts)
            tgt_pcd = make_point_cloud(tgt_keypts)
            estimate_normal(src_pcd, radius=self.downsample*2)
            estimate_normal(tgt_pcd, radius=self.downsample*2)
            src_normal = np.array(src_pcd.normals)
            tgt_normal = np.array(tgt_pcd.normals)
            src_normal = src_normal[src_sel_ind, :]
            tgt_normal = tgt_normal[tgt_sel_ind, :]
            input_src_normal = src_normal[corr[:, 0]]
            input_tgt_normal = tgt_normal[corr[:, 1]]
            corr_pos = np.concatenate([input_src_keypts, input_src_normal, input_tgt_keypts, input_tgt_normal], axis=-1)

        return corr_pos.astype(np.float32), \
            input_src_keypts.astype(np.float32), \
            input_tgt_keypts.astype(np.float32), \
            gt_trans.astype(np.float32), \
            labels.astype(np.float32),

    def process_batch(self, input_dict):

        xyz0=input_dict['pcd0']
        xyz1=input_dict['pcd1']
        iC0=input_dict['sinput0_C'].cuda()
        iC1=input_dict['sinput1_C'].cuda()
        iF0=input_dict['sinput0_F'].cuda()
        iF1=input_dict['sinput1_F'].cuda()
        T_gt = input_dict['T_gt']
        lens = input_dict['len_batch']

        sinput0 = ME.SparseTensor(iF0, coordinates=iC0, device=self.device)
        oF0 = self.feat_model(sinput0).F

        sinput1 = ME.SparseTensor(iF1, coordinates=iC1, device=self.device)
        oF1 = self.feat_model(sinput1).F

        res_list = []
        src_to = 0
        tgt_to = 0
        for i, (src_len, tgt_len) in enumerate(lens):
            src_from = src_to
            src_to = src_from + src_len
            tgt_from = tgt_to
            tgt_to = tgt_from + tgt_len
            src_keypts = xyz0[i].detach().cpu().numpy()
            tgt_keypts = xyz1[i].detach().cpu().numpy()
            src_features = oF0[src_from:src_to,:].detach().cpu().numpy()
            tgt_features = oF1[tgt_from:tgt_to,:].detach().cpu().numpy()
            orig_trans = T_gt[i].numpy()

            cur_res = self.get_pairs(src_keypts, tgt_keypts, src_features, tgt_features, orig_trans)
            cur_res += (src_features, tgt_features)
            res_list.append(cur_res)
        return collate_fn(res_list)
		
if __name__ == "__main__":
    phase = 'train'
    dset = KITTINMPairDataset(
        phase,
        transform=None,
        random_rotation=False,
        random_scale=False,
        manual_seed=False,
        config=None,
        rank=0)