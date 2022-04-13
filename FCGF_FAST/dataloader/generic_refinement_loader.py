import numpy as np
import MinkowskiEngine as ME
import torch
from torch.utils.data import Dataset

from utils.tools_3d import euler_angles_to_rotation_matrix, apply_transformation
from utils.visualization import draw_multiple_clouds

VOXEL_SIZE = 0.3

def collate_fn(list_data):
    GT_mot_list = []
    coarse_mot_list = []
    keys = [list_data[0][2][j].keys() for j in range(2)]
    arrs_list = [{k: [] for k in keys[j]} for j in range(2)]
    arr_lens = [[], []]
    for data in list_data:
        GT_mot_list.append(data[0])
        coarse_mot_list.append(data[1])
        for i in range(2):
            for fld in arrs_list[i].keys():
                arrs_list[i][fld].append(data[2][i][fld])
            arr_lens[i].append(arrs_list[i]['PC'][-1].shape[0])

    GT_mot_final = np.stack(GT_mot_list)
    coarse_mot_final = np.stack(coarse_mot_list)
    arrs_final = [{k: [] for k in keys[j]} for j in range(2)]
    for i in range(2):
        for fld in keys[i]:
            if arrs_list[i][fld][0].ndim > 0:
                arrs_final[i][fld] = np.concatenate(arrs_list[i][fld], axis=0)
            else:
                arrs_final[i][fld] = np.stack(arrs_list[i][fld])
        arrs_final[i]['len'] = np.array(arr_lens[i])

    return GT_mot_final, coarse_mot_final, arrs_final


class GenericRefinementLoader(Dataset):

    def __getitem__(self, item_ind):

        pair = self.U.pairs[item_ind,:]
        GT_motion, P, Q = self.U.get_pair(item_ind)                                
        coarse_motion = self.U.get_coarse_motion(item_ind)

        full_PCs = [P,Q]    
        arrs = [{}, {}]
        for i in [0, 1]:
            PC_np = full_PCs[i].astype(np.float32) # ME.utils.sparse_quantize sometimes returns strange output when input dtype is np.float64
            arrs[i]['PC'] = PC_np            
            arrs[i]['session_ind'] = pair[0]
            arrs[i]['cloud_ind'] = pair[i+1]

        return GT_motion, coarse_motion, arrs

    def __len__(self):
        return len(self.U.pairs)
