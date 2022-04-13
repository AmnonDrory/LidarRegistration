import numpy as np
import MinkowskiEngine as ME
import torch
from torch.utils.data import Dataset

from utils.tools_3d import euler_angles_to_rotation_matrix, apply_transformation
from utils.visualization import draw_multiple_clouds

VOXEL_SIZE = 0.3

def collate_fn(list_data):
    GT_mot_list = []
    keys = [list_data[0][1][j].keys() for j in range(2)]
    arrs_list = [{k: [] for k in keys[j]} for j in range(2)]
    arr_lens = [[], []]
    for data in list_data:
        GT_mot_list.append(data[0])
        for i in range(2):
            for fld in arrs_list[i].keys():
                arrs_list[i][fld].append(data[1][i][fld])
            arr_lens[i].append(arrs_list[i]['PC'][-1].shape[0])

    GT_mot_final = torch.stack(GT_mot_list)
    arrs_final = [{k: [] for k in keys[j]} for j in range(2)]
    for i in range(2):
        for fld in keys[i]:
            if fld == 'coords':
                arrs_final[i][fld] = ME.utils.batched_coordinates(arrs_list[i][fld], device=arrs_list[i][fld][0].device)
            elif arrs_list[i][fld][0].ndim > 0:
                arrs_final[i][fld] = torch.cat(arrs_list[i][fld], dim=0)
            else:
                arrs_final[i][fld] = torch.stack(arrs_list[i][fld])
        arrs_final[i]['len'] = torch.from_numpy(np.array(arr_lens[i]))

    return GT_mot_final, arrs_final

def random_rotation(P, Q,  M_GT):
    MAX_ROTATION_ANGLES_IN_DEGREES = [5,5,180]

    def generate_random_rotation(max_rotation_angles_in_degrees=MAX_ROTATION_ANGLES_IN_DEGREES):        
        random_rotation_angles_in_degrees = np.random.rand(3)*max_rotation_angles_in_degrees*np.sign(np.random.randn(3))
        R = euler_angles_to_rotation_matrix(random_rotation_angles_in_degrees, deg_or_rad='deg')
        M = np.eye(4)
        M[:3,:3] = R
        return M

    M_P = generate_random_rotation()
    P_new = apply_transformation(M_P, P).astype(np.float32)     
    M_Q = generate_random_rotation()
    Q_new = apply_transformation(M_Q, Q).astype(np.float32)     

    M_GT_new = M_Q @ M_GT @ M_P.T   
    return P_new, Q_new, M_GT_new

def send_arrs_to_device(arrs, device, keep_PC_on_cpu=False):
    new_arrs = []
    for i in range(len(arrs)):
        new_arrs.append({})
        for k in arrs[i].keys():
            if (k == 'PC') and keep_PC_on_cpu:
                new_arrs[i][k] = arrs[i][k]    
            else:
                new_arrs[i][k] = arrs[i][k].to(device)
    return new_arrs

class GenericBalancedLoader(Dataset):

    def __getitem__(self, item_ind):

        pair = self.U.pairs[item_ind,:]
        GT_mot_np, P, Q = self.U.get_pair(item_ind)        
        
        if self.random_rotation:
            P, Q, GT_mot_np = random_rotation(P, Q, GT_mot_np) 

        GT_motion = torch.tensor(GT_mot_np, requires_grad=False, dtype=torch.float)

        full_PCs = [P,Q]    
        arrs = [{}, {}]
        for i in [0, 1]:
            PC_np = full_PCs[i].astype(np.float32) # ME.utils.sparse_quantize sometimes returns strange output when input dtype is np.float64
            PC_torch = torch.tensor(PC_np, device = GT_motion.device).contiguous()
            _, sel_torch = ME.utils.sparse_quantize(PC_torch / VOXEL_SIZE, return_index=True)                             
            sel = sel_torch.numpy()
            arrs[i]['PC'] = torch.tensor(PC_np[sel, :] , requires_grad=False, dtype=torch.float)
            arrs[i]['coords'] = torch.floor(arrs[i]['PC'] / VOXEL_SIZE)        
            arrs[i]['session_ind'] = torch.tensor(pair[0], requires_grad=False, dtype=torch.int32)
            arrs[i]['cloud_ind'] = torch.tensor(pair[i+1], requires_grad=False, dtype=torch.int32)

        return GT_motion, arrs

    def __len__(self):
        return len(self.U.pairs)
