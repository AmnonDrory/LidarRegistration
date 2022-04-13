import numpy as np
import torch
import time
from utils.PointCloudUtils import dist, cdist, dist_torch, cdist_torch, square_dist, min_without_self_per_row, representative_neighbor_dist
from utils.net_tools import soft_argmin_on_rows, argmin_on_cols, argmin_on_cols_torch, argmin_on_rows_torch, argmin_on_rows, softargmin_rows_torch_new
from utils.tools_3d import dilate_randomly, apply_rot
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
EPS = 10.0**-12

def find_best_alpha(T,S):
    assert False, "Use guess_best_alpha instead"
    all_losses = []
    alphas = np.logspace(-6,2,num=20)
    for alpha in alphas:
        all_losses.append(bb_loss_soft(T,S,alpha))

    return alphas[np.nanargmax(np.abs(all_losses))]

def guess_best_alpha(A):
        """
        A good guess for the temperature of the soft argmin (alpha) can
        be calculated as a linear function of the representative (e.g. median)
        distance of points to their nearest neighbor in a point cloud.

        :param A: Point Cloud of size Nx3
        :return: Estimated value of alpha
        """

        if A.shape[1] != 3:
                A = A[:,:3]

        COEFF = 0.1 # was estimated using the code in experiments/alpha_regression.py
        rep = representative_neighbor_dist(A, dist_matrix_as_input=False)
        return COEFF * rep

def chamfer_dist(T,S):
        D = cdist(T, S)
        d1 = np.mean(np.min(D,axis=1))
        d2 = np.mean(np.min(D,axis=0))
        return d1, d2

def soft_bb_pairs(D, t, eps=EPS):
        R = soft_argmin_on_rows(D, t, eps)
        C = soft_argmin_on_rows(D.T, t, eps).T
        B = np.multiply(R, C)
        return B

def bb_loss_soft(T,S,t, eps=EPS, feature_mode='xyz'):
        D = cdist(T, S, feature_mode)
        B = soft_BB_pairs(D, t, eps)

        loss = -np.sum(B)
        loss /= np.mean([T.shape[0],S.shape[0]])
        return loss

def bb_dist_loss(T,S, feature_mode='xyz'):
        D = cdist(T, S, feature_mode)
        R = argmin_on_rows(D)
        C = argmin_on_cols(D)
        B = R * C # element-wise multiplication
        P = D * B
        loss = np.sum(P) / np.sum(B)
        return loss

def bb_dist_loss_soft(T,S,t, eps, feature_mode='xyz'):
        D = cdist(T, S, feature_mode)
        R = soft_argmin_on_rows(D, t, eps)
        # R = argmin_on_cols(D)
        C = soft_argmin_on_rows(D.T, t, eps).T
        # C = argmin_on_cols(D.T)
        B = R * C # element-wise multiplication
        P = D * B
        loss = np.sum(P) / np.sum(B)
        return loss

def bb_loss(T,S, feature_mode='xyz'):
        if feature_mode == 'xyz':
                loss = -fast_count_mutual_nearest_neighbors(T, S)
        else:
                D = cdist(T, S, feature_mode)
                R = argmin_on_cols(D)
                C = argmin_on_cols(D.T)
                B = R * (np.transpose(C))
                loss = -np.sum(B)

        loss /= np.mean([T.shape[0],S.shape[0]])
        return loss

def icp_loss(T,S):
        n_jobs = None
        if T.shape[0] >= 5000:  # for small clouds the overhead of a concurrent implementation is detrimental.
                n_jobs = 8

        NN = NearestNeighbors(n_jobs=n_jobs)
        NN.fit(T)
        dist_S_to_T, i_S_to_T = NN.kneighbors(S, 1, return_distance=True)
        return np.mean(dist_S_to_T)

def fast_count_mutual_nearest_neighbors(T,S):
    n_jobs = None
    if T.shape[0] >= 5000: # for small clouds the overhead of a concurrent implementation is detrimental.
        n_jobs = 8

    NN = NearestNeighbors(n_jobs=n_jobs)
    NN.fit(S)
    i_T_to_S = NN.kneighbors(T, 1, return_distance=False)
    mat_T_to_S = sparse.coo_matrix((np.ones(len(i_T_to_S)),
                                    (np.arange(len(i_T_to_S)), i_T_to_S.flatten())),
                                   [T.shape[0], S.shape[0]]).tocsr()
    NN.fit(T)
    i_S_to_T = NN.kneighbors(S, 1, return_distance=False)
    mat_S_to_T = sparse.coo_matrix((np.ones(len(i_S_to_T)),
                                    (i_S_to_T.flatten(), np.arange(len(i_S_to_T)))),
                                   [T.shape[0], S.shape[0]]).tocsr()

    BB_pairs = mat_T_to_S.multiply(mat_S_to_T)
    return float(BB_pairs.nnz)

def count_significant_soft_bb_pairs(B):
        RELATIVE_SIGNIFICANCE_THRESH=0.1
        b = B.detach().cpu().numpy()
        sorted_down = -np.sort(-b, axis=1)
        relative = sorted_down / sorted_down[:,0:1]
        num_significant_per_row = (relative >= RELATIVE_SIGNIFICANCE_THRESH).sum(axis=1, keepdims=True)
        return np.mean(num_significant_per_row)

def experimental_bb_dist_loss_torch_from_two_Ds(close_enough_mask, BBP_D, dist_D, t, eps=EPS):
        B = close_enough_mask * torch.exp(-BBP_D / t)
        softness = count_significant_soft_bb_pairs(B)
        mask_softness = count_significant_soft_bb_pairs(close_enough_mask)
        loss = torch.sum(torch.mul(B,dist_D))/ (torch.sum(B))
        return loss, softness, mask_softness

def bb_dist_loss_torch_from_two_Ds(BBP_D, dist_D, t, eps=EPS):
        BBP = torch_soft_best_buddy_pairs(BBP_D, t, eps)
        BBP_SQ = BBP**2
        loss = torch.sum(torch.mul(BBP_SQ,dist_D))/ (torch.sum(BBP_SQ))
        return loss, BBP

def bb_dist_loss_torch_from_D(D, t, eps=EPS):
        B = torch_soft_best_buddy_pairs(D, t, eps)
        loss = torch.sum(torch.mul(B,D))/ (torch.sum(B))
        return loss

def bb_dist_loss_torch(T, S, t, eps, feature_mode='xyz'):
        D = cdist_torch(T, S, feature_mode).double()
        return bb_dist_loss_torch_from_D(D,t, eps)

def torch_soft_best_buddy_pairs(D, t, eps=EPS):
        R = softargmin_rows_torch_new(D, t, eps)
        C = softargmin_rows_torch_new(D.t(), t, eps)
        B = torch.mul(R, C.t())
        return B

def bb_loss_torch_from_D(D, t, eps): # soft
        REPORT = False
        T_num_samples = D.shape[0]
        S_num_samples = D.shape[1]
        mean_num_samples = np.mean([T_num_samples, S_num_samples])
        B = torch_soft_best_buddy_pairs(D, t, eps)
        if REPORT:
                b = B.detach().cpu().numpy().flatten()
                r = []
                r.append(b.max())
                r.append(np.percentile(b, 90))
                r.append(np.percentile(b, 10))
                r.append(b.min())
                print("B percentiles (100,90,10,0) : " + str(r))
        loss = torch.div(-torch.sum(B), mean_num_samples)
        return loss

def bb_loss_torch(T, S, t, eps, feature_mode='xyz'): # soft
        D = cdist_torch(T, S, feature_mode)
        return bb_loss_torch_from_D(D, t, eps)

def torch_hard_best_buddy_pairs_from_dist(D):
        R = argmin_on_rows_torch(D)
        C = argmin_on_cols_torch(D)
        B = R * C
        return B

def torch_hard_best_buddy_pairs(T,S):
        D = cdist_torch(T, S)
        return torch_hard_best_buddy_pairs_from_dist(D)

def hard_BBS_loss_torch(T, S):
        B = torch_hard_best_buddy_pairs(T, S)
        loss = -B.sum()
        loss /= (0.5*(T.shape[0]+S.shape[0]))
        return loss
	
def add_outliers(A, A_outliers, n_outliers, translation=None, rotate=False, offset = [10,10,10]):
        '''
        Add the point cloud from A_outliers to A after rotating and translating it
        '''
        trans = []
        if translation is None:
                trans.append(np.mean(A[0]) + offset[0])
                trans.append(np.mean(A[1]) + offset[1])
                trans.append(np.mean(A[2]) + offset[2])
        else:
                trans = translation
        trans = np.array(trans)
        A_outliers += trans
        A_outliers = dilate_randomly(A_outliers, n_outliers)
        if rotate:
                angle = 30
                A_outliers = apply_rot(A_outliers,[0,0,angle]).T
        if A is None:
                return A_outliers
        if n_outliers:
                if np.max(np.shape(A)):
                        A = np.concatenate([A, A_outliers], axis=0)
                else:
                        A = A_outliers
        return A

def add_xyz_noise(PC, strength=1):
        if strength == 0:
                return PC
        print('add_xyz_noise: square_dist was replaced by dist, thus changing the meaning of "strength". Please change calling code accordingly')

        N =  PC.shape[0]

        SUBSET_SIZE = 1000
        if N > SUBSET_SIZE:
                inds = np.random.permutation(N)
                PC_sub = PC[inds[:SUBSET_SIZE],:]
                D = dist(PC_sub, PC_sub)
                neighbor_dist_for_sub = representative_neighbor_dist(D)
                ratio = float(N) / SUBSET_SIZE
                neighbor_dist = ratio**(-1./3.) * neighbor_dist_for_sub # simplified assumption: neighbor dist is the reciprocal of density. number of samples in a volume grows as the third power of density.
        else:
                D = dist(PC, PC)
                neighbor_dist = representative_neighbor_dist(D)

        requested_noise_sigma = strength * neighbor_dist
        noise = np.random.normal(0, requested_noise_sigma, PC.shape)
        return PC + noise


def calc_block_inds_soft(A, B, num_blocks_per_row_or_col):
        A, B = calc_block_inds(A, B, num_blocks_per_row_or_col)
        A[:, -1] -= 1  # we need indexing from 0, not 1
        B[:, -1] -= 1  # we need indexing from 0, not 1
        return A, B


def calc_block_inds(A, B, num_blocks_per_row_or_col):
        PC = np.vstack([A, B])
        n_A = A.shape[0]
        min_X = np.min(PC[:, 0])
        min_Y = np.min(PC[:, 1])
        PC_X_wid = np.max(PC[:, 0]) - min_X
        PC_Y_wid = np.max(PC[:, 1]) - min_Y
        X_block_wid = PC_X_wid / num_blocks_per_row_or_col
        Y_block_wid = PC_Y_wid / num_blocks_per_row_or_col
        X_ind = np.floor((PC[:, 0] - min_X) / X_block_wid)
        Y_ind = np.floor((PC[:, 1] - min_Y) / Y_block_wid)
        X_ind[X_ind >= num_blocks_per_row_or_col] = num_blocks_per_row_or_col - 1
        Y_ind[Y_ind >= num_blocks_per_row_or_col] = num_blocks_per_row_or_col - 1

        PC = np.hstack([PC, np.reshape(X_ind, [-1, 1]), np.reshape(Y_ind, [-1, 1])])

        A = PC[:n_A, :]
        B = PC[n_A:, :]
        A = np.hstack([A, np.reshape(np.arange(1, A.shape[0] + 1), [-1, 1])])
        B = np.hstack([B, np.reshape(np.arange(1, B.shape[0] + 1), [-1, 1])])
        return A, B


def current_subsets(A, B, x, y, c):
        is_cur_A = (A[:, c['x_ind']] == x) & (A[:, c['y_ind']] == y)
        cur_A = A[is_cur_A, :]

        is_cur_B = (B[:, c['x_ind']] == x) & (B[:, c['y_ind']] == y)
        for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                        if (dx == 0) and (dy == 0):
                                continue
                        is_cur_B |= (B[:, c['x_ind']] == (x + dx)) & (B[:, c['y_ind']] == (y + dy))

        cur_B = B[is_cur_B, :]

        return cur_A, cur_B


def NN_in_blocks(A, B, c, num_blocks_per_row_or_col):
        A_to_B = np.zeros(A.shape[0] + 1, dtype=int)
        for x in range(num_blocks_per_row_or_col):
                for y in range(num_blocks_per_row_or_col):
                        cur_A, cur_B = current_subsets(A, B, x, y, c)
                        if (cur_A.size == 0) or (cur_B.size == 0):
                                continue
                        D = square_dist(cur_A[:, :3], cur_B[:, :3])
                        nearest = np.argmin(D, axis=1)
                        nearest_B_inds = cur_B[nearest, c['ind']]
                        A_to_B[cur_A[:, c['ind']].astype(int)] = nearest_B_inds

        return A_to_B


def NN_in_blocks_soft(A, B, alpha, c, num_blocks_per_row_or_col, eps):
        R_data = []
        D_data = []
        i = []
        j = []
        for x in range(num_blocks_per_row_or_col):
                for y in range(num_blocks_per_row_or_col):
                        cur_A, cur_B = current_subsets(A, B, x, y, c)
                        if (cur_A.size == 0) or (cur_B.size == 0):
                                continue
                        D = dist(cur_A[:, :3], cur_B[:, :3])
                        R = soft_argmin_on_rows(D.copy(), alpha, eps)
                        I_B, I_A = np.meshgrid(cur_B[:, c['ind']], cur_A[:, c['ind']])
                        D_data.append(D.flatten())
                        R_data.append(R.flatten())
                        i.append(I_A.flatten())
                        j.append(I_B.flatten())
        D_data = np.hstack(D_data)
        R_data = np.hstack(R_data)
        i = np.hstack(i)
        j = np.hstack(j)
        A_to_B = sparse.coo_matrix((R_data, (i, j)), shape=(A.shape[0], B.shape[0]))
        A_to_B_D = sparse.coo_matrix((D_data, (i, j)), shape=(A.shape[0], B.shape[0]))

        return A_to_B


def block_bb_loss(A, B, feature_mode, num_blocks_per_row_or_col=10, output_pairs=False):
        """
        divide the xy-plane to blocks and assume that a point's nearest neighbor is in its
        block or, at farthest, an immediately neighboring block. If the initial guess is close,
        this could be reasonable. This method allows us to handle point clouds with many more point.

        :param A:
        :param B:
        :param init_motion:
        :return:
        """

        c = {'x': 0, 'y': 1, 'z': 2, 'x_ind': 3, 'y_ind': 4, 'ind': 5}
        A, B = calc_block_inds(A, B, num_blocks_per_row_or_col)
        A_to_B = NN_in_blocks(A, B, c, num_blocks_per_row_or_col)
        B_to_A = NN_in_blocks(B, A, c, num_blocks_per_row_or_col)
        bb_pairs_B = (A_to_B[B_to_A] == np.arange(len(B_to_A))).nonzero()[0]
        bb_pairs_A = B_to_A[bb_pairs_B]
        bb_pairs_B_final = bb_pairs_B[1:] - 1
        bb_pairs_A_final = bb_pairs_A[1:] - 1
        pairs = np.hstack([bb_pairs_A_final, bb_pairs_B_final]).T
        BBS_loss = -len(bb_pairs_B_final) / np.mean([A.shape[0], B.shape[0]])
        if output_pairs:
                return BBS_loss, pairs
        else:
                return BBS_loss


def block_bb_loss_soft(A, B, alpha, feature_mode, num_blocks_per_row_or_col=10):
        """
        divide the xy-plane to blocks and assume that a point's nearest neighbor is in its
        block or, at farthest, an immediately neighboring block. If the initial guess is close,
        this could be reasonable. This method allows us to handle point clouds with many more point.

        :param A:
        :param B:
        :param init_motion:
        :return:
        """

        c = {'x': 0, 'y': 1, 'z': 2, 'x_ind': 3, 'y_ind': 4, 'ind': 5}
        A, B = calc_block_inds_soft(A, B, num_blocks_per_row_or_col)
        A_to_B = NN_in_blocks_soft(A, B, alpha, c, num_blocks_per_row_or_col)
        B_to_A = NN_in_blocks_soft(B, A, alpha, c, num_blocks_per_row_or_col)
        BB_arr = A_to_B.multiply(B_to_A.T)
        BBS_loss = -BB_arr.sum() / np.mean([A.shape[0], B.shape[0]])
        return BBS_loss

def calc_cloud_spacing_torch(inp, input_is_distance_matrix=True):
    if input_is_distance_matrix:
        D_AA = inp
    else:
        D_AA = batch_dist_torch(inp, inp)

    spacing = torch.median(torch.kthvalue(D_AA, 2, dim=1).values)  # the typical distance to the nearest neighbor in A
    return spacing

def batch_dist_torch(P_, Q_):
    offs = len(P_.shape)-3
    P_Reg = P_.unsqueeze(2+offs)
    Q_Reg = Q_.unsqueeze(1+offs)
    sq_diff = (P_Reg - Q_Reg) ** 2
    sq_dist = torch.sum(sq_diff, dim=3+offs)
    D = torch.sqrt(sq_dist)
    return D