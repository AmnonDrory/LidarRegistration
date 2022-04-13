import numpy as np
import torch
import math
from torch.autograd import Variable
from copy import deepcopy
import open3d as o3d
from time import time

def point_to_plane_dist_sparse_torch(X, Y, X_normals, Y_normals, align_normals=True, do_sqrt=True):
    assert(X.shape[1]==3)
    assert(Y.shape[1] == 3)
    assert(X_normals.shape[1]==3)
    assert(Y_normals.shape[1]==3)

    N = X.shape[0]
    M = Y.shape[2]
    assert Y.shape[0] == N, "point_to_plane_dist_sparse_torch calculates distances between corresponding points in two lists (and not from every point in list A and to every point in list B)"
    assert X_normals.shape[0]==N
    assert Y_normals.shape[0] == N

    if align_normals:
        normal_product = torch.sum(X_normals*Y_normals,dim=1)
        product_sign = torch.sign(normal_product)
        product_sign[product_sign==0]=1
    else:
        product_sign = torch.ones([N,M], dtype=X.dtype, device=X.device)
    product_sign = product_sign.unsqueeze(1)

    normal_sum = X_normals + product_sign*Y_normals
    pts_diff = X-Y

    # inner product:
    inplace_mult = pts_diff * normal_sum
    dot = torch.sum(inplace_mult, dim=1, keepdim=True)
    D = dot**2

    if do_sqrt:
        D = D.clamp_min_(1e-30).sqrt_() #

    return D

def euler_angles_to_rotation_matrix_torch(theta, phi, psi):
    if torch.cuda.is_available():
        one = Variable(torch.ones(1, dtype=theta.dtype)).cuda()
        zero = Variable(torch.zeros(1, dtype=theta.dtype)).cuda()
    else:
        one = Variable(torch.ones(1, dtype=theta.dtype))
        zero = Variable(torch.zeros(1, dtype=theta.dtype))
    rot_x = torch.cat((
        torch.unsqueeze(torch.cat((one, zero, zero), 0), dim=1),
        torch.unsqueeze(torch.cat((zero, theta.cos(), theta.sin()), 0), dim=1),
        torch.unsqueeze(torch.cat((zero, -theta.sin(), theta.cos()), 0), dim=1),
    ), dim=1)
    rot_y = torch.cat((
        torch.unsqueeze(torch.cat((phi.cos(), zero, -phi.sin()), 0), dim=1),
        torch.unsqueeze(torch.cat((zero, one, zero), 0), dim=1),
        torch.unsqueeze(torch.cat((phi.sin(), zero, phi.cos()), 0), dim=1),
    ), dim=1)
    rot_z = torch.cat((
        torch.unsqueeze(torch.cat((psi.cos(), psi.sin(), zero), 0), dim=1),
        torch.unsqueeze(torch.cat((-psi.sin(), psi.cos(), zero), 0), dim=1),
        torch.unsqueeze(torch.cat((zero, zero, one), 0), dim=1),
    ), dim=1)
    A = torch.mm(rot_z, torch.mm(rot_y, rot_x))
    if torch.cuda.is_available():
        A = A.cuda()
    return A

def euler_angles_to_rotation_matrix(theta_vec, deg_or_rad='deg'):
    if deg_or_rad is 'deg':
        theta_vec = np.radians(theta_vec)
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta_vec[0]), -math.sin(theta_vec[0]) ],
                    [0,         math.sin(theta_vec[0]), math.cos(theta_vec[0])  ]
                    ])
    R_y = np.array([[math.cos(theta_vec[1]),    0,      math.sin(theta_vec[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta_vec[1]),   0,      math.cos(theta_vec[1])  ]
                    ])
    R_z = np.array([[math.cos(theta_vec[2]),    -math.sin(theta_vec[2]),    0],
                    [math.sin(theta_vec[2]),    math.cos(theta_vec[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R


def prep_sparse_subset_of_points_torch(torch_A, torch_B, torch_A_normals, torch_B_normals, torch_inds):

    torch_points = {'HARD_BEST_BUDDY_PAIRS': {}}
    torch_points['HARD_BEST_BUDDY_PAIRS']['A'] = torch_A[torch_inds['HARD_BEST_BUDDY_PAIRS']['A'], :]
    torch_points['HARD_BEST_BUDDY_PAIRS']['B'] = torch_B[torch_inds['HARD_BEST_BUDDY_PAIRS']['B'], :]
    torch_normals = {'HARD_BEST_BUDDY_PAIRS': {}}
    torch_normals['HARD_BEST_BUDDY_PAIRS']['A'] = torch_A_normals[torch_inds['HARD_BEST_BUDDY_PAIRS']['A'], :]
    torch_normals['HARD_BEST_BUDDY_PAIRS']['B'] = torch_B_normals[torch_inds['HARD_BEST_BUDDY_PAIRS']['B'], :]
    torch_points = {'points': torch_points, 'normals': torch_normals}

    return torch_points


def SG_apply_rot_trans_torch(torch_points, WT, transT):

    rotated_points = deepcopy(torch_points)
    for fld in ['points', 'normals']:
        rotated_points[fld]['HARD_BEST_BUDDY_PAIRS']['B'] = torch.matmul(torch_points[fld]['HARD_BEST_BUDDY_PAIRS']['B'].double(), WT)
        if fld == 'points':
            # also translate
            rotated_points[fld]['HARD_BEST_BUDDY_PAIRS']['B'] += transT

    return rotated_points


def calc_loss(torch_points, rotated_points):

    dist_Rus = point_to_plane_dist_sparse_torch(
        torch_points['points']['HARD_BEST_BUDDY_PAIRS']['A'].unsqueeze(2),
        rotated_points['points']['HARD_BEST_BUDDY_PAIRS']['B'].unsqueeze(2),
        torch_points['normals']['HARD_BEST_BUDDY_PAIRS']['A'].unsqueeze(2),
        rotated_points['normals']['HARD_BEST_BUDDY_PAIRS']['B'].unsqueeze(2))
    BD = dist_Rus.mean()

    return BD


def torch_intersect(Na, Nb, i_ab,j_ab,i_ba,j_ba):    
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


def knn_gpu(P, Q):
    nn_max_n = 5000

    def knn_dist(p, q):
        # Fast implementation with torch.einsum()
        with torch.no_grad():      
            # L2 distance:
            dist2 = torch.sum(p**2, dim=1).reshape([-1,1]) + torch.sum(q**2, dim=1).reshape([1,-1]) -2*torch.einsum('ac,bc->ab', p, q)
            dist = dist2.clamp_min(1e-30).sqrt_()
            # Cosine distance:
            # dist = 1-torch.einsum('ac,bc->ab', p, q)                  
            min_dist, ind = dist.min(dim=1, keepdim=True)      
        return ind
    
    N = len(P)
    C = int(np.ceil(N / nn_max_n))
    stride = nn_max_n
    inds = []
    for i in range(C):
        with torch.no_grad():
            ind = knn_dist(P[i * stride:(i + 1) * stride], Q)
            inds.append(ind)
    
    inds = torch.cat(inds)
    assert len(inds) == N

    corres_idx0 = torch.arange(len(inds), device=P.device).long().squeeze()
    corres_idx1 = inds.long().squeeze()
    return corres_idx0, corres_idx1


def gpu_BB(P,Q):
    corres_idx0, corres_idx1 = knn_gpu(P, Q)
    
    uniq_inds_1=torch.unique(corres_idx1)
    inv_corres_idx1, inv_corres_idx0 = knn_gpu(Q[uniq_inds_1,:], P)
    inv_corres_idx1 = uniq_inds_1

    final_corres_idx0, final_corres_idx1 = torch_intersect(
    P.shape[0], Q.shape[0],
    corres_idx0, corres_idx1,
    inv_corres_idx0, inv_corres_idx1)

    return final_corres_idx0, final_corres_idx1


def prerun_gpu(torch_A, torch_B, WT, transT):
    torch_B_rot = (torch_B.double() @ WT) + transT
    pairs0, pairs1 = gpu_BB(torch_A,torch_B_rot)
    inds = {}
    inds['HARD_BEST_BUDDY_PAIRS'] = {}
    inds['HARD_BEST_BUDDY_PAIRS']['A'] = pairs0
    inds['HARD_BEST_BUDDY_PAIRS']['B'] = pairs1
    return inds


def BBR_F_step(torch_A, torch_B,
               torch_A_normals, torch_B_normals,
               theta, phi, psi, 
               trans_x, trans_y, trans_z,
               optimizer,
               angles_np, trans_np, loss_np):

    angles_np, trans_np = record_in_logs(angles_np, trans_np, theta, phi, psi, trans_x, trans_y,
                                                   trans_z)

    W = torch.squeeze(euler_angles_to_rotation_matrix_torch(theta, phi, psi))
    trans = torch.cat([trans_x, trans_y, trans_z], dim=0).unsqueeze(1)

    WT = W.T.double()
    transT = trans.T.double()

    torch_inds = prerun_gpu(torch_A, torch_B, WT, transT)

    torch_points = prep_sparse_subset_of_points_torch(torch_A, torch_B, torch_A_normals, torch_B_normals, torch_inds)

    rotated_points = SG_apply_rot_trans_torch(torch_points, WT, transT)

    loss = calc_loss(torch_points, rotated_points)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    loss_np.append(loss.item())
    return optimizer, loss_np, angles_np, trans_np

def calc_normals(X, knn_for_normals=13, radius=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=knn_for_normals))
    return np.asarray(pcd.normals)

def record_in_logs(angles_np,trans_np, theta, phi, psi, trans_x, trans_y, trans_z):
    angle = np.expand_dims([theta.item(), phi.item(), psi.item()], axis=1)
    trans = np.expand_dims([trans_x.item(), trans_y.item(), trans_z.item()], axis=1)
    if angles_np != [] and np.shape(angles_np.shape)[0]<2:
        angles_np = np.expand_dims(angles_np, axis=1)
    if trans_np != [] and np.shape(trans_np.shape)[0]<2:
        trans_np = np.expand_dims(trans_np, axis=1)
    if angles_np == []:
        angles_np = angle
    else:
        angles_np = np.append(angles_np, angle, axis=1)
    if trans_np == []:
        trans_np = trans
    else:
        trans_np = np.append(trans_np, trans, axis=1)
    
    return angles_np,trans_np

def downsample(X, X_normals, num_samples):
    inds = np.random.permutation(X.shape[0])[:num_samples]
    X = X[inds,:]
    X_normals = X_normals[inds,:]
    return X, X_normals

def BBR_F(A,B):
    
    NUM_SAMPLES = 30000

    config = {
        'nIterations': 100,
        'angles_lr': 2e-4,
        'trans_lr': 2e-4
    }    

    start_time = time()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A_normals = calc_normals(A)
    B_normals = calc_normals(B)
    A, A_normals = downsample(A, A_normals, NUM_SAMPLES)
    B, B_normals = downsample(B, B_normals, NUM_SAMPLES)
    torch_A = torch.tensor(A, device=device, requires_grad=False)
    torch_A_normals = torch.tensor(A_normals, device=device, requires_grad=False)
    torch_B = torch.tensor(B, device=device, requires_grad=False)
    torch_B_normals = torch.tensor(B_normals, device=device, requires_grad=False)    

    theta = torch.tensor([0.0], device=device, requires_grad=True)        
    phi = torch.tensor([0.0], device=device, requires_grad=True)        
    psi = torch.tensor([0.0], device=device, requires_grad=True)        
    trans_x = torch.tensor([0.0], device=device, requires_grad=True)        
    trans_y = torch.tensor([0.0], device=device, requires_grad=True)        
    trans_z = torch.tensor([0.0], device=device, requires_grad=True)        

    optimizer = torch.optim.Adam([
        {'params':[theta, phi, psi], 'lr': config['angles_lr']}, 
        {'params':[trans_x, trans_y, trans_z], 'lr': config['trans_lr']}])

    loss_np = []
    angles_np = []
    trans_np = []
    for iter in range(config['nIterations']):
        optimizer, loss_np, angles_np, trans_np = BBR_F_step(torch_A, torch_B,
               torch_A_normals, torch_B_normals,
               theta, phi, psi, 
               trans_x, trans_y, trans_z,
               optimizer,
               angles_np, trans_np, loss_np)

    
    ind = np.argmin(loss_np)    
    angles = angles_np[:, ind]
    W = euler_angles_to_rotation_matrix(angles, deg_or_rad='rad')    
    trans = trans_np[:,ind]    
    B_to_A = np.eye(4)
    B_to_A[:3,:3] = W
    B_to_A[:3,3] = trans
    A_to_B = np.linalg.inv(B_to_A)    

    elapsed = time() - start_time
    return A_to_B, elapsed
