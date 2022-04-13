import numpy as np
import torch
from time import time
from sklearn.neighbors import NearestNeighbors

def square_dist(A, B):
    """
    Measure Squared Euclidean Distance from every point in point-cloud A, to every point in point-cloud B
    :param A: Point Cloud: Nx3 Array of real numbers, each row represents one point in x,y,z space
    :param B: Point Cloud: Mx3 Array of real numbers
    :return:  NxM array, where element [i,j] is the squared distance between the i'th point in A and the j'th point in B
    """

    AA = np.sum(A**2, axis=1, keepdims=True)
    BB = np.sum(B**2, axis=1, keepdims=True)
    inner = np.matmul(A,B.T)

    R = AA + (-2)*inner + BB.T

    return R

def big_square_dist(A,B):
    nA = A.shape[0]
    nB = B.shape[0]
    batch_size = np.maximum(nA,nB)
    res = np.zeros([nA,nB],dtype=np.float32)
    done = False
    while not done:
        try:
            num_batches_A = int(np.ceil(float(nA)/batch_size))
            num_batches_B = int(np.ceil(float(nB)/batch_size))
            A_to = 0
            for i in range(num_batches_A):
                A_from = A_to
                A_to += batch_size
                if A_to > nA:
                    A_to = nA
                B_to = 0
                for j in range(num_batches_B):
                    B_from = B_to
                    B_to += batch_size
                    if B_to > nB:
                        B_to = nB
                    cur_res = square_dist(A[A_from:A_to, :], B[B_from:B_to, :])
                    res[A_from:A_to, B_from:B_to] = cur_res
        except MemoryError:
            batch_size = int(batch_size / 2)
        else:
            done = True
    return res

def new_cdist(x1, x2, do_sqrt=True):
        x1 = x1.float()
        x2 = x2.float()
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True).float()
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True).float()
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add(x1_norm)
        if do_sqrt:
            res = res.clamp_min_(1e-30).sqrt_()
        return res

def PC_to_file(PC, filename):
    """
    Write point cloud to file

    :param PC: Point Cloud: an Nx3 array of real numbers
    :param filename: Name of file to write into
    """
    with open(filename, 'w') as fid:
        fid.write('%d\n' % PC.shape[0])
        for i in range(PC.shape[0]):
            fid.write("% 20.17f % 20.17f % 20.17f\n" % (PC[i, 0], PC[i, 1], PC[i, 2]))


def file_to_PC(filename):
    """
    Load point cloud from file

    :param filename: File to load from
    :return: Point Cloud: an Nx3 array of real numbers
    """
    with open(filename, 'r') as fid:
        text = fid.read().splitlines()

    num_points = int(text[0])
    PC = np.zeros([num_points, 3], dtype=float)

    for i, line in enumerate(text[1:]):
        PC[i, :] = np.array([float(x) for x in line.split()])

    return PC

def cdist(A, B, feature_mode='xyz'):
        from utils.subsampling import num_features
        if (A.shape[1] is not num_features[feature_mode]):
            A = A.T
        if (B.shape[1] is not num_features[feature_mode]):
            B = B.T
        if feature_mode in ['xyz', 'density']:
            C = dist(A, B)
        elif feature_mode == 'gaussian':
            C = mutual_mahalanobis(A, B)
        else:
            assert False, "Unknown feature_mode: " + feature_mode
        return C

def dist(A,B):
    """
    Measure Squared Euclidean Distance from every point in point-cloud A, to every point in point-cloud B
    :param A: Point Cloud: Nx3 Array of real numbers, each row represents one point in x,y,z space
    :param B: Point Cloud: Mx3 Array of real numbers
    :return:  NxM array, where element [i,j] is the squared distance between the i'th point in A and the j'th point in B
    """
    s = square_dist(A,B)
    s[s<0]=0
    return np.sqrt(s)

def old_dist_torch(A,B):
    """
    Measure Squared Euclidean Distance from every point in point-cloud A, to every point in point-cloud B
    :param A: Point Cloud: 3xN Array of real numbers, each row represents one point in x,y,z space
    :param B: Point Cloud: 3xM Array of real numbers
    :return:  NxM array, where element [i,j] is the squared distance between the i'th point in A and the j'th point in B
    """

    if(A.shape[1] is not 3):
        A = torch.transpose(A, dim0=0, dim1=1)
    if (B.shape[1] is not 3):
        B = torch.transpose(B, dim0=0, dim1=1)
    n_A = A.shape[0]
    n_B = B.shape[0]
    A3 = torch.reshape(A,[n_A,1,3]).double()
    B3 = torch.reshape(B, [1,n_B, 3]).double()
    dist = torch.mul((A3-B3),(A3-B3))
    R = torch.sqrt(torch.sum(dist, dim=2)+1e-30)
    return R

def cdist_torch(A,B, feature_mode='xyz', do_sqrt=True):
    from utils.subsampling import num_features
    if (A.shape[1] is not num_features[feature_mode]):
        A = torch.transpose(A, dim0=0, dim1=1)
    if (B.shape[1] is not num_features[feature_mode]):
        B = torch.transpose(B, dim0=0, dim1=1)
    A = A.double().contiguous()
    B = B.double().contiguous()
    if feature_mode in ['xyz', 'density']:
        C = new_cdist(A,B, do_sqrt)
    elif feature_mode == 'gaussian':
        C = mutual_mahalanobis_torch(A,B)
    else:
        assert False, "Unknown feature_mode: " + feature_mode
    return C

def one_directional_mahalanobis(A_xyz,A_inv_cov, B_xyz):

    res = np.zeros([A_xyz.shape[0], B_xyz.shape[0]])
    for i in range(A_xyz.shape[0]):
        cur_A_inv_cov = A_inv_cov[i, :, :]
        dif = B_xyz - A_xyz[i,:]
        left2 = np.matmul(dif,cur_A_inv_cov)
        right = left2 * dif
        res[i,:] = np.sum(right,axis=1).T
    return res

def mutual_mahalanobis(A,B):
    assert A.shape[1] == 12, "Wrong shape for A: " + str(A.shape)
    assert B.shape[1] == 12, "Wrong shape for B: " + str(B.shape)

    A_xyz = A[:, :3]
    A_inv_cov = A[:,3:].reshape([-1, 3, 3])
    B_xyz = B[:, :3]
    B_inv_cov = B[:, 3:].reshape([-1, 3, 3])

    A_to_B = one_directional_mahalanobis(A_xyz, A_inv_cov, B_xyz)
    B_to_A = one_directional_mahalanobis(B_xyz, B_inv_cov, A_xyz)

    C = np.sqrt(0.5*(A_to_B + B_to_A))

    return C

def one_directional_mahalanobis_torch(A_xyz,A_inv_cov, B_xyz):

    res_list = []
    for i in range(A_xyz.shape[0]):
        cur_A_inv_cov = A_inv_cov[i, :, :]
        dif = B_xyz - A_xyz[i,:]
        left2 = torch.matmul(dif,cur_A_inv_cov)
        right = left2 * dif
        cur_row = torch.sum(right,dim=1)
        res_list.append(cur_row.unsqueeze(0))
    res = torch.cat(res_list, dim=0)
    return res

def mutual_mahalanobis_torch(A,B):
    assert A.shape[1] == 12, "Wrong shape for A: " + str(A.shape)
    assert B.shape[1] == 12, "Wrong shape for B: " + str(B.shape)

    A_xyz = A[:, :3]
    A_inv_cov = A[:,3:].reshape([-1, 3, 3])
    B_xyz = B[:, :3]
    B_inv_cov = B[:, 3:].reshape([-1, 3, 3])

    A_to_B = one_directional_mahalanobis_torch(A_xyz,A_inv_cov, B_xyz)
    B_to_A = one_directional_mahalanobis_torch(B_xyz,B_inv_cov, A_xyz)

    C = torch.sqrt(0.5*( A_to_B + B_to_A ))

    return C


def dist_torch(A,B):
    if (A.shape[1] is not 3):
        A = torch.transpose(A, dim0=0, dim1=1)
    if (B.shape[1] is not 3):
        B = torch.transpose(B, dim0=0, dim1=1)
    A = A.double()
    B = B.double()
    AA = torch.unsqueeze(torch.sum(A**2, dim=1),1)
    BB = torch.unsqueeze(torch.sum(B**2, dim=1),1)
    PP = torch.transpose(BB, dim0=0, dim1=1)
    P = torch.transpose(B, dim0=0, dim1=1)
    inner = torch.matmul(A,P)
    R = AA + (-2)*inner + PP
    return torch.sqrt(R+1e-3)

def min_without_self_per_row(D):
    """
    Accepts a distance matrix between all points in a set. For each point,
    returns its distance from the closest point that is not itself.

    :param D: Distance matrix, where element [i,j] is the distance between i'th point in the set and the j'th point in the set. Should be symmetric with zeros on the diagonal.
    :return: vector of distances to nearest neighbor for each point.
    """
    E = D.copy()
    for i in range(E.shape[0]):
        E[i,i] = np.inf
    m = np.min(E,axis=1)
    return m

def representative_neighbor_dist(A, dist_matrix_as_input=True, allow_parallelism=True):
    if dist_matrix_as_input:
        return representative_neighbor_dist_from_dist_matrix(A)

    n_jobs = None
    if allow_parallelism and (A.shape[0] >= 5000): # for small clouds the overhead of a concurrent implementation is detrimental.
        n_jobs = 8

    NN = NearestNeighbors(n_neighbors=1,
                          radius=None,
                          algorithm='auto',
                          leaf_size=30,
                          metric='minkowski',
                          p=2,
                          metric_params=None,
                          n_jobs=n_jobs)
    NN.fit(A)
    d, i = NN.kneighbors(A, 2, return_distance=True)
    return np.median(d[:, -1])


def representative_neighbor_dist_from_dist_matrix(D):
    """
    Accepts a distance matrix between all points in a set,
    returns a number that is representative of the distances in this set.

    :param D: Distance matrix, where element [i,j] is the distance between i'th point in the set and the j'th point in the set. Should be symmetric with zeros on the diagonal.
    :return: The representative distance in this set
    """

    assert D.shape[0] == D.shape[1], "Input to representative_neighbor_dist should be a matrix of distances from a point cloud to itself"
    m = min_without_self_per_row(D)
    neighbor_dist = np.median(m)
    return neighbor_dist

def smooth_normals(PC, normals, smoothness=1):
    """

    :param PC:
    :param normals:
    :param smoothness:
    :return:
    """
    D = dist(PC,PC)
    neighbor_dist = representative_neighbor_dist(D)

    smoothness_sigma = smoothness * neighbor_dist
    G = np.exp(-((D**2)/(2*(smoothness_sigma**2))))
    W = G / np.sum(G,axis=1,keepdims=True)

    N = np.matmul(W, normals)
    return N


def crop_and_normalize(A, B):
    """
    Crop the bounding box that contains samples from both A and B,
    and normalize it to lie within the unit cube (without changing the aspect ratio).

    :param A: point cloud of size Nx3
    :param B: point cloud of size Mx3
    :return:
    """
    range = {'A': {}, 'B': {}, 'joint': {}}
    data = {'A': A, 'B': B}
    for data_name in ['A', 'B']:
        range[data_name]['min'] = np.min(data[data_name], axis=0)
        range[data_name]['max'] = np.max(data[data_name], axis=0)
    combining_function = {'min': np.maximum,
                          'max': np.minimum}  # seemingly inverted but correct, becuase we're looking for the interesection of ranges
    for cur in ['min', 'max']:
        range['joint'][cur] = combining_function[cur](
            range['A'][cur],
            range['B'][cur]
        )

    range['joint']['extreme'] = np.maximum(
        np.abs(range['joint']['min']),
        np.abs(range['joint']['max']))
    normalization_factor = np.max(range['joint']['extreme'])

    for data_name in ['A', 'B']:
        is_below = np.any(data[data_name] < range['joint']['min'], axis=1)
        is_above = np.any(data[data_name] > range['joint']['max'], axis=1)
        is_valid = ~is_below & ~is_above
        data[data_name] = data[data_name][is_valid, :]
        data[data_name] /= normalization_factor

    return data['A'], data['B']
