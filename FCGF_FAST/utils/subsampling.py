import numpy as np
import open3d as o3d
import pandas as pd
from copy import deepcopy
import torch
import sys
import MinkowskiEngine as ME

from utils.PointCloudUtils import square_dist, cdist_torch
from utils.visualization import draw_registration_result, visualize_gaussian

num_features = {'xyz': 3, 'density': 4, 'gaussian': 12}

def sparse_quantize_torch(points, voxel_size, deterministic=False, return_indices=False):
    """
    Fast voxel-grid filtering. 

    params:
    ------
    points - tensor of size Nx3 containing pointcloud. 
    voxel_size - in meters
    deterministic - faster if set to False. 
    return_indices - if true, second output is indices that were selected in 'points'.
    """
    device = points.device
        
    if deterministic:
        Q_rot = points.cpu().contiguous()
        Q_rot_norm = Q_rot / voxel_size.cpu()
        assert(Q_rot_norm.dtype == torch.float32) # it's possible that other types may cause errors in ME.utils.sparse_quantize
        _ , sel = ME.utils.sparse_quantize(Q_rot_norm.detach(), return_index=True)
        Q_rot_sel = Q_rot[sel, :]
        res = Q_rot_sel.to(device)
    else:
        # GPU implementation. Much faster. issues:
        # 1. Not deterministic due to scatter_() function
        # 2. Selects a random point in each voxel, unlike ME.utils.sparse_quantize() which always 
        #    selects the same point (possibly the one nearest the center or center of mass?).            
        x = torch.floor(points / voxel_size).int()
        unique, inverse = torch.unique(x, dim=0, return_inverse=True) # based on https://github.com/rusty1s/pytorch_unique
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        sel = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
        res = points[sel, :]                        

    if return_indices:
        return res, sel
    else:
        return res
    




def calc_bin_inds(PC, n_bins, axis, mode):
    N = PC.shape[0]
    if "adaptive" in mode:
        inds = np.round(np.linspace(0, N, n_bins + 1)).astype(int)
        s = np.sort(PC[:, axis])
        thresh = s[inds[1:]-1]
    else: # "equally_spaced"
        thresh = np.linspace(np.min(PC[:,axis]), np.max(PC[:,axis]),  n_bins + 1)
        thresh = thresh[1:]

    bin_ind = np.zeros(N) + np.nan
    for i in range(n_bins):
        is_cur = (PC[:, axis] <= thresh[i]) & np.isnan(bin_ind)
        bin_ind[is_cur] = i

    assert np.sum(np.isnan(bin_ind)) == 0, "Error: not all samples were assigned to a bin"

    return bin_ind

def voxelGrid_filter_inner(PC, num_samples, mode, return_inds):
    """
    if return_inds=True, the output is a _selection_ of samples from
    the original cloud. Otherwise, its the _mean_ of all points in each
    cell
    """

    if "equal_nbins_per_axis" in mode:
        n_bins = int(np.ceil(num_samples ** (1. / 3)))
        n_bins_x = n_bins
        n_bins_y = n_bins
        n_bins_z = n_bins
    else:
        span = []
        for axis in range(3):
            span.append( np.max(PC[:,axis])-np.min(PC[:,axis]) )
        normalized_num_samples = num_samples * (span[0]**2 / (span[1]*span[2]))
        n_bins_x = int(np.ceil(normalized_num_samples ** (1. / 3)))
        n_bins_y = int(np.ceil(n_bins_x * (span[1]/span[0])))
        n_bins_z = int(np.ceil(n_bins_x * (span[2] / span[0])))
        assert (n_bins_x * n_bins_y * n_bins_z) >= num_samples, "Error"
    x_bin_inds = calc_bin_inds(PC, n_bins_x, 0, mode)
    y_bin_inds = calc_bin_inds(PC, n_bins_y, 1, mode)
    z_bin_inds = calc_bin_inds(PC, n_bins_z, 2, mode)

    if return_inds:
        data = np.hstack([x_bin_inds.reshape([-1,1]),
                          y_bin_inds.reshape([-1,1]),
                          z_bin_inds.reshape([-1,1]),
                          np.arange(PC.shape[0]).reshape([-1,1]),
                          PC])

        df = pd.DataFrame(data, columns=['x_ind', 'y_ind', 'z_ind', 'index', 'x', 'y', 'z'])

        selection = np.array(df.groupby(['x_ind', 'y_ind', 'z_ind']).first())
        indices = selection[:,0].astype(int)
        newPC = selection[:,1:]
        return newPC, indices

    else:
        data = np.hstack([x_bin_inds.reshape([-1, 1]),
                          y_bin_inds.reshape([-1, 1]),
                          z_bin_inds.reshape([-1, 1]),
                          PC])

        df = pd.DataFrame(data, columns=['x_ind', 'y_ind', 'z_ind', 'x', 'y', 'z'])
        newPC = np.array(df.groupby(['x_ind', 'y_ind', 'z_ind']).mean())

        return newPC

def voxelGrid_filter(PC, num_requested_samples, mode, return_inds):
    """
    Sub-sample a point cloud by defining a grid of voxels, and returning the average point in each one.

    :param PC: Nx3 array, point cloud, each row is a sample
    :param num_samples: numbver of requested samples
    :param mode: list of strings, can contain any of the following:
                 "exact_number" - return exactly num_requested_samples, otherwise may return more than requested number (but never less)
                 "equal_nbins_per_axis" - same number of bins for each axis (x,y,z). Otherwise the bins are cube shaped, and usually a different number of bins fits in each of the dimensions.
                 "adaptive" - smaller bins where there is more data. Otherwise, all bins are the same size.
    :param return_inds - if true, return both subset of points and their inds in the original array
    :return: newPC - a point cloud with approximately num_requested_samples
    """
    num_samples = num_requested_samples
    N = PC.shape[0]
    done = False
    MAX_ATTEMPTS = 40
    ACCELERATION_FACTOR = 2
    MAX_DIVERGENCE_TIME = 4
    TOLERANCE = 0.05
    rel_history = []
    newPC_history = []
    while not done:
        res = voxelGrid_filter_inner(PC, num_samples, mode, return_inds)
        if return_inds:
            new_N = res[0].shape[0]
        else:
            new_N = res.shape[0]
        newPC_history.append(res)
        relative_error_in_size = (new_N/float(num_requested_samples)) -1
        rel_history.append(relative_error_in_size)
        if (relative_error_in_size < 0) or (relative_error_in_size > TOLERANCE):
            best_ind = np.argmin(np.abs(rel_history))
            if (len(rel_history) - best_ind > MAX_DIVERGENCE_TIME) and (np.max(rel_history) > 0):
                    done = True
            else:
                num_samples = int(np.ceil(num_samples*float(num_requested_samples)/new_N))
                if (np.max(rel_history) < 0):
                    num_samples = int(ACCELERATION_FACTOR*num_samples)

        else:
            done = True

        if len(rel_history) >= MAX_ATTEMPTS:
            done = True

    if len(rel_history) >= MAX_ATTEMPTS:
        assert False, "voxelGrid_filter could not supply required number of samples"
        print("Error: voxelGrid_filter could not supply required number of samples, recovering")
        best_ind = np.argmax(rel_history)
        return newPC_history[best_ind]

    rel_history_above_only = np.array(rel_history)
    rel_history_above_only[rel_history_above_only<0] = np.inf
    best_ind_above = np.argmin(rel_history_above_only)

    res = newPC_history[best_ind_above]
    if 'exact_number' in mode:
        if return_inds:
            N0 = res[0].shape[0]
        else:
            N0 = res.shape[0]
        p = np.random.permutation(N0)
        keep = p[:num_requested_samples]

        if return_inds:
            new_res = (
                res[0][keep, ...],
                res[1][keep, ...]
            )
            res = new_res
        else:
            res = res[keep, ...]

    return res

def voxel_filter(pcd, N):
    # pcd is of open3d point cloud class
    if "numpy" in str(type(pcd)):
        tmp = o3d.geometry.PointCloud()
        tmp.points = o3d.utility.Vector3dVector(pcd)
        pcd = tmp
    K = np.shape(pcd.points)[0]
    vs = 1e-3
    while K>N:
        pcd = o3d.geometry.voxel_down_sample(pcd, voxel_size=vs)
        vs *= 2
        K = np.shape(pcd.points)[0]
    return pcd

def farthest_point_sampling(PC, K):
    assert False, "This function was supposed to be a faster implementation of fps, but ended up being considerably slower"
    MAX_PC_SIZE = 10000
    if PC.shape[0] > MAX_PC_SIZE:
        print("Memory constraints prevent running farthest_point_sampling() on such a big point cloud. \nIt will first be reduced to %d samples using the r_squared_normalized method." % MAX_PC_SIZE)
        PC = get_random_subset(PC, MAX_PC_SIZE, mode="r_squared_normalized")
    N = PC.shape[0]
    D = square_dist(PC, PC)
    for i in range(D.shape[0]):
        D[i,i] = np.inf
    inds = np.arange(N)
    is_selected = np.zeros(N, dtype=bool)
    first_selected = int(np.random.choice(inds))
    is_selected[first_selected] = True
    for _ in range(1,K):
        curD = D[np.ix_(is_selected,~is_selected)]
        min_dist = np.min(curD,axis=0)
        selection_ind_in_set_of_unselected = np.argmax(min_dist)
        cur_inds = inds[~is_selected]
        selection_ind = cur_inds[selection_ind_in_set_of_unselected]
        is_selected[selection_ind] = True
    return PC[is_selected,:]

def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)

def fps_from_given_pc(pts, K, given_pc, return_inds=False):
    """
    taken from https://github.com/orendv/learning_to_sample/blob/master/reconstruction/src/sample_net_point_net_ae.py
    :param self:
    :param pts:
    :param K:
    :param given_pc:
    :return:
    """
    farthest_pts = np.zeros((K, 3))
    keep_inds = np.zeros([K])
    t = given_pc.shape[0]
    farthest_pts[0:t,:] = given_pc

    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, t):
        distances = np.minimum(distances, calc_distances(farthest_pts[i,:], pts))

    for i in range(t, K):
        best_ind = np.argmax(distances)
        farthest_pts[i,:] = pts[best_ind,:]
        keep_inds[i] = best_ind
        distances = np.minimum(distances, calc_distances(farthest_pts[i,:], pts))
    if return_inds:
        return farthest_pts, keep_inds.astype(int)
    else:
        return farthest_pts

def fps_pure_torch(pts_torch, K):
    """
    performs Farthest Point Sampling entirely in torch.

    based on https://github.com/orendv/learning_to_sample/blob/master/reconstruction/src/sample_net_point_net_ae.py
    :param self:
    :param pts:
    :param K:
    :param given_pc:
    :return: indices of selected subset
    """
    first_ind = 0

    farthest_pts_torch = torch.zeros([K, 3], dtype=pts_torch.dtype, device=pts_torch.device)
    keep_inds = torch.zeros([K], dtype=torch.long, device=pts_torch.device)
    keep_inds[0] = first_ind
    farthest_pts_torch[0, :] = pts_torch[first_ind, :]

    min_distances = None
    for i in range(1, K):
        cur_dist = torch.cdist(farthest_pts_torch[None, i - 1, :], pts_torch)
        if min_distances is None:
            min_distances = cur_dist.squeeze()
        else:
            min_dist_candidates = torch.cat([min_distances.unsqueeze(0), cur_dist], dim=0)
            min_distances, _ = torch.min(min_dist_candidates, dim=0)
        best_ind = torch.argmax(min_distances)
        farthest_pts_torch[i, :] = pts_torch[best_ind, :]
        keep_inds[i] = best_ind

    return keep_inds

def fps_torch(pts, K, return_inds=False, submode='randomize'):
    """
    taken from https://github.com/orendv/learning_to_sample/blob/master/reconstruction/src/sample_net_point_net_ae.py
    :param self:
    :param pts:
    :param K:
    :param given_pc:
    :return:
    """
    num_points = pts.shape[0]
    if submode == 'randomize':
        rand_ind = np.random.randint(0, num_points)
    else:
        rand_ind = 0

    try:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        pts_torch = torch.from_numpy(pts).to(device)
        farthest_pts_torch = torch.zeros([K,3],dtype=pts_torch.dtype,device=pts_torch.device)
        keep_inds = torch.zeros([K],dtype=pts_torch.dtype,device=pts_torch.device)
        keep_inds[0] = rand_ind
        farthest_pts_torch[0,:] = pts_torch[rand_ind,:]

        min_distances = None
        for i in range(1, K):
            cur_dist = cdist_torch(farthest_pts_torch[None,i-1,:], pts_torch)
            if min_distances is None:
                min_distances = cur_dist.squeeze()
            else:
                min_dist_candidates = torch.cat([min_distances.unsqueeze(0),cur_dist], dim=0)
                min_distances,_ = torch.min(min_dist_candidates, dim=0)
            best_ind = torch.argmax(min_distances)
            farthest_pts_torch[i, :] = pts_torch[best_ind,:]
            keep_inds[i] = best_ind

        farthest_pts = farthest_pts_torch.detach().cpu().numpy()
        if return_inds:
            return farthest_pts, keep_inds.detach().cpu().numpy().astype(int)
        else:
            return farthest_pts
    except RuntimeError:
        return fps_from_given_pc(pts, K, pts[rand_ind:(rand_ind+1), :], return_inds)

def unify_bins(R_ind, required_samples_per_bin, starts, centers):
    """
    Accepts a division of a point cloud into (cylindrical) bins. For each bin,
    there's a required number of samples, and an available number. The basic assumption is
    that the available number decreases with radius. Therefore, the far (high-radius) bins
    probably have requirements that cannot be satisfied. This function joins multiple bins
    into one joint bin, going from far to near. It stops when the joint bin has enough sample
    to satisfy joint requirement.

    :param R_ind: For each sample, the bin it belongs to.
    :param required_samples_per_bin:
    :param starts: bin starts
    :param centers: bin centers
    :return: modified version of inputs.
    """
    num_samples = required_samples_per_bin.sum()
    n_bins = len(required_samples_per_bin)
    non_empty_bin_inds, available_samples_per_bin_raw = np.unique(R_ind, return_counts=True)
    min_non_empty_bin_ind = int(non_empty_bin_inds[0])
    available_samples_per_bin = np.zeros(len(required_samples_per_bin),dtype=int)
    for ind, num in zip(non_empty_bin_inds, available_samples_per_bin_raw):
        available_samples_per_bin[int(ind)]=num

    # Handle case where some of the closest bins are empty:
    num_impossible = required_samples_per_bin[:min_non_empty_bin_ind].sum()
    required_samples_per_bin[:min_non_empty_bin_ind] = 0
    required_samples_per_bin[-1] += num_impossible # hack, send the impossible requirements from the center, to the farthest edge. We expect these to be a small number, so this doesn't matter much.

    # unify bins:
    for bin_ind in range(n_bins-1,min_non_empty_bin_ind,-1):
        if available_samples_per_bin[bin_ind] < required_samples_per_bin[bin_ind]:
            # unify with previous bin
            R_ind[R_ind==bin_ind] = (bin_ind-1)
            required_samples_per_bin[bin_ind-1] += required_samples_per_bin[bin_ind]
            required_samples_per_bin[bin_ind] = 0
            available_samples_per_bin[bin_ind-1] += available_samples_per_bin[bin_ind]
            available_samples_per_bin[bin_ind] = 0

    # handle special case where the closest non-empty bin has too many samples required:
    av = available_samples_per_bin
    req = required_samples_per_bin
    first = min_non_empty_bin_ind
    if av[first] < req[first]:
        extra = req[first] - av[first]
        req[first] = av[first]
        # send extra requirements to farthest bin that can satisfy them
        for bin_ind in range(n_bins-1,first-1,-1):
            if av[bin_ind] - req[bin_ind] > extra:
                req[bin_ind] += extra
                break

    assert(required_samples_per_bin.sum() == num_samples), "Unexpected, unification changes overall number of samples"
    assert (available_samples_per_bin>=required_samples_per_bin).all(), "Error: some bins cannot satisfy requirements"
    starts = None # no longer correct after unification
    centers = None # no longer correct after unification

    return R_ind, required_samples_per_bin, starts, centers

def measure_occupancy(PC, R_ind, n_bins, n_subbins_theta=None):
    """
    For each cylindrical bin, measures how much of it is occupied by samples, as a fraction
    of the the 360 degree range of angles.

    :param PC: point cloud
    :param R_ind: which cylindrical bin each sample belongs to
    :param n_bins: number of cylindrical bins
    :param n_subbins_theta: number of angular sub-bins in each cylindrical bin
    :return: for each cyclindrical bin, the fraction of the angle range actually occupied by samples.
    """

    #divide samples into angular sub-bins:
    if n_subbins_theta is None:
        n_subbins_theta = 50
    theta = np.arctan2(PC[:,0], PC[:,1])
    theta_subbin_width = float(np.max(theta) - np.min(theta)) / n_subbins_theta
    theta_ind = np.floor((theta - np.min(theta))/theta_subbin_width)
    theta_ind[theta_ind == n_subbins_theta] == n_subbins_theta - 1

    # measure occupancy (as fraction of angular sub-bins that are not empty)
    occupancy = []
    for cur_R_ind in range(n_bins):
        is_cur_R_bin = (R_ind == cur_R_ind)
        theta_inds_in_cur_R_bin = theta_ind[is_cur_R_bin]
        num_different_theta_inds_in_cur_R_ind = len(np.unique(theta_inds_in_cur_R_bin))
        cur_occupancy = float(num_different_theta_inds_in_cur_R_ind) / n_subbins_theta
        occupancy.append(cur_occupancy)

    return occupancy

def get_radius_normalized_subset(PC, num_samples, mode, submode, n_bins):
    # divide R into bins, within each bin get a random subset of points.
    # the number of points per bin is proportional to the size of the bin.
    # the bins are either cylindrical rings or sphere shells.

    if submode is None:
        submode = "farthest"
    points_per_bin = 50
    if n_bins is None:
        n_bins = int(np.floor(float(num_samples) / points_per_bin))

    if mode == "r_normalized":
        # cylindrical rings (ignore z axis)
        R = np.sqrt(np.sum(PC[:, :2] ** 2, axis=1))
    elif mode == "r_squared_normalized":
        # spherical shells
        R = np.sqrt(np.sum(PC ** 2, axis=1))
    bin_width = (np.max(R) / n_bins)
    R_ind = np.minimum(np.floor(R / bin_width), n_bins - 1)
    occupancy = measure_occupancy(PC, R_ind, n_bins)
    starts = np.arange(n_bins) * bin_width
    centers = starts + 0.5 * bin_width
    if mode == "r_normalized":
        volume = centers
    elif mode == "r_squared_normalized":
        volume = centers ** 2
    volume *= occupancy

    relative_volume = volume / volume.sum()
    required_samples_per_bin = np.floor(relative_volume * num_samples)
    extra_samples = int(num_samples - required_samples_per_bin.sum())
    additions = np.zeros(n_bins)
    additions[:extra_samples] = 1
    np.random.shuffle(additions)
    required_samples_per_bin += additions

    R_ind, required_samples_per_bin, starts, centers = unify_bins(R_ind, required_samples_per_bin, starts, centers)

    new_PC = []
    for cur_bin_ind, cur_num_required in enumerate(required_samples_per_bin):
        if cur_num_required == 0:
            continue
        is_in_bin = (R_ind == cur_bin_ind)
        curPC = PC[is_in_bin, :]
        assert curPC.shape[0] >= cur_num_required, "error: more samples are required than actually exist in bin"

        cur_selected_PC = get_random_subset(curPC, int(cur_num_required), mode=submode)
        new_PC.append(cur_selected_PC)
    result = np.vstack(new_PC)

    assert result.shape[0] == num_samples, "Failed to subsample the required number of samples"

    return result


def analyze_fps(pts):
    num_points = pts.shape[0]
    D = square_dist(pts,pts)
    dist_to_prev = np.zeros(num_points)
    for i in range(1,num_points):
        dist_to_prev[i-1] = np.min(D[i,:i])
    return dist_to_prev

def get_random_subset(PC, num_samples, mode="farthest", n_bins=None, submode=None, local_statistics=None, stat_weights=None, allow_overask=False, return_inds=False):
    """
    Subsample a point cloud, using either of various methods

    :param PC:
    :param num_samples:
    :param mode:
    :param n_bins:
    :param submode: Relevant for the "r_normalized" and "r_squared_normalized" methods.
    :return:
    """
    if num_samples > PC.shape[0]:
        if allow_overask:
            if return_inds:
                return PC, np.arange(PC.shape[0])
            else:
                return PC
        else:
            assert False, "Error: requesting more samples than there are"

    if PC.shape[0] == num_samples:
        result = PC
    if mode == "uniform":
        inds = np.random.permutation(PC.shape[0])[:num_samples]
        if return_inds:
            result = PC[inds, :], inds
        else:
            result = PC[inds, :]
    elif mode == "farthest":
        result = fps_torch(PC, num_samples, return_inds, submode)
    elif mode == "high":
        result = subsample_high_points(PC, num_samples)
    elif "voxel" in mode:
        if submode is None:
            submode = ["equal_nbins_per_axis"]

        # The voxelGrid subsampling algorithm has no randomality.
        # we force it to have some by randomly removing a small subset of the points

        keep_fraction = 0.9
        num_keep = int(PC.shape[0]*keep_fraction)
        if num_samples < num_keep:
            PC = get_random_subset(PC, num_keep, mode="uniform")
        result = voxelGrid_filter(PC, num_samples, submode, return_inds)

    elif "normalized" in mode:
        result = get_radius_normalized_subset(PC, num_samples, mode, submode, n_bins)
    else:
        assert False, "unknown mode"

    if local_statistics is not None:
        features = get_local_statistics(result, PC, mode=local_statistics)
        if stat_weights is not None:
            W = np.reshape(stat_weights, [1,features.shape[1]])
            features *= W
        result = np.hstack([result,features])
        print("Feature STD: " + str(np.std(result, axis=0)))

    return result

def get_local_statistics(A_sub, A, mode, radius=None, radius_factor=None):
    VISUALIZE = False

    if radius == None:
        if radius_factor is None:
            radius_factor = 0.04
            if mode == "gaussian":
                radius_factor = 0.15 # large enough so covariance can be calculated for all points
        radius = radius_factor * np.max(np.max(A,axis=0)-np.min(A,axis=0))
    d = square_dist(A_sub, A)
    is_close = d <= radius**2

    if mode == "density":
        density = np.sum(is_close,axis=1).astype(float)
        density /= np.sum(density)
        stats = density.reshape([-1,1])

    elif mode == "gaussian":
        stats = []
        for i in range(A_sub.shape[0]):
            if VISUALIZE:
                draw_registration_result(A_sub,A[is_close[i,:],:])

            if is_close[i,:].sum()<4: # not enough neighbors to fit a covariance matrix
                assert False, "not enough samples in neighborhood!"
                stats.append(np.array([[np.NaN]*9]))
            else:
                neighbors = A[is_close[i,:],:]
                norm_neighbors = neighbors - A_sub[i,:]
                covariance = np.cov(norm_neighbors,rowvar=False)
                inv_cov = np.linalg.inv(covariance)
                stats.append(inv_cov.reshape([1,-1]))
                if VISUALIZE:
                    visualize_gaussian(A_sub, np.hstack([A_sub[i,:] , stats[-1][0] ]))

        stats = np.vstack(stats)

    else:
        assert False, "Unknown mode " + mode

    return stats

def subsample_fraction(PC, fraction):
    N = PC.shape[0]
    subset_size = int(np.round(N * fraction))
    inds = np.random.permutation(N)[:subset_size]
    return PC[inds,:]


def keep_closest(PC, max_dist):
    R = np.sqrt(np.sum(PC ** 2, axis=1))
    return PC[R <= max_dist, :]


def fit_plane(PC):
    xy1 = deepcopy(PC)
    xy1[:, 2] = 1
    z = PC[:, 2]
    abc, _, _, _ = np.linalg.lstsq(xy1, z, rcond=None)
    return abc


def is_on_plane(PC, abc, thickness):
    all_xy1 = deepcopy(PC)
    all_xy1[:, 2] = 1
    predicted_road_z = np.matmul(all_xy1, abc.reshape([-1, 1])).flatten()
    res = np.abs(PC[:, 2] - predicted_road_z) <= thickness
    return res


def improved_road_removal__take_2_failed(PC):
    # Idea: divide xy plane into blocks. For each block separately:
    # find most common heihgt of points. first guess is a plane parallel
    # to the xy plane at this height. Next, take all point near this plane,
    # and fit a plane to them.

    road_thickness = 0.5  # meters
    num_cols_or_rows = 5
    PC_X_wid = np.max(PC[:, 0]) - np.min(PC[:, 0])
    PC_Y_wid = np.max(PC[:, 1]) - np.min(PC[:, 1])
    X_block_wid = PC_X_wid / num_cols_or_rows
    Y_block_wid = PC_Y_wid / num_cols_or_rows
    X_ind = np.round(PC[:, 0] / X_block_wid)
    Y_ind = np.round(PC[:, 1] / Y_block_wid)
    radius = np.maximum(np.abs(X_ind), np.abs(Y_ind))

    max_radius = int(np.max(radius))

    new_PC = []
    finished_blocks = []
    for cur_radius in range(0, max_radius + 1):
        if cur_radius == 0:
            cur_X = 0
            cur_Y = 0
            is_cur_block = (X_ind == cur_X) & (Y_ind == cur_Y)
            curPC = PC[is_cur_block, :]
            # draw_registration_result(PC,curPC)

            count, bin_edges = np.histogram(curPC[:, 2], 100)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ind_of_most_frequent = np.argmax(count)
            road_z = bin_centers[ind_of_most_frequent]
            raw_abc = np.array([0, 0, road_z])
            is_road = is_on_plane(curPC, raw_abc, road_thickness)
            raw_road_points = curPC[is_road, :]
            abc = fit_plane(raw_road_points)
            finished_blocks.append({"X": cur_X, "Y": cur_Y, "plane": abc})
            is_road_final = is_on_plane(curPC, abc, road_thickness)
            new_PC.append(curPC[~is_road_final, :])
        else:
            cands_in_cur_radius = []
            for cur_X in range(-num_cols_or_rows, num_cols_or_rows + 1):
                for cur_Y in range(-num_cols_or_rows, num_cols_or_rows + 1):
                    if np.maximum(np.abs(cur_X), np.abs(cur_Y)) == cur_radius:
                        cands_in_cur_radius.append([cur_X, cur_Y])
            while len(cands_in_cur_radius) > 0:
                # select next candidate:
                num_neighbors = []
                for cur_X, cur_Y in cands_in_cur_radius:
                    nei_count = 0
                    for fin_blk in finished_blocks:
                        if (np.abs(fin_blk["X"] - cur_X) <= 1) and (np.abs(fin_blk["Y"] - cur_Y) <= 1):
                            nei_count += 1
                    num_neighbors.append(nei_count)
                best_cand_ind = np.argmax(num_neighbors)
                cur_X, cur_Y = cands_in_cur_radius[best_cand_ind]
                cands_in_cur_radius.pop(best_cand_ind)

                is_cur_block = (X_ind == cur_X) & (Y_ind == cur_Y)
                curPC = PC[is_cur_block, :]
                # draw_registration_result(PC,curPC)

                count, bin_edges = np.histogram(curPC[:, 2], 100)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                ind_of_most_frequent = np.argmax(count)
                road_z = bin_centers[ind_of_most_frequent]
                raw_abc = np.array([0, 0, road_z])
                is_road = is_on_plane(curPC, raw_abc, road_thickness)
                raw_road_points = curPC[is_road, :]
                abc = fit_plane(raw_road_points)
                finished_blocks.append({"X": cur_X, "Y": cur_Y, "plane": abc})
                is_road_final = is_on_plane(curPC, abc, road_thickness)
                draw_registration_result(PC, curPC[is_road_final, :])
                new_PC.append(curPC[~is_road_final, :])

    new_PC = np.vstack(new_PC)
    draw_registration_result(PC, new_PC)
    return (new_PC)


def improved_road_removal__take_1_failed(PC):
    # Idea: divide xy plane into blocks. Start from central block (containing 0,0).
    # find local road plane. Now traverse blocks from inside out. The first guess for
    # road points in a block, are points that belong to the road plane of the neighboring blocks.
    # take these points, fit a plane to get the local road plane.
    # Failure mode: points inside trees, mid height to the ground, ended up being considered road.

    road_thickness = 0.5  # meters
    num_cols_or_rows = 5
    PC_X_wid = np.max(PC[:, 0]) - np.min(PC[:, 0])
    PC_Y_wid = np.max(PC[:, 1]) - np.min(PC[:, 1])
    X_block_wid = PC_X_wid / num_cols_or_rows
    Y_block_wid = PC_Y_wid / num_cols_or_rows
    X_ind = np.round(PC[:, 0] / X_block_wid)
    Y_ind = np.round(PC[:, 1] / Y_block_wid)
    radius = np.maximum(np.abs(X_ind), np.abs(Y_ind))

    max_radius = int(np.max(radius))

    new_PC = []
    finished_blocks = []
    for cur_radius in range(0, max_radius + 1):
        if cur_radius == 0:
            cur_X = 0
            cur_Y = 0
            is_cur_block = (X_ind == cur_X) & (Y_ind == cur_Y)
            curPC = PC[is_cur_block, :]
            # draw_registration_result(PC,curPC)

            count, bin_edges = np.histogram(curPC[:, 2], 100)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ind_of_most_frequent = np.argmax(count)
            road_z = bin_centers[ind_of_most_frequent]
            raw_abc = np.array([0, 0, road_z])
            is_road = is_on_plane(curPC, raw_abc, road_thickness)
            raw_road_points = curPC[is_road, :]
            abc = fit_plane(raw_road_points)
            finished_blocks.append({"X": cur_X, "Y": cur_Y, "plane": abc})
            is_road_final = is_on_plane(curPC, abc, road_thickness)
            new_PC.append(curPC[~is_road_final, :])
        else:
            cands_in_cur_radius = []
            for cur_X in range(-num_cols_or_rows, num_cols_or_rows + 1):
                for cur_Y in range(-num_cols_or_rows, num_cols_or_rows + 1):
                    if np.maximum(np.abs(cur_X), np.abs(cur_Y)) == cur_radius:
                        cands_in_cur_radius.append([cur_X, cur_Y])
            while len(cands_in_cur_radius) > 0:
                # select next candidate:
                num_neighbors = []
                for cur_X, cur_Y in cands_in_cur_radius:
                    nei_count = 0
                    for fin_blk in finished_blocks:
                        if (np.abs(fin_blk["X"] - cur_X) <= 1) and (np.abs(fin_blk["Y"] - cur_Y) <= 1):
                            nei_count += 1
                    num_neighbors.append(nei_count)
                best_cand_ind = np.argmax(num_neighbors)
                cur_X, cur_Y = cands_in_cur_radius[best_cand_ind]
                cands_in_cur_radius.pop(best_cand_ind)

                is_cur_block = (X_ind == cur_X) & (Y_ind == cur_Y)
                curPC = PC[is_cur_block, :]
                # draw_registration_result(PC, curPC)
                is_raw_road_point = None
                for fin_block in finished_blocks:
                    if (np.abs(cur_X - fin_block["X"]) > 1) or (np.abs(cur_Y - fin_block["Y"]) > 1):
                        continue
                    cur_is_road = is_on_plane(curPC, fin_block["plane"], road_thickness)
                    if is_raw_road_point is None:
                        is_raw_road_point = cur_is_road
                    else:
                        is_raw_road_point |= cur_is_road
                raw_road_points = curPC[is_raw_road_point, :]
                abc = fit_plane(raw_road_points)
                # draw_registration_result(PC, raw_road_points)
                finished_blocks.append({"X": cur_X, "Y": cur_Y, "plane": abc})
                is_road_final = is_on_plane(curPC, abc, road_thickness)
                new_PC.append(curPC[~is_road_final, :])

    new_PC = np.vstack(new_PC)
    draw_registration_result(PC, new_PC)
    return (new_PC)

def improved_road_removal__take_3(PC):
    """
    Divide xy-plane into blocks. At each corner of each block find the lowest point close to it.
    For each block, use these low corner points to fit a plane. That is the initial guess for the
    road plane in this block.

    :param PC:
    :return:
    """
    road_thickness = 0.5  # meters
    num_cols_or_rows_for_block_width_calculation = 20
    num_cols_or_rows = num_cols_or_rows_for_block_width_calculation + 1
    PC_X_wid = np.max(PC[:, 0]) - np.min(PC[:, 0])
    PC_Y_wid = np.max(PC[:, 1]) - np.min(PC[:, 1])
    X_block_wid = PC_X_wid / num_cols_or_rows_for_block_width_calculation
    Y_block_wid = PC_Y_wid / num_cols_or_rows_for_block_width_calculation
    X_ind = np.round(PC[:, 0] / X_block_wid)
    Y_ind = np.round(PC[:, 1] / Y_block_wid)
    X_ind_half = np.ceil(PC[:, 0]/ X_block_wid)
    Y_ind_half = np.ceil(PC[:, 1] / Y_block_wid)

    data = np.hstack([
        X_ind_half.reshape([-1,1]),
        Y_ind_half.reshape([-1,1]),
        PC
    ])


    df = pd.DataFrame(data, columns=['X_ind_half', 'Y_ind_half', 'x','y','z'])
    grouped = df.groupby(['X_ind_half', 'Y_ind_half']).min()
    min_X_ind_half = np.min(X_ind_half)
    min_Y_ind_half = np.min(Y_ind_half)
    min_X_ind = int(np.min(X_ind))
    min_Y_ind = int(np.min(Y_ind))
    max_X_ind = int(np.max(X_ind))
    max_Y_ind = int(np.max(Y_ind))

    local_lowest_point = np.nan + np.zeros([num_cols_or_rows + 1, num_cols_or_rows + 1])
    local_lowest_point_vals = np.array(grouped)
    for row_ind, row in enumerate(grouped.iterrows()):
        inds = np.array(row[0]) - np.array([min_X_ind_half, min_Y_ind_half])
        local_lowest_point[int(inds[1]),int(inds[0])] = row_ind

    non_road_points = []
    for y in range(min_X_ind,max_X_ind+1):
        for x in range(min_Y_ind,max_Y_ind+1):
            is_cur = (X_ind == x) & (Y_ind == y)
            curPC = PC[is_cur,:]
            if curPC.size == 0:
                continue

            corner_points = []
            for dy in [0,1]:
                for dx in [0,1]:
                    inds = [int(y+dy-min_Y_ind_half), int(x+dx-min_X_ind_half)]
                    data_ind = local_lowest_point[inds[0],inds[1]]
                    if not np.isnan(data_ind):
                        corner_points.append(local_lowest_point_vals[int(data_ind),:])
            if len(corner_points)<3:
                non_road_points.append(curPC)
                continue
            corner_points = np.vstack(corner_points)
            raw_abc = fit_plane(corner_points)
            is_raw_road = is_on_plane(curPC, raw_abc, road_thickness)
            raw_road_points = curPC[is_raw_road, :]
            abc = fit_plane(raw_road_points)
            is_road = is_on_plane(curPC, abc, road_thickness)
            cur_non_road_points = curPC[~is_road,:]
            non_road_points.append(cur_non_road_points)
    resPC = np.vstack(non_road_points)
    return resPC

def remove_road(PC):
    mode = "plane"  # "constant_height"
    local_PC = keep_closest(PC, 10)
    count, bin_edges = np.histogram(local_PC[:, 2], 100)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ind_of_most_frequent = np.argmax(count)
    road_z = bin_centers[ind_of_most_frequent]
    road_thickness = 0.5  # meters
    if mode == "constant_height":
        is_road = np.abs(PC[:, 2] - road_z) <= road_thickness
    elif mode == "plane":
        raw_is_road = np.abs(local_PC[:, 2] - road_z) <= road_thickness
        raw_road_points = local_PC[raw_is_road, :]
        xy1 = deepcopy(raw_road_points)
        xy1[:, 2] = 1
        z = raw_road_points[:, 2]
        abc, _, _, _ = np.linalg.lstsq(xy1, z, rcond=None)
        all_xy1 = deepcopy(PC)
        all_xy1[:, 2] = 1
        predicted_road_z = np.matmul(all_xy1, abc.reshape([-1, 1])).flatten()
        is_road = np.abs(PC[:, 2] - predicted_road_z) <= road_thickness
    else:
        assert False, "unknown mode"

    keep_mask = ~is_road

    return PC[keep_mask, :], keep_mask

def subsample_high_points(A, num_samples, quantile=None):
    if quantile is None:
        quantile = 0.7

    adjusted_num_samples = int(num_samples * (1/(1-quantile) +1)) # the +1 is too make sure we end up with enough points
    A_FPS = get_random_subset(A, adjusted_num_samples, "farthest")
    A_tops = remove_local_low_points_K_neighbors(A_FPS, quantile=quantile)

    if A_tops.shape[0] > num_samples:
        p = np.random.permutation(A_tops.shape[0])
        inds = p[:num_samples]
        res = A_tops[inds, :]
    else:
        res = A_tops

    return res

def remove_local_low_points(PC, quantile=None):
    return remove_local_low_points_K_neighbors(PC, quantile=None)

def remove_local_low_points_K_neighbors(PC, quantile=None):
    from utils.experiment_utils import get_quantiles
    if quantile is None:
        quantile = 0.7

    N = PC.shape[0]
    D = square_dist(PC,PC)
    NEI_SIZE = 20

    keep = np.zeros(N, dtype=bool)
    for i in range(N):
        s = np.argsort(D[i,:])
        is_local_neighbor = np.zeros(N, dtype=bool)
        is_local_neighbor[s[:NEI_SIZE]] = True
        closest_neighbors_heights = PC[is_local_neighbor,2]
        self_height = PC[i,2]
        thresh = get_quantiles(closest_neighbors_heights, quantile)
        if self_height > thresh:
            keep[i] = True

    return PC[keep,:]

def remove_local_low_points_radius(PC, quantile=None, radius=None, radius_factor=None):
    from utils.experiment_utils import get_quantiles

    if quantile is None:
        quantile = 0.7

    if radius == None:
        if radius_factor is None:
            radius_factor = 0.02
        radius = radius_factor * np.max(np.max(PC,axis=0)-np.min(PC,axis=0))
    D = square_dist(PC,PC)
    is_close = D <= radius**2

    N = PC.shape[0]

    keep = np.zeros(N, dtype=bool)
    for i in range(N):
        is_local_neighbor = is_close[i,:]
        closest_neighbors_heights = PC[is_local_neighbor,2]
        self_height = PC[i,2]
        thresh = get_quantiles(closest_neighbors_heights, quantile)
        if self_height > thresh:
            keep[i] = True

    return PC[keep,:]
