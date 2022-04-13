# Written by Amnon Drory (amnondrory@mail.tau.ac.il), Tel-Aviv University, 2021.
# Please cite the following paper if you use this code:
# - Amnon Drory, Shai Avidan, Raja Giryes, Stress Testing LiDAR Registration

# This code creates a challenging and balanced set of LiDAR scan pairs, 
# for training and testing point-cloud registration algorithms. 
# The input is a LiDAR dataset (e.g. KITTI, NuScenens, Apollo-Southbay), 
# which consists of sequences of point clouds recorded by a vehicle
# during a driving session. From these sequences, this algorithms 
# selects a subset of pairs which provides a diverse set of registration
# challenges. This set contains pairs with a variety of time offsets, 
# relative motions and source sequences. This is achieved by:
# 1. Selecting a set of candidate pairs from all source sequences, with 
#    diverse time offsets, and sufficient overlap between the two point clouds.  
# 2. Repeatedly:
#    a. sampling uniformy at random a 6DOF motion. 
#    b. Selecting a candidate pair whose relative motion is close to
#       the motion from (a). 

import numpy as np
import pickle as pickle
import os
import open3d as o3d
from matplotlib import pyplot as plt
import multiprocessing as mp
from scipy.spatial import cKDTree
from easydict import EasyDict as edict
from copy import deepcopy

from utils.tools_3d import motion_to_fields, apply_transformation
from utils.TicToc import *
try:
    from datasets.KITTI import KITTI_full, KITTI_balanced
except Exception as E:
    print("ignoring exception: " + repr(E))
try:
    from datasets.ApolloSouthbay import ApolloSouthbay_full, ApolloSouthbay_balanced
except Exception as E:
    print("ignoring exception: " + repr(E))
try:
    from datasets.NuScenes import NuScenes_full, NuScenes_balanced
except Exception as E:
    print("ignoring exception: " + repr(E))
try:
    from datasets.LyftLEVEL5 import LyftLEVEL5_full, LyftLEVEL5_balanced
except Exception as E:
    print("ignoring exception: " + repr(E))


default_config = edict({})

# refine_GT_for_candidate: if True, refine the motion estimation for each candidate (e.g. by performing symmetrical-icp)
default_config.refine_GT_for_candidate = False 

# refine_GT_for_entire_session: pre-process GT for the entire session by sequentially registering each consecutive pair of frames
default_config.refine_GT_for_entire_session = False

# refine_GT_Z_only: when refining GT, assume errors exist only in the up/down direction (this is the case in some NuScenes sequences)
default_config.refine_GT_Z_only = False 

# round_sizes_to_multiple: round up requested sizes of registration_sets to make them divisible by this value. Set to None for no rounding.
default_config.round_sizes_to_multiple = None

# output_dir: directories containing balanced registration sets and intermediate values will be placed here
default_config.output_dir = 'output/'

# candidates_per_sample: number of candidates is larger than requested number of samples by this factor
default_config.candidates_per_sample = 4

# max_spacing_in_sec: the maximum separation in seconds between source and target. Used in order to save some processing time. Can be set to np.inf.
default_config.max_spacing_in_sec = 60

# report_interval: how often to print status (for each concurrent process separately)
default_config.report_interval = 20

# minimum_overlap: what fraction of the source and target point clouds must be overlapping in a valid pair (at least)
default_config.minimum_overlap = 0.2

# overlap_measure: could be either one directional ('src_to_tgt') or 'symmetric'.
default_config.overlap_measure = 'symmetric' 


def ensure_path(name):
    name = name.replace('//','/')
    s = os.path.split(name)
    if '.' in s[-1]:
        dirname =  s[0]
    else:
        dirname = name
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    return name

class PerSessionCounter():
    """
    Auxiliary class that keeps track of how many candidates and samples were taken
    from each session, to help in achieving fair representation.
    """
    def __init__(self, sessions_list): 
        self.num_cands = {si: 0 for si in sessions_list}
        self.num_selected = {si: 0 for si in sessions_list}
    
    def record_num_cands(self, session_ind, num_cands):
        self.num_cands[session_ind] = num_cands

    def get_fullness(self, session_inds):
        res = [float(self.num_selected[si])/self.num_cands[si] for si in session_inds]
        return res

    def record_selected(self, session_ind):        
        self.num_selected[session_ind] += 1

class BalancedSetGenerator():
    """
    Main class
    """
    def __init__(self, DS_full, subset_sizes, subset_names, config=None):
        """
        DS_full - LiDAR point-cloud dataset, allows access to all point clouds and provides the ground-truth relative motion between any pair of point-clouds. 
        subset_sizes - the requested number of pairs to generate. This can be a list if multiple sets are to be created.
        subset_names - the names of the set/sets to create. This could be a string (e.g. 'train') or a list (e.g. ['train', 'validation']) of the same length as subset_sizes.
        config - see for example default_config. 
        """
        def to_list(x):
            return list(np.array([x]).flatten())

        if config is None:
            self.config = default_config
        else:
            self.config = config

        self.config.max_spacing = int(self.config.max_spacing_in_sec / DS_full.time_step)

        self.DS_full = DS_full        
        self.subset_sizes = to_list(subset_sizes)
        self.subset_names = to_list(subset_names)

        if self.config.round_sizes_to_multiple is not None:
            m = self.config.round_sizes_to_multiple
            self.subset_sizes = [int(np.ceil(s/m)*m) for s in self.subset_sizes]
            print(f"{self.DS_full.name} {self.DS_full.phase} adjusted requested size from {subset_sizes} to {self.subset_sizes}")

    def downsample(self, X, voxel_size):
        x = self.make_open3d_point_cloud(X)
        x_ = o3d.geometry.PointCloud.voxel_down_sample(x, voxel_size=voxel_size)
        X_ = np.array(x_.points)    
        return X_

    def NN(self, A,B): 
        # for each point in A, finds its nearest-neighbor in B.
        feat1tree = cKDTree(B)
        d, inds = feat1tree.query(A, k=1, workers=-1)
        return d, inds

    def overlap_fraction(self, A,B):
        """
        Estimates what part of the point clouds is overlapping. 

        Inputs:
            A - source point cloud
            B - target point cloud

        Outputs:
            overlap_frac - the fraction of points in A that have 
                           a corresponding point in B.
            overlap_frac_symmetric - the minimum between A->B overlap and B->A overlap.
        """

        # Downsample both clouds with voxel-grid filter, then find for each point
        # in A whether it has a partner in B that is at most a voxel away:        
        voxel_size = 1 # meters
        A_ = self.downsample(A, voxel_size)
        B_ = self.downsample(B, voxel_size)
        d, _ = self.NN(A_, B_)
        is_in_overlap = (d < np.sqrt(2)*voxel_size) # treating pointcloud as essentially planar, so length of diagonal of square voxel
        num_overlapping = is_in_overlap.sum()
        overlap_frac = float(num_overlapping) / A_.shape[0]
        overlap_frac_symmetric = min(overlap_frac, float(num_overlapping) / B_.shape[0])
        return overlap_frac, overlap_frac_symmetric

    def make_open3d_point_cloud(self, xyz):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        return pcd

    def calc_GT_overlap(self, A, B, GT_mot, return_both=False):
        """
        Align clouds according to GT_mot and meaure overlap

        Inputs:
            A - source point cloud
            B - target point cloud
            GT_mot - GT relative motion between clouds
            return_both - if True, returns two measures of overlap (source-to-target and symmetric).
                          Otherwise, returns the one defined by config.overlap_measure
        """
        A_corr = apply_transformation(GT_mot, A)        
        overlap_frac, overlap_frac_symmetric = self.overlap_fraction(A_corr,B)
        if return_both:
            return overlap_frac, overlap_frac_symmetric
        assert self.config.overlap_measure in ['src_to_tgt', 'symmetric'], "config.overlap_measure should be set to either 'src_to_tgt' or 'symmetric'"
        if self.config.overlap_measure == 'src_to_tgt':
            return overlap_frac
        if self.config.overlap_measure == 'symmetric':
            return overlap_frac_symmetric        
    
    def icp(self, A, B, voxel_size):
        xyz0 = A
        xyz1 = B
        xyz0_np = xyz0.astype(np.float64)
        xyz1_np = xyz1.astype(np.float64)
        pcd0 = self.make_open3d_point_cloud(xyz0_np) 
        pcd1 = self.make_open3d_point_cloud(xyz1_np)
        T = o3d.pipelines.registration.registration_icp(
            pcd0,
            pcd1, voxel_size * 2, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()).transformation        
        return T

    def refine_motion(self, GT_mot_orig, A, B, downsample=True, voxel_size=0.3):
        """
        Estimate a refined motion, using GT_mot_orig as an initial guess.

        Inputs:
            GT_mot_orig - a potentially imperfect estimate of the relative motion between
                          the point clouds
            A - source point cloud
            B - target point cloud

        Outputs:
            GT_mot - a refined estimate of the relative motion
        """
        if downsample:
            b = self.downsample(B, voxel_size)
            a = self.downsample(A, voxel_size)
        else:            
            b = B
            a = A
        
        if self.config.refine_GT_Z_only:
            GT_mot = self.refine_motion_Z_only(GT_mot_orig, a, b, voxel_size)
        else:
            a_corr = apply_transformation(GT_mot_orig, a)
            icp_mot = self.icp(a_corr, b, voxel_size)
            GT_mot = icp_mot @ GT_mot_orig
        return GT_mot

    def get_relative_motion(self, session_ind, i, j):
        if self.config.refine_GT_for_entire_session:
            position_i = self.positions[session_ind][i]
            position_j = self.positions[session_ind][j]
            mot = position_j @ np.linalg.inv(position_i)
            return mot
        else:
            return self.DS_full.get_relative_motion(session_ind, i, j)

    def refine_motion_Z_only(self, raw_mot, A, B, voxel_size):
        # Register point clouds to find refined motion in Z axis only (using a variant of IRLS).
        # Useful for datasets such as NuScenes where supplied "GT motion"
        # is incorrect in Z dimension. 
        A_ = apply_transformation(raw_mot, A)                
        B_ = B
        
        MAX_REPEATS = 10
        MIN_CHANGE = 10**-6
        dz = 0
        for repeat in range(MAX_REPEATS):
            d, ind = self.NN(A_, B_)
            A_set = A_
            B_set = B_[ind,:]
            xy_dist = np.sqrt(np.sum( (A_set[:,:2]-B_set[:,:2])**2, axis=1))
            
            is_valid = xy_dist <= voxel_size
            A_z = A_set[is_valid,2]
            B_z = B_set[is_valid,2]
            z_dist = A_z - B_z
            abs_z_dist = np.abs(z_dist)
            w = 1/abs_z_dist
            med_w = np.median(w)
            w[w>med_w]=med_w

            mean_z_dist = np.sum(w*z_dist)/np.sum(w)
            A_[:,2] -= mean_z_dist
            dz -= mean_z_dist
            if np.abs(mean_z_dist) < MIN_CHANGE:
                break
        
        res_mot = raw_mot
        res_mot[2,3] += dz

        return res_mot

    def refine_session_GT(self, session_ind):
        """
        Register each frame to the next to achieve improved ground-truth motions
        for the entire session. 
        """
        voxel_size = 0.3 # meters
        positions = {}
        offset = self.DS_full.indexing_from()     
        first_ind = offset   
        num_clouds = self.DS_full.session_length(session_ind)
        B = self.DS_full.load_PC(session_ind, first_ind)
        B_ = self.downsample(B, voxel_size)
        B_pos = np.eye(4, dtype=np.float64)
        positions[first_ind] = B_pos
        for i in range(first_ind+1, num_clouds+offset):            
            raw_mot = self.DS_full.get_relative_motion(session_ind, i-1, i)
            A_ = B_
            A_pos = B_pos
            B = self.DS_full.load_PC(session_ind, i)
            B_ = self.downsample(B, voxel_size)
            mot = self.refine_motion(raw_mot, A_, B_, downsample=False, voxel_size=voxel_size)
            B_pos = mot @ A_pos            
            positions[i] = B_pos
        
        if not hasattr(self, 'positions'):
            self.positions = {}
        self.positions[session_ind] = positions

    def find_farthest_overlapping_partner(self, session_ind, i, A, N, previous_spacing=None):        
        """
        Find the largest index j_max for which the overlap between 
        the clouds at index i and index j_max is significant enough.
        Overlap is assumed to be monotonically decreasing over time, 
        and it typically is, for reasonably short time spans.

        Inputs:
            session_ind
            i - index of source frame
            A - source point cloud
            N - number of frames in session
            previous_spacing - optional. the spacing between i' and j' that
                               was found for the previous source frame, i' 
        """
        
        # This function uses several constants, but is quite
        # insensitive to their specific values. 
        relative_overlap_error_close_enough_for_early_stop = 0.1
        initial_spacing = 50         
        close_enough_spacing = 5         

        # 1. try the same spacing as for the previous source frame.         
        if previous_spacing is not None:
            j = min(N-1, i + previous_spacing)            
            B = self.DS_full.load_PC(session_ind, j)
            GT_mot = self.get_relative_motion(session_ind, i, j)
            overlap = self.calc_GT_overlap(A, B, GT_mot)
            if (i < j) and np.abs((overlap / self.config.minimum_overlap) - 1) < relative_overlap_error_close_enough_for_early_stop:
                return j

        # 2. perform binary search. 
        if not previous_spacing is None:            
            initial_spacing = previous_spacing
        
        high = min(N-1, i+self.config.max_spacing)
        low = i+1 
        j = max(low+1, min(high-1, i + initial_spacing))        
        while (high - low) > close_enough_spacing:
            B = self.DS_full.load_PC(session_ind, j)
            GT_mot = self.get_relative_motion(session_ind, i, j)
            overlap = self.calc_GT_overlap(A, B, GT_mot)
            if overlap > self.config.minimum_overlap:
                low = j+1
            else:
                high = j-1
            j = int((low + high)/2)
        if (low-1)>i:
            return low-1
        else:
            return None

    def prep_candidate_record(self, session_ind, i, j, A):
        """
        Collect all data regarding the candidate point-cloud pair, this includes
        overlap-fraction and relative motion in different representations.        

        Inputs:
            session_ind
            i - index of source cloud
            j - index of target cloud
            A - source point cloud
        """
        B = self.DS_full.load_PC(session_ind, j)
        GT_mot = self.get_relative_motion(session_ind, i, j)
        if self.config.refine_GT_for_candidate:
            GT_mot = self.refine_motion(GT_mot, A, B)
        overlap_frac, overlap_frac_symmetric = self.calc_GT_overlap(A,B,GT_mot, return_both=True)
        # double check that overlap is valid:
        if (((self.config.overlap_measure == 'src_to_tgt') and (overlap_frac < self.config.minimum_overlap)) or
            ((self.config.overlap_measure == 'symmetric') and (overlap_frac_symmetric < self.config.minimum_overlap))) :
            return None
        mot_fields = motion_to_fields(GT_mot)
        record = np.array( [session_ind, i, j] + list(GT_mot.flatten()) + list(mot_fields) + [overlap_frac, overlap_frac_symmetric])
        return record
    
    def get_record_field_names(self):
        return 'session_ind src_ind tgt_ind mot0 mot1 mot2 mot3 mot4 mot5 mot6 mot7 mot8 mot9 mot10 mot11 mot12 mot13 mot14 mot15 trans_x trans_y trans_z roll pitch yaw overlap overlap_symmetric'

    def get_cands_list_file_name(self, session_ind, phase):
        candidates_dir = ensure_path(self.config.output_dir + '/candidates/' + self.DS_full.name + '/' + phase + '/')
        pickle_file =  candidates_dir + 'session_%d.pickle' % session_ind
        return pickle_file
    
    def create_candidate_set(self, session_ind):
        """
        Select a set of point-cloud pairs from the session, with diverse 
        time-offsets and overlap ratios. Only pairs whose overlap is 
        above a minimum are selected.
        """
        pickle_file = self.get_cands_list_file_name(session_ind, self.DS_full.phase)

        if os.path.isfile(pickle_file):
            print(f"Session {session_ind}: using candidates list from file {pickle_file}. Delete this file if you would like the candidate list to be recreated.")            
            return 
                
        if self.config.refine_GT_for_entire_session:
            self.refine_session_GT(session_ind)            

        total_samples_requested = np.sum(self.subset_sizes)
        total_num_available = self.DS_full.total_num_of_clouds()
        num_needed_cands = total_samples_requested * self.config.candidates_per_sample
        src_step = max(1, int(total_num_available / num_needed_cands))
        
        offset = self.DS_full.indexing_from()
        N = self.DS_full.session_length(session_ind)
        rows = []
        src_cands = range(offset, N+offset-1, src_step)
        prev_spacing = None
        for num, i in enumerate(src_cands): # for each source frame:
            tic()
            A = self.DS_full.load_PC(session_ind, i)
            
            # find group of all frames that have a valid overlap with source frame:
            max_j = self.find_farthest_overlapping_partner(session_ind, i, A, N, prev_spacing)            
            if max_j is None:
                continue
            prev_spacing = (max_j - i)
            j_cands = range(i+1,max_j+1) 

            # randomly select target frame from group (this ensures a diversity of time-offsets and overlap-ratios):
            j = np.random.choice(j_cands, 1)[0]
            
            record = self.prep_candidate_record(session_ind, i, j, A)
            if record is not None:
                rows.append(record)

            if (num % self.config.report_interval) == (session_ind % self.config.report_interval):
                toc('%d: iteration %d of %d' % (session_ind, num, len(src_cands)))

        res = np.vstack(rows)

        with open(pickle_file, 'wb') as fid:
            pickle.dump(res, fid)

    def to_points_in_hyper_cube(self, cands):
        """
        Represent each candidate as a point in a 6-dimensional unit hyper-cube
        """
        fields = cands[:,19:19+6] # [trans_x, trans_y, trans_z, roll, pitch, yaw] (see get_record_field_names)
        M = np.max(fields, axis=0, keepdims=True)
        m = np.min(fields, axis=0, keepdims=True)
        points = (fields - m)/(M-m)
        return points

    def select_sample(self, cands, points, P):
        """
        Try to select a sample for the registration set, by randomly generating
        a point in the hypercube, and if there is a sample close enough to
        this point, selecting it. If there are multiple samples that are
        close enough, also take into consideration fair representation
        for all sessions. 
        """
        THRESH = 0.1
        def calc_d(x,y):
            return np.sqrt(np.sum((x - y)**2, axis=1))
        
        # generate random point in hyper-cube:
        r = np.random.rand(6)

        # find the group of samples that are close enough to the point:        
        d = calc_d(r, points)
        is_close_enough = (d < THRESH)
        if is_close_enough.sum() == 0:   
            return None
        group_inds = is_close_enough.nonzero()[0]
        group = cands[group_inds,:]

        # find the samples in the group that belong to under-represented sessions:
        fullness = P.get_fullness(group[:,0].astype(int))        
        min_fullness = np.min( fullness )   
        is_rare = (fullness == min_fullness)
        rare_inds = group_inds[is_rare]
        
        # of these, select the one that is closest to the point:
        dd = calc_d(r, points[rare_inds, :])
        sel_ind = rare_inds[np.argmin(dd)]

        # record the selection and remove the selected from candidates:
        session_ind = int(cands[sel_ind,0])
        P.record_selected(session_ind)
        selected_mask = np.zeros([cands.shape[0]], dtype=bool)
        selected_mask[sel_ind] = True
        selected_samples = cands[selected_mask,:]
        cands = cands[~selected_mask,:]
        points = self.to_points_in_hyper_cube(cands) # re-scale the 6-dimensional hypercube to fit to the remaining candidates
        return selected_samples, cands, points, P

    def save_set(self, registration_set, subset_name):
        """
        Save registration set to file
        """
        o1 = np.argsort(registration_set[:,1])
        registration_set = registration_set[o1,:]
        o0 = np.argsort(registration_set[:,0], kind='stable')
        registration_set = registration_set[o0,:]
        output_file = ensure_path(self.config.output_dir + '/balanced_sets/' + self.DS_full.name + '/' + subset_name + '.txt')
        outfile = open(output_file, 'w')
        outfile.write(self.get_record_field_names() + '\n')
        for row in registration_set:
            s = "%d %d %d " % (row[0], row[1], row[2])
            for i in range(3,len(row)-1):
                s += '%.16f ' % row[i]
            s += '%.16f\n' % row[-1]
            outfile.write(s)
        outfile.close()

    def select_balanced_set(self):
        """
        From the candidates, select a balanced subset of the requested size.
        """       
        P = PerSessionCounter(self.DS_full.sessions_list)

        # load candidates from files:
        cands_list = []
        for session_ind in self.DS_full.sessions_list:
            pickle_file = self.get_cands_list_file_name(session_ind, self.DS_full.phase)
            with open(pickle_file, 'rb') as fid:
                cur_cands = pickle.load(fid)
            P.record_num_cands(session_ind, cur_cands.shape[0])
            cands_list.append(cur_cands)
        cands = np.vstack(cands_list)
        
        # iteratively select samples:
        points = self.to_points_in_hyper_cube(cands)
        registration_set = []
        num_selected = 0
        num_no_selection = 0
        it = 0
        total_samples_requested = np.sum(self.subset_sizes)
        while num_selected < total_samples_requested:
            it += 1
            if it % 100 == 0:
                print(f"{self.DS_full.name} {self.subset_names} # attempts with no selection = {num_no_selection}, # selected = {num_selected}")
            res = self.select_sample(cands, points, P)
            if res is None:
                num_no_selection += 1
                continue
            else:
                selected_samples, cands, points, P = res
                registration_set.append(selected_samples)
                num_selected += selected_samples.shape[0]

        # write out registration set to files:
        registration_set = np.vstack(registration_set)        
        for size, name in zip(self.subset_sizes, self.subset_names):
            cur_inds = np.random.choice(registration_set.shape[0], size, replace=False)
            is_cur = np.zeros(registration_set.shape[0], dtype=bool)
            is_cur[cur_inds] = True
            cur_registration_set = registration_set[is_cur,:]
            registration_set = registration_set[~is_cur,:]
            self.save_set(cur_registration_set, name)
    
    def create_set(self):
        """
        Main function.
        """
        
        # step 1: extract candidates from each sequence (driving session):                
        max_processes = 10 # The number of concurrent processes. Set this manually for best performance on your machine.
        used_processes = 0
        ps = []

        for session_ind in self.DS_full.sessions_list:
            if used_processes >= max_processes:
                for p in ps:
                    p.join()
                    used_processes -= 1
                ps = []

            p = mp.Process(target=self.create_candidate_set, args=(session_ind,))
            ps.append(p)
            p.start()
            used_processes += 1

        for p in ps:
            p.join()

        # step 2: create balanced set
        self.select_balanced_set()        


def analyze_registration_set(DS_balanced):
    """
    Display various statistic of the registration set.
    """

    title = DS_balanced.name + ' ' + DS_balanced.phase

    trans = DS_balanced.pairs[:,19:22]
    roll = DS_balanced.pairs[:,22]
    pitch = DS_balanced.pairs[:,23]
    yaw = DS_balanced.pairs[:,24]
    overlap = DS_balanced.pairs[:,25]
    symmetric_overlap = DS_balanced.pairs[:,25]
    d_time = (DS_balanced.pairs[:,2]-DS_balanced.pairs[:,1])*DS_balanced.time_step
    
    plt.figure()
    plt.subplot(2,3,1)
    xs = np.sqrt(np.sum(trans**2, axis=1))
    plt.hist(xs, bins=np.arange(np.min(xs),np.max(xs),5))
    plt.title('dist (m)')

    plt.subplot(2,3,6)    
    plt.hist(yaw, bins=np.arange(np.min(yaw),np.max(yaw),5))
    plt.title('yaw (deg)')

    plt.subplot(2,3,3)    
    plt.hist(symmetric_overlap, bins=20)
    plt.title('symmetric_overlap')

    plt.subplot(2,3,2)    
    plt.hist(d_time, bins=20)
    plt.title('time diff (s)')    

    plt.subplot(2,3,4)    
    plt.hist(roll, bins=20)
    plt.title('roll (deg)')    

    plt.subplot(2,3,5)    
    plt.hist(pitch, bins=20)
    plt.title('pitch (deg)')    

    plt.suptitle(title)
    plt.show()

def ApolloSouthbay():
    """
    Create balanced registration set for  Apollo-Southbay 
    """
    config = deepcopy(default_config)
    config.round_sizes_to_multiple = 96

    apollo_train = ApolloSouthbay_full('train')
    E = BalancedSetGenerator( apollo_train,
                            subset_sizes=[4000, 200],
                            subset_names=['train', 'validation'],
                            config=config)
    E.create_set()

    apollo_test = ApolloSouthbay_full('test')
    E = BalancedSetGenerator(apollo_test, 7000, 'test', config)
    E.create_set()    

    apollo_balanced_test = ApolloSouthbay_balanced('test')
    analyze_registration_set(apollo_balanced_test)        

def KITTI():
    """
    Create balanced registration set for  KITTI
    """
    config = deepcopy(default_config)
    config.round_sizes_to_multiple = 96
    config.refine_GT_for_candidate = True

    kitti_train = KITTI_full('train')
    E = BalancedSetGenerator(kitti_train, 1400, 'train', config)    
    E.create_set()

    kitti_validation = KITTI_full('validation')
    E = BalancedSetGenerator(kitti_validation, 200, 'validation', config)
    E.create_set()

    kitti_test = KITTI_full('test')
    E = BalancedSetGenerator(kitti_test, 600, 'test', config)
    E.create_set()    

    kitti_balanced_test = KITTI_balanced('test')
    analyze_registration_set(kitti_balanced_test)    

def LyftLEVEL5():
    """
    Create balanced registration set for  LyftLEVEL5
    """
    config = deepcopy(default_config)
    config.round_sizes_to_multiple = 96    
    config.refine_GT_for_candidate = True
    config.refine_GT_Z_only = True

    DS = LyftLEVEL5_full('train')
    E = BalancedSetGenerator( DS,
                            subset_sizes=[2000, 200],
                            subset_names=['train', 'validation'],
                            config=config)
    E.create_set()

    DS = LyftLEVEL5_full('test')
    E = BalancedSetGenerator(DS, 2500, 'test', config)
    E.create_set()    

    DS = LyftLEVEL5_balanced('test')
    analyze_registration_set(DS)    

def NuScenes():
    """
    Create balanced registration sets for NuScenes-Boston and NuScenes-Singapore
    """
    config = deepcopy(default_config)
    config.round_sizes_to_multiple = 96
    config.refine_GT_for_candidate = True
    config.refine_GT_Z_only = True

    DS = NuScenes_full('boston', 'train')
    E = BalancedSetGenerator(DS, 4000, 'train', config)    
    E.create_set()

    DS = NuScenes_full('boston', 'validation')
    E = BalancedSetGenerator(DS, 300, 'validation', config)    
    E.create_set()    

    DS = NuScenes_full('boston', 'test')
    E = BalancedSetGenerator(DS, 2500, 'test', config)    
    E.create_set()

    DS = NuScenes_full('singapore', 'train')
    E = BalancedSetGenerator(DS, 4000, 'train', config)    
    E.create_set()

    DS = NuScenes_full('singapore', 'validation')
    E = BalancedSetGenerator(DS, 300, 'validation', config)    
    E.create_set()    

    DS = NuScenes_full('singapore', 'test')
    E = BalancedSetGenerator(DS, 2500, 'test', config)    
    E.create_set()

    DS = NuScenes_balanced('boston', 'test')
    analyze_registration_set(DS)    

    DS = NuScenes_balanced('singapore', 'test')
    analyze_registration_set(DS)    


if __name__ == "__main__":
    ApolloSouthbay()
    NuScenes()


