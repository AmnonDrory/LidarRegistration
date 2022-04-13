# Written by Amnon Drory (amnondrory@mail.tau.ac.il), Tel-Aviv University, 2021.
# Please cite the following paper if you use this code:
# - Amnon Drory, Shai Avidan, Raja Giryes, Stress Testing LiDAR Registration

import pandas as pd
import numpy as np
import pykitti

from datasets.paths import paths
from utils.visualization import draw_multiple_clouds

ORIGINAL_DATASET_PATH = paths['KITTI']
BALANCED_SETS_PATH = paths['balanced_sets']

class KITTI_utils():
    def __init__(self):
        self.dataset = None

    def init_session(self, session_ind):
        try:
            requested_sequence_path = self.get_session_path_raw(session_ind)
            if self.dataset.sequence_path == requested_sequence_path:
                return
        except:
            pass

        sequence = "%02d" % session_ind
        self.dataset = pykitti.odometry(self.dataset_directory(), sequence)

    def poses2velo(self, poses_cam0):
        # Transform poses in cam0 frame to velodyne frame
        calib = self.dataset.calib
        Tr = calib.T_cam0_velo
        R = Tr[0:3, 0:3]
        T = Tr[0:3, 3]
        Rt = np.concatenate((R.T, np.expand_dims(np.matmul(-R.T, T.T), axis=1)), axis=1)
        TrI = np.concatenate((Rt, (np.expand_dims([0, 0, 0, 1], axis=1).T)))
        poses_velo = np.matmul(np.matmul(TrI, poses_cam0), Tr)
        return poses_velo

    def load_GT_poses(self, session_ind=None):
        if session_ind is not None:
            assert session_ind == int(self.dataset.sequence)
        poses_raw = self.dataset.poses
        calibrated = [self.poses2velo(pose) for pose in poses_raw]
        return calibrated

    def load_PC(self, session_ind, index):
        self.init_session(session_ind)
        PC_dir = self.get_session_path(session_ind) + 'velodyne/'
        PC = self.dataset.get_velo(int(index))[:,:3]
        return PC

    def get_session_path_raw(self, session_ind):
        requested_sequence_path = self.dataset_directory() + 'sequences/%02d' % session_ind
        return requested_sequence_path

    def get_session_path(self, session_ind):
        requested_sequence_path = self.get_session_path_raw(session_ind)
        assert self.dataset.sequence_path == requested_sequence_path, "Don't forget to call init_session() first"
        return requested_sequence_path + '/'

    @staticmethod
    def dataset_directory():
        return ORIGINAL_DATASET_PATH

    @staticmethod
    def dataset_name():
        return 'KITTI'


class KITTI_full():
    def __init__(self, phase):
        assert phase in ['train', 'validation', 'test']
        self.name = "KITTI"
        self.phase = phase
        self.time_step = 0.1 # seconds between consecutive frames
        self.U = KITTI_utils()        
        if self.phase == 'train':
            self.sessions_list = [0,1,2,3,4,5]            
        elif self.phase == 'validation':
            self.sessions_list = [6,7]            
        elif self.phase == 'test':
            self.sessions_list = [8,9,10]
        self.GT_poses = {}
        for sess_ind in self.sessions_list:
            self.U.init_session(sess_ind)            
            self.GT_poses[sess_ind] = self.U.load_GT_poses(sess_ind)

    def total_num_of_clouds(self):
        return np.sum([self.session_length(ses_i) for ses_i in self.sessions_list])

    def session_length(self, session_ind):
        assert session_ind in self.sessions_list, f"requested session ({session_ind}) is not available"
        return len(self.GT_poses[session_ind])

    def load_PC(self, session_ind, index):
        self.U.init_session(session_ind)
        return self.U.load_PC(session_ind, index)

    def get_relative_motion(self, session_ind, index_src, index_tgt):
        src_to_0 = self.GT_poses[session_ind][index_src]
        tgt_to_0 = self.GT_poses[session_ind][index_tgt]
        relative_mot = np.linalg.inv(tgt_to_0) @ src_to_0
        return relative_mot

    def indexing_from(self):
        return 0

class KITTI_balanced:
    def __init__(self, phase):
        assert phase in ['train', 'validation', 'test']
        self.name = 'KITTI'
        self.time_step = 0.1 # seconds between consecutive frames
        self.phase = phase
        self.U = KITTI_utils()        
        pairs_file = BALANCED_SETS_PATH + '/' + self.name + '/' + phase + '.txt'
        self.pairs = pd.read_csv(pairs_file, sep=" ", header=0).values                

    def get_pair(self, ind):        
        pair = self.pairs[ind]        
        session_ind = int(pair[0])
        src_ind = int(pair[1])
        tgt_ind = int(pair[2])
        mot = pair[3:(3+16)].reshape([4,4])
        self.U.init_session(session_ind)
        A = self.U.load_PC(session_ind, src_ind)
        B = self.U.load_PC(session_ind, tgt_ind)
        return mot, A, B


        
        




