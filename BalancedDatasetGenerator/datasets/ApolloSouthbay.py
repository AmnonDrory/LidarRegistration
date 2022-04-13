# Written by Amnon Drory (amnondrory@mail.tau.ac.il), Tel-Aviv University, 2021.
# Please cite the following paper if you use this code:
# - Amnon Drory, Shai Avidan, Raja Giryes, Stress Testing LiDAR Registration

import numpy as np
import open3d as o3d
import os
import pandas as pd
from utils.tools_3d import quaternion_to_euler, euler_angles_to_rotation_matrix
from datasets.paths import paths

ORIGINAL_DATASET_PATH = paths['ApolloSouthbay']
BALANCED_SETS_PATH = paths['balanced_sets']

class Apollo_utils():
    def __init__(self):
        self.find_session_paths()

    def find_session_paths(self):
                
        keys = [
            "MapData/HighWay237/2018-10-05/",
            "MapData/SunnyvaleBigloop/Caspian_and_Geneva/2017-12-13/",
            "MapData/SunnyvaleBigloop/Borrgas/2017-12-13/",
            "MapData/SunnyvaleBigloop/Java/2017-12-13/",
            "MapData/SunnyvaleBigloop/Mathilda_Moffet/2017-12-28/",
            "MapData/SunnyvaleBigloop/Crossman/2017-12-13/",
            "MapData/SunnyvaleBigloop/Mathilda_Carribean/2017-12-14/",
            "MapData/SunnyvaleBigloop/Bordeaux/2017-12-13/",
            "MapData/MathildaAVE/2018-09-25/",
            "MapData/SanJoseDowntown/2018-10-02/",
            "MapData/BaylandsToSeafood/2018-09-26/",
            "MapData/ColumbiaPark/2018-09-21/2/",
            "MapData/ColumbiaPark/2018-09-21/4/",
            "MapData/ColumbiaPark/2018-09-21/1/",
            "MapData/ColumbiaPark/2018-09-21/3/",
            "TrainData/HighWay237/2018-10-12/",
            "TrainData/MathildaAVE/2018-10-04/",
            "TrainData/SanJoseDowntown/2018-10-11/",
            "TrainData/BaylandsToSeafood/2018-10-05/",
            "TrainData/ColumbiaPark/2018-10-03/",
            "TestData/HighWay237/2018-10-12/",
            "TestData/SunnyvaleBigloop/2018-10-03/",
            "TestData/MathildaAVE/2018-10-12/",
            "TestData/SanJoseDowntown/2018-10-11/2/",
            "TestData/SanJoseDowntown/2018-10-11/1/",
            "TestData/BaylandsToSeafood/2018-10-12/",
            "TestData/ColumbiaPark/2018-10-11/"]

        def list_dirs():
            subdirs = [ORIGINAL_DATASET_PATH]
            for cur_dir in subdirs:
                new_dirs = glob(cur_dir + '/*/')
                for d in new_dirs:
                    subdirs.append(d)
            return subdirs
        
        subdirs = list_dirs()

        self.session_paths = []        
        for key in keys:
            path = []
            for dir in subdirs:
                if dir.endswith(key):
                    path.append(dir)

            if len(path)==0:
                self.session_paths.append(None)
            else:
                assert len(path)==1, "Error: multiple directories match key"
                self.session_paths.append(path[0])

    def load_GT_poses(self, session_ind):
        session_path = self.get_session_path(session_ind)
        GT_filename = session_path + "poses/gt_poses.txt"
        res = pd.read_csv(GT_filename, sep=" ", header=None).values
        return res


    def is_train_session(self, session_ind):
        session_path = self.get_session_path(session_ind)
        return ("TrainData" in session_path)

    def is_test_session(self, session_ind):
        session_path = self.get_session_path(session_ind)
        return ("Test" in session_path)

    def is_phase_session(self, session_ind, phase):
        if phase == 'train':
            return self.is_train_session(session_ind)
        elif phase == 'test':
            return self.is_test_session(session_ind)
        else:
            assert False, "unexpected value for parameter 'phase': " + phase

    def dataset_directory(self):
        return ORIGINAL_DATASET_PATH

    def dataset_name(self):
        return 'Apollo'

    def get_all_session_paths(self):
        return self.session_paths

    def get_session_path(self, session_ind):
        session_paths = self.get_all_session_paths()
        return session_paths[session_ind]

    def num_sessions(self):
        session_paths = self.get_all_session_paths()
        return len(session_paths)


    def convert_GT(self,raw):
        trans = raw[2:5]
        angles = quaternion_to_euler(*(raw[5:9]))
        mat = np.eye(4)
        mat[:3,3] = trans
        mat[:3,:3] = euler_angles_to_rotation_matrix(angles)
        return mat
    
    def extract_GT(self, GT_raw, ind):
        GT_ind = np.where(GT_raw[:, 0] == ind)[0][0]
        gt = self.convert_GT(GT_raw[GT_ind, :])
        return gt

    def load_PC(self, session_ind, index):
        session_path = self.get_session_path(session_ind)
        filename = session_path + "pcds/%d.pcd" % index
        assert os.path.isfile(filename), "Error: could not find file " + filename
        pcd = o3d.io.read_point_cloud(filename)
        return np.asarray(pcd.points)

    
class ApolloSouthbay_full():
    def __init__(self, phase):
        assert phase in ['train', 'test']
        self.name = "ApolloSouthbay"
        self.phase = phase
        self.time_step = 0.1 # seconds between consecutive frames
        self.U = Apollo_utils()
        self.sessions_list = []
        for sess_ind in range(self.U.num_sessions()):
            if self.U.is_phase_session(sess_ind, self.phase):
                self.sessions_list.append(sess_ind)
        self.GT_poses = {}
        for sess_ind in self.sessions_list:
            self.GT_poses[sess_ind] = self.U.load_GT_poses(sess_ind)

    def total_num_of_clouds(self):
        return np.sum([self.session_length(ses_i) for ses_i in self.sessions_list])

    def session_length(self, session_ind):
        assert session_ind in self.sessions_list, f"requested session ({session_ind}) is not available"
        return len(self.GT_poses[session_ind])

    def load_PC(self, session_ind, index):
        return self.U.load_PC(session_ind, index)

    def get_relative_motion(self, session_ind, index_src, index_tgt):
        GT_raw = self.GT_poses[session_ind]
        src_to_0 = self.U.extract_GT(GT_raw, index_src)
        tgt_to_0 = self.U.extract_GT(GT_raw, index_tgt)
        relative_mot = np.linalg.inv(tgt_to_0) @ src_to_0
        return relative_mot

    def indexing_from(self):
        return 1

class ApolloSouthbay_balanced:
    def __init__(self, phase):
        assert phase in ['train', 'validation', 'test']
        self.name = 'ApolloSouthbay'
        self.time_step = 0.1 # seconds between consecutive frames
        self.phase = phase
        self.U = Apollo_utils()                
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
