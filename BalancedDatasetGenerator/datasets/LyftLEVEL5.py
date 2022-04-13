# Written by Amnon Drory (amnondrory@mail.tau.ac.il), Tel-Aviv University, 2021.
# Please cite the following paper if you use this code:
# - Amnon Drory, Shai Avidan, Raja Giryes, Stress Testing LiDAR Registration

import os
os.environ["OMP_NUM_THREADS"] = "1" # Disable multiprocesing for numpy/opencv.

import pandas as pd
import numpy as np

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
from datasets.paths import paths

DATASET_ROOT = paths["LyftLEVEL5"]
BALANCED_SETS_PATH = paths["balanced_sets"]

class LyftLEVEL5_utils():
    def __init__(self, phase):
        assert phase in ['train','test']
        self.phase = phase
        self.prep_list_of_sessions()

    def load_PC(self, session_ind, cloud_ind):
        assert session_ind < self.num_sessions
        assert cloud_ind < self.session_lengths[session_ind], f"Requested cloud {cloud_ind}, but session {session_ind} only contains {self.session_lengths[session_ind]} clouds"
        lidar_token = self.cloud_tokens[session_ind][cloud_ind]
        return self.load_cloud_raw(lidar_token)

    def get_relative_motion_A_to_B(self, session_ind, cloud_ind_A, cloud_ind_B):
        token_A = self.cloud_tokens[session_ind][cloud_ind_A]
        posA = self.load_position_raw(token_A)
        token_B = self.cloud_tokens[session_ind][cloud_ind_B]
        posB = self.load_position_raw(token_B)        
        mot_A_to_B = np.linalg.inv(posB) @ posA
        return mot_A_to_B

    def prep_list_of_sessions(self):
        self.level5data = LyftDataset(json_path=DATASET_ROOT + "/%s_data" % self.phase, data_path=DATASET_ROOT, verbose=True)
        self.num_sessions = len(self.level5data.scene)
        self.cloud_tokens = []
        self.session_lengths = []

        for session_ind in range(self.num_sessions):
            
            record = self.level5data.scene[session_ind]
            session_token = record['token']            
            sample_token = record["first_sample_token"]
            sample = self.level5data.get("sample", sample_token)
            lidar_token = sample["data"]["LIDAR_TOP"]
            cur_lidar_tokens = []
            while len(lidar_token) > 0:
                cur_lidar_tokens.append(lidar_token)
                lidar_data = self.level5data.get("sample_data", lidar_token)
                lidar_token = lidar_data["next"]
            self.cloud_tokens.append(cur_lidar_tokens)
            self.session_lengths.append(len(cur_lidar_tokens))

    def load_position_raw(self, sample_lidar_token):
        lidar_data = self.level5data.get("sample_data", sample_lidar_token)

        ego_pose = self.level5data.get("ego_pose", lidar_data["ego_pose_token"])
        calibrated_sensor = self.level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

        # Homogeneous transformation matrix from car frame to world frame.
        global_from_car = transform_matrix(ego_pose['translation'],
                                        Quaternion(ego_pose['rotation']), inverse=False)

        return global_from_car
    
    def load_cloud_raw(self, sample_lidar_token):  
        lidar_data = self.level5data.get("sample_data", sample_lidar_token)  
        lidar_filepath = self.level5data.get_sample_data_path(sample_lidar_token)
        ego_pose = self.level5data.get("ego_pose", lidar_data["ego_pose_token"])
        calibrated_sensor = self.level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                            inverse=False)

        lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)

        # The lidar pointcloud is defined in the sensor's reference frame.
        # We want it in the car's reference frame, so we transform each point
        lidar_pointcloud.transform(car_from_sensor)

        return lidar_pointcloud.points[:3,:].T    

class LyftLEVEL5_full():
    def __init__(self, phase):
        assert phase in ['train', 'test']
        self.name = "LyftLEVEL5" 
        self.phase = phase        
        self.time_step = 0.2 # seconds between consecutive frames
        self.U = LyftLEVEL5_utils(self.phase)       
        self.sessions_list = np.arange(self.U.num_sessions)
        if self.phase == 'train':
            self.sessions_list = np.delete(self.sessions_list, [21]) # this session contain corrupted data


    def total_num_of_clouds(self):
        return np.sum([self.session_length(ses_i) for ses_i in self.sessions_list])

    def session_length(self, session_ind):
        assert session_ind in self.sessions_list, f"requested session ({session_ind}) is not available"
        return self.U.session_lengths[session_ind]

    def load_PC(self, session_ind, index):
        return self.U.load_PC(session_ind, index)

    def get_relative_motion(self, session_ind, index_src, index_tgt):
        return self.U.get_relative_motion_A_to_B(session_ind, index_src, index_tgt)

    def indexing_from(self):
        return 0


class LyftLEVEL5_balanced:
    def __init__(self, phase):
        assert phase in ['train', 'validation', 'test']
        self.name = "LyftLEVEL5" 
        self.time_step = 0.2 # seconds between consecutive frames
        self.phase = phase
        self.U = LyftLEVEL5_utils(self.phase)
        pairs_file = BALANCED_SETS_PATH + '/' + self.name + '/' + phase + '.txt'
        self.pairs = pd.read_csv(pairs_file, sep=" ", header=0).values                

    def get_pair(self, ind):        
        pair = self.pairs[ind]        
        session_ind = int(pair[0])
        src_ind = int(pair[1])
        tgt_ind = int(pair[2])
        mot = pair[3:(3+16)].reshape([4,4])
        A = self.U.load_PC(session_ind, src_ind)
        B = self.U.load_PC(session_ind, tgt_ind)
        return mot, A, B


