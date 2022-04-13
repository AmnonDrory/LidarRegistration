# Written by Amnon Drory (amnondrory@mail.tau.ac.il), Tel-Aviv University, 2021.
# Please cite the following paper if you use this code:
# - Amnon Drory, Shai Avidan, Raja Giryes, Stress Testing LiDAR Registration

import os
os.environ["OMP_NUM_THREADS"] = "1" # Disable multiprocesing for numpy/opencv

import pandas as pd
import numpy as np

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import train as train_split, val as val_split, test as test_split

from datasets.paths import paths

DATASET_ROOT = paths["NuScenes"]
BALANCED_SETS_PATH = paths["balanced_sets"]

class NuScenes_utils():
    def __init__(self, phase):
        assert phase in ['train','validation', 'test']        
        self.phase = phase
        self.prep_list_of_sessions()
        self.splits = {'train': train_split, 'validation': val_split, 'test': test_split}
    
    def is_phase_session(self, phase, session_ind=None):
        if session_ind is None:
            res = np.array([n in self.splits[phase] for n in self.session_names])              
        else:
            res = self.session_names[session_ind] in self.splits[phase]
        return res 

    def is_location_session(self, location, session_ind=None):
        if session_ind is None:
            res = np.array([location in l for l in self.session_locations])
        else:
            res = location in self.session_locations[session_ind]
        return res         
            
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
        if self.phase in ['train', 'validation']:
            version = 'v1.0-trainval'
        elif self.phase == 'test':
            version = 'v1.0-test'
        self.NuScenes_data = NuScenes(version=version, dataroot=DATASET_ROOT, verbose=True)
        self.num_sessions = len(self.NuScenes_data.scene)

        self.cloud_tokens = []
        self.session_lengths = []
        self.session_locations = []
        self.session_names = []

        for session_ind in range(self.num_sessions):
            
            record = self.NuScenes_data.scene[session_ind]
            session_token = record['token']   
            self.session_names.append(record['name'])
            location = self.NuScenes_data.get('log', record['log_token'])['location']         
            self.session_locations.append(location)
            sample_token = record["first_sample_token"]
            sample = self.NuScenes_data.get("sample", sample_token)
            lidar_token = sample["data"]["LIDAR_TOP"]
            cur_lidar_tokens = []
            while len(lidar_token) > 0:
                cur_lidar_tokens.append(lidar_token)
                lidar_data = self.NuScenes_data.get("sample_data", lidar_token)
                lidar_token = lidar_data["next"]
            self.cloud_tokens.append(cur_lidar_tokens)
            self.session_lengths.append(len(cur_lidar_tokens))

    def load_position_raw(self, sample_lidar_token):
        lidar_data = self.NuScenes_data.get("sample_data", sample_lidar_token)

        ego_pose = self.NuScenes_data.get("ego_pose", lidar_data["ego_pose_token"])
        calibrated_sensor = self.NuScenes_data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

        # Homogeneous transformation matrix from car frame to world frame.
        global_from_car = transform_matrix(ego_pose['translation'],
                                        Quaternion(ego_pose['rotation']), inverse=False)

        return global_from_car
    
    def load_cloud_raw(self, sample_lidar_token):  
        lidar_data = self.NuScenes_data.get("sample_data", sample_lidar_token)  
        lidar_filepath = self.NuScenes_data.get_sample_data_path(sample_lidar_token)
        ego_pose = self.NuScenes_data.get("ego_pose", lidar_data["ego_pose_token"])
        calibrated_sensor = self.NuScenes_data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                            inverse=False)

        lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)

        # The lidar pointcloud is defined in the sensor's reference frame.
        # We want it in the car's reference frame, so we transform each point
        lidar_pointcloud.transform(car_from_sensor)

        return lidar_pointcloud.points[:3,:].T    


class NuScenes_full():
    def __init__(self, location, phase):
        assert phase in ['train', 'validation', 'test']
        assert location in ['boston', 'singapore']
        self.name = "NuScenes_" + location
        self.phase = phase
        self.location = location
        self.time_step = 0.05 # seconds between consecutive frames
        self.U = NuScenes_utils(self.phase)

        is_correct_location = self.U.is_location_session(self.location)        
        is_correct_phase = self.U.is_phase_session(self.phase)        
        self.sessions_list = (is_correct_location & is_correct_phase).nonzero()[0]
    
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


class NuScenes_balanced:
    def __init__(self, location, phase):
        assert phase in ['train', 'validation', 'test']
        self.name = "NuScenes_" + location
        self.time_step = 0.05 # seconds between consecutive frames
        self.phase = phase
        self.location = location
        try:
            self.U = NuScenes_utils(self.phase)
        except AssertionError as E:
            print("Could not load NuScenes_utils: " + str(E))            
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

