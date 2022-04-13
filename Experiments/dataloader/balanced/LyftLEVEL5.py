import os
import pandas as pd
import numpy as np
from glob import glob
import errno

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix

from dataloader.paths import LyftLEVEL5_dir, balanced_sets_base_dir, cache_dir

DATASET_ROOT = LyftLEVEL5_dir
BALANCED_SETS_PATH = balanced_sets_base_dir
CACHE_DIR = cache_dir # set to None to avoid caching point-clouds

class LyftLEVEL5_utils():
    def __init__(self, phase):
        assert phase in ['train','test']
        self.phase = phase
        self.prep_list_of_sessions()

    def load_PC(self, session_ind, cloud_ind, cache_file=None):
        assert session_ind < self.num_sessions
        assert cloud_ind < self.session_lengths[session_ind], f"Requested cloud {cloud_ind}, but session {session_ind} only contains {self.session_lengths[session_ind]} clouds"
        lidar_token = self.cloud_tokens[session_ind][cloud_ind]
        cloud = self.load_cloud_raw(lidar_token)
        if (cache_file is not None) and not os.path.isfile(cache_file):
            np.save(cache_file, cloud)
        return cloud

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
        pairs_file = BALANCED_SETS_PATH + '/' + self.name + '/' + phase + '.txt'
        self.pairs = pd.read_csv(pairs_file, sep=" ", header=0).values                
        
        self.init_cache_dir()

        if not self.fully_cached:
            self.U = LyftLEVEL5_utils(self.phase)

    def init_cache_dir(self):

        if CACHE_DIR is None:
            self.cache_dir = None        
            self.cached_files = []
            self.fully_cached = False
            return 

        self.cache_dir = CACHE_DIR + '/' + self.name + '/' + self.phase + '/'
        if not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise e

        cache_files_raw = glob(self.cache_dir + '*.npy')        
        cache_files = [os.path.split(f)[-1] for f in cache_files_raw]
        
        self.fully_cached = True
        for pair in self.pairs:
            cache_file_src = '%d_%d.npy' % (pair[0], pair[1])
            if not cache_file_src in cache_files:
                self.fully_cached = False
                break
            code_tgt = '%d_%d.npy' % (pair[0], pair[2])
            if not code_tgt in cache_files:
                self.fully_cached = False
                break

    def get_pair(self, ind):        
        pair = self.pairs[ind]        
        session_ind = int(pair[0])
        src_ind = int(pair[1])
        tgt_ind = int(pair[2])
        mot = pair[3:(3+16)].reshape([4,4])
        
        if self.cache_dir is None:            
            A = self.U.load_PC(session_ind, src_ind)            
            B = self.U.load_PC(session_ind, tgt_ind)
        else:
            cache_file_src = self.cache_dir + '%d_%d.npy' % (session_ind, src_ind)
            if os.path.isfile(cache_file_src):
                A = np.load(cache_file_src)
            else:
                A = self.U.load_PC(session_ind, src_ind, cache_file_src)
            
            cache_file_tgt = self.cache_dir + '%d_%d.npy' % (session_ind, tgt_ind)
            if os.path.isfile(cache_file_tgt):
                B = np.load(cache_file_tgt)
            else:            
                B = self.U.load_PC(session_ind, tgt_ind, cache_file_tgt)

        return mot, A, B


