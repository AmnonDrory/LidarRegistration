import numpy as np
import open3d as o3d
import os
import pandas as pd
from dataloader.paths import ApolloSouthbay_dir, balanced_sets_base_dir, cache_dir
from glob import glob
import errno

ORIGINAL_DATASET_PATH = ApolloSouthbay_dir
BALANCED_SETS_PATH = balanced_sets_base_dir
CACHE_DIR = cache_dir # set to None to avoid caching point-clouds

class Apollo_utils():


    @staticmethod
    def dataset_directory():
        return ORIGINAL_DATASET_PATH

    @staticmethod
    def get_all_session_paths():
        session_paths_file = Apollo_utils.dataset_directory() + 'session_paths.txt'
        with open(session_paths_file, "r") as fid:
            session_paths_relative = fid.read().splitlines()

        session_paths = [Apollo_utils.dataset_directory() + p for p in session_paths_relative]
        return session_paths

    @staticmethod
    def get_session_path(session_ind):
        session_paths = Apollo_utils.get_all_session_paths()
        return session_paths[session_ind]

    @staticmethod
    def load_PC(session_ind, index, cache_file=None): 
        session_path = Apollo_utils.get_session_path(session_ind)
        filename = session_path + "pcds/%d.pcd" % index
        assert os.path.isfile(filename), "Error: could not find file " + filename
        pcd = o3d.io.read_point_cloud(filename)
        cloud = np.asarray(pcd.points)
        if (cache_file is not None) and not os.path.isfile(cache_file):
            np.save(cache_file, cloud)
        return cloud

    
class ApolloSouthbay_balanced:
    def __init__(self, phase):
        assert phase in ['train', 'validation', 'test']
        self.name = 'ApolloSouthbay'
        self.time_step = 0.1 # seconds between consecutive frames
        self.phase = phase                        
        pairs_file = BALANCED_SETS_PATH + '/' + self.name + '/' + phase + '.txt'
        self.pairs = pd.read_csv(pairs_file, sep=" ", header=0).values          

        coarse_motions_file = BALANCED_SETS_PATH + '/' + self.name + '/' + phase + '.coarse_motions.txt'
        if os.path.isfile(coarse_motions_file):
            self.coarse_motions = pd.read_csv(coarse_motions_file, sep=" ", header=0).values
        else:
            self.coarse_motions = None    
		
        self.init_cache_dir()

        if not self.fully_cached:
            self.U = Apollo_utils()

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

    def get_coarse_motion(self, ind):
        if self.coarse_motions is None:
            return None
        
        row = self.coarse_motions[ind,:]
        gt_row = self.pairs[ind,:]
        if not (row[:3]==gt_row[:3]).all():
            assert False, "coarse_motions do not correspond to ground truths. Check datafiles."

        coarse_motion = row[3:(3+16)].reshape([4,4])
        return coarse_motion


