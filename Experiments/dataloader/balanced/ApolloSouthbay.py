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

    def dataset_directory(self):
        return ORIGINAL_DATASET_PATH

    def get_all_session_paths(self):
        return self.session_paths

    def get_session_path(self, session_ind):
        session_paths = self.get_all_session_paths()
        return session_paths[session_ind]


    def load_PC(self, session_ind, index, cache_file=None): 
        session_path = self.get_session_path(session_ind)
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
