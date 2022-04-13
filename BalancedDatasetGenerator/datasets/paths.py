import os
from os.path import isdir, realpath, dirname, split
from os import makedirs

BASE_DIR = split(dirname(realpath(__file__)))[0]

paths = {
	'KITTI': 'XXX/kitti/dataset/', # change this to a directory containing the KITTI dataset
	'ApolloSouthbay': 'XXX/apollo/', # change this to a directory containing the Apollo-Southbay dataset
	'NuScenes': 'XXX/NuScenes', # change this to a directory containing the NuScenes dataset
	'balanced_sets': BASE_DIR + '/output/balanced_sets/'        
}

