import numpy as np
import os
import datetime
from time import time
import pandas as pd
import sys
import tempfile
tempdir = tempfile.gettempdir()

def save_to_ply(PC, filename, *args, **kwargs):
    parent_path,_ = os.path.split(filename)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
    if not filename.endswith('ply'):
        filename += '.ply'

    header =    "ply\n" + \
                "format ascii 1.0\n" +  \
                "element vertex %d\n" % PC.shape[0] +  \
                "property float x\n" +  \
                "property float y\n" +  \
                "property float z\n" +  \
                "end_header\n"
    
    s = (''.join(["%.14f %.14f %.14f\n"] * PC.shape[0])) % tuple(PC.flatten())
    
    with open(filename, 'w') as ply:
        ply.write(header)
        ply.write(s)

def read_sym_icp_output(path2):
    out_path, name = os.path.split(path2)
    out = os.path.join(out_path,os.path.splitext(name)[0]+'.xf')
    if not os.path.isfile(out):
        return None
    with open(out, 'r') as fid:
        result = fid.read().splitlines()
    os.remove(out)
    numbers = []
    for line in result:
        numbers.append(line.split(' '))        
    inv_mot = np.asarray(numbers, dtype=float) # motion B->A    
    mot = np.linalg.inv(inv_mot) # motion A->B
    return mot


def symmetric_icp(PC_to, PC_from):
    start_time = time()
    cur_time_str = datetime.datetime.now().strftime("%H_%M_%S_%f")
    A_path = tempdir + '/symmetric_icp.A.%s.%d.%.6f.ply' % (cur_time_str,int(10000000*np.random.rand()),PC_from.mean())
    B_path = tempdir + '/symmetric_icp.B.%s.%d.%.6f.ply' % (cur_time_str,int(10000000*np.random.rand()),PC_to.mean())
    save_to_ply(PC_to, A_path)
    save_to_ply(PC_from, B_path)
    res = run_symmetric_icp(A_path, B_path)
    os.remove(A_path)
    os.remove(B_path)
    elapsed = time() - start_time
    return res, elapsed


def run_symmetric_icp(A_path, B_path):
    trimesh2_dir = '/trimesh2'
    exe = trimesh2_dir + '/bin.Linux64/mesh_align '
    MAX_ATTEMPTS = 10

    for attempt_ind in range(MAX_ATTEMPTS):

        cmd = exe + ' ' + A_path + ' ' + B_path + '> /dev/null 2>&1'
        assert(os.path.isfile(B_path))
        assert(os.path.isfile(A_path))
        x = os.system(cmd)
        mot = read_sym_icp_output(B_path)
        if (mot is None) or np.any(np.isnan(mot)):
            print('run_symmetric_icp attempt %d failed' % attempt_ind)
        else:
            break

    if (mot is None) or np.any(np.isnan(mot)):
        print("run_symmetric_icp failed for the maximum allowed attempts: %d" % MAX_ATTEMPTS)
        return np.eye(4)        

    return mot
