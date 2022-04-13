# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import torch
import numpy as np
import random
from scipy.linalg import expm, norm
import math

# Rotation matrix along axis with angle theta
def M(axis, theta):
  return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def sample_random_trans(pcd, randg, rotation_range=360):
  T = np.eye(4)
  R = M(randg.rand(3) - 0.5, rotation_range * np.pi / 180.0 * (randg.rand(1) - 0.5))
  T[:3, :3] = R
  T[:3, 3] = R.dot(-np.mean(pcd, axis=0))
  return T


class Compose:
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, coords, feats):
    for transform in self.transforms:
      coords, feats = transform(coords, feats)
    return coords, feats


class Jitter:
  def __init__(self, mu=0, sigma=0.01):
    self.mu = mu
    self.sigma = sigma

  def __call__(self, coords, feats):
    if random.random() < 0.95:
      feats += self.sigma * torch.randn(feats.shape[0], feats.shape[1])
      if self.mu != 0:
        feats += self.mu
    return coords, feats


class ChromaticShift:
  def __init__(self, mu=0, sigma=0.1):
    self.mu = mu
    self.sigma = sigma

  def __call__(self, coords, feats):
    if random.random() < 0.95:
      feats[:, :3] += torch.randn(self.mu, self.sigma, (1, 3))
    return coords, feats

def sample_almost_planar_rotation(randg):
  
    MAX_ROTATION_ANGLES_IN_DEGREES = [5,5,180]

    def euler_angles_to_rotation_matrix(theta_vec, deg_or_rad='deg'):
        if deg_or_rad is 'deg':
            theta_vec = np.radians(theta_vec)
        R_x = np.array([[1,         0,                  0                   ],
                        [0,         math.cos(theta_vec[0]), -math.sin(theta_vec[0]) ],
                        [0,         math.sin(theta_vec[0]), math.cos(theta_vec[0])  ]])
        R_y = np.array([[math.cos(theta_vec[1]),    0,      math.sin(theta_vec[1])  ],
                        [0,                     1,      0                   ],
                        [-math.sin(theta_vec[1]),   0,      math.cos(theta_vec[1])  ]])
        R_z = np.array([[math.cos(theta_vec[2]),    -math.sin(theta_vec[2]),    0],
                        [math.sin(theta_vec[2]),    math.cos(theta_vec[2]),     0],
                        [0,                     0,                      1]])
        R = np.dot(R_z, np.dot( R_y, R_x ))
        return R

    def empty_motion_matrix():
        M = np.zeros([4,4], dtype=np.float32)
        M[3,3] = 1
        return M

    def generate_random_rotation(randg, max_rotation_angles_in_degrees=MAX_ROTATION_ANGLES_IN_DEGREES):        
        random_rotation_angles_in_degrees = randg.rand(3)*max_rotation_angles_in_degrees*np.sign(randg.randn(3))
        R = euler_angles_to_rotation_matrix(random_rotation_angles_in_degrees, deg_or_rad='deg')
        M = empty_motion_matrix()
        M[:3,:3] = R
        return M
    
    T = generate_random_rotation(randg)
    return T
