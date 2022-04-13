# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019
import torch
import torch.functional as F
import numpy as np


def rotation_mat2angle(R):
  return torch.acos(torch.clamp((torch.trace(R) - 1) / 2, -0.9999, 0.9999))


def rotation_error(R1, R2):
  assert R1.shape == R2.shape
  return torch.acos(torch.clamp((torch.trace(torch.mm(R1.t(), R2)) - 1) / 2, -0.9999, 0.9999))


def translation_error(t1, t2):
  assert t1.shape == t2.shape
  return torch.sqrt(((t1 - t2)**2).sum())


def batch_rotation_error(rots1, rots2):
  r"""
  arccos( (tr(R_1^T R_2) - 1) / 2 )
  rots1: B x 3 x 3 or B x 9
  rots1: B x 3 x 3 or B x 9
  """
  assert len(rots1) == len(rots2)
  trace_r1Tr2 = (rots1.reshape(-1, 9) * rots2.reshape(-1, 9)).sum(1)
  side = (trace_r1Tr2 - 1) / 2
  return torch.acos(torch.clamp(side, min=-0.999, max=0.999))


def batch_translation_error(trans1, trans2):
  r"""
  trans1: B x 3
  trans2: B x 3
  """
  assert len(trans1) == len(trans2)
  return torch.norm(trans1 - trans2, p=2, dim=1, keepdim=False)



def eval_metrics(output, target):
  output = (F.sigmoid(output) > 0.5)
  target = target
  return torch.norm(output - target)


def corr_dist(est, gth, xyz0, xyz1, weight=None, max_dist=1):
  xyz0_est = xyz0 @ est[:3, :3].t() + est[:3, 3]
  xyz0_gth = xyz0 @ gth[:3, :3].t() + gth[:3, 3]
  dists = torch.clamp(torch.sqrt(((xyz0_est - xyz0_gth).pow(2)).sum(1)), max=max_dist)
  if weight is not None:
    dists = weight * dists
  return dists.mean()

def calc_d2_array(X,Y):
  NUM_PARTS = 3
  x_sep = np.round(np.linspace(0,X.shape[0], NUM_PARTS+1)).astype(int)            
  d2_list = []
  for p in range(NUM_PARTS):
    cur_X = X[x_sep[p]:x_sep[p+1],:]                
    cur_d2 = torch.sum((cur_X.unsqueeze(1) - Y.unsqueeze(0))**2, dim=2)                    
    d2_list.append(cur_d2)
  d2 = torch.cat(d2_list)
  return d2

def pdist(A, B, dist_type='L2'):
  if dist_type == 'L2':
    D2 = calc_d2_array(A,B)
    return torch.sqrt(D2 + 1e-7)
  elif dist_type == 'SquareL2':
    return calc_d2_array(A,B)
  else:
    raise NotImplementedError('Not implemented')


def get_loss_fn(loss):
  if loss == 'corr_dist':
    return corr_dist
  else:
    raise ValueError(f'Loss {loss}, not defined')
