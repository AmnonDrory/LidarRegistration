import torch

def w_by_inverse_dist(X,Y):
    D = torch.sqrt(((X-Y)**2).sum(dim=1,keepdim=True))
    med = torch.median(D)
    min_thresh = 0.1*med
    w = 1/D
    w[D<min_thresh] = 1/min_thresh
    return w
    
def procrustes(X, Y, w_in=None):
  """
  Adapted from DGR code (https://chrischoy.github.io/publication/dgr/)
  X: torch tensor N x 3
  Y: torch tensor N x 3
  """
  if False:      
    w = w_by_inverse_dist(X,Y)          
  else:
    w = torch.ones_like(X[:,0:1])

  if w_in is not None:
    w_in = w_in.reshape(w.shape).to(w.device)
    w *= w_in

  R, T = weighted_procrustes(X, Y, w)

  M = torch.eye(4)
  M[:3,:3] = R
  M[:3,3] = T

  return M

def weighted_procrustes(X, Y, w, eps=10**-6):
  """
  Taken from DGR code (https://chrischoy.github.io/publication/dgr/)
  X: torch tensor N x 3
  Y: torch tensor N x 3
  w: torch tensor N
  """
  # https://ieeexplore.ieee.org/document/88573
  assert len(X) == len(Y)
  W1 = torch.abs(w).sum()
  w_norm = w / (W1 + eps)
  mux = (w_norm * X).sum(0, keepdim=True)
  muy = (w_norm * Y).sum(0, keepdim=True)

  # Use CPU for small arrays
  Sxy = (Y - muy).t().mm(w_norm * (X - mux)).cpu().double()
  U, D, V = Sxy.svd()
  S = torch.eye(3).double()
  if U.det() * V.det() < 0:
    S[-1, -1] = -1

  R = U.mm(S.mm(V.t())).float()
  t = (muy.cpu().squeeze() - R.mm(mux.cpu().t()).squeeze()).float()
  return R, t