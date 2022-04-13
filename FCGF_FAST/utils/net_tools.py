import numpy as np
import torch
EPS = 10**-12

def soft_argmin_on_rows(X, t, eps=EPS):
    y = []
    for x in X:
        y.append(np.exp(-x/t)/(np.sum(np.exp(-x/t),axis=0)+eps))
    return np.array(y)

def my_softmax(x, eps = EPS, dim=0):
    x_exp = torch.exp(x)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp/(x_exp_sum + eps)
	
def my_softmax_test(x, eps = EPS, dim=0):
# Test how many of the nearest neighbors are axtually needed for good results.
    MAX_K = None
    if MAX_K is not None:
        assert dim == 1, "Unexpected dim"
        res_rows = []
        for i in range(x.shape[0]):
            row = x[i,:]
            x_exp_row = torch.exp(row)
            s = torch.sort(row).values
            thresh = s[-MAX_K]
            mask = row >= thresh
            y = torch.where(mask, x_exp_row, 0*row[0])
            res_rows.append(torch.unsqueeze(y,0))

        x_exp = torch.cat(res_rows,dim=0)
    else:
        x_exp = torch.exp(x)

    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)

    res = x_exp/(x_exp_sum + eps)

    return res

def softargmin_rows_torch_new(X, t, eps=EPS):
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=X.device)
    t = t.double()
    X = X.double()
    weights = my_softmax(-X/t, eps=eps, dim=1)
    return weights

def softargmin_cols_torch_new(X, t, eps=EPS):
    t = t.double()
    X = X.double()
    weights = my_softmax(-X/t, eps=eps, dim=0)
    return weights

def argmin_on_cols(X):
    one_hot_y = np.zeros_like(X)
    Y = np.argmin(X, axis=0)
    one_hot_y[Y,np.arange(X.shape[1])] = 1
    return one_hot_y

def argmin_on_rows(X):
    one_hot_y = np.zeros_like(X)
    Y = np.argmin(X, axis=1)
    one_hot_y[np.arange(X.shape[0]),Y] = 1
    return one_hot_y

def argmin_on_cols_torch(X):
    one_hot_y = torch.zeros_like(X)
    Y = torch.argmin(X, dim=0)
    one_hot_y[Y,torch.arange(X.shape[1])] = 1
    return one_hot_y

def argmin_on_rows_torch(X):
    one_hot_y = torch.zeros_like(X)
    Y = torch.argmin(X, dim=1)
    one_hot_y[torch.arange(X.shape[0]),Y] = 1
    return one_hot_y

