import torch
import random
import numpy as np
import scipy.io as scio
from scipy.stats import beta
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_mask(X, sample_rate):
    mask_train = torch.zeros(X.shape).type(dtype)
    for i in range(X.shape[2]):
        slice_mask = (torch.rand(X.shape[0], X.shape[1]) > 1 - sample_rate).to(dtype)
        mask_train[:, :, i] = slice_mask

    mask_comp = (1 - mask_train).type(dtype)
    splitter = torch.rand_like(mask_comp)
    mask_val = (mask_comp * (splitter > 0.5).to(dtype)).type(dtype)
    mask_test = (mask_comp - mask_val).clamp(min=0, max=1).type(dtype)

    return mask_train.to(device), mask_test.to(device), mask_val.to(device)

def calcu_nre(val, rec, mask):
    num = torch.norm(val * mask - rec * mask)
    den = torch.norm(val * mask)
    return num / (den + 1e-12)

def sample(aa, divide = 3, t = None):
    indexes = beta.rvs(aa[0], aa[1], size=t//divide) * (t-1)
    indexes = np.floor(indexes).astype(int)

    return indexes

def read_data(data = 'data/condition.mat', sample_rate = 0.1):
    mat = scio.loadmat(data)
    X = mat['Ohsi']
    X = torch.from_numpy(X).to(device)
    mask_train, mask_test, mask_val= prepare_mask(X, sample_rate)
    X_train = X * mask_train
    X_test = X * mask_test
    X_val = X * mask_val
    return X_train, X_test, X_val, X, mask_train, mask_test, mask_val

def max_update(A_T,B_T,C_T,A_ini,B_ini,C_ini,A_delta,B_delta,C_delta):
    A_update_num = 1e6
    B_update_num = 1e6
    C_update_num = 1e6

    if A_delta != 0:
        A_update_num = (A_T-A_ini)//A_delta
    if B_delta != 0:
        B_update_num = (B_T-B_ini)//B_delta
    if C_delta != 0:
        C_update_num = (C_T-C_ini)//C_delta
    
    max_update_num = min(A_update_num, B_update_num, C_update_num)

    return max_update_num
