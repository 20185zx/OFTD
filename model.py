
import math
import time
import torch
import numpy as np
from torch import nn
from torch import optim
from thop import profile
from utils import *
from affine import *
dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                is_first=False, omega_0=1.5): 
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Online_CP_single_net_affine(nn.Module): 
    def __init__(self,n_1,n_2,R=100,mid_channel=256,omega_0=1.5):
        super(Online_CP_single_net_affine, self).__init__()
        self.A = nn.Parameter(torch.Tensor(R,n_1,1))
        self.B = nn.Parameter(torch.Tensor(R,1,n_2))

        self.C_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True, omega_0 = omega_0),
                                SineLayer(mid_channel, mid_channel, is_first=True, omega_0 = omega_0),
                                nn.Linear(mid_channel, R))
        
        self.x_net = nn.Sequential(SineLayer(R, R, omega_0=0.2, is_first=True),
                                       nn.Linear(R, 1, bias = False))
        
        self.y_net = nn.Sequential(SineLayer(R, R, omega_0=0.2, is_first=True),
                                       nn.Linear(R, 1, bias = False))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.A.size(0))
        self.A.data.uniform_(-stdv, stdv)
        self.B.data.uniform_(-stdv, stdv)
                                    
    def forward(self,C_input):
        x = torch.matmul(self.A, self.B).permute(1,2,0)
        C = self.C_net(C_input).permute(1,0)
        x1 = self.x_net(C.permute(1,0))
        y = self.y_net(C.permute(1,0))
        return x @ C, x1.squeeze(-1), y.squeeze(-1)
  

class Online_CP_single_net(nn.Module): 
    def __init__(self,n_1,n_2,R=100,mid_channel=256,omega_0=1.5):
        super(Online_CP_single_net, self).__init__()
        self.A = nn.Parameter(torch.Tensor(R,n_1,1))
        self.B = nn.Parameter(torch.Tensor(R,1,n_2))

        self.C_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True, omega_0 = omega_0),
                                SineLayer(mid_channel, mid_channel, is_first=True, omega_0 = omega_0),
                                nn.Linear(mid_channel, R))
        


        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.A.size(0))
        self.A.data.uniform_(-stdv, stdv)
        self.B.data.uniform_(-stdv, stdv)
                                    
    def forward(self,C_input):
        x = torch.matmul(self.A, self.B).permute(1,2,0)
        C = self.C_net(C_input).permute(1,0)
        return x @ C
    

    
class Online_CP_multi_net(nn.Module): 
    def __init__(self,R1=100,R2=100,R3=100,mid_channel=256,omega_A=1.5,omega_B=1.5,omega_C=1.5):
        super(Online_CP_multi_net, self).__init__()

        self.A_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True, omega_0 = omega_A),
                                nn.Linear(mid_channel, R1))
        
        self.B_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True, omega_0 = omega_B),
                                nn.Linear(mid_channel, R2))

        self.C_net = nn.Sequential(SineLayer(1, mid_channel, is_first=True, omega_0 = omega_C),
                                SineLayer(mid_channel, mid_channel, is_first=True, omega_0 = omega_C),
                                nn.Linear(mid_channel, R3))
        self.core = nn.Parameter(torch.Tensor(R1,R2,R3))
        size = self.core.size(0)
        self.core = nn.Parameter(torch.zeros(size, size, size), requires_grad=False)
        for i in range(size):
            self.core[i, i, i] = 1
        
    def forward(self, A_input, B_input, C_input):
        A = self.A_net(A_input)
        B = self.B_net(B_input)
        C = self.C_net(C_input)
        
        centre = self.core.permute(1,2,0)
        centre = centre @ A.t()
        centre = centre.permute(2,1,0)
        centre = centre @ B.t()
        centre = centre.permute(0,2,1)
        centre = centre @ C.t()
        return centre


def online_update_single_affine(aa, model, X_train, X_test, mask_train, 
                 mask_test, t, delta_t, divide, flops_all = None, every_iter = 100):    

    start = time.time()
    rotate_theta = torch.zeros((103), device=0, requires_grad=True)
    Scale_factor = torch.ones((103), device=0, requires_grad=True)
    x = torch.ones((103), device=0, requires_grad=True)
    y = torch.ones((103), device=0, requires_grad=True)
    params = []
    params += [rotate_theta]
    params += [Scale_factor]

    optimizier = optim.Adam([{'params':model.parameters(), 'lr':0.001, 'weight_decay': 10e-8},
                                        {'params':params, 'lr':0.002*0.001}])   

    t = t + delta_t
    if t > X_train.shape[2]:
        t = X_train.shape[2]

    X_t = X_train[:, :, :t]
    X_t_test = X_test[:, :, :t]

    mask_t_train = mask_train[:, :, :t]
    mask_t_test = mask_test[:, :, :t]
    
    C_input = torch.from_numpy(np.arange(t)).reshape(t, 1).type(dtype).to(device)
    loss_best = 1e10

    new_data_ind = np.arange(t-delta_t, t)
    
    for iter in range(every_iter):
        indexes = sample(aa, divide = divide, t = t, )
        ind = np.concatenate([indexes, new_data_ind], axis=0)

        mask_train_here = mask_t_train[:,:,ind]
        X_t_here = X_t[:,:,ind]
        optimizier.zero_grad()
        C_input_here = torch.from_numpy(ind).unsqueeze(-1).type(dtype).to(device)

        X_Out_real,x,y = model(C_input_here)

        rotate_theta1 = rotate_theta[ind]
        Scale_factor1 = Scale_factor[ind]

        x1= x
        y1 = y

        X_Out_real = X_Out_real.permute(2,0,1)
        X_Out_real = affine_B1(X_Out_real.unsqueeze(0), x1, y1,
                        rotate_theta1, Scale_factor1, 32).squeeze(0)
        X_Out_real = X_Out_real.permute(1,2,0)
        if iter == 0:
            flops_update, params = profile(model, inputs=(C_input_here,), verbose=False)
            flops_all.append(flops_update)
        loss = ((X_Out_real*mask_train_here - X_t_here*mask_train_here)**2).sum()
        if loss.item() < loss_best:
            loss_best = loss.item()
            best_params = model.state_dict()

        loss.backward() 
        optimizier.step()


    model.load_state_dict(best_params)

    X_Out_real,x,y = model(C_input)
    X_Out_real = X_Out_real.permute(2,0,1)
    c = C_input.detach().cpu().numpy().squeeze()
    x2 = x[c]
    y2 = y[c]
    rotate_theta2 = rotate_theta[c]
    Scale_factor2 = Scale_factor[c]
    X_Out_real = affine_B1(X_Out_real.unsqueeze(0), x2, y2,
                    rotate_theta2, Scale_factor2, 32).squeeze(0)
    X_Out_real = X_Out_real.permute(1,2,0)

    if torch.eq(mask_t_test[:, :, -1], 0).all():
        mask_t_test[:, :, -1] = (mask_t_test[:, :, -2] + mask_test[:, :, t]) / 2
        X_t_test[:, :, -1] = (X_t_test[:, :, -2] + X_test[:, :, t]) / 2
    
    nre = calcu_nre(X_t[:,:,:], X_Out_real[:,:,:], mask_t_train[:,:,:]).item()
    nre_test = calcu_nre(X_t_test[:,:,:], X_Out_real[:,:,:], mask_t_test[:,:,: ]).item()
    end = time.time()
    time_cost = end - start

    return model, t, time_cost, nre, nre_test


def online_update_single(aa, model, X_train, X_test, mask_train, 
                 mask_test, t, delta_t, divide, flops_all = None, every_iter = 100):    

    start = time.time()
    params = []
    params += [x for x in model.parameters()]
    optimizier = optim.Adam(params, lr=0.001, weight_decay=10e-8)

    t = t + delta_t
    if t > X_train.shape[2]:
        t = X_train.shape[2]

    X_t = X_train[:, :, :t]
    X_t_test = X_test[:, :, :t]

    mask_t_train = mask_train[:, :, :t]
    mask_t_test = mask_test[:, :, :t]
    
    C_input = torch.from_numpy(np.arange(t)).reshape(t, 1).type(dtype).to(device)
    loss_best = 1e10
    new_data_ind = np.arange(t-delta_t, t)
   
    indexes = sample(aa, divide = divide, t = t, )
    ind = np.concatenate([indexes, new_data_ind], axis=0)
    # To further reduce the computational overhead, fix the size of the memory Buffer (this is an option)
    # if len(indexes) > 100:
    #     indexes = np.random.choice(indexes, 100, replace=False)
    for iter in range(every_iter):

        mask_train_here = mask_t_train[:,:,ind]
        X_t_here = X_t[:,:,ind]
        optimizier.zero_grad()
        C_input_here = torch.from_numpy(ind).unsqueeze(-1).type(dtype).to(device)
        X_Out_real = model(C_input_here)
        if iter == 0:
            flops_update, params = profile(model, inputs=(C_input_here,), verbose=False)
            flops_all.append(flops_update)

        loss = ((X_Out_real*mask_train_here - X_t_here*mask_train_here)**2).sum()

        if loss.item() < loss_best:
            loss_best = loss.item()
            best_params = model.state_dict()

        loss.backward() 
        optimizier.step()


    model.load_state_dict(best_params)
    X_Out_real = model(C_input)

    if torch.eq(mask_t_test[:, :, -1], 0).all():
        mask_t_test[:, :, -1] = (mask_t_test[:, :, -2] + mask_test[:, :, t]) / 2
        X_t_test[:, :, -1] = (X_t_test[:, :, -2] + X_test[:, :, t]) / 2
    
    nre = calcu_nre(X_t[:,:,:], X_Out_real[:,:,:], mask_t_train[:,:,:]).item()
    nre_test = calcu_nre(X_t_test[:,:,:], X_Out_real[:,:,:], mask_t_test[:,:,:]).item()
    end = time.time()
    time_cost = end - start

    return model, t, time_cost, nre, nre_test


def online_update_multi(alpha_beta, model, X_train, X_test, mask_train, mask_test, 
             A_t, B_t, C_t, A_delta, B_delta, C_delta, divide, 
             flops_all = None, every_iter = 500):
    params = []
    params += [x for x in model.parameters()]
    optimizier = optim.Adam(params, lr=0.001, weight_decay=10e-8)

    A_t = A_t + A_delta
    if A_t > X_train.shape[0]:
        A_t = X_train.shape[0]
    B_t = B_t + B_delta
    if B_t > X_train.shape[1]:
        B_t = X_train.shape[1]
    C_t = C_t + C_delta
    if C_t > X_train.shape[2]:
        C_t = X_train.shape[2]

    X_t = X_train[:A_t, :B_t, :C_t]
    X_t_test = X_test[:A_t, :B_t, :C_t]

    mask_t_train = mask_train[:A_t, :B_t, :C_t]
    mask_t_test = mask_test[:A_t, :B_t, :C_t]
    
    A_input = torch.from_numpy(np.arange(A_t)).reshape(A_t, 1).type(dtype).to(device)
    B_input = torch.from_numpy(np.arange(B_t)).reshape(B_t, 1).type(dtype).to(device)
    C_input = torch.from_numpy(np.arange(C_t)).reshape(C_t, 1).type(dtype).to(device)
    new_data_ind_A = np.arange(A_t-A_delta, A_t)
    new_data_ind_B = np.arange(B_t-B_delta, B_t)
    new_data_ind_C = np.arange(C_t-C_delta, C_t)

    loss_best = 1e10
    start_time = time.perf_counter()
    for iter in range(every_iter):
        indexes_A = sample(alpha_beta, divide = divide, t = A_t)
        indexes_B = sample(alpha_beta, divide = divide, t = B_t)
        indexes_C = sample(alpha_beta, divide = divide, t = C_t)

        ind_A = np.concatenate([indexes_A, new_data_ind_A], axis=0)
        ind_B = np.concatenate([indexes_B, new_data_ind_B], axis=0)
        ind_C = np.concatenate([indexes_C, new_data_ind_C], axis=0)

        mask_train_here = mask_t_train[:,:,ind_C]
        mask_train_here = mask_train_here[:,ind_B,:]
        mask_train_here = mask_train_here[ind_A,:,:]
        X_t_here = X_t[:,:,ind_C]
        X_t_here = X_t_here[:,ind_B,:]
        X_t_here = X_t_here[ind_A,:,:]

        optimizier.zero_grad()

        A_input_here = torch.from_numpy(ind_A).unsqueeze(-1).type(dtype).to(device)
        B_input_here = torch.from_numpy(ind_B).unsqueeze(-1).type(dtype).to(device)
        C_input_here = torch.from_numpy(ind_C).unsqueeze(-1).type(dtype).to(device)
        X_Out_real = model(A_input_here, B_input_here, C_input_here)
        if iter == 0:
            flops_update, params = profile(model, inputs=(A_input_here,B_input_here,C_input_here,), verbose=False)
            flops_all.append(flops_update)

        loss = ((X_Out_real*mask_train_here - X_t_here*mask_train_here)**2).sum()
        if loss.item() < loss_best:
            loss_best = loss.item()
            best_params = model.state_dict()
        loss.backward()
        optimizier.step()

    time_cost = time.perf_counter() - start_time
    model.load_state_dict(best_params)
    X_Out_real = model(A_input,B_input,C_input)

    if torch.eq(mask_t_test[:, :, -1], 0).all():
        mask_t_test[:, :, -1] = mask_t_test[:, :, -2]
        X_t_test[:, :, -1] = X_t_test[:, :, -2]
        
    nre = calcu_nre(X_t[:A_t,:B_t,:C_t], X_Out_real[:A_t,:B_t,:C_t], mask_t_train[:A_t,:B_t,:C_t]).item()
    nre_test = calcu_nre(X_t_test[:A_t,:B_t,:C_t], X_Out_real[:A_t,:B_t,:C_t], mask_t_test[:A_t,:B_t,:C_t]).item()
    
    return model, A_t, B_t, C_t, time_cost, nre, nre_test
