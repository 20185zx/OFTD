import math
import time
import random
import torch
import numpy as np
from torch import optim 
from model import online_update_multi, Online_CP_multi_net
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# prepare data
X_train, X_test, X_val, X, mask_train, mask_test, mask_val= read_data(data = 'data/foreman.mat', sample_rate = 0.3)


# set hyperparameters
p = 0.1
R1 = 100
R2 = 100
R3 = 100
A_ini = math.floor(p*(X.shape[0]))
B_ini = math.floor(p*(X.shape[1]))
C_ini = math.floor(p*(X.shape[2]))
A_delta = math.floor(p*(X.shape[0]))
B_delta = math.floor(p*(X.shape[1]))
C_delta = math.floor(p*(X.shape[2]))
mid_channel = 128
omega_A = 1.5
omega_B = 1.5
omega_C = 0.6


# initial stage
print('Initial stage start')
A_t = A_ini
B_t = B_ini
C_t = C_ini

X_t = X_train[:A_t, :B_t, :C_t]
X_t_val = X_val[:A_t, :B_t, :C_t]
X_t_test = X_test[:A_t, :B_t, :C_t]
mask_t_train = mask_train[:A_t, :B_t, :C_t]
mask_t_test = mask_test[:A_t, :B_t, :C_t]
mask_t_val = mask_val[:A_t, :B_t, :C_t]

model = Online_CP_multi_net(R1, R2, R3, mid_channel, omega_A=omega_A, omega_B=omega_B, omega_C=omega_C).to(device)
A_input = torch.from_numpy(np.array(range(A_t)).reshape(A_t,1)).type(dtype).to(device)
B_input = torch.from_numpy(np.array(range(B_t)).reshape(B_t,1)).type(dtype).to(device)
C_input = torch.from_numpy(np.array(range(C_t))).reshape(C_t,1).type(dtype).to(device)
params = []
params += [x for x in model.parameters()]

optimizier = optim.Adam(params, lr=0.001, weight_decay=10e-8)

best_nre_val = float('inf')
patience = 10
wait = 0
best_state = None
for iter in range(4000):
    optimizier.zero_grad()
    X_Out_real = model(A_input,B_input,C_input)
    loss = ((X_Out_real * mask_t_train - X_t * mask_t_train) ** 2).sum()
    loss.backward() 
    optimizier.step()

    with torch.no_grad():
        nre_initial_train = calcu_nre(X_t, X_Out_real, mask_t_train)
        nre_initial_test = calcu_nre(X_t_test, X_Out_real, mask_t_test)
        nre_val = calcu_nre(X_t_val, X_Out_real, mask_t_val)

    if nre_val < best_nre_val - 1e-6:
        best_nre_val = nre_val
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            break
if best_state is not None:
    model.load_state_dict(best_state)

print(f'Initial stage end, nre train: {nre_initial_train:.3f}, nre test: {nre_initial_test:.3f}\n')


# Online update stage
print('Online update stage start')
divide = 3
alpha_beta = [1,1.2]
[A_T,B_T,C_T] = X_train.shape
nres_train = []
nres_test = []
time_costs = []
flops_all = []

time_cost_all_start = time.perf_counter()
max_update_num = max_update(A_T,B_T,C_T,A_ini,B_ini,C_ini,A_delta,B_delta,C_delta)
for i in range(max_update_num):
    model, A_t, B_t, C_t, time_cost, nre_train, nre_test = online_update_multi(alpha_beta, model, X_train, X_test, 
                                                                    mask_train, mask_test, A_t, B_t, C_t, A_delta, 
                                                                    B_delta, C_delta, divide, flops_all = flops_all, 
                                                                    every_iter = 500)
    time_costs.append(time_cost)
    nres_train.append(nre_train)
    nres_test.append(nre_test)
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f'time step: {i+1}/{max_update_num}, nre train: {nre_train:.3f}, nre test: {nre_test:.3f}, time cost: {time_cost:.2f}')
time_cost_all_end = time.perf_counter()

average_flops = round(np.mean(flops_all)/1e6, 2)
average_nre_train = np.mean(nres_train)
average_nre_test = np.mean(nres_test)
average_time_cost = (time_cost_all_end - time_cost_all_start) / max_update_num

print('Online update stage end\n',
    'FLOPs:', average_flops, 'M\t', 
      'average nre train:', f'{average_nre_train:.3f}\t', 
      'average nre test:', f'{average_nre_test:.3f}\t', 
      'average time cost:', f'{average_time_cost:.3f}\n')
