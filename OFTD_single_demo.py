import time
import math
import random
import torch
import numpy as np
from torch import optim
from utils import *
from model import Online_CP_single_net, online_update_single
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# prepare data
X_train, X_test, X_val, X, mask_train, mask_test, mask_val = read_data(data='data/condition.mat', sample_rate=0.3)


# Initial stage
print('Initial stage start')
t_initial = 5
delta_t = 1
t_total = X_train.shape[2]
t = t_initial
X_t = X_train[:, :, :t]
X_t_val = X_val[:, :, :t]
X_t_test = X_test[:, :, :t]
mask_t_train = mask_train[:, :, :t]
mask_t_test = mask_test[:, :, :t]
mask_t_val = mask_val[:, :, :t]

model = Online_CP_single_net(X_t.shape[0], X_t.shape[1], R=100, mid_channel=128, omega_0=0.3).to(device)
C_input = torch.from_numpy(np.array(range(t))).reshape(t,1).type(dtype).to(device)
params = []
params += [x for x in model.parameters()]
optimizier = optim.Adam(params, lr=0.001, weight_decay=10e-8)

best_nre_val = float('inf')
patience = 10
wait = 0
best_state = None
for iter in range(4000):
    optimizier.zero_grad()
    X_Out_real = model(C_input)
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
alpha_beta = [1,1.2]
nres_train = []
nres_test = []
time_costs = []
flops_all = []
time_cost_all_start = time.perf_counter()
while t < t_total:
    model, t, time_cost, nre_train, nre_test = online_update_single(alpha_beta, model, X_train, X_test, 
                                        mask_train, mask_test, t, delta_t, divide = 3, 
                                        flops_all = flops_all, every_iter = 100)
    time_costs.append(time_cost)
    if math.isnan(nre_train)==False:
        nres_train.append(nre_train)
    nres_test.append(nre_test)
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print(f'time step: {t}/{t_total}, nre train: {nre_train:.3f}, nre test: {nre_test:.3f}, time cost: {time_cost:.2f}', end='\r')

time_cost_all_end = time.perf_counter()
average_time_cost = (time_cost_all_end - time_cost_all_start) / ((t_total - t_initial) // delta_t)
average_flops = round(np.mean(flops_all)/1e6, 2)

nres_train = [x for x in nres_train if math.isfinite(x)]
nres_test = [x for x in nres_test if math.isfinite(x)]
average_nre_train = np.mean(nres_train)
average_nre_test = np.mean(nres_test)

print('Online update stage end\n',
    'FLOPs:', average_flops, 'M\t', 
    'average nre train:', f'{average_nre_train:.4f}\t', 
    'average nre test:', f'{average_nre_test:.4f}\t', 
    'average time cost:', f'{average_time_cost:.4f}\n')

