import time
import math
import random
import torch
import numpy as np
from torch import optim
from utils import *
from model import Online_CP_single_net_affine, online_update_single_affine
from affine import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)
seed=42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# prepare data
X_train, X_test, X_val, X, mask_train, mask_test, mask_val = read_data(data='./data/foreman.mat', sample_rate=0.3)

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


affinex = 18
affiney = 4

Affinex = 144+64
Affiney = 176+64
rotate_theta = torch.zeros((5), device=0, requires_grad=True)
Scale_factor = torch.ones((5), device=0, requires_grad=True)
x = torch.ones((5), device=0, requires_grad=True)
y = torch.ones((5), device=0, requires_grad=True)
params = []
params += [rotate_theta]
params += [Scale_factor]

model = Online_CP_single_net_affine(Affinex, Affiney, R=100, mid_channel=128, omega_0=0.3).to(device)
C_input = torch.from_numpy(np.array(range(t))).reshape(t,1).type(dtype).to(device)

optimizier = optim.Adam([{'params':model.parameters(), 'lr':0.001, 'weight_decay': 10e-8},
                                    {'params':params, 'lr':0.8*0.001}])  
best_nre_val = float('inf')
patience = 10
wait = 0
best_state = None
for iter in range(4000):
    optimizier.zero_grad()
    X_Out_real,x,y = model(C_input)
    X_Out_real = X_Out_real.permute(2,0,1)

    X_Out_real = affine_B1(X_Out_real.unsqueeze(0), x, y,
                    rotate_theta, Scale_factor, 32).squeeze(0)
    X_Out_real = X_Out_real.permute(1,2,0)

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
    model, t, time_cost, nre_train, nre_test = online_update_single_affine(alpha_beta, model, X_train, X_test, 
                                        mask_train, mask_test, t, delta_t, divide = 3, 
                                        flops_all = flops_all, every_iter = 100)
    time_costs.append(time_cost)
    if math.isnan(nre_train)==False:
        nres_train.append(nre_train)
    nres_test.append(nre_test)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print(f'time step: {t}/{t_total}, nre train: {nre_train:.3f}, nre test: {nre_test:.3f}, time cost: {time_cost:.2f}')

time_cost_all_end = time.perf_counter()
average_time_cost = (time_cost_all_end - time_cost_all_start) / ((t_total - t_initial) // delta_t)
average_flops = round(np.mean(flops_all)/1e6, 2)
average_nre_train = np.mean(nres_train)
average_nre_test = np.mean(nres_test)

print('Online update stage end\n',
      'FLOPs:', average_flops, 'M\t', 
      'average nre train:', f'{average_nre_train:.3f}\t', 
      'average nre test:', f'{average_nre_test:.3f}\t', 
      'average time cost:', f'{average_time_cost:.3f}\n')
