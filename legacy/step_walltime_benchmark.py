import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.dual_arm_env import DualArmEnv

env = DualArmEnv()
env.reset()

dt_list = []
ncon_list = []
nefc_list = []
niter_list = []

for _ in range(1000):
    t0 = time.perf_counter()
    env.step()
    dt_list.append(time.perf_counter() - t0)
    ncon_list.append(int(env.data.ncon))
    nefc_list.append(int(env.data.nefc))
    niter_list.append(int(np.sum(env.data.solver_niter)))

def summary(xs):
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "p95": float(np.percentile(xs, 95))}

step_time_sec = summary(dt_list),
ncon = summary(ncon_list),
nefc = summary(nefc_list),
solver_niter = summary(niter_list)

print(f'step_time_sec: {step_time_sec}')
print(f'ncon: {ncon}')
print(f'nefc: {nefc}')
print(f'solver_niter: {solver_niter}')