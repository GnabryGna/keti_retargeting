import time
import mujoco
import numpy as np
# import torch
# import yaml

from env import dual_arm_mjcf

# def load_config(config_file):
#     with open(config_file) as file:
#         config = yaml.safe_load(file)
#     return config

def set_seed(seed):
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def run_bench(model, data, n_steps=10000, warmup_steps=1000):
    for _ in range(warmup_steps):
        mujoco.mj_step(model, data)

    dt_list = []
    ncon_list = []
    nefc_list = []
    niter_list = []

    for _ in range(n_steps):
        t0 = time.perf_counter()
        mujoco.mj_step(model, data)
        dt_list.append(time.perf_counter() - t0)
        ncon_list.append(int(data.ncon))
        nefc_list.append(int(data.nefc))
        niter_list.append(int(np.sum(data.solver_niter)))

    def summary(xs):
        return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "p95": float(np.percentile(xs,95))}

    return {
        "step_time_sec": summary(dt_list),
        "ncon": summary(ncon_list),
        "nefc": summary(nefc_list),
        "solver_niter": summary(niter_list)
    }

if __name__ == '__main__':
    mjcf = dual_arm_mjcf.load()
    model = mjcf.compile()
    print(model.ngeom)
    data = mujoco.MjData(model)

    initial_sate = model.key('initial_state').id
    mujoco.mj_resetDataKeyframe(model, data, initial_sate)
    mujoco.mj_forward(model, data)

    result = run_bench(model, data)
    print(result)