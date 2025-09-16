import pickle

from mujoco import viewer

from env.dual_arm_env import DualArmEnv
from utils import set_seed

# with open("retargeting_data.pkl","rb") as f:
#     raw = pickle.load(f)

set_seed(0)
env = DualArmEnv(save_video=False)

# viewer.launch(env.model, env.data)
with viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False) as viewer:
    observation = env.reset()
    while viewer.is_running():
        action = None
        next_observation, reward= env.step(action)
        observation = next_observation
        viewer.sync()

env.close()