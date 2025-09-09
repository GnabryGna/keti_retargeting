import pickle

import mujoco
from mujoco import viewer

from env.dual_arm_env import DualArmEnv

# with open("retargeting_data.pkl","rb") as f:
#     raw = pickle.load(f)

'''
env = DualArmEnv()

with viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False) as viewer:
    observation = env.reset()
    while viewer.is_running():
        action = None
        next_observation, reward= env.step(action)
        observation = next_observation

        viewer.sync()
'''

env = DualArmEnv(save_video=True)

for _ in range(2):
    observation = env.reset()
    while env.data.time <= 5:
        action = None
        next_observation, reward= env.step(action)
        observation = next_observation

env.close()