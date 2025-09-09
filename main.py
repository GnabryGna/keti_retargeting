import pickle

from mujoco import viewer

from env.dual_arm_env import DualArmEnv

# with open("retargeting_data.pkl","rb") as f:
#     raw = pickle.load(f)

env = DualArmEnv()
# agent = Agent()

with viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False) as viewer:
    observation = env.reset()
    while viewer.is_running():
        action = None # agent.get_action(observation)
        next_observation, reward= env.step(action)
        # agent.update(observation, action, reward, next_observation)
        observation = next_observation

        viewer.sync()