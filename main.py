from mujoco import viewer

from env.dual_arm_env import DualArmEnv

env = DualArmEnv()
# agent = Agent()

with viewer.launch_passive(env.model, env.data) as viewer:
    observation = env.reset()
    while viewer.is_running():
        action = None # agent.get_action(obs)
        next_observation, reward= env.step(action)

        # agent.update(observation, action, reward, next_observation)

        observation = next_observation

        viewer.sync()