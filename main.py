from mujoco import viewer

from env.dual_arm_env import DualArmEnv
from utils import set_seed

set_seed(0)
env = DualArmEnv()
env.reset()

viewer.launch(env.model, env.data)
# with viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False) as viewer:
#     while viewer.is_running():
#         env.step(None)
#         viewer.sync()
# env.close()