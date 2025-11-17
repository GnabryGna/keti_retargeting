from utils import se3_utils as S
from utils import transform as T
from utils.torch_utils import to_numpy, to_torch
from retargeting import FingerRetargetor, OPERATOR2VP_RIGHT, OPERATOR2VP_LEFT

from VisionProTeleop.avp_stream import VisionProStreamer

import torch
import numpy as np
import time, subprocess
from absl import app, flags
from tqdm import tqdm
from env.dual_arm_env import DualArmEnv
from utils.env_utils import set_seed

from collections import defaultdict
from pathlib import Path
from mujoco import viewer
from pytransform3d import rotations
from sapien import Pose
import glfw
import mujoco
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string("ip", "172.16.0.86", "VisionPro ip.")
flags.DEFINE_bool("record", False, "Record VisionPro data.")


def filter_data(data, fps, duration):
    init_time = data[0]["time"]
    all_times = np.array([d["time"] for d in data]) - init_time
    step = 1.0 / fps
    new_data = []
    for i in range(fps * duration):
        current_time = i * step
        diff = np.abs(all_times - current_time)
        best_match = np.argmin(diff)
        new_data.append(data[best_match])
    return new_data


# allegro_real_to_isaac_indices = (0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7)


def main(_):
    # Environment
    set_seed(0)
    env = DualArmEnv()
    # Finger retargeting
    robot_dir = Path(__file__).absolute().parent / "dex-urdf" / "robots" / "hands"
    retargetor = FingerRetargetor(robot_dir=robot_dir)

    # VisionPro
    vp = VisionProStreamer(ip=FLAGS.ip, record=FLAGS.record)
    to_real_right = (
        0,
        1,
        2,
        3,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        4,
        5,
        6,
        7,
    )
    to_real_left = (
        12,
        13,
        14,
        15,
        8,
        9,
        10,
        11,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    )
    obs = env.reset()
    # viewer.launch(env.model, env.data)
    actions = defaultdict()

    init_info = vp.latest
    # datasets = defaultdict(list)
    with viewer.launch_passive(
        env.model, env.data, show_left_ui=False, show_right_ui=False
    ) as v:
        # Set initial camera pose
        time.sleep(0.5)  # 창 뜰 시간 (필요 시 0.1~1.0 조정)
        subprocess.run(["wmctrl", "-r", ":ACTIVE:", "-b", "add,fullscreen"])

        try:
            v.cam.azimuth = 270.0
            # v.cam.elevation = -25.0
            # v.cam.distance = 1.6
            v.cam.lookat[:] = env.grasping_area_pos - [0.3, -0.1, 0]
        except Exception:
            pass
        while v.is_running():
            latest = vp.latest
            latest["time"] = time.time()

            # Map operator wrist frames into robot frame consistently with finger mapping
            lw = to_torch(latest["left_wrist"][0].copy())
            rw = to_torch(latest["right_wrist"][0].copy())

            lw[:3, :3] = lw[:3, :3] @ OPERATOR2VP_LEFT
            rw[:3, :3] = rw[:3, :3] @ OPERATOR2VP_RIGHT
            # actions["left_wrist_qpos"] = S.mat2posquat(to_torch(lw))[0]
            # actions["right_wrist_qpos"] = S.mat2posquat(to_torch(rw))[0]
            left_quat = to_torch(rotations.quaternion_from_matrix(lw[:3, :3]))
            right_quat = to_torch(rotations.quaternion_from_matrix(rw[:3, :3]))

            def quat_mul(q2, q1):  # q = q2 ⊗ q1
                x1, y1, z1, w1 = q1
                x2, y2, z2, w2 = q2
                return torch.tensor(
                    [
                        w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
                        w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
                        w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
                        w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
                    ],
                    dtype=q1.dtype,
                    device=q1.device,
                )

            qx_pi = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=left_quat.dtype, device=left_quat.device
            )  # Rx(π)
            # qy_pi = torch.tensor(
            #     [0.0, 1.0, 0.0, 0.0], dtype=left_quat.dtype, device=left_quat.device
            # )  # Ry(π)

            left_quat = quat_mul(qx_pi, left_quat)  # 또는 quat_mul(qy_pi, left_quat)
            right_quat = quat_mul(qx_pi, right_quat)
            actions["left_wrist_qpos"] = torch.cat([lw[:3, 3], left_quat], dim=-1)
            actions["right_wrist_qpos"] = torch.cat([rw[:3, 3], right_quat], dim=-1)

            # actions["right_wrist_qpos"][0] += 0.4
            actions["left_wrist_qpos"][0] -= 0.1
            actions["right_wrist_qpos"][0] += 0.02
            actions["left_wrist_qpos"][1] += 0.07
            actions["right_wrist_qpos"][1] += 0.07
            actions["left_wrist_qpos"][2] -= 0.6
            actions["right_wrist_qpos"][2] -= 0.6
            print(
                "left_x, left_y: ",
                actions["left_wrist_qpos"][0],
                actions["left_wrist_qpos"][1],
                "right_x, right_y : ",
                actions["right_wrist_qpos"][0],
                actions["right_wrist_qpos"][1],
            )

            # datasets["actions"].append(np.concatenate([to_numpy(actions["left_wrist_qpos"]), to_numpy(actions["right_wrist_qpos"])]))
            # Finger
            left_finger = latest["left_fingers"]
            left_finger_qpos = retargetor.retarget(data=left_finger, hand="left")
            left_finger_qpos = np.expand_dims(left_finger_qpos, 0)
            left_finger_qpos = left_finger_qpos[:, to_real_left]

            right_finger = latest["right_fingers"]
            right_finger_qpos = retargetor.retarget(data=right_finger, hand="right")
            right_finger_qpos = np.expand_dims(right_finger_qpos, 0)
            right_finger_qpos = right_finger_qpos[:, to_real_right]

            actions["left_robot_hand_qpos"] = to_torch(left_finger_qpos)
            actions["right_robot_hand_qpos"] = to_torch(right_finger_qpos)

            obs, reward = env.step(actions)
            mujoco.mj_step(env.model, env.data)
            v.sync()

    # datasets["actions"] = np.array(datasets["actions"])
    # print(datasets["actions"].shape)
    # with open("pre_datasets.pkl", "wb") as f:
    #     pickle.dump(datasets, f, protocol=4)


if __name__ == "__main__":

    app.run(main)
