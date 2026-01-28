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
import os
import threading
import sys
import select
from utils.logger import Logger

FLAGS = flags.FLAGS

flags.DEFINE_string("ip", "172.16.0.86", "VisionPro ip.")
flags.DEFINE_bool("record", False, "Record VisionPro data.")


def get_init_dataset():
    return {
        "observations": defaultdict(list),
        "next_observations": defaultdict(list),
        "actions": defaultdict(list),
        "reward": [],
    }


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
    actions = defaultdict()
    datasets = get_init_dataset()
    iter_idx = 0
    data_path = "./datasets"
    
    # Create renderer for image capture
    # width = 1200
    # height = 800
    # image_renderer = mujoco.Renderer(env.model, width=width, height=height)

    dataset_name_list = os.listdir(data_path)
    max_idx = -1
    for dataset_name in dataset_name_list:
        dataset_idx = int(dataset_name.split("_")[1].split(".")[0])
        max_idx = max(dataset_idx, max_idx)
    task_idx = max_idx + 1

    reset_viewer = False
    terminal = False
    Logger.debug(f"Current idx: {task_idx}")

    # Option 1: Terminal input monitoring (background thread)
    def monitor_terminal_input():
        nonlocal reset_viewer
        nonlocal terminal
        while True:
            try:
                # Check if input is available (non-blocking for Unix/Linux)
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    line = sys.stdin.readline().strip().lower()
                    if line in ("reset", "r"):
                        reset_viewer = True
                        print("Terminal input detected: Resetting environment...")
                    elif line == "t":
                        terminal = True
                        print(
                            "Terminal state detected: Saving dataset and restarting viewer..."
                        )
            except (OSError, ValueError):
                # Windows doesn't support select on stdin, use threading approach
                try:
                    line = input().strip().lower()
                    if line in ("reset", "r"):
                        reset_viewer = True
                        print("Terminal input detected: Resetting environment...")
                    elif line == "t":
                        terminal = True
                        print(
                            "Terminal state detected: Saving dataset and restarting viewer..."
                        )
                except:
                    time.sleep(0.5)

    input_thread = threading.Thread(target=monitor_terminal_input, daemon=True)
    input_thread.start()
    print("Terminal input monitoring enabled.")
    print("  - Type 'reset' or 'r' + Enter to reset environment")
    print("  - Type 't' + Enter to save dataset and restart (terminal state)")

    init_info = vp.latest
    print("Data Collection Started.")
    while True:
        with viewer.launch_passive(
            env.model,
            env.data,
            show_left_ui=False,
            show_right_ui=False,
        ) as view:
            # Set initial camera pose
            time.sleep(0.5)
            subprocess.run(["wmctrl", "-r", ":ACTIVE:", "-b", "add,fullscreen"])

            view.cam.azimuth = -270.0
            view.cam.lookat[:] = env.grasping_area_pos - [0.3, -0.2, 0]
            view.cam.elevation = -70.0
            # Main simulation loop
            while view.is_running() and not reset_viewer and not terminal:
                latest = vp.latest
                latest["time"] = time.time()

                lw = to_torch(latest["left_wrist"][0].copy())
                rw = to_torch(latest["right_wrist"][0].copy())

                lw[:3, :3] = lw[:3, :3] @ OPERATOR2VP_LEFT
                rw[:3, :3] = rw[:3, :3] @ OPERATOR2VP_RIGHT
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

                left_quat = quat_mul(
                    qx_pi, left_quat
                )  # 또는 quat_mul(qy_pi, left_quat)
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

                # Finger
                left_finger = latest["left_fingers"]
                left_finger_qpos = retargetor.retarget(data=left_finger, hand="left")
                left_finger_qpos = np.expand_dims(left_finger_qpos, 0)
                left_finger_qpos = left_finger_qpos[:, to_real_left]

                right_finger = latest["right_fingers"]
                right_finger_qpos = retargetor.retarget(data=right_finger, hand="right")
                right_finger_qpos = np.expand_dims(right_finger_qpos, 0)
                right_finger_qpos = right_finger_qpos[:, to_real_right]

                actions["left_hand_qpos"] = to_torch(left_finger_qpos)
                actions["right_hand_qpos"] = to_torch(right_finger_qpos)
                next_obs, reward = env.step(actions)
                if iter_idx > 4:
                    for k, v in obs.items():
                        datasets["observations"][k].append(v)
                    for k, v in next_obs.items():
                        datasets["next_observations"][k].append(v)
                    for k, v in actions.items():
                        datasets["actions"][k].append(v)
                    datasets["reward"].append(reward)
                    
                    # Capture image from view.cam
                    # Update scene first
                    # image_renderer.update_scene(env.data)
                    # # Copy camera settings from viewer to renderer
                    # image_renderer.scene.camera.azimuth = view.cam.azimuth
                    # image_renderer.scene.camera.elevation = view.cam.elevation
                    # image_renderer.scene.camera.distance = view.cam.distance
                    # image_renderer.scene.camera.lookat[:] = view.cam.lookat[:]
                    # # Render image
                    # # image = image_renderer.render()
                    # if "image" not in datasets["observations"]:
                    #     datasets["observations"]["image"] = []
                    #     datasets["next_observations"]["image"] = []
                    # datasets["observations"]["image"].append(image.copy())
                    # datasets["next_observations"]["image"].append(image.copy())
                mujoco.mj_step(env.model, env.data)
                # reward_val = 0.0 if reward is None else float(reward)
                env.add_triad(view)
                view.sync()
                obs = next_obs
                iter_idx += 1

            if reset_viewer or terminal:
                try:
                    view.close()
                except:
                    pass
                time.sleep(0.5)

        if terminal:
            # Save current dataset if any data was collected
            if iter_idx > 4:
                with open(f"{data_path}/datasets_{task_idx}.pkl", "wb") as f:
                    pickle.dump(datasets, f, protocol=4)
                print(f"Dataset saved: datasets_{task_idx}.pkl")
                task_idx += 1
            # Reset environment and prepare for restart
            obs = env.reset()
            datasets = get_init_dataset()
            iter_idx = 0
            reset_viewer = False
            terminal = False  # Reset terminal flag
            print("=" * 25)
            print(
                "Terminal state processed. Environment reset. Waiting before restarting viewer..."
            )
            Logger.debug(f"Current idx: {task_idx}")
            print("=" * 25)
            # Wait longer to ensure viewer is fully closed and resources released
            time.sleep(1.5)
            # Continue outer while loop to restart viewer
            continue
        if reset_viewer:
            # Reset environment and prepare for restart
            obs = env.reset()
            datasets = get_init_dataset()
            iter_idx = 0
            reset_viewer = False
            # Update file mtime for file-based trigger
            print("=" * 25)
            print("Environment reset. Waiting before restarting viewer...")
            Logger.debug(f"Current idx: {task_idx}")
            print("=" * 25)
            # Wait longer to ensure viewer is fully closed and resources released
            time.sleep(1.5)
            # Continue outer while loop to restart viewer
            continue


if __name__ == "__main__":

    app.run(main)
