import os
from datetime import datetime

import imageio.v2 as imageio
import mujoco
import numpy as np

from env import dual_arm_mjcf
from env.robot import AllegroHandV4

# from env.robot import InspireRH56DFTP
from env.robot import xArm7


class DualArmEnv:
    BARCODE_SCANNER_NAME = "barcode_scanner/barcode_scanner"
    YCB_OBJECT_NAMES = [
        "003_cracker_box/003_cracker_box",
        "004_sugar_box/004_sugar_box",
        "005_tomato_soup_can/005_tomato_soup_can",
        "006_mustard_bottle/006_mustard_bottle",
        "010_potted_meat_can/010_potted_meat_can",
        "021_bleach_cleanser/021_bleach_cleanser",
    ]

    def __init__(self, save_video=False):
        mjcf = dual_arm_mjcf.load()
        self.model = mjcf.compile()
        self.data = mujoco.MjData(self.model)

        self.left_robot_arm = xArm7(self.model, self.data, "xarm7_left")
        self.right_robot_arm = xArm7(self.model, self.data, "xarm7_right")
        self.left_robot_hand = AllegroHandV4(
            self.model, self.data, "xarm7_left/allegro_left"
        )
        self.right_robot_hand = AllegroHandV4(
            self.model, self.data, "xarm7_right/allegro_right"
        )
        # self.left_robot_hand = InspireRH56DFTP(self.model, self.data, 'xarm7_left/inspire_rh56dftp_left')
        # self.right_robot_hand = InspireRH56DFTP(self.model, self.data, 'xarm7_right/inspire_rh56dftp_right')

        self.grasping_area_pos = self.model.site("grasping_area").pos
        self.grasping_area_size = self.model.site("grasping_area").size
        self.grasping_area = [
            (
                self.grasping_area_pos[i] - self.grasping_area_size[i],
                self.grasping_area_pos[i] + self.grasping_area_size[i],
            )
            for i in range(2)
        ]

        self.barcode_scanner_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.BARCODE_SCANNER_NAME
        )
        self.ycb_object_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.YCB_OBJECT_NAMES
        ]

        self.save_video = save_video
        # TODO: to utils.py
        self.writer = None
        if self.save_video:
            self.video_fps = 60
            self.frame_dt = 1.0 / self.video_fps
            self.next_frame_time = 0.0
            width = self.model.vis.global_.offwidth
            height = self.model.vis.global_.offheight
            self.renderer = mujoco.Renderer(self.model, width=width, height=height)

    def reset(self):
        initial_sate = self.model.key("initial_state").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, initial_sate)

        self.spawn_ycb_object()
        mujoco.mj_step(self.model, self.data, nstep=10000)  # 10s

        observation = self.get_observation()

        if self.save_video:
            if self.writer is not None:
                self.writer.close()
            time_stamp = datetime.now().strftime("%y%m%d_%H_%M_%S")
            file_path = os.path.join("logs", f"{time_stamp}.mp4")
            self.writer = imageio.get_writer(
                file_path,
                fps=self.video_fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=None,
            )
            self.next_frame_time = self.data.time

        return observation

    def step(self, action):
        self.left_robot_arm.set_tcp_pose(action["left_wrist_qpos"])
        self.right_robot_arm.set_tcp_pose(action["right_wrist_qpos"])

        # self.left_robot_arm.servoj(action['left_robot_arm_q_pos'])
        # self.right_robot_arm.servoj(action['right_robot_arm_q_pos'])
        self.left_robot_hand.servoj(action["left_robot_hand_qpos"])
        self.right_robot_hand.servoj(action["right_robot_hand_qpos"])
        mujoco.mj_step(self.model, self.data, nstep=10)  # 1/(0.001*10) = 100Hz
        observation = self.get_observation()
        reward = self.get_reward()

        if self.save_video:
            if self.data.time >= self.next_frame_time:
                self.renderer.update_scene(self.data)
                frame = self.renderer.render()
                self.writer.append_data(frame)
                self.next_frame_time += self.frame_dt

        return observation, reward

    def get_observation(self):
        left_robot_arm_q_pos = self.left_robot_arm.get_q_pos()  # (7,)
        right_robot_arm_q_pos = self.right_robot_arm.get_q_pos()  # (7,)
        left_robot_hand_q_pos = (
            self.left_robot_hand.get_q_pos()
        )  # (16,) for allegro, (12,) for inspire
        right_robot_hand_q_pos = (
            self.right_robot_hand.get_q_pos()
        )  # (16,) for allegro, (12,) for inspire

        barcode_scanner_pose = np.empty(7)  # (7,)
        barcode_scanner_pose[:3] = self.data.xpos[self.barcode_scanner_body_id].copy()
        barcode_scanner_pose[3:] = self.data.xquat[self.barcode_scanner_body_id].copy()

        ycb_object_poses = np.empty((len(self.ycb_object_body_ids), 7))  # (7,)
        for i, ycb_objetc_body_id in enumerate(self.ycb_object_body_ids):
            ycb_object_poses[i, :3] = self.data.xpos[ycb_objetc_body_id].copy()
            ycb_object_poses[i, 3:] = self.data.xquat[ycb_objetc_body_id].copy()

        # TODO: Add tactile/force data

        return {
            "left_robot_arm_q_pos": left_robot_arm_q_pos,
            "right_robot_arm_q_pos": right_robot_arm_q_pos,
            "left_robot_hand_q_pos": left_robot_hand_q_pos,
            "right_robot_hand_q_pos": right_robot_hand_q_pos,
            "barcode_scanner_pose": barcode_scanner_pose,
            "ycb_object_poses": ycb_object_poses,
        }

    def get_reward(self):
        return None

    def spawn_ycb_object(self):
        nonoverlap_pos_dict = self.sample_nonoverlap_pos(
            self.grasping_area[0], self.grasping_area[1]
        )

        for ycb_object_name, pos in nonoverlap_pos_dict.items():
            yaw = np.random.uniform(-np.pi, np.pi)
            quat = np.array([np.cos(yaw * 0.5), 0, 0, np.sin(yaw * 0.5)])
            pose = np.concatenate([pos, quat])

            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, ycb_object_name
            )
            joint_qpos_adr = self.model.jnt_qposadr[joint_id]
            joint_qvel_adr = self.model.jnt_dofadr[joint_id]

            self.data.qpos[joint_qpos_adr : joint_qpos_adr + 7] = pose
            self.data.qvel[joint_qvel_adr : joint_qvel_adr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def sample_nonoverlap_pos(self, x_range, y_range, max_tries=1000):
        radius_dict = {}
        z_dict = {}
        for ycb_object_name in self.YCB_OBJECT_NAMES:
            quat = self.model.geom(ycb_object_name).quat
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, quat)
            rotation = rotation.reshape(3, 3)
            geom_size = np.abs(rotation) @ self.model.geom(ycb_object_name).size

            radius_dict[ycb_object_name] = max(geom_size[:2])
            z_dict[ycb_object_name] = geom_size[2]

        order = sorted(radius_dict.keys(), key=lambda k: -radius_dict[k])

        nonoverlap_pos_dict = {}  # {ycb_object_name:[x, y, z]}

        for name in order:
            radius = radius_dict[name]
            x_min, x_max = x_range[0] + radius, x_range[1] - radius
            y_min, y_max = y_range[0] + radius, y_range[1] - radius

            placed = False
            tries = 0
            while tries < max_tries:
                tries += 1
                overlap = False

                xy_pos_candidate = np.array(
                    [np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)]
                )

                for j, pos in nonoverlap_pos_dict.items():
                    if np.linalg.norm(xy_pos_candidate - pos[:2]) < (
                        radius + radius_dict[j]
                    ):
                        overlap = True
                        break

                if not overlap:
                    z = z_dict[name]
                    nonoverlap_pos_dict[name] = np.array(
                        [xy_pos_candidate[0], xy_pos_candidate[1], z]
                    )
                    placed = True
                    break

            if not placed:
                raise RuntimeError(f"Failed to place object without overlap: {name}")

        return nonoverlap_pos_dict

    # def set_state():

    def close(self):
        if self.save_video:
            self.renderer.close()
            self.writer.close()
