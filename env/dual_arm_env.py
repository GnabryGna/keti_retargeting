from datetime import datetime

import imageio.v2 as imageio
import mujoco
import numpy as np

from env import dual_arm_mjcf
# from env.robot import AllegroHandV4
from env.robot import InspireRH56DFTP
from env.robot import xArm7


class DualArmEnv:
    BARCODE_SCANNER_BODY_NAME = 'barcode_scanner/barcode_scanner'
    YCB_OBJECT_BODY_NAMES = [
        '003_cracker_box/003_cracker_box',
        '004_sugar_box/004_sugar_box',
        '005_tomato_soup_can/005_tomato_soup_can',
        '006_mustard_bottle/006_mustard_bottle',
        '010_potted_meat_can/010_potted_meat_can',
        '021_bleach_cleanser/021_bleach_cleanser'
    ]

    def __init__(self, save_video=False):
        mjcf = dual_arm_mjcf.load()
        self.model = mjcf.compile()
        self.data = mujoco.MjData(self.model)

        self.left_robot_arm = xArm7(self.model, self.data, 'xarm7_left')
        self.right_robot_arm = xArm7(self.model, self.data, 'xarm7_right')
        # self.left_robot_hand = AllegroHandV4(self.model, self.data, 'xarm7_left/allegro_left')
        # self.right_robot_hand = AllegroHandV4(self.model, self.data 'xarm7_right/allegro_right')
        self.left_robot_hand = InspireRH56DFTP(self.model, self.data, 'xarm7_left/inspire_rh56dftp_left')
        self.right_robot_hand = InspireRH56DFTP(self.model, self.data, 'xarm7_right/inspire_rh56dftp_right')

        self.grasping_area_pos = self.model.site('grasping_area').pos
        self.grasping_area_size = self.model.site('grasping_area').size
        self.grasping_area = [(self.grasping_area_pos[i] - self.grasping_area_size[i],
                               self.grasping_area_pos[i] + self.grasping_area_size[i]) for i in range(2)]
        
        self.barcode_scanner_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.BARCODE_SCANNER_BODY_NAME)
        self.ycb_object_body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in self.YCB_OBJECT_BODY_NAMES]

        self.save_video = save_video
        self.writer = None
        if self.save_video:
            self.video_fps = 60
            self.frame_dt = 1.0/self.video_fps
            self.next_frame_time = 0.0
            width = self.model.vis.global_.offwidth
            height = self.model.vis.global_.offheight
            self.renderer = mujoco.Renderer(self.model, width=width, height=height)

    def reset(self):
        initial_sate = self.model.key('initial_state').id
        mujoco.mj_resetDataKeyframe(self.model, self.data, initial_sate)
        # self.spawn_ycb_object()
        mujoco.mj_step(self.model, self.data, nstep=1000) # TODO: adjust settle steps
        observation = self.get_observation()

        if self.save_video:
            if self.writer is not None:
                self.writer.close()
            timestamp = datetime.now().strftime('%y%m%d_%H_%M_%S')
            filename = f'{timestamp}.mp4'
            self.writer = imageio.get_writer(filename, fps=self.video_fps, codec="libx264", pixelformat="yuv420p", macro_block_size=None)
            self.next_frame_time = self.data.time

        return observation

    def step(self, action):
        # self.left_robot_arm.servoj(action['left_robot_arm_q_pos'])
        # self.right_robot_arm.servoj(action['right_robot_arm_q_pos'])
        # self.left_robot_hand.servoj(action['left_robot_hand_q_pos'])
        # self.right_robot_hand.servoj(action['right_robot_hand_q_pos'])
        mujoco.mj_step(self.model, self.data)
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
        left_robot_arm_q_pos = self.left_robot_arm.get_q_pos() # (7,)
        right_robot_arm_q_pos = self.right_robot_arm.get_q_pos() # (7,)
        left_robot_hand_q_pos = self.left_robot_hand.get_q_pos() # (16,)
        right_robot_hand_q_pos = self.right_robot_hand.get_q_pos() # (16,)

        barcode_scanner_pose = np.empty(7) # (7,)
        barcode_scanner_pose[:3] = self.data.xpos[self.barcode_scanner_body_id].copy()
        barcode_scanner_pose[3:] = self.data.xquat[self.barcode_scanner_body_id].copy()

        ycb_object_poses = np.empty((len(self.ycb_object_body_ids), 7)) # (7,)
        for i, ycb_objetc_body_id in enumerate(self.ycb_object_body_ids):
            ycb_object_poses[i, :3] = self.data.xpos[ycb_objetc_body_id].copy()
            ycb_object_poses[i, 3:] = self.data.xquat[ycb_objetc_body_id].copy()

        # TODO: Add tactile(force) data

        return {
            'left_robot_arm_q_pos': left_robot_arm_q_pos,
            'right_robot_arm_q_pos': right_robot_arm_q_pos,
            'left_robot_hand_q_pos': left_robot_hand_q_pos,
            'right_robot_hand_q_pos': right_robot_hand_q_pos,
            'barcode_scanner_pose': barcode_scanner_pose,
            'ycb_object_poses': ycb_object_poses
        }

    def get_reward(self):
        return None
    
    def close(self):
        if self.save_video:
            self.renderer.close()
            self.writer.close()

    def spawn_ycb_object(self, settle_steps=1000):
        nonoverlap_xy_pos_dict = self.sample_nonoverlap_xy_pos(self.grasping_area[0], self.grasping_area[1])

        for ycb_object_name, xy_pos in nonoverlap_xy_pos_dict.items():
            x = xy_pos[0]
            y = xy_pos[1]
            z = 0.12
            pos = np.array([x, y, z])

            # yaw = np.random.uniform(-np.pi, np.pi)

            qpos_adr = self.model.joint(ycb_object_name).qposadr.item()
            dof_adr = self.model.joint(ycb_object_name).dofadr.item()
            
            self.data.qpos[qpos_adr:qpos_adr + 3] = pos
            # self.data.qpos[qpos_adr + 3:qpos_adr + 7] = quat
            self.data.qvel[dof_adr:dof_adr + 6] = 0.0


        mujoco.mj_forward(self.model, self.data)

        # for _ in range(settle_steps):
            # mujoco.mj_step(self.model, self.data, nstep=1)
            
    def sample_random_pose(self, x_range, y_range, z_range):
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        z = np.random.uniform(z_range[0], z_range[1])
        pos = np.array([x, y, z])

        # Shoemake/Marsaglia method
        u1, u2, u3 = np.random.random(3)
        quat = np.array([ # [w, x, y, z]
            np.sqrt(u1)*np.cos(2*np.pi*u3),
            np.sqrt(1 - u1)*np.sin(2*np.pi*u2),
            np.sqrt(1 - u1)*np.cos(2*np.pi*u2),
            np.sqrt(u1)*np.sin(2*np.pi*u3)
        ])

        return pos, quat
    
    # def sample_random_pose_upright(self, x_range, y_range):
    #     x = np.random.uniform(x_range[0], x_range[1])
    #     y = np.random.uniform(y_range[0], y_range[1])
    #     z = table_z + lift                     # 테이블 바로 위에 놓기
    #     yaw = np.random.uniform(-np.pi, np.pi) # yaw만 랜덤
    #     c, s = np.cos(yaw/2), np.sin(yaw/2)
    #     quat = np.array([c, 0.0, 0.0, s])  # (w,x,y,z)
    #     return np.array([x, y, z]), quat

    def sample_nonoverlap_xy_pos(self, x_range, y_range, max_tries=1000):
        radius_dict = {ycb_object_name: float(self.model.geom(ycb_object_name).rbound)*0.01 for ycb_object_name in self.YCB_OBJECT_BODY_NAMES}
        nonoverlap_xy_pos_dict = {} # {ycb_object_name:[x, y]}
        tries = 0

        for i, radius in radius_dict.items():
            overlap = False
            while tries < max_tries:
                tries += 1
                xy_pos_candidate = np.array([np.random.uniform(x_range[0] + radius, x_range[1] - radius),
                                             np.random.uniform(y_range[0] + radius, y_range[1] - radius)])
                for j, xy_pos in nonoverlap_xy_pos_dict.items():
                    if np.linalg.norm(xy_pos_candidate - xy_pos) < (radius + radius_dict[j]):
                        overlap = True
                        break
                if overlap == False:
                    nonoverlap_xy_pos_dict[i] = xy_pos_candidate
                    break
            
            if overlap == True:
                raise RuntimeError('Failed to place object without overlap.')
        
        return nonoverlap_xy_pos_dict