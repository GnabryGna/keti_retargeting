import mujoco
import numpy as np

from env import dual_arm_mjcf
# from env.robot import AllegroHandV4
# from env.robot import xArm7


class DualArmEnv:
    BARCODE_SCANNER_NAME = 'barcode_scanner/barcode_scanner'
    YCB_OBJECT_NAMES = [
        '003_cracker_box/003_cracker_box',
        '004_sugar_box/004_sugar_box',
        '005_tomato_soup_can/005_tomato_soup_can',
        '006_mustard_bottle/006_mustard_bottle',
        '010_potted_meat_can/010_potted_meat_can',
        '021_bleach_cleanser/021_bleach_cleanser'
    ]

    def __init__(self):
        mjcf = dual_arm_mjcf.load()
        self.model = mjcf.compile()
        self.data = mujoco.MjData(self.model)
        # self.left_robot_arm = xArm7()
        # self.right_robot_arm = xArm7()
        # self.left_robot_hand = AllegroHandV4()
        # self.right_robot_hand = AllegroHandV4()

        self.grasping_area_pos = self.model.site('grasping_area').pos
        self.grasping_area_size = self.model.site('grasping_area').size
        self.grasping_area = [(self.grasping_area_pos[i] - self.grasping_area_size[i],
                               self.grasping_area_pos[i] + self.grasping_area_size[i]) for i in range(2)]

    def reset(self):
        initial_sate = self.model.key('initial_state').id
        # TODO: YCB object의 초기 위치를 spawn 영역 밖으로 수정
        mujoco.mj_resetDataKeyframe(self.model, self.data, initial_sate)
        
        self.spawn_ycb_object()

        observation = self.get_observation()

        return observation

    def step(self, action):
        mujoco.mj_step(self.model, self.data, nstep=1)
        observation = self.get_observation()
        reward = self.get_reward()

        return reward, observation

    def get_observation(self):
        # left_robot_arm_qpos = self.left_robot_arm.get_qpos() # (7,)
        # right_robot_arm_qpos = self.right_robot_arm.get_qpos() # (7,)
        # left_robot_hand_qpos = self.left_robot_hand.get_qpos() # (16,)
        # right_robot_hand_qpos = self.right_robot_hand.get_qpos() # (16,)
        barcode_scanner_pose = np.concatenate([ # (7,)
            self.data.body(self.BARCODE_SCANNER_NAME).xpos.copy(), # (3,)
            self.data.body(self.BARCODE_SCANNER_NAME).xquat.copy() # (4,)
        ])
        ycb_object_pose_list = []
        for ycb_object_name in self.YCB_OBJECT_NAMES:
            ycb_object_pose = np.concatenate([ # (7,)
                self.data.body(ycb_object_name).xpos.copy(), # (3,)
                self.data.body(ycb_object_name).xquat.copy() # (4,)
            ])
            ycb_object_pose_list.append(ycb_object_pose)
        # TODO: Add tactile(force) data

        return np.concatenate([
            # left_robot_arm_qpos,
            # right_robot_arm_qpos,
            # left_robot_hand_qpos,
            # right_robot_hand_qpos,
            barcode_scanner_pose,
            ycb_object_pose
        ])

    def get_reward(self):
        pass

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
        radius_dict = {ycb_object_name:float(self.model.geom(ycb_object_name).rbound)*0.01 for ycb_object_name in self.YCB_OBJECT_NAMES} # {ycb_object_name:radius}
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