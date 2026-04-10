import os
from datetime import datetime
from sqlite3 import adapt

import imageio.v2 as imageio
import mujoco
import numpy as np

from env import dual_arm_mjcf
from env.robot import AllegroHandV4
from env.robot import AllegroHandV4, InspireRH56F1, xArm7
# from env.robot import InspireRH56DFTP
from env.robot import xArm7
from env import dual_arm_mjcf
from collections import defaultdict
y_transform = np.array(
    [[0, 1, 0],
     [-1, 0, 0],
     [0, 0 , 1]]
)
class DualArmEnv:
    BARCODE_SCANNER_NAME = "barcode_scanner/barcode_scanner"
    YCB_OBJECT_NAMES = [
        # '003_cracker_box',
        # '004_sugar_box',
        '006_mustard_bottle',
        '010_potted_meat_can',
        '021_bleach_cleanser'
    ]

    def __init__(self, save_video=False, control_sim_dt=None):
        # mjcf = dual_arm_mjcf.load()
        self.target_ycb_objects = self.YCB_OBJECT_NAMES
        self.model = dual_arm_mjcf.build_model('allegro', self.target_ycb_objects)

        self.data = mujoco.MjData(self.model)

        self.control_sim_dt = control_sim_dt
        if control_sim_dt is not None:
            dt = float(self.model.opt.timestep)
            self._physics_substeps = max(1, int(round(control_sim_dt / dt)))
        else:
            self._physics_substeps = 10

        self.left_robot_arm = xArm7(self.model, self.data, "xarm7_left")
        self.right_robot_arm = xArm7(self.model, self.data, "xarm7_right")
        self.left_robot_hand = InspireRH56F1(self.model, self.data, 'xarm7_left/inspire_rh56f1_left')

        self.right_robot_hand = AllegroHandV4(
            self.model, self.data, "xarm7_right/allegro_right"
        )

        grasping_area_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'grasping_area')
        # self.left_ik_target_site_id = mujoco.mj_name2id(
        #     self.model, mujoco.mjtObj.mjOBJ_SITE, 'left_ik_target'
        # )
        
        self.grasping_area_pos = self.model.site_pos[grasping_area_site_id]
        self.grasping_area_size = self.model.site_size[grasping_area_site_id]
        self.grasping_area = [
            (
                self.grasping_area_pos[i] - self.grasping_area_size[i],
                self.grasping_area_pos[i] + self.grasping_area_size[i]
            )
            for i in range(2)
        ]
        
        self.barcode_scanner_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'barcode_scanner/body')

        self.ycb_object_barcode_site_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f'{name}/barcode') for name in self.target_ycb_objects]

        self.ycb_object_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{name}/body")
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

        # Track if any object has passed the z threshold
        self.threshold = False
        self.target_index = None

        # Axis length for coordinate visualization
        self.axis_length = 0.1  # 10cm axes
        self.distance = 1

        # Geom id sets for filtering contacts (names after MJCF attach prefixes)
        self._right_arm_geom_ids = set()
        self._ycb_geom_ids = set()
        for gid in range(self.model.ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if not gname:
                continue
            if gname.startswith("xarm7_right/"):
                self._right_arm_geom_ids.add(gid)
            for ycb_name in self.target_ycb_objects:
                if gname.startswith(f"{ycb_name}/"):
                    self._ycb_geom_ids.add(gid)
                    break

    def _ycb_object_name_for_geom(self, geom_id: int) -> str:
        gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        for ycb_name in self.target_ycb_objects:
            if gname.startswith(f"{ycb_name}/"):
                return ycb_name
        return "unknown"

    def apply_left_initial_state(self, left_arm_qpos=None, left_hand_qpos=None):
        self.left_robot_arm.servoj(left_arm_qpos)
        self.left_robot_hand.servo_joint(left_hand_qpos)

    def reset(self, left_arm_qpos=None, left_hand_qpos=None):
        initial_sate = self.model.key("initial_state").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, initial_sate)

        self._spawn_ycb_object(self.grasping_area[0], self.grasping_area[1])
        self.apply_left_initial_state(
            left_arm_qpos=left_arm_qpos, left_hand_qpos=left_hand_qpos
        )
        mujoco.mj_step(self.model, self.data, nstep=10000)  # 10s

        self.threshold = False
        self.target_index = None
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
        self.right_robot_arm.servoj(action["right_wrist_qpos"])
        self.right_robot_hand.servo_joint(action["right_hand_qpos"])
        # self.target_index = 2
        if self.threshold:
            if self.distance > 0.05:
                barcode_site_id = self.ycb_object_barcode_site_ids[self.target_index]
                target_pos = self.data.site_xpos[barcode_site_id].copy()
                # print("target_pos: ", target_pos)
                self.distance = self.left_arm_servo_joint(target_pos=target_pos)
            else:
                interp_steps = 5
                initial_pos = np.array([-0.3636,  0.3787,  0.4319])
                initial_quat = np.array([-0.3482, -0.1951, -0.5898, -0.7021])
                tcp_pose = self.left_robot_arm.get_tcp_pose()
                current_pos = tcp_pose[:3]
                current_quat = tcp_pose[3:]
                next_pos = current_pos + (initial_pos - current_pos) / interp_steps

                # Smooth quaternion update with one-step SLERP toward initial_quat.
                q0 = current_quat / (np.linalg.norm(current_quat) + 1e-12)
                q1 = initial_quat / (np.linalg.norm(initial_quat) + 1e-12)
                dot = float(np.dot(q0, q1))
                if dot < 0.0:
                    q1 = -q1
                    dot = -dot
                dot = np.clip(dot, -1.0, 1.0)

                t = 1.0 / interp_steps
                if dot > 0.9995:
                    next_quat = q0 + t * (q1 - q0)
                    next_quat /= (np.linalg.norm(next_quat) + 1e-12)
                else:
                    theta_0 = np.arccos(dot)
                    sin_theta_0 = np.sin(theta_0)
                    theta = theta_0 * t
                    s0 = np.sin(theta_0 - theta) / (sin_theta_0 + 1e-12)
                    s1 = np.sin(theta) / (sin_theta_0 + 1e-12)
                    next_quat = s0 * q0 + s1 * q1
                    next_quat /= (np.linalg.norm(next_quat) + 1e-12)

                self.left_robot_arm.servoj(np.concatenate([next_pos, next_quat]))
            
        
        observation = self.get_observation()
        mujoco.mj_step(self.model, self.data, nstep=self._physics_substeps)
        reward = self.get_reward()

        # if reward != -1.0:
            # print("reward : ", reward)
        return observation, reward


    def left_arm_servo_joint(self, target_pos):
        tcp_pose = self.left_robot_arm.get_tcp_pose()
        current_pos = tcp_pose[:3].copy()
        current_quat = tcp_pose[3:].copy()

        # Build desired orientation so TCP local +Z points to target direction.
        desired_z = target_pos - current_pos
        desired_z = desired_z /np.linalg.norm(desired_z)

        current_rot_flat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(current_rot_flat, current_quat)
        current_rot = current_rot_flat.reshape(3, 3)
        ref_x = current_rot[:, 0]
        desired_x = ref_x - np.dot(ref_x, desired_z) * desired_z

        alt = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        desired_x = alt - np.dot(alt, desired_z) * desired_z
    
        desired_x /= np.linalg.norm(desired_x)
        desired_y = np.cross(desired_z, desired_x)
        desired_y /= (np.linalg.norm(desired_y) + 1e-12)
        desired_rot = np.column_stack([desired_x, desired_y, desired_z])
        desired_rot = desired_rot @ y_transform

        desired_quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(desired_quat, desired_rot.reshape(-1))
        
        z_axis_length = 0.3
        desired_tcp_pos = target_pos - z_axis_length * desired_z

        interp_steps = 25
        next_pos = current_pos + (desired_tcp_pos - current_pos) / interp_steps
        self.left_robot_arm.servoj(np.concatenate([next_pos, desired_quat]))
        
        distance = np.linalg.norm(desired_tcp_pos - current_pos)
        distance = round(distance, 3)
        # print("distance : ", distance)
        return distance
        
    def get_observation(self):
        right_robot_arm_q_pos = self.right_robot_arm.get_q_pos()  # (7,)
        right_robot_hand_q_pos = (
            self.right_robot_hand.get_joint_pos()
        )  # (16,) for allegro, (12,) for inspire

        barcode_scanner_pose = np.empty(7)  # (7,)
        barcode_scanner_pose[:3] = self.data.xpos[self.barcode_scanner_body_id].copy()
        barcode_scanner_pose[3:] = self.data.xquat[self.barcode_scanner_body_id].copy()
        
        barcode_pose = np.empty((len(self.ycb_object_body_ids), 7))
        for i, barcode_site_id in enumerate(self.ycb_object_barcode_site_ids):
            barcode_pos = self.data.site_xpos[barcode_site_id].copy()
            barcode_xmat = self.data.site_xmat[barcode_site_id].copy()
            barcode_quat = np.empty(4)
            mujoco.mju_mat2Quat(barcode_quat, barcode_xmat)
            barcode_pose[i] = np.concatenate([barcode_pos, barcode_quat])
            
        ycb_object_poses = np.empty((len(self.ycb_object_body_ids), 7))  # (7,)
        for i, ycb_object_body_id in enumerate(self.ycb_object_body_ids):
            ycb_object_poses[i, :3] = self.data.xpos[ycb_object_body_id].copy()
            ycb_object_poses[i, 3:] = self.data.xquat[ycb_object_body_id].copy()

        # contact_data = []
        # for i in range(self.data.ncon):
        #     contact = self.data.contact[i]
        #     contact_force = np.zeros(6)
        #     mujoco.mj_contactForce(self.model, self.data, i, contact_force)
        #     contact_data.append(
        #         {
        #             "pos": contact.pos.copy(),
        #             "frame": contact.frame.copy(),
        #             "friction": contact.friction.copy(),
        #             "mu": contact.mu,
        #             "dim": contact.dim,
        #             "geom1": contact.geom[0],
        #             "geom2": contact.geom[1],
        #             "contact_force": contact_force,
        #         }
        #     )

        # Right arm <-> YCB: defaultdict[ycb_name, list[contact_dict]]
        # right_ycb_contacts = defaultdict(list)
        # for i in range(self.data.ncon):
        #     c = self.data.contact[i]
        #     g1, g2 = int(c.geom[0]), int(c.geom[1])
        #     right_touch_ycb = (g1 in self._right_arm_geom_ids and g2 in self._ycb_geom_ids) or (
        #         g2 in self._right_arm_geom_ids and g1 in self._ycb_geom_ids
        #     )
        #     if not right_touch_ycb:
        #         continue
        #     ycb_gid = g1 if g1 in self._ycb_geom_ids else g2
        #     ycb_key = self._ycb_object_name_for_geom(ycb_gid)
        #     f6 = np.zeros(6)
        #     mujoco.mj_contactForce(self.model, self.data, i, f6)
        #     R = np.asarray(c.frame, dtype=np.float64).reshape(3, 3, order="F")
        #     f_world = R @ f6[:3]
        #     right_ycb_contacts[ycb_key].append(
        #         {
        #             "geom1": g1,
        #             "geom2": g2,
        #             "geom1_name": mujoco.mj_id2name(
        #                 self.model, mujoco.mjtObj.mjOBJ_GEOM, g1
        #             ),
        #             "geom2_name": mujoco.mj_id2name(
        #                 self.model, mujoco.mjtObj.mjOBJ_GEOM, g2
        #             ),
        #             "contact_force6d": f6.copy(),
        #             "force_world": f_world.copy(),
        #             "pos": c.pos.copy(),
        #         }
        #     )

        return {
            # "left_robot_arm_q_pos": left_robot_arm_q_pos,
            "right_wrist_qpos": right_robot_arm_q_pos,
            # "left_robot_hand_q_pos": left_robot_hand_q_pos,
            "right_hand_qpos": right_robot_hand_q_pos,
            "barcode_scanner_pose": barcode_scanner_pose,
            'tactile_data': self.data.sensordata.copy(),
            "ycb_object_poses": ycb_object_poses,
            'barcode_pose': barcode_pose,
            # "contact": contact_data,
            # "contact": right_ycb_contacts,
        }

    def get_reward(self):
        # Check all YCB objects
        reward = 0
        tactile = self.data.sensordata
        wrist_pos = self.right_robot_arm.get_tcp_pose()[:3]
        if not self.threshold and self.target_index is None:
            min_distance = 100.0

            for i, ycb_object_body_id in enumerate(self.ycb_object_body_ids):
                obj_pos = self.data.xpos[ycb_object_body_id].copy()
                if obj_pos[0] > -0.1:
                    target_distance = float(np.linalg.norm(obj_pos[:3] - wrist_pos))
                    if min_distance >= target_distance:
                        min_distance = target_distance
                        self.target_id = ycb_object_body_id
                        self.target_index = i
                    
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.target_id)
            name = name.split('/')[0]
            print(f"target = {name}, target_index: {self.target_index}")
            print(self.threshold)
            
        if self.target_index is not None:
            ycb_object_body_id = self.ycb_object_body_ids[self.target_index]
            target_pos = self.data.xpos[ycb_object_body_id].copy()
        
            if target_pos[0] > -0.1 and target_pos[2] > 0.3 and \
                not self.threshold:
                self.threshold = True
                reward = 5.0

            elif target_pos[0] <= -0.1 and target_pos[2] <= 0.25 and \
                    self.threshold:
                self.threshold = False
                self.target_index = None
                self.distance = 1
                reward = 10.0  

            elif target_pos[0] > -0.1  and target_pos[2] <= 0.18 and \
                    self.threshold:
                self.threshold = False
                self.distance = 1
                reward = -4.0
            
            elif tactile.sum() /4 >= 1 and wrist_pos[0] > - 0.1 and \
                    not self.threshold:
                reward = tactile.sum() / 10 
                
            elif self.threshold:
                box_pos = [-0.25/2 - 0.1, 0.34/2 + 0.3, 0.3]
                reward = -round(float(np.linalg.norm(target_pos[:3] - box_pos)), 2) / 2.0
                
            else:
                reward = -round(float(np.linalg.norm(target_pos[:3] - wrist_pos)), 2) / 2.0
                
        print("reward: ", reward)
        return reward

    def _spawn_ycb_object(self, x_range, y_range):
        nonoverlap_pos_dict = self._sample_nonoverlap_pos(x_range, y_range)

        for ycb_object_name, pos in nonoverlap_pos_dict.items():
            # yaw = np.random.uniform(-np.pi, np.pi)  # TODO: restore
            yaw = np.random.uniform(0, np.pi/4)
            quat = np.array([np.cos(yaw*0.5), 0, 0, np.sin(yaw*0.5)])
            pose = np.concatenate([pos, quat])

            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'{ycb_object_name}/joint')
            joint_qpos_adr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[joint_qpos_adr:joint_qpos_adr + 7] = pose
            joint_qvel_adr = self.model.jnt_dofadr[joint_id]
            self.data.qvel[joint_qvel_adr:joint_qvel_adr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def _sample_nonoverlap_pos(self, x_range, y_range, max_steps: int = 1000):
        radius_dict = {}
        z_dict= {}
        for ycb_object_name in self.target_ycb_objects:
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'{ycb_object_name}/visual_geom')
            quat = self.model.geom_quat[geom_id]
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, quat)
            rotation = rotation.reshape(3, 3)
            geom_size = np.abs(rotation) @ self.model.geom_size[geom_id]
            radius_dict[ycb_object_name] = max(geom_size[:2])
            z_dict[ycb_object_name] = geom_size[2]
        ordered_ycb_object_name_list = sorted(radius_dict, key=radius_dict.get, reverse=True)

        nonoverlap_pos_dict = {}
        for ycb_object_name in ordered_ycb_object_name_list:
            radius = radius_dict[ycb_object_name]
            x_min, x_max = x_range[0] + radius, x_range[1] - radius
            y_min, y_max = y_range[0] + radius, y_range[1] - radius

            placed = False
            step = 0
            while step < max_steps:
                step += 1
                overlap = False

                xy_pos_candidate = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max)])
                
                for placed_ycb_object_name, placed_ycb_object_pos in nonoverlap_pos_dict.items():
                    if np.linalg.norm(xy_pos_candidate - placed_ycb_object_pos[:2]) < (radius + radius_dict[placed_ycb_object_name]):
                        overlap = True
                        break

                if not overlap:
                    z = z_dict[ycb_object_name]
                    nonoverlap_pos_dict[ycb_object_name] = np.array([xy_pos_candidate[0], xy_pos_candidate[1], z])
                    placed = True
                    break
            
            if not placed:
                raise RuntimeError(f'Failed to place object without overlap: {ycb_object_name}')
        
        return nonoverlap_pos_dict

    def add_triad(self, viewer, length=0.25, radius=0.003, z_length=0.25):
        """
        viewer.user_scn 에 (pos, quat) 기준의 3축 선/캡슐을 추가.
        quat은 MuJoCo 포맷(w, x, y, z).
        z_length: z축의 길이 (기본값 0.25m)
        """
        pos = self.data.xpos[self.barcode_scanner_body_id].copy()
        quat_wxyz = self.data.xquat[self.barcode_scanner_body_id].copy()
        # 매 프레임 누적 방지: user_scn의 geoms를 리셋
        viewer.user_scn.ngeom = 0

        # 로컬 좌표축 단위벡터
        ex = np.array([1.0, 0.0, 0.0], dtype=np.float64) #x
        ey = np.array([0.0, 1.0, 0.0], dtype=np.float64) #y
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float64) #z

        # quat로 축 회전: mju_rotVecQuat(out, vec, quat)
        q = np.asarray(quat_wxyz, dtype=np.float64)
        rx = np.zeros(3)
        ry = np.zeros(3)
        rz = np.zeros(3)
        mujoco.mju_rotVecQuat(rx, ex, q)
        mujoco.mju_rotVecQuat(ry, ey, q)
        mujoco.mju_rotVecQuat(rz, ez, q)

        p0 = np.asarray(pos, dtype=np.float64)
        p1x = p0 + length * rx
        p1y = p0 + length * ry
        p1z = p0 + z_length * rz

        # 커스텀 geom 하나 추가하는 헬퍼
        def _add_segment(pa, pb, rgba):
            g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            viewer.user_scn.ngeom += 1

            # "connector" 타입: 두 점을 잇는 캡슐/실린더/화살표 등을 만들어줌
            mujoco.mjv_initGeom(
                g,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                pos=np.array([0.0, 0.0, 0.0], dtype=np.float64),
                mat=np.eye(3, dtype=np.float64).reshape(-1),
                rgba=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            )

            # 2) from-to 커넥터 설정 (mjv_makeConnector 대신 mjv_connector)  [oai_citation:3‡mujoco.readthedocs.io](https://mujoco.readthedocs.io/en/2.3.7/APIreference/APIfunctions.html?utm_source=chatgpt.com)
            mujoco.mjv_connector(
                g,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                width=radius,
                from_=np.asarray(pa, dtype=np.float64),
                to=np.asarray(pb, dtype=np.float64),
            )

            # 3) 색 지정
            g.rgba[:] = rgba

        # x/y/z 축: 색은 취향대로 (RGBA)
        _add_segment(p0, p1x, np.array([1.0, 0.2, 0.2, 1.0]))  # x
        _add_segment(p0, p1y, np.array([0.2, 1.0, 0.2, 1.0]))  # y
        _add_segment(p0, p1z, np.array([0.2, 0.2, 1.0, 1.0]))  # z

        # viewer에 반영
        viewer.sync()
        
    def close(self):
        if self.save_video:
            self.renderer.close()
            self.writer.close()
