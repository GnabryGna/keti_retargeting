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
        # "003_cracker_box/003_cracker_box",
        # "004_sugar_box/004_sugar_box",
        # "005_tomato_soup_can/005_tomato_soup_can",
        "006_mustard_bottle/006_mustard_bottle",
        # "010_potted_meat_can/010_potted_meat_can",
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

        # Track if any object has passed the z threshold
        self.has_object_passed_threshold = False
        self.scan_touch_reward_given = np.zeros(
            len(self.ycb_object_body_ids), dtype=bool
        )
        self.target_ycb_x = None
        self.target_ycb_z = None
        self.target_index = None
        # Axis length for coordinate visualization
        self.axis_length = 0.1  # 10cm axes

    def reset(self):
        initial_sate = self.model.key("initial_state").id
        mujoco.mj_resetDataKeyframe(self.model, self.data, initial_sate)

        self.spawn_ycb_object()
        mujoco.mj_step(self.model, self.data, nstep=10000)  # 10s

        # Reset threshold flag
        self.has_object_passed_threshold = False
        self.scan_touch_reward_given = np.zeros(
            len(self.ycb_object_body_ids), dtype=bool
        )
        self.target_ycb_x = None
        self.target_ycb_z = None
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
        # self.left_robot_arm.set_tcp_pose(action["left_wrist_qpos"])
        # self.right_robot_arm.set_tcp_pose(action["right_wrist_qpos"])

        self.left_robot_arm.servoj(action["left_wrist_qpos"])
        self.right_robot_arm.servoj(action["right_wrist_qpos"])
        self.left_robot_hand.servoj(action["left_hand_qpos"])
        self.right_robot_hand.servoj(action["right_hand_qpos"])
        observation = self.get_observation()
        # for objects in observation["ycb_object_poses"]:
        #     print("ycb pose : ", objects[:3])
        mujoco.mj_step(self.model, self.data, nstep=10)  # 1/(0.001*10) = 100Hz
        reward = self.get_reward()
        if reward != -1.0:
            print("reward : ", reward)
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
        # Get barcode scanner position and z-axis direction
        scanner_pos = self.data.xpos[self.barcode_scanner_body_id].copy()
        scanner_quat = self.data.xquat[self.barcode_scanner_body_id].copy()

        # Calculate z-axis direction in world frame
        ez_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        z_axis_world = np.zeros(3, dtype=np.float64)
        mujoco.mju_rotVecQuat(z_axis_world, ez_local, scanner_quat)
        z_axis_world = z_axis_world / np.linalg.norm(z_axis_world)

        # Define z-axis line segment: from scanner_pos to scanner_pos + 0.25 * z_axis
        z_axis_length = 0.25
        z_axis_end = scanner_pos + z_axis_length * z_axis_world

        # Check all YCB objects
        if not self.has_object_passed_threshold:
            for i, ycb_object_body_id in enumerate(self.ycb_object_body_ids):
                # Skip already scanned objects
                if self.scan_touch_reward_given[i]:
                    continue
                
                obj_pos = self.data.xpos[ycb_object_body_id].copy()
                x_pos = obj_pos[0]  # x coordinate (index 0)
                z_pos = obj_pos[2]  # z coordinate (index 2)
                if z_pos > 0.3 and not self.has_object_passed_threshold:
                    self.target_ycb_x = x_pos
                    self.target_ycb_z = z_pos
                    self.target_index = i
                    break
        
        # Update target object's current position on every step
        if self.target_index is not None:
            ycb_object_body_id = self.ycb_object_body_ids[self.target_index]
            obj_pos = self.data.xpos[ycb_object_body_id].copy()
            self.target_ycb_x = obj_pos[0]
            self.target_ycb_z = obj_pos[2]
        
        if self.target_ycb_x is not None or self.target_ycb_z is not None and self.target_index is not None:
            # Condition 1: z position > 0.3 (first success - pick up object)
            if self.target_ycb_z > 0.3 and not self.has_object_passed_threshold:
                self.has_object_passed_threshold = True
                return 0.0

            # Condition 3: Object touches barcode scanner z-axis (0.25 length) AND z_pos >= 0.3
            if (self.target_ycb_z >= 0.3) and (
                not self.scan_touch_reward_given[self.target_index]
            ):

                # 선분 endpoints
                p0 = scanner_pos
                p1 = z_axis_end
                v = p1 - p0
                vv = float(np.dot(v, v)) + 1e-12  # divide-by-zero 방지

                # 이 body에 속한 geom들 전부 검사 (표면 기준 접촉 판정)
                # 실제 body id 가져오기 (target_index는 리스트 위치, body_id는 MuJoCo 인덱스)
                ycb_body_id = self.ycb_object_body_ids[self.target_index]
                gadr = self.model.body_geomadr[ycb_body_id]
                gnum = self.model.body_geomnum[ycb_body_id]

                # z축(blue) 자체의 두께(너가 그린 triad 캡슐 radius랑 맞추면 좋음)
                axis_radius = 0.004  # 예: triad 그릴 때 radius=0.004 썼다면 동일하게

                for gid in range(gadr, gadr + gnum):
                    # geom 중심(world)
                    cg = self.data.geom_xpos[gid].copy()

                    # 선분-점 최소거리(선분으로 clamp)
                    w = cg - p0
                    t = float(np.dot(w, v) / vv)
                    t = max(0.0, min(1.0, t))
                    closest = p0 + t * v
                    dist = float(np.linalg.norm(cg - closest))

                    # geom "대략 반경" 계산 (shape별로 다름)
                    gtype = int(self.model.geom_type[gid])
                    size = self.model.geom_size[gid].copy()

                    # MuJoCo geom_size 의미(대표):
                    # sphere: [r,0,0], capsule: [r, halfLength, 0], cylinder: [r, halfLength, 0], box: [hx,hy,hz]
                    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                        r_geom = float(size[0])
                    elif gtype in (
                        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
                    ):
                        r_geom = float(size[0])
                    elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                        # box는 half-extent라서 대각 반경을 사용
                        r_geom = float(np.linalg.norm(size[:3]))
                    else:
                        # mesh 등은 대충 안전하게 좀 크게(또는 0.05 같은 상수) 잡는게 현실적
                        r_geom = 0.05

                    # "닿음" 판정: 중심-선분 거리 <= (geom반경 + 축반경 + 여유)
                    margin = 0.015  # 15mm 여유로 증가 (감지 거리 확대)
                    if dist <= (r_geom + axis_radius + margin):
                        self.scan_touch_reward_given[self.target_index] = True
                        return 0.0

            # Condition 2: x position < -0.2 AND z position < 0.3 (second success - place in box)
            if (
                self.target_ycb_x < -0.2
                and self.target_ycb_z < 0.3
                and self.has_object_passed_threshold
            ):
                self.has_object_passed_threshold = False
                self.target_ycb_x = None
                self.target_ycb_z = None
                self.target_index = None
                return 0.0

            # Drop penalty: object fell (but only if scan not completed)
            if self.target_ycb_z < 0.2 and self.has_object_passed_threshold:
                # Don't penalize if scan already completed (placing in box)
                if not self.scan_touch_reward_given[self.target_index]:
                    self.has_object_passed_threshold = False
                    return -2.0

        return -1.0

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
                    # Ensure object spawns on table surface
                    # z_dict[name] is the half-height of the object
                    # Table surface is at grasping_area_pos[2] = 0
                    # So object center should be at z = table_height + object_half_height
                    table_height = self.grasping_area_pos[2]  # Should be 0
                    object_half_height = z_dict[name]
                    z = table_height + object_half_height

                    # Additional check: ensure position is within grasping area bounds
                    # (x, y) is already constrained, but double-check z
                    nonoverlap_pos_dict[name] = np.array(
                        [xy_pos_candidate[0], xy_pos_candidate[1], z]
                    )
                    placed = True
                    break

            if not placed:
                raise RuntimeError(f"Failed to place object without overlap: {name}")

        return nonoverlap_pos_dict

    # def set_state():

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
        ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)

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
        p1z = p0 + z_length * rz  # z축만 0.25 길이로 설정

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
