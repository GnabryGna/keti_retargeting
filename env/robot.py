import mujoco
import numpy as np


class RobotArmBase:
    def __init__(self, model, data, joint_names, actuator_names, tcp_site_name):
        self.model = model
        self.data = data
        self.joint_names = joint_names
        self.actuator_names = actuator_names
        self.tcp_site_name = tcp_site_name

        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) for joint_name in joint_names]
        self.actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name) for actuator_name in actuator_names]
        self.tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tcp_site_name)

        self.joint_qpos_adr = model.jnt_qposadr[self.joint_ids]
        self.joint_dof_adr = model.jnt_dofadr[self.joint_ids]
        
    def get_q_pos(self):
        return self.data.qpos[self.joint_qpos_adr].copy() # (njoint,)
    
    def get_q_vel(self):
        return self.data.qvel[self.joint_dof_adr].copy() # (njoint,)
    
    def get_tcp_pose(self):
        pos = self.data.site_xpos[self.tcp_site_id].copy() # (3,)
        xmat = self.data.site_xmat[self.tcp_site_id].copy() # (9,)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, xmat) # [w, x, y, z]
        pose = np.concatenate([pos, quat])
        return pose
    
    def set_q_pos(self, target_q_pos):
        self.data.qpos[self.joint_qpos_adr] = target_q_pos
        self.data.ctrl[self.actuator_ids] = target_q_pos
        mujoco.mj_forward(self.model, self.data)
        self.data.qfrc_applied[self.joint_dof_adr] = self.data.qfrc_bias[self.joint_dof_adr]

    def set_tcp_pose(self, target_tcp_pose):
        ik_result = self.qpos_from_site_pose(target_pos=target_tcp_pose[:3], target_quat=target_tcp_pose[3:])
        self.set_q_pos(ik_result['qpos'])
    
    def servoj(self, target_q_pos):
        self.data.ctrl[self.actuator_ids] = target_q_pos
        self.data.qfrc_applied[self.joint_dof_adr] = self.data.qfrc_bias[self.joint_dof_adr]

    def qpos_from_site_pose(
        self,
        target_pos=None,
        target_quat=None,
        tol=1e-14,
        rot_weight=1.0,
        regularization_threshold=0.1,
        regularization_strength=3e-2,
        max_update_norm=2.0,
        progress_thresh=20.0,
        max_steps=100
    ):  
        dtype = self.data.qpos.dtype

        if target_pos is not None and target_quat is not None:
            jac = np.empty((6, self.model.nv), dtype=dtype)
            err = np.empty(6, dtype=dtype)
            jac_pos, jac_rot = jac[:3], jac[3:]
            err_pos, err_rot = err[:3], err[3:]
        else:
            jac = np.empty((3, self.model.nv), dtype=dtype)
            err = np.empty(3, dtype=dtype)
            if target_pos is not None:
                jac_pos, jac_rot = jac, None
                err_pos, err_rot = err, None
            elif target_quat is not None:
                jac_pos, jac_rot = None, jac
                err_pos, err_rot = None, err
            else:
                raise ValueError('At least one of `target_pos` or `target_quat` must be specified.')

        update_nv = np.zeros(self.model.nv, dtype=dtype)

        if target_quat is not None:
            site_xquat = np.empty(4, dtype=dtype)
            neg_site_xquat = np.empty(4, dtype=dtype)
            err_rot_quat = np.empty(4, dtype=dtype)

        data = mujoco.MjData(self.model)
        data.qpos[:] = self.data.qpos
        mujoco.mj_fwdPosition(self.model, data)

        site_id = self.tcp_site_id
        site_xpos = data.site_xpos[site_id]
        site_xmat = data.site_xmat[site_id]

        dof_indices = self.joint_dof_adr

        success = False
        steps = 0

        for steps in range(max_steps):
            err_norm = 0.0

            if target_pos is not None:
                err_pos[:] = target_pos - site_xpos
                err_norm += float(np.linalg.norm(err_pos))

            if target_quat is not None:
                mujoco.mju_mat2Quat(site_xquat, site_xmat)
                mujoco.mju_negQuat(neg_site_xquat, site_xquat)
                mujoco.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
                mujoco.mju_quat2Vel(err_rot, err_rot_quat, 1)
                err_norm += float(np.linalg.norm(err_rot)*rot_weight)

            if err_norm < tol:
                success = True
                break

            mujoco.mj_jacSite(self.model, data, jac_pos, jac_rot, site_id)
            jac_joints = jac[:, dof_indices]
            reg_strength = regularization_strength if err_norm > regularization_threshold else 0.0
            update_joints = self.nullspace_method(jac_joints, err, regularization_strength=reg_strength)

            update_norm = float(np.linalg.norm(update_joints))
            if update_norm <= 0.0:
                print(f'Stopping due to zero update at step {steps}')
                break

            progress_criterion = err_norm / update_norm
            if progress_criterion > progress_thresh:
                print(
                    f'Stopping due to insufficient progress at step {steps}: '
                    f'err_norm/update_norm={progress_criterion:g} > {progress_thresh:g}'
                )
                break

            if update_norm > max_update_norm:
                update_joints *= (max_update_norm / update_norm)

            update_nv[...] = 0.0
            update_nv[dof_indices] = update_joints

            mujoco.mj_integratePos(self.model, data.qpos, update_nv, 1)
            mujoco.mj_fwdPosition(self.model, data)

            # For debug
            # print(f'Step {steps:2d}: err_norm={err_norm:<10.3g} update_norm={update_norm:<10.3g}')

        if not success and steps == max_steps - 1:
            print(f'Failed to converge after {max_steps} steps: err_norm={err_norm:g}')

        qpos = data.qpos[self.joint_qpos_adr].copy()

        return {'qpos': qpos, 'err_norm': err_norm, 'steps': steps, 'success': success}

    def nullspace_method(self, jac_joints, delta, regularization_strength=0.0):
        hess_approx = jac_joints.T @ jac_joints
        joint_delta = jac_joints.T @ delta
        if regularization_strength > 0.0:
            hess_approx = hess_approx + np.eye(hess_approx.shape[0], dtype=hess_approx.dtype)*regularization_strength
            return np.linalg.solve(hess_approx, joint_delta)
        else:
            return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]


class xArm7(RobotArmBase):
    JOINT_NAMES = [
        'joint1',
        'joint2',
        'joint3',
        'joint4',
        'joint5',
        'joint6',
        'joint7'
    ]
    ACTUATOR_NAMES = [
        'act1',
        'act2',
        'act3',
        'act4',
        'act5',
        'act6',
        'act7'
    ]
    TCP_SITE_NAME = "attachment_site"

    def __init__(self, model, data, prefix=None):
        if prefix:
            joint_names = []
            for joint_id in range(model.njnt):
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                if joint_name.startswith(prefix) and "/" not in joint_name[len(prefix) + 1:]:
                    joint_names.append(joint_name)
            
            actuator_names = []
            for actuator_id in range(model.nu):
                actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
                if actuator_name.startswith(prefix) and "/" not in actuator_name[len(prefix) + 1:]:
                    actuator_names.append(actuator_name)
            
            tcp_site_name = f"{prefix}/{self.TCP_SITE_NAME}"

        else:
            joint_names = self.JOINT_NAMES
            actuator_names = self.ACTUATOR_NAMES
            tcp_site_name = self.TCP_SITE_NAME

        super().__init__(model, data, joint_names, actuator_names, tcp_site_name)


class RobotHandBase:
    def __init__(self, model, data, joint_names, actuator_names, passive_joint_index=None):
        self.model = model
        self.data = data
        self.joint_names = joint_names
        self.actuator_names = actuator_names

        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) for joint_name in joint_names]
        self.actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name) for actuator_name in actuator_names]

        self.joint_qpos_adr = model.jnt_qposadr[self.joint_ids]
        self.joint_dof_adr = model.jnt_dofadr[self.joint_ids]

        self.underactuated = len(joint_names) != len(actuator_names)
        if self.underactuated:
            self.passive_joint_index = passive_joint_index

    def get_q_pos(self):
        return self.data.qpos[self.joint_qpos_adr].copy() # (njoint,)
    
    def get_q_vel(self):
        return self.data.qvel[self.joint_dof_adr].copy() # (njoint,)
    
    def set_q_pos(self, target_q_pos):
        self.data.qpos[self.joint_qpos_adr] = target_q_pos
        if self.underactuated:
            actuator_input = np.delete(target_q_pos, self.passive_joint_index)
            self.data.ctrl[self.actuator_ids] = actuator_input
        else:
            self.data.ctrl[self.actuator_ids] = target_q_pos
        mujoco.mj_forward(self.model, self.data)
    
    def servoj(self, target_q_pos):
        if self.underactuated:
            actuator_input = np.delete(target_q_pos, self.passive_joint_index)
            self.data.ctrl[self.actuator_ids] = actuator_input
        else:
            self.data.ctrl[self.actuator_ids] = target_q_pos


class AllegroHandV4(RobotHandBase):
    JOINT_NAMES = [
        'ffj0',
        'ffj1',
        'ffj2',
        'ffj3',
        'mfj0',
        'mfj1',
        'mfj2',
        'mfj3',
        'rfj0',
        'rfj1',
        'rfj2',
        'rfj3',
        'thj0',
        'thj1',
        'thj2',
        'thj3'
    ]
    ACTUATOR_NAMES = [
        'ffa0',
        'ffa1',
        'ffa2',
        'ffa3',
        'mfa0',
        'mfa1',
        'mfa2',
        'mfa3',
        'rfa0',
        'rfa1',
        'rfa2',
        'rfa3',
        'tha0',
        'tha1',
        'tha2',
        'tha3'
    ]

    def __init__(self, model, data, prefix=None):
        if prefix:
            joint_names = []
            for joint_id in range(model.njnt):
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                if joint_name.startswith(prefix) and "/" not in joint_name[len(prefix) + 1:]:
                    joint_names.append(joint_name)
            
            actuator_names = []
            for actuator_id in range(model.nu):
                actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
                if actuator_name.startswith(prefix) and "/" not in actuator_name[len(prefix) + 1:]:
                    actuator_names.append(actuator_name)

        else:
            joint_names = self.JOINT_NAMES
            actuator_names = self.ACTUATOR_NAMES

        super().__init__(model, data, joint_names, actuator_names)


class InspireRH56DFTP(RobotHandBase):
    JOINT_NAMES = [
        'thumb_proximal_rotation_joint',
        'thumb_proximal_bending_joint',
        'thumb_middle_joint',
        'thumb_distal_joint',
        'index_proximal_joint',
        'index_distal_joint',
        'middle_proximal_joint',
        'middle_distal_joint',
        'ring_proximal_joint',
        'ring_distal_joint',
        'little_proximal_joint',
        'little_distal_joint',
    ]
    ACTUATOR_NAMES = [
        'thumb_proximal_rotation',
        'thumb_proximal_bending',
        'index_proximal',
        'middle_proximal',
        'ring_proximal',
        'little_proximal',
    ]
    PASSIVE_JOINT_INDEX = [2, 3, 5, 7, 9, 11]

    def __init__(self, model, data, prefix=None):
        if prefix:
            joint_names = []
            for joint_id in range(model.njnt):
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                if joint_name.startswith(prefix) and "/" not in joint_name[len(prefix) + 1:]:
                    joint_names.append(joint_name)
            
            actuator_names = []
            for actuator_id in range(model.nu):
                actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
                if actuator_name.startswith(prefix) and "/" not in actuator_name[len(prefix) + 1:]:
                    actuator_names.append(actuator_name)

        else:
            joint_names = self.JOINT_NAMES
            actuator_names = self.ACTUATOR_NAMES
        
        super().__init__(model, data, joint_names, actuator_names, self.PASSIVE_JOINT_INDEX)