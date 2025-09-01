import mujoco


class RobotArmBase:
    def __init__(self, model, data, joint_names, actuator_names, tcp_site_name):
        self.model = model
        self.data = data
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) for joint_name in joint_names]
        self.joint_qpos_adr = self.model.jnt_qposadr[self.joint_ids]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name) for actuator_name in actuator_names]
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, tcp_site_name)
        
    def get_q_pos(self):
        q_pos = [self.data.qpos[qpos_adr].copy().item() for qpos_adr in self.joint_qpos_adr] # (njoint,)
        return q_pos
    
    def get_q_vel(self):
        q_vel = [self.data.qvel[qpos_adr].copy().item() for qpos_adr in self.joint_qpos_adr] # (njoint,)
        return q_vel
    
    def get_tcp_pose(self):
        p = self.data.site_xpos[self.tcp_site_id].copy() # (3,)
        R = self.data.site_xmat[self.tcp_site_id].reshape(3, 3).copy() # (3, 3)
        return p, R
    
    def set_control(self, input):
        for index, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = input[index]
    
    def set_q_pos(self, target_q_pos):
        for index, qpos_adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[qpos_adr] = target_q_pos[index]
        mujoco.mj_forward(self.model, self.data)
        self.set_control(target_q_pos)

    '''
    def set_tcp_pose(self, target_tcp_pose):
        target_q_pos = ik(target_tcp_pose) # TODO: Inverse kinematics
        self.set_q_pos(target_q_pos)
    '''
    
    def servoj(self, target_q_pos):
        self.set_control(target_q_pos)


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

    def __init__(self, model, data):
        super().__init__(model, data, self.JOINT_NAMES, self.ACTUATOR_NAMES, self.TCP_SITE_NAME)


class RobotHandBase:
    def __init__(self, model, data, joint_names, actuator_names):
        self.model = model
        self.data = data
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) for joint_name in joint_names]
        self.joint_qpos_adr = self.model.jnt_qposadr[self.joint_ids]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name) for actuator_name in actuator_names]

    def get_q_pos(self):
        q_pos = [self.data.qpos[qpos_adr].copy().item() for qpos_adr in self.joint_qpos_adr] # (njoint,)
        return q_pos
    
    def get_q_vel(self):
        q_vel = [self.data.qvel[qpos_adr].copy().item() for qpos_adr in self.joint_qpos_adr] # (njoint,)
        return q_vel
    
    def set_control(self, input):
        for index, actuator_id in enumerate(self.actuator_ids):
            self.data.ctrl[actuator_id] = input[index]
    
    def set_q_pos(self, target_q_pos):
        for index, qpos_adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[qpos_adr] = target_q_pos[index]
        mujoco.mj_forward(self.model, self.data)
        self.set_control(target_q_pos)
    
    def servoj(self, target_q_pos):
        self.set_control(target_q_pos)


# Allegro Hand V4 MuJoCo Interface
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

    def __init__(self, model, data):
        super().__init__(model, data, self.JOINT_NAMES, self.ACTUATOR_NAMES)


class InspireRH56DFTP(RobotHandBase):
    JOINT_NAMES = [
        'thumb_proximal_1_joint',
        'thumb_proximal_2_joint',
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
        'thumb_proximal_1',
        'thumb_proximal_2',
        'thumb_middle',
        'thumb_distal',
        'index_proximal',
        'index_distal',
        'middle_proximal',
        'middle_distal',
        'ring_proximal',
        'ring_distal',
        'little_proximal',
        'little_distal',
    ]

    def __init__(self, model, data):
        super().__init__(model, data, self.JOINT_NAMES, self.ACTUATOR_NAMES)