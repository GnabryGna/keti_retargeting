# Adapted from https://github.com/iMSquared/def-oricorn/blob/main/util/transform_util.py
from mujoco_inspire.utils import control_utils as C
from mujoco_inspire.utils import transform as T

import torch


def qaction(quat, pos):
    return C.quat_mul(
        C.quat_mul(quat, torch.cat([pos, torch.zeros_like(pos[..., :1])], dim=-1)),
        torch.tensor(T.quat_inverse(quat)),
    )[..., :3]


def pq_multi(pos1, quat1, pos2, quat2):
    return qaction(quat1, pos2) + pos1, C.quat_mul(quat1, quat2)


def fr3_dh_params(joint_state):
    dh_params = torch.tensor(
        [
            [0.0, 0.333, 0.0, joint_state[0]],
            [0.0, 0.0, -torch.pi / 2, joint_state[1]],
            [0.0, 0.316, torch.pi / 2, joint_state[2]],
            [0.0825, 0.0, torch.pi / 2, joint_state[3]],
            [-0.0825, 0.384, -torch.pi / 2, joint_state[4]],
            [0.0, 0.0, torch.pi / 2, joint_state[5]],
            [0.088, 0.0, torch.pi / 2, joint_state[6]],
            [0.0, 0.107, 0.0, 0.0],
            [0.0, 0.0, 0.0, -torch.pi / 4],
            [0.0, 0.1034, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return dh_params


def get_tf_mat(i, dh):
    a = dh[i][0]
    d = dh[i][1]
    alpha = dh[i][2]
    theta = dh[i][3]
    q = theta
    return torch.tensor(
        [
            [torch.cos(q), -torch.sin(q), 0, a],
            [
                torch.sin(q) * torch.cos(alpha),
                torch.cos(q) * torch.cos(alpha),
                -torch.sin(alpha),
                -torch.sin(alpha) * d,
            ],
            [
                torch.sin(q) * torch.sin(alpha),
                torch.cos(q) * torch.sin(alpha),
                torch.cos(alpha),
                torch.cos(alpha) * d,
            ],
            [0, 0, 0, 1],
        ],
    )


def get_ee_pose(joint_state: torch.Tensor):
    dh_params = fr3_dh_params(joint_state)
    T_EE = torch.eye(4, dtype=torch.float32)
    for i in range(10):
        T_EE = T_EE @ get_tf_mat(i, dh_params)
    position = T_EE[:3, 3]
    orientation = C.mat2quat(T_EE[:3, :3])
    return position, orientation


def get_jacobian(joint_state: torch.Tensor):
    dh_params = fr3_dh_params(joint_state)
    T_EE = torch.eye(4, dtype=torch.float32)
    for i in range(10):
        T_EE = T_EE @ get_tf_mat(i, dh_params)
    J = torch.zeros(size=(6, 10), dtype=torch.float32)
    T = torch.eye(4, dtype=torch.float32)
    for i in range(10):
        T = T @ get_tf_mat(i, dh_params)
        p = T_EE[:3, 3] - T[:3, 3]
        z = T[:3, 2]
        J[:3, i] = torch.linalg.cross(z, p)
        J[3:, i] = z
    return J[:, :7]


@torch.jit.script
def ik_controller(jacobian, action):
    damping = 0.1
    jacobian_T = jacobian.transpose(0, 1)
    lmbda = torch.eye(6) * (damping**2)
    u = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lmbda) @ action
    return u


@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


# def orientation_error(desired, current):
#     cc = quat_conjugate(current)
#     q_r = quat_mul(desired, cc)
#     error = q_r[0:3] * torch.sign(q_r[3])
#     return error.to(torch.float32)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
