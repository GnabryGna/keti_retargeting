import pickle
from pathlib import Path
from typing import List

import numpy as np
import tqdm
import tyro
from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting


OPERATOR2VP_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)
OPERATOR2VP_LEFT = np.array(
    [
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ]
)


def two_mat_batch_mul(batch_mat: np.ndarray, left_rot: np.ndarray):
    result = np.tile(np.eye(4), [batch_mat.shape[0], 1, 1])
    result[:, :3, :3] = np.matmul(left_rot[None, ...], batch_mat[:, :3, :3])
    result[:, :3, 3] = batch_mat[:, :3, 3] @ left_rot.T
    return result


def joint_avp2hand(finger_mat: np.ndarray):
    finger_index = np.array(
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
    )
    finger_mat = finger_mat[finger_index]
    return finger_mat


class FingerRetargetor(object):
    def __init__(self, robot_dir: Path):
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

        left_config_path = get_default_config_path(
            robot_name=RobotName.allegro,
            retargeting_type=RetargetingType.dexpilot,
            hand_type=HandType.left,
        )

        right_config_path = get_default_config_path(
            robot_name=RobotName.allegro,
            retargeting_type=RetargetingType.dexpilot,
            hand_type=HandType.right,
        )

        self.left_retargeting = RetargetingConfig.load_from_file(
            left_config_path
        ).build()

        self.right_retargeting = RetargetingConfig.load_from_file(
            right_config_path
        ).build()

    def retarget(self, data: np.ndarray, hand: str):
        if hand == "right":
            joint_pose = two_mat_batch_mul(data, OPERATOR2VP_RIGHT.T)
            joint_pos = joint_avp2hand(joint_pose)[:, :3, 3]

            indices = self.right_retargeting.optimizer.target_link_human_indices
            origin_indices = indices[0, :]
            task_indices = indices[1, :]

            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = self.right_retargeting.retarget(ref_value)

        else:
            joint_pose = two_mat_batch_mul(data, OPERATOR2VP_LEFT.T)
            joint_pos = joint_avp2hand(joint_pose)[:, :3, 3]

            indices = self.left_retargeting.optimizer.target_link_human_indices
            origin_indices = indices[0, :]
            task_indices = indices[1, :]

            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = self.left_retargeting.retarget(ref_value)

        return qpos
