import mujoco
import numpy as np

from env.dual_arm_env import DualArmEnv


def set_seed(seed):
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def site_matching_mesh_aabb(model, geom_name):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    mesh_id = model.geom_dataid[geom_id]
    adr = model.mesh_vertadr[mesh_id]
    n = model.mesh_vertnum[mesh_id]
    V = model.mesh_vert[adr : adr + n]
    pmin = V.min(axis=0)
    pmax = V.max(axis=0)
    center_local = 0.5 * (pmin + pmax)
    half_ext_local = 0.5 * (pmax - pmin)

    geom_pos = model.geom_pos[geom_id]
    geom_quat = model.geom_quat[geom_id]

    geom_rotation = np.empty(9)
    mujoco.mju_quat2Mat(geom_rotation, geom_quat)
    geom_rotation = geom_rotation.reshape(3, 3)

    site_pos = geom_pos + geom_rotation @ center_local
    # site_pos += geom_rotation[:, 2]*eps
    site_quat = geom_quat.copy()

    print(f"size: {half_ext_local}")
    print(f"pos: {site_pos}")
    print(f"quat: {site_quat}")


if __name__ == "__main__":
    env = DualArmEnv()

    name_list = [
        "palm_tactile_sensor",
        "thumb_pad_tactile_sensor",
        "thumb_middle_section_tactile_sensor",
        "thumb_nail_tactile_sensor",
        "thumb_tip_tactile_sensor",
        "index_pad_tactile_sensor",
        "index_nail_tactile_sensor",
        "index_tip_tactile_sensor",
        "middle_pad_tactile_sensor",
        "middle_nail_tactile_sensor",
        "middle_tip_tactile_sensor",
        "ring_pad_tactile_sensor",
        "ring_nail_tactile_sensor",
        "ring_tip_tactile_sensor",
        "little_pad_tactile_sensor",
        "little_nail_tactile_sensor",
        "little_tip_tactile_sensor",
    ]
    for name in name_list:
        print(name)
        geom_name = f"xarm7_left/inspire_rh56dftp_left/{name}"
        site_matching_mesh_aabb(env.model, geom_name)
        print("-" * 50)
