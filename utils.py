import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np
# import torch
# import yaml

from env import dual_arm_mjcf
from env.dual_arm_env import DualArmEnv


# def load_config(config_file):
#     with open(config_file) as file:
#         config = yaml.safe_load(file)
#     return config


def set_seed(seed):
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def run_bench(model, data, n_steps=10000):
    dt_list = []
    ncon_list = []
    nefc_list = []
    niter_list = []

    for _ in range(n_steps):
        t0 = time.perf_counter()
        mujoco.mj_step(model, data)
        dt_list.append(time.perf_counter() - t0)
        ncon_list.append(int(data.ncon))
        nefc_list.append(int(data.nefc))
        niter_list.append(int(np.sum(data.solver_niter)))

    def summary(xs):
        return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "p95": float(np.percentile(xs, 95))}

    return {
        "step_time_sec": summary(dt_list),
        "ncon": summary(ncon_list),
        "nefc": summary(nefc_list),
        "solver_niter": summary(niter_list)
    }


def save_data(duration, dt):
    env = DualArmEnv.load()
    env.reset()
    
    settle_time = env.data.time
    next_sample_time = settle_time
    end_time = settle_time + duration

    time = []
    tcp_pose = []

    while env.data.time <= end_time:
        while env.data.time >= next_sample_time and next_sample_time <= end_time:
            right_robot_arm_tcp_pose = env.right_robot_arm.get_tcp_pose() # Edit here
            time.append(env.data.time - settle_time)
            tcp_pose.append(right_robot_arm_tcp_pose)
            np.savez("tcp_data_with_gc.npz", time=time, tcp_pose=tcp_pose)
            next_sample_time += dt

        env.step(None)


def site_matching_mesh_aabb(model, geom_name):
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    mesh_id = model.geom_dataid[geom_id]
    adr = model.mesh_vertadr[mesh_id]
    n = model.mesh_vertnum[mesh_id]
    V = model.mesh_vert[adr:adr+n]
    pmin = V.min(axis=0)
    pmax = V.max(axis=0)
    center_local = 0.5*(pmin + pmax)
    half_ext_local = 0.5*(pmax - pmin)

    geom_pos = model.geom_pos[geom_id]
    geom_quat = model.geom_quat[geom_id]

    geom_rotation = np.empty(9)
    mujoco.mju_quat2Mat(geom_rotation, geom_quat)
    geom_rotation = geom_rotation.reshape(3, 3)

    site_pos  = geom_pos + geom_rotation @ center_local
    # site_pos += geom_rotation[:, 2]*eps
    site_quat = geom_quat.copy()

    print(f'size: {half_ext_local}')
    print(f'pos: {site_pos}')
    print(f'quat: {site_quat}')


if __name__ == '__main__':
    mjcf = dual_arm_mjcf.load()
    model = mjcf.compile()
    data = mujoco.MjData(model)

    initial_sate = model.key('initial_state').id
    mujoco.mj_resetDataKeyframe(model, data, initial_sate)
    mujoco.mj_forward(model, data)

    result = run_bench(model, data)
    print(result)

    # name_list = [
    #     "palm_tactile_sensor",
    #     "thumb_pad_tactile_sensor",
    #     "thumb_middle_section_tactile_sensor",
    #     "thumb_nail_tactile_sensor",
    #     "thumb_tip_tactile_sensor",
    #     "index_pad_tactile_sensor",
    #     "index_nail_tactile_sensor",
    #     "index_tip_tactile_sensor",
    #     "middle_pad_tactile_sensor",
    #     "middle_nail_tactile_sensor",
    #     "middle_tip_tactile_sensor",
    #     "ring_pad_tactile_sensor",
    #     "ring_nail_tactile_sensor",
    #     "ring_tip_tactile_sensor",
    #     "little_pad_tactile_sensor",
    #     "little_nail_tactile_sensor",
    #     "little_tip_tactile_sensor"
    # ]
    # for name in name_list:
    #     print(name)
    #     geom_name = f'xarm7_left/inspire_rh56dftp_left/{name}'
    #     site_matching_mesh_aabb(model, geom_name)
    #     print('-'*50)
    
    '''
    # Load data
    tcp_data_without_gc = np.load("tcp_data_without_gc.npz")
    tcp_data_with_gc = np.load("tcp_data_with_gc.npz")

    time_without_gc = tcp_data_without_gc["time"]
    tcp_pose_without_gc = tcp_data_without_gc["tcp_pose"]
    time_with_gc = tcp_data_with_gc["time"]
    tcp_pose_with_gc = tcp_data_with_gc["tcp_pose"]
    
    # Position
    # tcp_x_pos_without_gc = tcp_pose_without_gc[:, 0]
    # tcp_y_pos_without_gc = tcp_pose_without_gc[:, 1]
    tcp_z_pos_without_gc = tcp_pose_without_gc[:, 2]
    # tcp_x_pos_with_gc = tcp_pose_with_gc[:, 0]
    # tcp_y_pos_with_gc = tcp_pose_with_gc[:, 1]
    tcp_z_pos_with_gc = tcp_pose_with_gc[:, 2]

    plt.figure()
    plt.plot(time_with_gc, tcp_z_pos_with_gc, label="with gravity compensation", color='b')
    plt.plot(time_without_gc, tcp_z_pos_without_gc, label="w/o gravity compensation", color='r')
    plt.xlabel("Simulation time [s]")
    plt.ylabel("TCP z position [m]")
    plt.title("TCP Position with/without Gravity Compensation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tcp_z_position.png")
    plt.show()

    # Position error
    pos_ref = np.array([0.2, 0.4, 0.4])

    pos_without_gc = tcp_pose_without_gc[:, :3]  # (N,3)
    pos_with_gc = tcp_pose_with_gc[:, :3]

    err_vec_without_gc = pos_without_gc - pos_ref
    err_vec_with_gc = pos_with_gc - pos_ref

    err_norm_without_gc = np.linalg.norm(err_vec_without_gc, axis=1)
    err_norm_with_gc = np.linalg.norm(err_vec_with_gc, axis=1)
    
    plt.figure()
    plt.plot(time_with_gc, err_norm_with_gc, label="with gravity compensation", color='b')
    plt.plot(time_without_gc, err_norm_without_gc, label="w/o gravity compensation", color='r')
    plt.xlabel("Simulation time [s]")
    plt.ylabel("Position error norm [m]")
    plt.title("TCP Position Error with/without Gravity Compensation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tcp_position_error.png")
    plt.show()

    # Oreintation error
    q_ref = np.array([0.7071068, -0.7071068, 0.0, 0.0])  # [qw, qx, qy, qz]

    def quat_angle_error(q, q_ref):
        q = q / np.linalg.norm(q)
        q_ref = q_ref / np.linalg.norm(q_ref)
        dot = np.clip(np.abs(np.dot(q, q_ref)), -1.0, 1.0)
        return 2 * np.arccos(dot)

    ori_err_without_gc = np.array([quat_angle_error(q[3:], q_ref) for q in tcp_pose_without_gc])
    ori_err_with_gc = np.array([quat_angle_error(q[3:], q_ref) for q in tcp_pose_with_gc])

    ori_err_without_gc_deg = np.degrees(ori_err_without_gc)
    ori_err_with_gc_deg = np.degrees(ori_err_with_gc)

    plt.figure()
    plt.plot(time_with_gc, ori_err_with_gc_deg, label="with gravity compensation", color='b')
    plt.plot(time_without_gc, ori_err_without_gc_deg, label="w/o gravity compensation", color='r')
    plt.xlabel("Simulation time [s]")
    plt.ylabel("Orientation error [deg]")
    plt.title("TCP Orientation Error with/without Gravity Compensation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tcp_orientation_error.png")
    plt.show()
    '''