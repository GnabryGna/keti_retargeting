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


def run_bench(model, data, n_steps=10000, warmup_steps=1000):
    for _ in range(warmup_steps):
        mujoco.mj_step(model, data)

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


if __name__ == '__main__':
    # ------------------------------
    # run_bench()
    # ------------------------------
    mjcf = dual_arm_mjcf.load()
    model = mjcf.compile()
    print(model.ngeom)
    data = mujoco.MjData(model)

    initial_sate = model.key('initial_state').id
    mujoco.mj_resetDataKeyframe(model, data, initial_sate)
    mujoco.mj_forward(model, data)

    result = run_bench(model, data)
    print(result)

    # ------------------------------
    # Gravity compensation
    # ------------------------------
    # Save data
    # env = DualArmEnv()

    # next_sample_time = 3.0
    # end_time = next_sample_time + 3
    # time = []
    # tcp_pose = []

    # observation = env.reset()
    # settle_time= env.data.time
    # while env.data.time <= end_time:
    #     while env.data.time >= next_sample_time and next_sample_time <= end_time:
    #         right_robot_arm_tcp_pose = env.right_robot_arm.get_tcp_pose()
    #         # time.append(env.data.time - settle_time)
    #         # tcp_pose.append(right_robot_arm_tcp_pose)
    #         # np.savez("tcp_data_without_gc.npz", time=time, tcp_pose=tcp_pose)
    #         next_sample_time += 0.01

    #     env.step(None)
    
    # Load data
    # tcp_data_without_gc = np.load("tcp_data_without_gc.npz")
    # tcp_data_with_gc = np.load("tcp_data_with_gc.npz")

    # time_without_gc = tcp_data_without_gc["time"]
    # tcp_pose_without_gc = tcp_data_without_gc["tcp_pose"]
    # time_with_gc = tcp_data_with_gc["time"]
    # tcp_pose_with_gc = tcp_data_with_gc["tcp_pose"]
    
    # # Position
    # # tcp_x_pos_without_gc = tcp_pose_without_gc[:, 0]
    # # tcp_y_pos_without_gc = tcp_pose_without_gc[:, 1]
    # tcp_z_pos_without_gc = tcp_pose_without_gc[:, 2]
    # # tcp_x_pos_with_gc = tcp_pose_with_gc[:, 0]
    # # tcp_y_pos_with_gc = tcp_pose_with_gc[:, 1]
    # tcp_z_pos_with_gc = tcp_pose_with_gc[:, 2]

    # plt.figure()
    # plt.plot(time_with_gc, tcp_z_pos_with_gc, label="with gravity compensation", color='b')
    # plt.plot(time_without_gc, tcp_z_pos_without_gc, label="w/o gravity compensation", color='r')
    # plt.xlabel("Simulation time [s]")
    # plt.ylabel("TCP z position [m]")
    # plt.title("TCP Position with/without Gravity Compensation")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("tcp_z_position.png")
    # plt.show()

    # Position error
    # pos_ref = np.array([0.2, 0.4, 0.4])

    # pos_without_gc = tcp_pose_without_gc[:, :3]  # (N,3)
    # pos_with_gc = tcp_pose_with_gc[:, :3]

    # err_vec_without_gc = pos_without_gc - pos_ref
    # err_vec_with_gc = pos_with_gc - pos_ref

    # err_norm_without_gc = np.linalg.norm(err_vec_without_gc, axis=1)
    # err_norm_with_gc = np.linalg.norm(err_vec_with_gc, axis=1)
    
    # plt.figure()
    # plt.plot(time_with_gc, err_norm_with_gc, label="with gravity compensation", color='b')
    # plt.plot(time_without_gc, err_norm_without_gc, label="w/o gravity compensation", color='r')
    # plt.xlabel("Simulation time [s]")
    # plt.ylabel("Position error norm [m]")
    # plt.title("TCP Position Error with/without Gravity Compensation")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("tcp_position_error.png")
    # plt.show()

    # Oreintation error
    # q_ref = np.array([0.7071068, -0.7071068, 0.0, 0.0])  # [qw, qx, qy, qz]

    # def quat_angle_error(q, q_ref):
    #     q = q / np.linalg.norm(q)
    #     q_ref = q_ref / np.linalg.norm(q_ref)
    #     dot = np.clip(np.abs(np.dot(q, q_ref)), -1.0, 1.0)
    #     return 2 * np.arccos(dot)

    # ori_err_without_gc = np.array([quat_angle_error(q[3:], q_ref) for q in tcp_pose_without_gc])
    # ori_err_with_gc = np.array([quat_angle_error(q[3:], q_ref) for q in tcp_pose_with_gc])

    # ori_err_without_gc_deg = np.degrees(ori_err_without_gc)
    # ori_err_with_gc_deg = np.degrees(ori_err_with_gc)

    # plt.figure()
    # plt.plot(time_with_gc, ori_err_with_gc_deg, label="with gravity compensation", color='b')
    # plt.plot(time_without_gc, ori_err_without_gc_deg, label="w/o gravity compensation", color='r')
    # plt.xlabel("Simulation time [s]")
    # plt.ylabel("Orientation error [deg]")
    # plt.title("TCP Orientation Error with/without Gravity Compensation")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("tcp_orientation_error.png")
    # plt.show()