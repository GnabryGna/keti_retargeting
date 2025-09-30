import os
import pickle
import sys

import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.dual_arm_env import DualArmEnv

target_tcp_pose = [0.2, 0.4, 0.4, 0.7071068, -0.7071068, 0.0, 0.0]

# Load data
with open("data_with_gc.pkl", "rb") as f:
    data_with_gc = pickle.load(f)
time_with_gc = data_with_gc['time']
tcp_pose_with_gc = data_with_gc['tcp_pose']
print(tcp_pose_with_gc.shape)

with open("data_without_gc.pkl", "rb") as f:
    data_without_gc = pickle.load(f)
time_without_gc = data_without_gc['time']
tcp_pose_without_gc = data_without_gc['tcp_pose']


def save_data():
    env = DualArmEnv()
    env.reset()

    env.right_robot_arm.set_tcp_pose(target_tcp_pose)

    duration = 3
    start_time = env.data.time
    end_time = start_time + duration

    time_list = []
    tcp_pose_list = []

    while env.data.time <= end_time:
        right_robot_arm_tcp_pose = env.right_robot_arm.get_tcp_pose()
        time_list.append(env.data.time - start_time)
        tcp_pose_list.append(right_robot_arm_tcp_pose)
        
        env.step(None)

    with open("data_with_gc.pkl", "wb") as f:
        pickle.dump({"time": np.array(time_list), "tcp_pose": np.array(tcp_pose_list)}, f)


def plot_graph(data_with_gc, data_without_gc, title, ylabel, filename):
    plt.figure()
    plt.plot(time_with_gc, data_with_gc, label='with gravity compensation', color='b')
    plt.plot(time_without_gc, data_without_gc, label='w/o gravity compensation', color='r')
    plt.title(f'{title}')
    plt.xlabel('Simulation Time [s]')
    plt.ylabel(f'{ylabel}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig(f'{filename}.png')
    plt.show()


def plot_x_position(plot=True):
    tcp_x_pos_with_gc = tcp_pose_with_gc[:, 0]
    tcp_x_pos_without_gc = tcp_pose_without_gc[:, 0]

    if plot:
        plot_graph(tcp_x_pos_with_gc, tcp_x_pos_without_gc, title='TCP X Position', ylabel='X Position [m]', filename='x_position.png')

    return tcp_x_pos_with_gc, tcp_x_pos_without_gc


def plot_y_position(plot=True):
    tcp_y_pos_with_gc = tcp_pose_with_gc[:, 1]
    tcp_y_pos_without_gc = tcp_pose_without_gc[:, 1]

    if plot:
        plot_graph(tcp_y_pos_with_gc, tcp_y_pos_without_gc, title='TCP Y Position', ylabel='Y Position [m]', filename='y_position.png')

    return tcp_y_pos_with_gc, tcp_y_pos_without_gc


def plot_z_position(plot=True):
    tcp_z_pos_with_gc = tcp_pose_with_gc[:, 2]
    tcp_z_pos_without_gc = tcp_pose_without_gc[:, 2]

    if plot:
        plot_graph(tcp_z_pos_with_gc, tcp_z_pos_without_gc, title='TCP Z Position', ylabel='Z Position [m]', filename='z_position.png')

    return tcp_z_pos_with_gc, tcp_z_pos_without_gc


def plot_position_error(plot=True): # L2 norm
    pos_ref = target_tcp_pose[:3]

    pos_with_gc = tcp_pose_with_gc[:, :3]
    pos_without_gc = tcp_pose_without_gc[:, :3]

    pos_err_vec_with_gc = pos_with_gc - pos_ref
    pos_err_vec_without_gc = pos_without_gc - pos_ref

    pos_err_norm_with_gc = np.linalg.norm(pos_err_vec_with_gc, axis=1)
    pos_err_norm_without_gc = np.linalg.norm(pos_err_vec_without_gc, axis=1)

    if plot:
        plot_graph(pos_err_norm_with_gc, pos_err_norm_without_gc, title='TCP Position Error', ylabel='Position Error [m]', filename='tcp_position_error.png')

    return pos_err_norm_with_gc, pos_err_norm_without_gc


def plot_roll_error(plot=True):
    q_ref = target_tcp_pose[3:]
    q_ref_inv = np.empty(4)
    mujoco.mju_negQuat(q_ref_inv, q_ref)

    def euler_error(q):
        q_error = np.empty(4)
        mujoco.mju_mulQuat(q_error, q_ref_inv, q)

        R_err = np.empty(9)
        mujoco.mju_quat2Mat(R_err, q_error)
        R_err = R_err.reshape(3, 3)

        roll = np.arctan2(R_err[2, 1], R_err[2, 2])
        roll_deg = np.degrees(roll)
        roll_deg = (roll_deg + 180.0) % 360.0 - 180.0

        return roll_deg
    
    roll_err_with_gc = [euler_error(q[3:]) for q in tcp_pose_with_gc]
    roll_err_without_gc = [euler_error(q[3:]) for q in tcp_pose_without_gc]

    if plot:
        plot_graph(roll_err_with_gc, roll_err_without_gc, title='TCP Roll Error', ylabel='Roll Error [deg]', filename='tcp_roll_error.png')

    return roll_err_with_gc, roll_err_without_gc


def plot_pitch_error(plot=True):
    q_ref = target_tcp_pose[3:]
    q_ref_inv = np.empty(4)
    mujoco.mju_negQuat(q_ref_inv, q_ref)

    def euler_error(q):
        q_error = np.empty(4)
        mujoco.mju_mulQuat(q_error, q_ref_inv, q)

        R_err = np.empty(9)
        mujoco.mju_quat2Mat(R_err, q_error)
        R_err = R_err.reshape(3, 3)

        pitch = np.arcsin(-R_err[2, 0])
        pitch_deg = np.degrees(pitch)
        pitch_deg = (pitch_deg + 180.0) % 360.0 - 180.0

        return pitch_deg
    
    pitch_err_with_gc = [euler_error(q[3:]) for q in tcp_pose_with_gc]
    pitch_err_without_gc = [euler_error(q[3:]) for q in tcp_pose_without_gc]
    
    if plot:
        plot_graph(pitch_err_with_gc, pitch_err_without_gc, title='TCP Pitch Error', ylabel='Pitch Error [deg]', filename='tcp_pitch_error.png')

    return pitch_err_with_gc, pitch_err_without_gc


def plot_yaw_error(plot=True):
    q_ref = target_tcp_pose[3:]
    q_ref_inv = np.empty(4)
    mujoco.mju_negQuat(q_ref_inv, q_ref)

    def euler_error(q):
        q_error = np.empty(4)
        mujoco.mju_mulQuat(q_error, q_ref_inv, q)

        R_err = np.empty(9)
        mujoco.mju_quat2Mat(R_err, q_error)
        R_err = R_err.reshape(3, 3)

        yaw = np.arctan2(R_err[1, 0], R_err[0, 0])
        yaw_deg = np.degrees(yaw)
        yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0

        return yaw_deg
    
    yaw_err_with_gc = [euler_error(q[3:]) for q in tcp_pose_with_gc]
    yaw_err_without_gc = [euler_error(q[3:]) for q in tcp_pose_without_gc]

    if plot:
        plot_graph(yaw_err_with_gc, yaw_err_without_gc, title='TCP Yaw Error', ylabel='Yaw Error [deg]', filename='tcp_yaw_error.png')

    return yaw_err_with_gc, yaw_err_without_gc


def plot_orientation_error(plot=True):
    q_ref = target_tcp_pose[3:]
    
    def quat_error(q):
        dot = np.clip(abs(np.dot(q_ref, q)), -1, 1)
        return np.degrees(2*np.arccos(dot))

    ori_err_with_gc = [quat_error(q[3:]) for q in tcp_pose_with_gc]
    ori_err_without_gc = [quat_error(q[3:]) for q in tcp_pose_without_gc]

    if plot:
        plot_graph(ori_err_with_gc, ori_err_without_gc, title='TCP Orientation Error', ylabel='Orientation Error [deg]', filename='tcp_orientation_error.png')

    return ori_err_with_gc, ori_err_without_gc


def plot_all():
    nrows = 2
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))

    # Position Error (L2 norm)
    pos_err_norm_with_gc, pos_err_norm_without_gc = plot_position_error(plot=False)
    axes[0, 0].plot(time_with_gc, pos_err_norm_with_gc, label='with gravity compensation', color='b')
    axes[0, 0].plot(time_without_gc, pos_err_norm_without_gc, label='w/o gravity compensation', color='r')
    axes[0, 0].set_title('TCP Position Error')
    axes[0, 0].set_xlabel('Simulation Time [s]')
    axes[0, 0].set_ylabel('Position Error [m]')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # X Position
    tcp_x_pos_with_gc, tcp_x_pos_without_gc = plot_x_position(plot=False)
    axes[0, 1].plot(time_with_gc, tcp_x_pos_with_gc, label='with gravity compensation', color='b')
    axes[0, 1].plot(time_without_gc, tcp_x_pos_without_gc, label='w/o gravity compensation', color='r')
    axes[0, 1].set_title('TCP X Position')
    axes[0, 1].set_xlabel('Simulation Time [s]')
    axes[0, 1].set_ylabel('X Position [m]')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Y Position
    tcp_y_pos_with_gc, tcp_y_pos_without_gc = plot_y_position(plot=False)
    axes[0, 2].plot(time_with_gc, tcp_y_pos_with_gc, label='with gravity compensation', color='b')
    axes[0, 2].plot(time_without_gc, tcp_y_pos_without_gc, label='w/o gravity compensation', color='r')
    axes[0, 2].set_title('TCP Y Position')
    axes[0, 2].set_xlabel('Simulation Time [s]')
    axes[0, 2].set_ylabel('Y position [m]')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Z Position
    tcp_z_pos_with_gc, tcp_z_pos_without_gc = plot_z_position(plot=False)
    axes[0, 3].plot(time_with_gc, tcp_z_pos_with_gc, label='with gravity compensation', color='b')
    axes[0, 3].plot(time_without_gc, tcp_z_pos_without_gc, label='w/o gravity compensation', color='r')
    axes[0, 3].set_title('TCP Z Position')
    axes[0, 3].set_xlabel('Simulation Time [s]')
    axes[0, 3].set_ylabel('Z position [m]')
    axes[0, 3].legend()
    axes[0, 3].grid(True)

    # Orientation Error
    ori_err_with_gc, ori_err_without_gc = plot_orientation_error(plot=False)
    axes[1, 0].plot(time_with_gc, ori_err_with_gc, label='with gravity compensation', color='b')
    axes[1, 0].plot(time_without_gc, ori_err_without_gc, label='w/o gravity compensation', color='r')
    axes[1, 0].set_title('TCP Orientation Error')
    axes[1, 0].set_xlabel('Simulation Time [s]')
    axes[1, 0].set_ylabel('Orientation Error [deg]')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Roll Error
    roll_err_with_gc, roll_err_without_gc = plot_roll_error(plot=False)
    axes[1, 1].plot(time_with_gc, roll_err_with_gc, label='with gravity compensation', color='b')
    axes[1, 1].plot(time_without_gc, roll_err_without_gc, label='w/o gravity compensation', color='r')
    axes[1, 1].set_title('TCP Roll Error')
    axes[1, 1].set_xlabel('Simulation Time [s]')
    axes[1, 1].set_ylabel('Roll Error [deg]')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Pitch Error
    pitch_err_with_gc, pitch_err_without_gc = plot_pitch_error(plot=False)
    axes[1, 2].plot(time_with_gc, pitch_err_with_gc, label='with gravity compensation', color='b')
    axes[1, 2].plot(time_without_gc, pitch_err_without_gc, label='w/o gravity compensation', color='r')
    axes[1, 2].set_title('TCP Pitch Error')
    axes[1, 2].set_xlabel('Simulation Time [s]')
    axes[1, 2].set_ylabel('Pitch Error [deg]')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    # Yaw Error
    yaw_err_with_gc, yaw_err_without_gc = plot_yaw_error(plot=False)
    axes[1, 3].plot(time_with_gc, yaw_err_with_gc, label='with gravity compensation', color='b')
    axes[1, 3].plot(time_without_gc, yaw_err_without_gc, label='w/o gravity compensation', color='r')
    axes[1, 3].set_title('TCP Yaw Error')
    axes[1, 3].set_xlabel('Simulation Time [s]')
    axes[1, 3].set_ylabel('Yaw Error [deg]')
    axes[1, 3].legend()
    axes[1, 3].grid(True)

    plt.tight_layout()
    # plt.savefig("tcp_pose.png")
    plt.show()


if __name__ == '__main__':
    # save_data()
    # plot_position_error()
    # plot_x_position()
    # plot_y_position()
    # plot_z_position()
    # plot_orientation_error()
    # plot_roll_error()
    # plot_pitch_error()
    # plot_yaw_error()
    plot_all()