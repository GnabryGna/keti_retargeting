# MuJoCo Dual-Arm Environment

## Description
`mujoco_dual_arm` is a MuJoCo-based dual-arm robotic simulation environment designed for research on reinforcement learning (RL), imitation learning (IL), and advanced control of robotic manipulation.

It provides:

* Two UFactory xArm7 robotic arms
* Multiple dexterous robot hands (Inspire RH56DFTP, Wonik Allegro Hand)
* Rich environments with YCB objects, table, and robot torso

## Installation

This implementation requires the following dependencies (tested on Windows 11):

* Python >= 3.9
* [MuJoCo](https://mujoco.org/) >= 3.3.3
* [imageio](https://imageio.readthedocs.io/) >= 2.37.0
* [Matplotlib](https://matplotlib.org/) >= 3.10.6

You can quickly install/update dependencies by running the following:

```shell
pip install -r requirements.txt
```

## Usage

```shell
python main.py
```