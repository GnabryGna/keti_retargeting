import os

import numpy as np
from dm_control import mjcf
from dm_control.utils import transformations


if __name__ == '__main__':
    # Scene
    scene_xml_path = os.path.join(os.getcwd(), 'scene.xml')
    scene = mjcf.from_path(scene_xml_path)

    # Table
    table_xml_path = os.path.join(os.getcwd(), 'table', 'table.xml')
    table = mjcf.from_path(table_xml_path)
    table.default.geom.rgba = [1, 1, 1, 1]
    table_attachment_frame = scene.attach(table)

    # Robot torso
    robot_torso_xml_path = os.path.join(os.getcwd(), 'robot_torso', 'robot_torso.xml')
    robot_torso = mjcf.from_path(robot_torso_xml_path)
    robot_torso.default.geom.rgba = [0.2, 0.2, 0.2, 1]
    robot_torso_attachment_frame = scene.attach(robot_torso)

    # Left robot arm (xarm7)
    robot_xml_path = os.path.join(os.getcwd(), 'ufactory_xarm7', 'xarm7_nohand.xml')
    left_robot_arm = mjcf.from_path(robot_xml_path)
    left_robot_arm.model = 'xarm7_left'
    left_robot_arm_attachment_frame = scene.attach(left_robot_arm)
    left_robot_arm_attachment_frame.pos = [-0.05692, 0, 0.44761]
    left_robot_arm_attachment_frame.quat = transformations.euler_to_quat([np.pi/2, 0, -np.pi*5/9], ordering='ZYX')

    # Right robot arm (xarm7)
    right_robot_arm = mjcf.from_path(robot_xml_path)
    right_robot_arm.model = 'xarm7_right'
    right_robot_arm_attachment_frame = scene.attach(right_robot_arm)
    right_robot_arm_attachment_frame.pos = [0.05692, 0, 0.44761]
    right_robot_arm_attachment_frame.quat = transformations.euler_to_quat([np.pi/2, 0, np.pi*5/9], ordering='ZYX')

    # # Left robot hand (inspire)
    # left_robot_hand_xml_path = os.path.join(os.getcwd(), 'inspire_robots_rh56dftp_left', 'rh56dftp_left.xml')
    # left_robot_hand = mjcf.from_path(left_robot_hand_xml_path)
    # left_robot_arm_attachment_site = left_robot_arm.find('site', 'attachment_site')
    # left_robot_arm_attachment_site.attach(left_robot_hand)

    # # Right robot hand (inspire)
    # right_robot_hand_xml_path = os.path.join(os.getcwd(), 'inspire_robots_rh56dftp_right', 'rh56dftp_right.xml')
    # right_robot_hand = mjcf.from_path(right_robot_hand_xml_path)
    # right_robot_arm_attachment_site = right_robot_arm.find('site', 'attachment_site')
    # right_robot_arm_attachment_site.attach(right_robot_hand)

    # Left robot hand (allegro)
    left_robot_arm_eef = left_robot_arm.find('body', 'link7')
    left_robot_arm_mount = left_robot_arm_eef.add('body',
                                                  name='left_robot_arm_mount',
                                                  pos=[0, 0, 0.01])
    left_robot_arm_mount.add('geom',
                             name='mount_geom',
                             type='cylinder',
                             size=[0.03773072, 0.005],
                             rgba=[0.2, 0.2, 0.2, 1],
                             pos=[0, 0, -0.005])
    left_robot_arm_mount.add('site',
                             name='mount_site')
    left_robot_hand_xml_path = os.path.join(os.getcwd(), 'wonik_allegro', 'left_hand.xml')
    left_robot_hand = mjcf.from_path(left_robot_hand_xml_path)
    left_robot_hand_attachment_site = left_robot_arm_mount.find('site', 'mount_site')
    left_robot_hand_attachment_frame = left_robot_hand_attachment_site.attach(left_robot_hand)
    left_robot_hand_attachment_frame.pos = [0, 0, 0.095]
    left_robot_hand_attachment_frame.euler = [0, -np.pi/2, 0]

    # Right robot hand (allegro)
    right_robot_arm_eef = right_robot_arm.find('body', 'link7')
    right_robot_arm_mount = right_robot_arm_eef.add('body',
                                                    name='right_robot_arm_mount',
                                                    pos=[0, 0, 0.01])
    right_robot_arm_mount.add('geom',
                              name='mount_geom',
                              type='cylinder',
                              size=[0.03773072, 0.005],
                              rgba=[0.2, 0.2, 0.2, 1],
                              pos=[0, 0, -0.005])
    right_robot_arm_mount.add('site',
                              name='mount_site')
    right_robot_hand_xml_path = os.path.join(os.getcwd(), 'wonik_allegro', 'right_hand.xml')
    right_robot_hand = mjcf.from_path(right_robot_hand_xml_path)
    right_robot_hand_attachment_site = right_robot_arm_mount.find('site', 'mount_site')
    right_robot_hand_attachment_frame = right_robot_hand_attachment_site.attach(right_robot_hand)
    right_robot_hand_attachment_frame.pos = [0, 0, 0.095]
    right_robot_hand_attachment_frame.euler = [0, -np.pi/2, 0]

    # Barcode scanner
    barcode_scanner_xml_path = os.path.join(os.getcwd(), 'barcode_scanner', 'barcode_scanner.xml')
    barcode_scanner = mjcf.from_path(barcode_scanner_xml_path)
    barcode_scanner.default.mesh.scale = [0.001, 0.001, 0.001]
    # barcode_scanner.default.mesh.inertia = 'exact'
    barcode_scanner.default.geom.condim = 6
    barcode_scanner.default.geom.priority = 1
    barcode_scanner_attachment_frame = scene.attach(barcode_scanner)
    barcode_scanner_attachment_frame.add('freejoint')
    barcode_scanner_attachment_frame.pos = [-0.5, 0.45, 0.2]
    barcode_scanner_attachment_frame.euler = [np.pi, 0, np.pi]

    # Object grasping area
    grasping_area_width = 0.34
    grasping_area_length = 0.72
    grasping_area = scene.worldbody.add('site',
                                        name='grasping_area',
                                        type='box',
                                        group=4,
                                        rgba=[0, 1, 0, 1],
                                        size=[grasping_area_length/2, grasping_area_width/2, 0.001],
                                        pos=[0, 0.2 + grasping_area_width/2, 0])

    # Box
    box_width = 0.25
    box_length = 0.34
    box_height = 0.205 # 실제 높이는 0.21 (box_height + box_thickness)
    box_thickness = 0.005
    box_rgba = [0.7, 0.6, 0.4, 1]
    box = scene.worldbody.add('body',
                              name='box',
                              pos=[0.36 + 0.125, 0.2 + 0.17, 0])
    box.add('geom',
            name='base',
            type='box',
            rgba=box_rgba,
            size=[box_width/2, box_length/2, box_thickness/2],
            pos=[0, 0, box_thickness/2])
    box.add('geom',
            name='side_1',
            type='box',
            rgba=box_rgba,
            size=[box_thickness/2, box_length/2, box_height/2],
            pos=[box_width/2 - box_thickness/2, 0, box_height/2 + box_thickness])
    box.add('geom',
            name='side_2',
            type='box',
            rgba=box_rgba,
            size=[box_width/2 - box_thickness, box_thickness/2, box_height/2],
            pos=[0, -box_length/2 + box_thickness/2, box_height/2 + box_thickness])
    box.add('geom',
            name='side_3',
            type='box',
            rgba=box_rgba,
            size=[box_thickness/2, box_length/2, box_height/2],
            pos=[-box_width/2 + box_thickness/2, 0, box_height/2 + box_thickness])
    box.add('geom',
            name='side_4',
            type='box',
            rgba=box_rgba,
            size=[box_width/2 - box_thickness, box_thickness/2, box_height/2],
            pos=[0, box_length/2 - box_thickness/2, box_height/2 + box_thickness])

    # Object
    ycb_object_init_pose = [ # [x, y, z, roll, pitch, yaw] relative to world frame
        [0.1, 0.25, 0.12, 0, 0, np.pi/2], # 003_cracker_box
        [-0.1, 0.25, 0.12, 0, 0, np.pi/2], # 004_sugar_box
        [0.1, 0.35, 0.12, 0, 0, np.pi/2], # 005_tomato_soup_can
        [-0.1, 0.35, 0.12, 0, 0, np.pi/2], # 006_mustard_bottle
        [0.1, 0.45, 0.12, np.pi/2, np.pi/2, 0], # 010_potted_meat_can
        [-0.1, 0.45, 0.12, 0, 0, -np.pi/2] # 021_bleach_cleanser
    ]
    ycb_object_barcode_pose = [ # [x, y, z, roll, pitch, yaw] relative to body frame
        [-0.010259, 0.050995, -0.10498, np.pi, 0, -np.pi/2], # 003_cracker_box
        [0.004512, 0.025381, -0.085659, np.pi, 0, -np.pi/2], # 004_sugar_box
        [-0.025168, -0.023411, -0.028418, np.pi/2, -np.pi/4, 0], # 005_tomato_soup_can
        [-0.026312, -0.000525, -0.043586, 0, -np.pi/2, -np.pi/2], # 006_mustard_bottle
        [-0.00522, 0.005731, 0.047954, 0, 0, np.pi/2], # 010_potted_meat_can
        [0.031735, 0.01317, -0.073959, 0, np.pi/2, np.pi*88/180] # 021_bleach_cleanser
    ]

    ycb_object_file_dir = os.path.join(os.getcwd(), 'ycb')
    ycb_object_name_list = os.listdir(ycb_object_file_dir)
    for id, ycb_object_name in enumerate(ycb_object_name_list):
        file_names = os.listdir(os.path.join(ycb_object_file_dir, ycb_object_name))
        xml_file_name = [file_name for file_name in file_names if file_name.lower().endswith('.xml')][0]
        ycb_object_xml_path = os.path.join(ycb_object_file_dir, ycb_object_name, xml_file_name)
        ycb_object = mjcf.from_path(ycb_object_xml_path)
        # ycb_object.default.mesh.inertia = 'exact'
        ycb_object.default.geom.condim = 6
        ycb_object.default.geom.priority = 1
        
        # Add barcode site
        ycb_object_body = ycb_object.find('body', f'{ycb_object_name}')
        ycb_object_body.add('site',
                            name='barcode', 
                            group=4,
                            size=[0.005],
                            rgba=[1, 0, 0, 1],
                            pos=ycb_object_barcode_pose[id][:3],
                            euler=ycb_object_barcode_pose[id][3:])

        ycb_object_attachment_frame = scene.attach(ycb_object)
        ycb_object_attachment_frame.add('freejoint')
        ycb_object_attachment_frame.pos = ycb_object_init_pose[id][:3]
        ycb_object_attachment_frame.euler = ycb_object_init_pose[id][3:]

    # Option
    scene.option.timestep= 0.001
    scene.option.o_solref = [0.002, 1]
    scene.option.integrator = 'implicit'
    scene.option.cone = 'elliptic'
    scene.option.flag.override = 'enable'
    scene.option.flag.multiccd = 'enable'
    scene.compiler.inertiagrouprange = [0, 2]
    
    # Initial state (allegro)
    initial_state = scene.keyframe.add('key')
    initial_state.name = 'initial_state'
    initial_state.qpos = (
        [-np.pi, 0, -np.pi, np.pi/6, 0, 0, 0,] # left arm
        + [0, 1.6, 1.3, 0.5, 0, 1.6, 1.3, 0.5, 0, 1.6, 1.3, 0.5, 0, 1.1, 1.1, 0.5] # left hand
        + [0, 0, 0, np.pi/6, -np.pi, 0, np.pi] # right arm
        + [0]*16 # right hand
        + [-0.18, 0.45, 0.49, 0.410932, 0.312626, 0.60471, 0.606404] # barcode scanner
        + ycb_object_init_pose[0][:3] + transformations.euler_to_quat(ycb_object_init_pose[0][3:]).tolist() # 003_cracker_box
        + ycb_object_init_pose[1][:3] + transformations.euler_to_quat(ycb_object_init_pose[1][3:]).tolist() # 004_sugar_box
        + ycb_object_init_pose[2][:3] + transformations.euler_to_quat(ycb_object_init_pose[2][3:]).tolist() # 005_tomato_soup_can
        + ycb_object_init_pose[3][:3] + transformations.euler_to_quat(ycb_object_init_pose[3][3:]).tolist() # 006_mustard_bottle
        + ycb_object_init_pose[4][:3] + transformations.euler_to_quat(ycb_object_init_pose[4][3:]).tolist() # 010_potted_meat_can
        + ycb_object_init_pose[5][:3] + transformations.euler_to_quat(ycb_object_init_pose[5][3:]).tolist() # 021_bleach_cleanser
    )
    initial_state.ctrl = initial_state.qpos[:46]

    mjcf.export_with_assets(mjcf_model=scene, out_dir= os.path.join(os.getcwd(), 'output'))
    print('Complete')