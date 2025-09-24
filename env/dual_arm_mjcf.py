import os

import mujoco
import numpy as np


def load():
    # Scene
    scene_xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'scene.xml')
    scene = mujoco.MjSpec.from_file(scene_xml_path)
    attachment_frame = scene.worldbody.add_frame()

    # Table
    table_xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'table', 'table.xml')
    table = mujoco.MjSpec.from_file(table_xml_path)
    scene.attach(child=table,
                 prefix=table.modelname + '/',
                 frame=attachment_frame)
    
    # Robot torso
    robot_torso_xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'robot_torso', 'robot_torso.xml')
    robot_torso = mujoco.MjSpec.from_file(robot_torso_xml_path)
    scene.attach(child=robot_torso,
                 prefix=robot_torso.modelname + '/',
                 frame=attachment_frame)
    
    # Left robot arm (xArm 7)
    robot_xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'ufactory_xarm7', 'xarm7.xml')
    left_robot_arm = mujoco.MjSpec.from_file(robot_xml_path)
    left_robot_arm.modelname = 'xarm7_left'
    left_robot_arm_attachment_frame = scene.worldbody.add_frame(pos=[-0.05692, 0, 0.44761],
                                                                euler=[0, -np.pi*5/9, np.pi/2])
    
    # Right robot arm (xArm 7)
    right_robot_arm = mujoco.MjSpec.from_file(robot_xml_path)
    right_robot_arm.modelname = 'xarm7_right'
    right_robot_arm_attachment_frame = scene.worldbody.add_frame(pos=[0.05692, 0, 0.44761],
                                                                 euler=[0, np.pi*5/9, np.pi/2])
    
    # Left robot hand (Allegro Hand V4)
    # left_robot_hand_xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'wonik_allegro', 'left_hand.xml')
    # left_robot_hand = mujoco.MjSpec.from_file(left_robot_hand_xml_path)
    # left_adaptor = left_robot_arm.body('link7').add_body(name='adaptor',
    #                                                      pos=[0, 0, 0.01])
    # left_adaptor.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    #                       size=[0.03773072, 0.005, 0], # TODO: 3번째 값은 필요 없음 (mujoco 오류?)
    #                       rgba=[0.2, 0.2, 0.2, 1],
    #                       pos=[0, 0, -0.005])
    # left_robot_hand_attachment_frame = left_adaptor.add_frame(pos=[0, 0, 0.095],
    #                                                           euler=[0, -np.pi/2, 0])
    # left_robot_arm.attach(child=left_robot_hand,
    #                       prefix=left_robot_hand.modelname + '/',
    #                       frame=left_robot_hand_attachment_frame)
    # scene.attach(child=left_robot_arm,
    #              prefix=left_robot_arm.modelname + '/',
    #              frame=left_robot_arm_attachment_frame)
    
    # Right robot hand (Allegro Hand V4)
    # right_robot_hand_xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'wonik_allegro', 'right_hand.xml')
    # right_robot_hand = mujoco.MjSpec.from_file(right_robot_hand_xml_path)
    # right_adaptor = right_robot_arm.body('link7').add_body(name='adaptor',
    #                                                        pos=[0, 0, 0.01])
    # right_adaptor.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    #                        size=[0.03773072, 0.005, 0], # TODO: 3번째 값은 필요 없음 (mujoco 오류?)
    #                        rgba=[0.2, 0.2, 0.2, 1],
    #                        pos=[0, 0, -0.005])
    # right_robot_hand_attachment_frame = right_adaptor.add_frame(pos=[0, 0, 0.095],
    #                                                             euler=[0, -np.pi/2, 0])
    # right_robot_arm.attach(child=right_robot_hand,
    #                        prefix=right_robot_hand.modelname + '/',
    #                        frame=right_robot_hand_attachment_frame)
    # scene.attach(child=right_robot_arm,
    #              prefix=right_robot_arm.modelname + '/',
    #              frame=right_robot_arm_attachment_frame)

    # Left robot hand (Inspire RH56DFTP)
    left_robot_hand_xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'inspire_rh56dftp_left', 'rh56dftp_left.xml')
    left_robot_hand = mujoco.MjSpec.from_file(left_robot_hand_xml_path)
    left_adaptor = left_robot_arm.body('link7').add_body(name='adaptor',
                                                         pos=[0, 0, 0.01])
    left_adaptor.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                          size=[0.042, 0.005, 0], # TODO: 3번째 값은 필요 없음 (mujoco 오류?)
                          rgba=[0.2, 0.2, 0.2, 1],
                          pos=[0, 0, -0.005])
    left_robot_hand_attachment_frame = left_adaptor.add_frame()
    left_robot_arm.attach(child=left_robot_hand,
                          prefix=left_robot_hand.modelname + '/',
                          frame=left_robot_hand_attachment_frame)
    scene.attach(child=left_robot_arm,
                 prefix=left_robot_arm.modelname + '/',
                 frame=left_robot_arm_attachment_frame)

    # Right robot hand (Inspire RH56DFTP4)
    right_robot_hand_xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'inspire_rh56dftp_right', 'rh56dftp_right.xml')
    right_robot_hand = mujoco.MjSpec.from_file(right_robot_hand_xml_path)
    right_adaptor = right_robot_arm.body('link7').add_body(name='adaptor',
                                                           pos=[0, 0, 0.01])
    right_adaptor.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=[0.042, 0.005, 0], # TODO: 3번째 값은 필요 없음 (mujoco 오류?)
                           rgba=[0.2, 0.2, 0.2, 1],
                           pos=[0, 0, -0.005])
    right_robot_hand_attachment_frame = right_adaptor.add_frame()
    right_robot_arm.attach(child=right_robot_hand,
                           prefix=right_robot_hand.modelname + '/',
                           frame=right_robot_hand_attachment_frame)
    scene.attach(child=right_robot_arm,
                 prefix=right_robot_arm.modelname + '/',
                 frame=right_robot_arm_attachment_frame)

    # Box
    box_width = 0.34
    box_length = 0.25
    box_height = 0.21
    box_thickness = 0.005
    box_rgba = [0.7, 0.6, 0.4, 1]
    box = scene.worldbody.add_body(name='box',
                                   pos=[-box_length/2 - 0.11, box_width/2 + 0.3, 0])
    box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                 rgba=box_rgba,
                 size=[box_length/2, box_width/2, box_thickness/2],
                 pos=[0, 0, 0])
    box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                 rgba=box_rgba,
                 size=[box_thickness/2, box_width/2, box_height/2],
                 pos=[box_length/2 - box_thickness/2, 0, box_height/2])
    box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                 rgba=box_rgba,
                 size=[box_thickness/2, box_width/2, box_height/2],
                 pos=[-box_length/2 + box_thickness/2, 0, box_height/2])
    box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                 rgba=box_rgba,
                 size=[box_length/2 - box_thickness, box_thickness/2, box_height/2],
                 pos=[0, -box_width/2 + box_thickness/2, box_height/2])
    box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                 rgba=box_rgba,
                 size=[box_length/2 - box_thickness, box_thickness/2, box_height/2],
                 pos=[0, box_width/2 - box_thickness/2, box_height/2])
    
    # Object grasping area
    grasping_area_width = 0.34
    grasping_area_length = 0.72
    scene.worldbody.add_site(name='grasping_area',
                             type=mujoco.mjtGeom.mjGEOM_BOX,
                             group=4,
                             pos=[grasping_area_length/2 - 0.11, grasping_area_width/2 + 0.3, 0],
                             size=[grasping_area_length/2, grasping_area_width/2, 0.001],
                             rgba=[0, 1, 0, 1])
    
    # Barcode scanner
    barcode_scanner_xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'barcode_scanner', 'barcode_scanner.xml')
    barcode_scanner = mujoco.MjSpec.from_file(barcode_scanner_xml_path)
    barcode_scanner_attachment_frame = scene.worldbody.add_frame(pos=[-0.5, 0.55, 0.2],
                                                                 euler=[np.pi, 0, np.pi])
    scene.attach(child=barcode_scanner,
                 prefix=barcode_scanner.modelname + '/',
                 frame=barcode_scanner_attachment_frame)

    # YCB object
    ycb_object_init_pose = {
        '003_cracker_box': [0.35, 0.37, 0.12, 0.707107, 0, 0, 0.707107],
        '004_sugar_box': [0.15, 0.37, 0.12, 0.707107, 0, 0, 0.707107],
        '005_tomato_soup_can': [0.35, 0.47, 0.12, 0.707107, 0, 0, 0.707107],
        '006_mustard_bottle': [0.15, 0.47, 0.12, 0.707107, 0, 0, 0.707107],
        '010_potted_meat_can': [0.35, 0.57, 0.12, 0.707107, 0, 0, 0.707107],
        '021_bleach_cleanser': [0.15, 0.57, 0.12, 0.707107, 0, 0, 0.707107]
    }
    ycb_object_file_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'ycb')
    ycb_object_name_list = os.listdir(ycb_object_file_dir)
    for ycb_object_name in ycb_object_name_list:
        file_names = os.listdir(os.path.join(ycb_object_file_dir, ycb_object_name))
        xml_file_name = next(file_name for file_name in file_names if file_name.lower().endswith('.xml'))
        ycb_object_xml_path = os.path.join(ycb_object_file_dir, ycb_object_name, xml_file_name)
        ycb_object = mujoco.MjSpec.from_file(ycb_object_xml_path)
        ycb_object_attachment_frame = scene.worldbody.add_frame(pos=ycb_object_init_pose[ycb_object_name][:3],
                                                                quat=ycb_object_init_pose[ycb_object_name][3:])
        scene.attach(child=ycb_object,
                     prefix=ycb_object.modelname + '/',
                     frame=ycb_object_attachment_frame)
        
    # Initial state (Allegro Hand V4)
    # initial_qpos = (
    #     [0, 0, 0, np.pi/6, np.pi, 0, np.pi] # left arm
    #     + [0, 1.6, 1.3, 0.5, 0, 1.6, 1.3, 0.5, 0, 1.6, 1.3, 0.5, 0, 1.1, 1.1, 0.5] # left hand
    #     + [0, 0, 0, np.pi/6, 0, 0, -np.pi/2] # right arm
    #     + [0]*16 # right hand
    #     + [-0.112, 0.312, 0.516, 0.417242, 0.264029, 0.483108, 0.723052] # barcode scanner
    #     + [0.517, 0.253, 0.112, 0.707107, 0, 0, 0.707107] # 003_cracker_box
    #     + [0.442, 0.340, 0.092, 0.707107, 0, 0, 0.707107] # 004_sugar_box
    #     + [0.549, 0.334, 0.058, 0.707107, 0, 0, 0.707107] # 005_tomato_soup_can
    #     + [0.543, 0.472, 0.085, 0.707107, 0, 0, 0.707107] # 006_mustard_bottle
    #     + [0.542, 0.403, 0.048, 0.707107, 0, 0, 0.707107] # 010_potted_meat_can
    #     + [0.423, 0.465, 0.110, 0.707107, 0, 0, 0.707107] # 021_bleach_cleanser
    # )
    # initial_ctrl = initial_qpos[:46]
    # scene.add_key(name='initial_state', qpos=initial_qpos, ctrl=initial_ctrl)

    # Initial state (Inspire RH56DFTP)
    initial_qpos = (
        [0, 0, 0, np.pi/6, 0, 0, np.pi/2] # left arm
        + [1.1624, 0, 0, 0, 1.25, 1.35, 1.25, 1.35, 1.25, 1.35, 1.25, 1.35] # left hand
        + [0, 0, 0, np.pi/6, 0, 0, 0] # right arm
        + [0]*12 # right hand
        + [-0.14742, 0.484668, 0.496348, 0.411075, 0.231993, 0.544672, 0.693202] # barcode scanner
        + [-0.203, 0.353, 0.112, 0.707107, 0, 0, 0.707107] # 003_cracker_box
        + [-0.278, 0.440, 0.092, 0.707107, 0, 0, 0.707107] # 004_sugar_box
        + [-0.171, 0.434, 0.058, 0.707107, 0, 0, 0.707107] # 005_tomato_soup_can
        + [-0.177, 0.572, 0.085, 0.707107, 0, 0, 0.707107] # 006_mustard_bottle
        + [-0.178, 0.503, 0.048, 0.707107, 0, 0, 0.707107] # 010_potted_meat_can
        + [-0.297, 0.565, 0.110, 0.707107, 0, 0, 0.707107] # 021_bleach_cleanser
    )
    initial_ctrl = (
        [0, 0, 0, np.pi/6, 0, 0, np.pi/2] # left arm
        + [1.1624, 0, 1.4244, 1.4244, 1.4244, 1.4244] # left hand
        + [0, 0, 0, np.pi/6, 0, 0, 0] # right arm
        + [0, 0, 0, 0, 0, 0] # right hand
    )
    scene.add_key(name='initial_state', qpos=initial_qpos, ctrl=initial_ctrl)

    scene.compile()

    # For debugging
    scene.to_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dual_arm_mjcf.xml'))

    return scene
