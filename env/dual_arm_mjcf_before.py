from pathlib import Path
from typing import Literal

import mujoco


def build_model(
        hand_type: Literal['allegro', 'inspire', 'tesollo'] = 'allegro',
        target_ycb_object: list[str] = None
    ):
    initial_qpos = []
    initial_ctrl = []

    # Arena
    arena_xml_path = Path(__file__).resolve().parent / 'assets' / 'arena.xml'
    spec = mujoco.MjSpec.from_file(str(arena_xml_path))
    attachment_frame = spec.worldbody.add_frame()

    # Table
    table_xml_path = Path(__file__).resolve().parent / 'assets' / 'table' / 'table.xml'
    table = mujoco.MjSpec.from_file(str(table_xml_path))
    spec.attach(
        child=table,
        prefix=table.modelname + '/',
        frame=attachment_frame
    )
    
    # Robot torso
    robot_torso_xml_path = Path(__file__).resolve().parent / 'assets' / 'robot_torso' / 'robot_torso.xml'
    robot_torso = mujoco.MjSpec.from_file(str(robot_torso_xml_path))
    spec.attach(
        child=robot_torso,
        prefix=robot_torso.modelname + '/',
        frame=attachment_frame
    )
    
    # Left robot arm (xArm 7)
    robot_xml_path = Path(__file__).resolve().parent / 'assets' / 'ufactory_xarm7' / 'xarm7.xml'
    left_robot_arm = mujoco.MjSpec.from_file(str(robot_xml_path))
    left_robot_arm.modelname = 'xarm7_left'
    left_robot_arm_attachment_frame = spec.worldbody.add_frame(
        pos=[-0.05692, 0, 0.64761],
        euler=[0, -mujoco.mjPI*5/9, mujoco.mjPI/2]
    )
    initial_qpos += [-0.03450384, -0.46866212, 0.67489503, 0.81920009, -1.86433262, 0.58700317, 4.00370058]
    initial_ctrl += [-0.03450384, -0.46866212, 0.67489503, 0.81920009, -1.86433262, 0.58700317, 4.00370058]

    # Left robot hand (Inspire RH56F1)
    left_robot_hand_xml_path = Path(__file__).resolve().parent / 'assets' / 'inspire_rh56f1' / 'left_hand.xml'
    left_robot_hand = mujoco.MjSpec.from_file(str(left_robot_hand_xml_path))
    left_adaptor = left_robot_arm.body('link7').add_body(
        name='adaptor',
        pos=[0, 0, 0.005]
    )
    left_adaptor.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.042, 0.005, 0],
        rgba=[0.2, 0.2, 0.2, 1],
    )
    left_robot_hand_attachment_frame = left_adaptor.add_frame(
        pos=[0, 0, 0.005],
        euler=[0, 0, mujoco.mjPI]
    )
    left_robot_arm.attach(
        child=left_robot_hand,
        prefix=left_robot_hand.modelname + '/',
        frame=left_robot_hand_attachment_frame
    )
    spec.attach(
        child=left_robot_arm,
        prefix=left_robot_arm.modelname + '/',
        frame=left_robot_arm_attachment_frame
    )
    initial_qpos += [1.5, 0, 0, 0, 1.2, 1.38, 1.22, 1.4, 1.25, 1.44, 1.28, 1.48]
    initial_ctrl += [1.5, 0, 1.2, 1.22, 1.25, 1.28]
    
    # Right robot arm (xArm 7)
    right_robot_arm = mujoco.MjSpec.from_file(str(robot_xml_path))
    right_robot_arm.modelname = 'xarm7_right'
    right_robot_arm_attachment_frame = spec.worldbody.add_frame(
        pos=[0.05692, 0, 0.64761],
        euler=[0, mujoco.mjPI*5/9, mujoco.mjPI/2]
    )
    initial_qpos += [-0.28620468, -0.05743847, -0.60686597, 0.69537941, -4.19273322, 1.11166329, -1.13839255]
    initial_ctrl += [-0.28620468, -0.05743847, -0.60686597, 0.69537941, -4.19273322, 1.11166329, -1.13839255]
    
    if hand_type == 'allegro':
        # Right robot hand (Allegro V4)
        right_robot_hand_xml_path = Path(__file__).resolve().parent / 'assets' / 'wonik_allegro' / 'right_hand.xml'
        right_robot_hand = mujoco.MjSpec.from_file(str(right_robot_hand_xml_path))
        right_adaptor = right_robot_arm.body('link7').add_body(
            name='adaptor',
            pos=[0, 0, 0.005]
        )
        right_adaptor.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.03773072, 0.005, 0],
            rgba=[0.2, 0.2, 0.2, 1],
        )
        right_robot_hand_attachment_frame = right_adaptor.add_frame(
            pos=[0, 0, 0.1],  # 0.095 + 0.005
            euler=[-mujoco.mjPI/2, -mujoco.mjPI/2, -mujoco.mjPI/2]
        )
        right_robot_arm.attach(
            child=right_robot_hand,
            prefix=right_robot_hand.modelname + '/',
            frame=right_robot_hand_attachment_frame
        )
        spec.attach(
            child=right_robot_arm,
            prefix=right_robot_arm.modelname + '/',
            frame=right_robot_arm_attachment_frame
        )
        initial_qpos += [0]*16
        initial_ctrl += [0]*16
    
    elif hand_type == 'inspire':
        # Right robot hand (Inspire RH56F1)
        right_robot_hand_xml_path = Path(__file__).resolve().parent / 'assets' / 'inspire_rh56f1' / 'right_hand.xml'
        right_robot_hand = mujoco.MjSpec.from_file(str(right_robot_hand_xml_path))
        right_adaptor = right_robot_arm.body('link7').add_body(
            name='adaptor',
            pos=[0, 0, 0.005]
        )
        right_adaptor.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.042, 0.005, 0],
            rgba=[0.2, 0.2, 0.2, 1],
        )
        right_robot_hand_attachment_frame = right_adaptor.add_frame(
            pos=[0, 0, 0.005],
            euler=[0, 0, mujoco.mjPI/2]
        )
        right_robot_arm.attach(
            child=right_robot_hand,
            prefix=right_robot_hand.modelname + '/',
            frame=right_robot_hand_attachment_frame
        )
        spec.attach(
            child=right_robot_arm,
            prefix=right_robot_arm.modelname + '/',
            frame=right_robot_arm_attachment_frame
        )
        initial_qpos += [0]*12
        initial_ctrl += [0]*6

    elif hand_type == 'tesollo':
        # Right robot hand (Tesollo DG-5F)
        right_robot_hand_xml_path = Path(__file__).resolve().parent / 'assets' / 'tesollo_dg5f' / 'right_hand_short.xml'
        right_robot_hand = mujoco.MjSpec.from_file(str(right_robot_hand_xml_path))
        right_robot_hand_attachment_frame = right_robot_arm.body('link7').add_frame(euler=[0, 0, mujoco.mjPI/2])
        right_robot_arm.attach(
            child=right_robot_hand,
            prefix=right_robot_hand.modelname + '/',
            frame=right_robot_hand_attachment_frame
        )
        spec.attach(
            child=right_robot_arm,
            prefix=right_robot_arm.modelname + '/',
            frame=right_robot_arm_attachment_frame
        )
        initial_qpos += [0]*20
        initial_ctrl += [0]*20
    
    else:
        raise ValueError(f'Invalid hand type: "{hand_type}". Supported hand types are "allegro" and "tesollo".')

    # Box
    box_width = 0.34
    box_length = 0.25
    box_height = 0.21
    box_thickness = 0.005
    box_rgba = [0.7, 0.6, 0.4, 1]
    box = spec.worldbody.add_body(
        name='box',
        pos=[-box_length/2 - 0.1, box_width/2 + 0.3, 0]
    )
    box.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        rgba=box_rgba,
        size=[box_length/2, box_width/2, box_thickness/2],
        pos=[0, 0, 0]
    )
    box.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        rgba=box_rgba,
        size=[box_thickness/2, box_width/2, box_height/2],
        pos=[box_length/2 - box_thickness/2, 0, box_height/2]
    )
    box.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        rgba=box_rgba,
        size=[box_thickness/2, box_width/2, box_height/2],
        pos=[-box_length/2 + box_thickness/2, 0, box_height/2]
    )
    box.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        rgba=box_rgba,
        size=[box_length/2 - box_thickness, box_thickness/2, box_height/2],
        pos=[0, -box_width/2 + box_thickness/2, box_height/2]
    )
    box.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        rgba=box_rgba,
        size=[box_length/2 - box_thickness, box_thickness/2, box_height/2],
        pos=[0, box_width/2 - box_thickness/2, box_height/2]
    )
    
    # Object grasping area
    grasping_area_width = 0.34
    grasping_area_length = 0.61
    spec.worldbody.add_site(
        name='grasping_area',
        type=mujoco.mjtGeom.mjGEOM_BOX,
        group=4,
        pos=[grasping_area_length/2 - 0.1, grasping_area_width/2 + 0.3, 0],
        size=[grasping_area_length/2, grasping_area_width/2, 0.001],
        rgba=[0, 1, 0, 1]
    )
    
    # Barcode scanner
    barcode_scanner_xml_path = Path(__file__).resolve().parent / 'assets' / 'barcode_scanner' / 'barcode_scanner.xml'
    barcode_scanner = mujoco.MjSpec.from_file(str(barcode_scanner_xml_path))
    barcode_scanner_attachment_frame = spec.worldbody.add_frame(
        pos=[-0.5, 0.55, 0.2],
        euler=[mujoco.mjPI, 0, mujoco.mjPI]
    )
    spec.attach(
        child=barcode_scanner,
        prefix=barcode_scanner.modelname + '/',
        frame=barcode_scanner_attachment_frame
    )
    # initial_qpos += [-0.426257, 0.476465, 0.360466, -0.0246973, -0.0351318, 0.693698, 0.718985]
    initial_qpos += [-0.18032681,  0.38294931,  0.45375707,  0.55113209,  0.25831389,  0.68542889, 0.39964307]
    # YCB object
    ycb_object_init_pose = {
        '003_cracker_box': [-0.193, 0.353, 0.112, 0.707107, 0, 0, 0.707107],
        '004_sugar_box': [-0.268, 0.440, 0.092, 0.707107, 0, 0, 0.707107],
        '006_mustard_bottle': [-0.167, 0.572, 0.085, 0.707107, 0, 0, 0.707107],
        '010_potted_meat_can': [-0.168, 0.503, 0.048, 0.707107, 0, 0, 0.707107],
        '021_bleach_cleanser': [-0.287, 0.565, 0.110, 0.707107, 0, 0, 0.707107]
    }
    ycb_object_dir = Path(__file__).resolve().parent / "assets" / "ycb"
    for name in target_ycb_object:
        dir = ycb_object_dir / name
        if not dir.exists():
            raise RuntimeError(f'YCB object directory not found: {dir}')
        ycb_object_xml_path = next(file_name for file_name in dir.iterdir() if file_name.suffix.lower() == ".xml")
        ycb_object = mujoco.MjSpec.from_file(str(ycb_object_xml_path))
        ycb_object_attachment_frame = spec.worldbody.add_frame(
            pos=ycb_object_init_pose[name][:3],
            quat=ycb_object_init_pose[name][3:]
        )
        spec.attach(
            child=ycb_object,
            prefix=ycb_object.modelname + "/",
            frame=ycb_object_attachment_frame,
        )
        initial_qpos += ycb_object_init_pose[name]

    spec.add_key(name='initial_state', qpos=initial_qpos, ctrl=initial_ctrl)

    model = spec.compile()

    # For debugging
    # spec.to_file(str(Path(__file__).resolve().parent / 'dual_arm_mjcf.xml'))

    return model
