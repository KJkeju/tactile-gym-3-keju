"""
python launch_servo.py -r sim -s tactip -e edge_yRz_shear -t pose_yRz -m simple_cnn -o circle -n 160 -v test
"""
import os
import itertools as it
import time
import time as t
import cv2
import ipdb
import numpy as np
from cri.transforms import inv_transform_euler, transform_euler

from tg3.data.collect.setup_embodiment import setup_embodiment
from tg3.learning.supervised.image_to_feature.cnn.label_encoder_regress import LabelEncoder
from tg3.learning.supervised.image_to_feature.cnn.setup_model import setup_model
from tg3.tasks.servo.servo_utils.controller import PIDController
from tg3.tasks.servo.servo_utils.labelled_model import LabelledModel
from tg3.tasks.servo.setup_servo import setup_servo, setup_parse
from tg3.utils.utils import load_json_obj, make_dir
from tg3.simulator.utils.setup_pb_utils import load_stim
from tg3.simulator.assets import add_assets_path
from tg3.simulator.utils.pybullet_draw_utils import draw_link_frame
from tg3.simulator.sensors.tactile_sensor import build_virtual_tripod_from_contacts
import pybullet as p
from tg3.simulator.sensors.tactile_sensor import TactileSensor
# from user_input.slider import Slider
# from ipdb import set_trace



def load_pose_model(model_dir, device):
    model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
    model_image_params = load_json_obj(os.path.join(model_dir, 'model_image_params'))
    label_params = load_json_obj(os.path.join(model_dir, 'model_label_params'))

    label_encoder = LabelEncoder(label_params, device)

    model = setup_model(
        in_dim=model_image_params['image_processing']['dims'],
        in_channels=1,
        out_dim=label_encoder.out_dim,
        model_params=model_params,
        saved_model_dir=model_dir,
        device=device
    )
    model.eval()

    return LabelledModel(model, model_image_params['image_processing'], label_encoder, device=device)

def generate_random_target():
    x = np.random.uniform(-200, 300)
    y = np.random.uniform(-300, 300)
    pose = np.array([x, y, 100, 0, 0, 0])
    indicator = np.array([(600 - pose[0]) / 1000, pose[1] / 1000, 0.06, 0, 0, 0])
    return pose, indicator

# def setup_environment(target_indicator):
def setup_environment(target_indicator, sensor_type):
    env_params = {
        'robot': 'sim_ur',
        'stim_name': 'cube',
        'speed': 50,
        'work_frame': (600, 0, 300, 180, 0, 0),
        'tcp_pose': (0, 0, -140, 0, 0, 0),
        'stim_pose': (350, -300, 60, 0, 0, 0),
        'show_gui': True,
        'load_target': target_indicator
    }

    sensor_params = {
        'left': {
            'type': 'standard_tactip',
            'core': 'fixed',
            'dynamics': {'stiffness': 300, 'damping': 10, 'friction': 10},
            'image_size': [128, 128],
            'turn_off_border': True,
            'show_tactile': True,
            # 'body_link': 'left_inner_finger',  # 按实际结构
            # 'tip_link': 'left_tactip_tip',
            'body_link': 'left_tactip_body_link',
            'tip_link':  'left_tactip_tip_link',   # 或保持你现有的名字
            # 'adapter_link': 'left_tactip_adapter_link',
        },
        'right': {
            'type': 'standard_tactip',
            'core': 'fixed',
            'dynamics': {'stiffness': 300, 'damping': 10, 'friction': 10},
            'image_size': [128, 128],
            'turn_off_border': True,
            'show_tactile': True,
            # 'body_link': 'right_inner_finger',
            # 'tip_link': 'right_tactip_tip',
            'body_link': 'right_tactip_body_link',
            'tip_link': 'right_tactip_tip_link',
            # 'adapter_link': 'right_tactip_adapter_link',
        }
    }
    # env_params = {
    #     'robot': 'sim_ur',
    #     'stim_name': 'cube',
    #     'speed': 50,
    #     'work_frame': (600, 0, 200, 180, 0, 180),
    #     'tcp_pose': (0, 0, 0, 0, 0, 0),
    #     'stim_pose': (350, -300, 60, 0, 0, 0),
    #     'show_gui': True,
    #     'load_target': target_indicator
    # }
    #
    # sensor_params = {
    #     "type": sensor_type,
    #     "sensor_type": "right_angle_tactip",
    #     "image_size": (256, 256),
    #     "show_tactile": False
    # }

    return setup_embodiment(env_params, sensor_params)



def create_controllers():
    align = PIDController(
        kp=[0, 0.03, 0, 0, 0, 0],
        ki=[0, 0.03, 0, 0, 0, 0],
        kd=[0, 0, 0, 0, 0, 0],
        ei_clip=[[-5], [5]],
        error='lambda y, r: transform_euler(r, y)',
        ref=[0]
    )
    pointing = PIDController(
        kp=[0.5, 0, 0, 0, 0, 0.5],
        ki=[0.01, 0, 0, 0, 0, 0.1],
        kd=[0, 0, 0, 0, 0, 0],
        ei_clip=[[-5], [5]],
        error='lambda y, r: transform_euler(r, y)',
        ref=[2, 0, 0, 0, 0, 0]
    )
    return align, pointing


def run_control_loop(robot, sensor, pose_model, target_pose, align_ctrl, point_ctrl):
    robot.controller.servo_mode = True
    robot.controller.time_delay = 10

    # Initial positioning
    robot.move_linear((0, 0, 0, 0, 0, 0))
    robot.move_linear((300, -300, 0, 0, 0, 0))
    robot.move_linear((300, -300, 100, 0, 0, 0))

    forward = np.array([-5, 0, 0, 0, 0, 0])
    back = np.array([4, 0, 0, 0, 0, 0])

    for step in range(5000):
        robot.move_linear(inv_transform_euler(forward, robot.pose))
        tactile_image = sensor.process()
        cv2.imshow('Tactile Image', tactile_image)
        cv2.waitKey(1)

        pred_pose = pose_model.predict(tactile_image)[:6]

        current_pose = robot.pose
        current_xy = np.array(current_pose[:2])
        target_xy = np.array(target_pose[:2])

        angle_to_target = np.rad2deg(np.arctan2(*(-(target_xy - current_xy))[::-1]))
        angle_diff = (current_pose[5] - angle_to_target + 180) % 360 - 180

        if np.linalg.norm(target_xy - current_xy) < 50:
            print("\nTarget reached.")
            break

        pose_sensor = np.array([0, pred_pose[0], pred_pose[1], pred_pose[5], pred_pose[3], pred_pose[4]])
        pointing = point_ctrl.update(pose_sensor, np.zeros(6))
        align = align_ctrl.update(np.array([0, angle_diff, 0, 0, 0, 0]), np.zeros(6))

        robot.move_linear(inv_transform_euler(back, robot.pose))

        if step < 5:
            pointing = align = np.zeros(6)


        robot.move_linear(inv_transform_euler(pointing + align, robot.pose))

    time.sleep(5)



def tactile_pushing(args):
    pose_model = load_pose_model(args.model_dir, args.device)


    target_pose, target_indicator = generate_random_target()


    # robot, sensor = setup_environment(target_indicator)
    pb, robot, sensor = setup_environment(target_indicator, sensor_type=args.sensor)
    embodiment = sensor.embodiment
    # 先把夹爪打开
    # open_gripper(pb, embodiment, angle=0.45)
    # if not hasattr(p, "COV_ENABLE_COLLISION_SHAPES"):
    #     p.COV_ENABLE_COLLISION_SHAPES = 6  # 官方源码里这个值就是 6
    # p.configureDebugVisualizer(p.COV_ENABLE_COLLISION_SHAPES, 1, physicsClientId=pb._client)
    # open_gripper(pb, embodiment, angle=0.7)
    robot.move_linear(np.array([0, 0, 0, 0, 0, 0]))
    spawn_stim_between_fingers(pb, embodiment, stim_name="cube", z_offset=0.005, use_tactip_tip=True)

    # robot.move_linear(np.array([0, -100, 0, 0, 0, 0]))


    # —— 在 pb, robot, sensor, embodiment 都就绪之后（spawn cube 之后也行）:
    # left_tip_id = embodiment.link_name_to_index["left_tactip_tip_link"]
    # right_tip_id = embodiment.link_name_to_index["right_tactip_tip_link"]

    # 让 tip 半透明，便于观察内部调试图形（RGBA：黄色 + 30% 透明）
    # pb.changeVisualShape(embodiment.embodiment_id, left_tip_id, rgbaColor=[1.0, 0.85, 0.0, 0.3])
    # pb.changeVisualShape(embodiment.embodiment_id, right_tip_id, rgbaColor=[1.0, 0.85, 0.0, 0.3])



    while True:
        # pb.stepSimulation()  # 推进物理仿真一步
        # ……创建好 pb, embodiment, robot, sensor 之后：

        # 1) 拿到左右传感器实例（只做一次）
        # try:
        #     sensor_left = sensor.tactile_sensors['left']
        #     sensor_right = sensor.tactile_sensors['right']
        # except AttributeError:
        #     sensor_left = embodiment.tactile_sensors['left']
        #     sensor_right = embodiment.tactile_sensors['right']
        #
        # tip_id_left = embodiment.link_name_to_index["left_tactip_tip_link"]
        # tip_id_right = embodiment.link_name_to_index["right_tactip_tip_link"]
        #
        # # 2) 仿真循环
        # while True:
        #     # 如果你没有自动步进，取消注释
        #     # pb.stepSimulation()
        #
        #     # 画出左右 tip 坐标系
        #     draw_link_frame(embodiment.embodiment_id, tip_id_left, lifetime=0.1)
        #     draw_link_frame(embodiment.embodiment_id, tip_id_right, lifetime=0.1)
        #
        #     # 3) 从接触点拟合三脚架（可视化法向与三点）
        #     tripod_L = build_virtual_tripod_from_contacts(pb, sensor_left, r=0.006, visualize=True)
        #     tripod_R = build_virtual_tripod_from_contacts(pb, sensor_right, r=0.006, visualize=True)
        #
        #
        #     # 4)（可选）粗略估计力矩 —— 只对左侧演示
        #     if tripod_L:
        #         tip_pos_L = np.array(pb.getLinkState(embodiment.embodiment_id,
        #                                              sensor_left.tactile_link_ids['tip'])[0])
        #         cfs_L = sensor_left.get_contact_features() or []
        #         Fn = sum(max(c['normal_force'], 0.0) for c in cfs_L)
        #         f_each = (Fn / 3.0) * tripod_L['normal']
        #         tau = np.zeros(3)
        #         for q in tripod_L['points']:
        #             rvec = q - tip_pos_L
        #             tau += np.cross(rvec, f_each)
        #         # 这里的 tau 就是对 tip 的一个粗略力矩估计（仅法向均分假设）
        #
        #     time.sleep(1.0 / 240.0)
        tactile_image_l, tactile_image_r = sensor.process()

        pred_pose_l = pose_model.predict(tactile_image_l)[:6]
        pred_pose_r = pose_model.predict(tactile_image_r)[:6]
        print(f'left pred {pred_pose_l}, \n right pred {pred_pose_r}')
        try:
            sensor_left = sensor.tactile_sensors['left']
            sensor_right = sensor.tactile_sensors['right']
        except AttributeError:
            sensor_left = embodiment.tactile_sensors['left']
            sensor_right = embodiment.tactile_sensors['right']

        tip_id_left = embodiment.link_name_to_index["left_tactip_tip_link"]
        tip_id_right = embodiment.link_name_to_index["right_tactip_tip_link"]
        draw_link_frame(embodiment.embodiment_id, tip_id_left, lifetime=0.1)
        draw_link_frame(embodiment.embodiment_id, tip_id_right, lifetime=0.1)
        current_pose = robot.pose
        target_pose = current_pose.copy()
        step = np.array([0, -1, 0, 0, 0, 0])
        target_pose += step
        robot.move_linear(target_pose),



    # align_ctrl, point_ctrl = create_controllers()
    #
    # run_control_loop(robot, sensor, pose_model, target_pose, align_ctrl, point_ctrl)

# def simple_animation_demo(args):
#     # 加载场景
#     _, target_indicator = generate_random_target()
#     pb, robot, sensor = setup_environment(target_indicator, sensor_type=args.sensor)

    # 动作1：初始位姿（比如工作空间中央上方）
    # home_pose = ((600, 0, 200, 180, 0, 180)
    # print("Moving to home pose:", home_pose)
    # robot.move_linear(home_pose)
    # time.sleep(1)
    #
    # # 动作2：前移一段距离
    # next_pose = ((700, 0, 200, 180, 0, 180)
    # print("Moving to next pose:", next_pose)
    # robot.move_linear(next_pose)
    # time.sleep(1)

    # # 动作2：夹爪关闭（夹紧）
    # print("Closing gripper")
    # robot.close_gripper()
    # time.sleep(1)
    #
    # # 动作3：夹爪打开
    # print("Opening gripper")
    # robot.open_gripper()
    # time.sleep(1)
    #
    # # 动作3：回到home
    # # print("Returning to home pose")
    # # robot.move_linear(home_pose)
    # # time.sleep(1)
    #
    # print("Animation complete. Window will close in 5 seconds.")
    # time.sleep(5)
def spawn_stim_between_fingers(pb, embodiment, stim_name="cube", z_offset=0.0, use_tactip_tip=True):
    """
    在夹爪左右指中间生成一个stim（默认cube）。
    - pb: pybullet client
    - embodiment: 由 setup_environment 返回/内部持有的 embodiment 对象
    - stim_name: 资产目录下 stimuli/{name}/{name}.urdf
    - z_offset: 沿世界Z方向微调高度（米）
    - use_tactip_tip: True 则用 tactip tip 链接，False 用 inner_finger
    """
    # 取 robot id 和 link 索引
    body_id = embodiment.arm.embodiment_id
    link_map = embodiment.arm.link_name_to_index

    if use_tactip_tip:
        left_link  = "left_tactip_tip_link"
        right_link = "right_tactip_tip_link"
    else:
        left_link  = "left_inner_finger"
        right_link = "right_inner_finger"

    if left_link not in link_map or right_link not in link_map:
        raise RuntimeError(f"Link name not found: {left_link} or {right_link}. Check your URDF/link_map keys.")

    l_pos, l_orn, *_ = pb.getLinkState(body_id, link_map[left_link], computeForwardKinematics=True)
    r_pos, r_orn, *_ = pb.getLinkState(body_id, link_map[right_link], computeForwardKinematics=True)

    l_pos = np.array(l_pos)
    r_pos = np.array(r_pos)
    mid = (l_pos + r_pos) / 2.0
    mid[2] += z_offset  # 需要高一点就给正值，比如 0.005
    # mid[0] -= 0.012
    # mid[1] -= 0.015
    # 物体URDF路径（项目内置 stimuli）
    stim_urdf = add_assets_path(r"H:\tactile-gym-3-bowen\tg3\simulator\stimuli\cube\cube.urdf")

    # 加载在中点处；orientation 给单位四元数即可（小物块不重要）
    mid = np.asarray(mid, dtype=float)  # 确保是 float
    stim_rpy = np.array([0.0, 0, 0.0], dtype=float)  # 先用水平姿态
    stim_pose6 = np.concatenate([mid, stim_rpy])  # 6 维: 位置 + 姿态

    load_stim(pb, stim_urdf, stim_pose6, fixed_base=False, enable_collision=True, scale=0.35)
    print(f"[INFO] Spawned {stim_name} at", mid.tolist())

def open_gripper(pb, embodiment, angle=0.45):
    """把 Robotiq85 打开到一个固定角度（弧度）。"""
    eid = embodiment.arm.embodiment_id
    jidx = embodiment.arm.joint_name_to_index  # 里头有 finger_joint 等名字

    # 若 pybullet 正常处理 mimic，只设 finger_joint 就够了；否则按 mimic 关系分别设定
    if 'finger_joint' in jidx:
        pb.resetJointState(eid, jidx['finger_joint'], angle)

    # 保险起见，把其它几个关节也按 mimic 关系写一遍（符号和上面的 URDF 一致）
    def setj(name, val):
        if name in jidx: pb.resetJointState(eid, jidx[name], val)

    setj('left_inner_knuckle_joint',  +angle)   # multiplier = +1
    setj('left_inner_finger_joint',   -angle)   # multiplier = -1
    setj('right_inner_knuckle_joint', -angle)   # multiplier = -1
    setj('right_inner_finger_joint',  +angle)   # multiplier = +1
    setj('right_outer_knuckle_joint', -angle)   # multiplier = -1

if __name__ == "__main__":

    args = setup_parse(
        path='./../tactile-data',
        robot='sim',
        sensor='tactip',
        experiments=['surface_zRxy'],
        predicts=['pose_zRxRy'],
        models=['simple_cnn'],
        objects=['circle', 'square'],
        sample_nums=[160, 190],
        # run_version=['test'], # to not overwrite previous runs
        # device='cpu' # 'cuda' or 'cpu'
    )
    # args.model_dir = r'E:\ai\repo\tactile-gym-3\tg3\tasks\servo\tactile_data\sim_tactip\surface_zRxy\regress_pose_zRxRy\simple_cnn'
    args.model_dir = r"H:\tactile-gym-3-bowen\tactile_data\sim_tactip\surface_zRxy\regress_pose_zRxy\simple_cnn"
    tactile_pushing(args)
    # simple_animation_demo(args)
