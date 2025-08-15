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
        'stim_pose': (600, 150, 60, 0, 0, 0),
        'show_gui': True,
        'load_target': target_indicator
    }

    sensor_params = {
        'left': {
            'type': 'standard_tactip',
            'core': 'fixed',
            'dynamics': {'stiffness': 60, 'damping': 2.3, 'friction': 2.3},
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
            'dynamics': {'stiffness': 60, 'damping': 2.3, 'friction': 2.3},
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

    # 先把夹爪打开到你期望的角度，比如 48°
    # set_gripper_angle(pb, embodiment, np.deg2rad(20), use_motors=False)

    # 然后打印
    # debug_gripper_limits_and_opening(pb, embodiment)
    # 先把夹爪打开
    # 打开到 35°
    # set_gripper_opening_deg(pb, embodiment, 35, use_motors=False)
    #
    # # …对准后夹紧到 12°
    # set_gripper_opening_deg(pb, embodiment, 12, use_motors=False)
    #
    # # 如果仍有轻微歪斜（1°级别），给微偏置（1°≈0.01745 rad）
    # set_gripper_angle(pb, embodiment, np.deg2rad(12), bias_left=+0.0, bias_right=-0.0)

    # open_gripper(pb, embodiment, angle=0.45)
    # emb = sensor.embodiment
    # drive_gripper(pb, emb, angle_rad=0.35, use_motors=False)   # <<< 用工具函数，严格按 mimic 符号
    # if not hasattr(p, "COV_ENABLE_COLLISION_SHAPES"):
    #     p.COV_ENABLE_COLLISION_SHAPES = 6  # 官方源码里这个值就是 6
    # p.configureDebugVisualizer(p.COV_ENABLE_COLLISION_SHAPES, 1, physicsClientId=pb._client)
    # open_gripper(pb, embodiment, angle=0.7)
    #加了这一段之后就没有发生卡顿了
    pb.setPhysicsEngineParameter(
        numSolverIterations=60,  # 迭代次数 ↑
        numSubSteps=4,  # 子步 ↑
        fixedTimeStep=1.0 / 240.0,  # 固定步长
        contactERP=0.2,  # 接触误差修正
        erp=0.2
    )
    robot.move_linear(np.array([0, 0, 0, 0, 0, 0]))

    # 初始化后：断掉了手指间的碰撞
    disable_gripper_self_collision(pb, embodiment)
    # set_initial_gripper_opening(pb, embodiment, opening_deg=40)  # 40°左右一般能放下 scale=1 的 cube

    best_a, best_w = set_to_max_opening(pb, embodiment, samples=16)
    print(f"[Init] best angle = {np.rad2deg(best_a):.1f} deg, width = {best_w * 1000:.1f} mm")
    debug_gripper_limits_and_opening(pb, embodiment)
    # robot.move_linear(np.array([0, 0, 150, 0, 0, 0]))
    # spawn_stim_between_fingers(pb, embodiment, stim_name="long_edge_flat", z_offset=-0.002, use_tactip_tip=True)

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

        # debug_tip_frames(pb, emb)

        tactile_image_l, tactile_image_r = sensor.process()

        # pred_pose_l = pose_model.predict(tactile_image_l)[:6]
        # pred_pose_r = pose_model.predict(tactile_image_r)[:6]
        # print(f'left pred {pred_pose_l}, \n right pred {pred_pose_r}')
        # 调试时看看还有谁在撞：
        print('\n contacts')
        print_self_contacts(pb, embodiment.arm.embodiment_id, embodiment.arm.link_name_to_index)
        print_robot_external_contacts(pb, embodiment.arm.embodiment_id, embodiment.arm.link_name_to_index)

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
        # set_gripper_opening_deg(pb, embodiment, 0.875, use_motors=False)


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
    stim_rpy = np.array([0.0, 1.57, 0.0], dtype=float)  # 先用水平姿态
    stim_pose6 = np.concatenate([mid, stim_rpy])  # 6 维: 位置 + 姿态

    load_stim(pb, stim_urdf, stim_pose6, fixed_base=False, enable_collision=True, scale=1)
    print(f"[INFO] Spawned {stim_name} at", mid.tolist())

# def open_gripper(pb, embodiment, angle=0.45):
#     """把 Robotiq85 打开到一个固定角度（弧度）。"""
#     eid = embodiment.arm.embodiment_id
#     jidx = embodiment.arm.joint_name_to_index  # 里头有 finger_joint 等名字
#
#     # 若 pybullet 正常处理 mimic，只设 finger_joint 就够了；否则按 mimic 关系分别设定
#     if 'finger_joint' in jidx:
#         pb.resetJointState(eid, jidx['finger_joint'], angle)
#
#     # 保险起见，把其它几个关节也按 mimic 关系写一遍（符号和上面的 URDF 一致）
#     def setj(name, val):
#         if name in jidx: pb.resetJointState(eid, jidx[name], val)
#
#     setj('left_inner_knuckle_joint',  +angle)   # multiplier = +1
#     setj('left_inner_finger_joint',   -angle)   # multiplier = -1
#     setj('right_inner_knuckle_joint', -angle)   # multiplier = -1
#     setj('right_inner_finger_joint',  +angle)   # multiplier = +1
#     setj('right_outer_knuckle_joint', -angle)   # multiplier = -1


def set_gripper_angle(pb, emb, angle_rad, use_motors=False,
                      bias_left=0.0, bias_right=0.0, force=1000.0):
    """
    按 URDF 的 mimic 关系**对称**开合到 angle_rad（弧度）。
    - angle_rad: 等效主关节 finger_joint 的角度（≈张开角）
    - bias_left/right: 左右微小校正(弧度)，默认 0
    - use_motors: False 用 resetJointState（快、便于调试）; True 用 POSITION_CONTROL
    """
    eid  = emb.arm.embodiment_id
    jidx = emb.arm.joint_name_to_index

    # 你这份 urdf 的“真·名字”与符号（已对齐 mimic multiplier）
    targets = {
        'finger_joint':              angle_rad,
        'left_inner_knuckle_joint':  +angle_rad + bias_left,
        'left_inner_finger_joint':   -angle_rad + bias_left,
        'right_inner_knuckle_joint': -angle_rad + bias_right,
        'right_inner_finger_joint':  +angle_rad + bias_right,
        'right_outer_knuckle_joint': -angle_rad + bias_right,
    }

    # 先关掉这些关节的速度电机，避免被 VELOCITY_CONTROL 顶住
    for name in targets:
        if name in jidx:
            pb.setJointMotorControl2(eid, jidx[name],
                                     pb.VELOCITY_CONTROL, force=0.0)

    # 写目标
    for name, val in targets.items():
        if name not in jidx:
            continue
        jid = jidx[name]
        if use_motors:
            pb.setJointMotorControl2(eid, jid, pb.POSITION_CONTROL,
                                     targetPosition=float(val),
                                     force=float(force))
        else:
            pb.resetJointState(eid, jid, float(val))


def set_gripper_opening_deg(pb, emb, opening_deg, **kwargs):
    """用“张开角（度）”来控制，内部转弧度。典型范围 0~45°。"""
    set_gripper_angle(pb, emb, np.deg2rad(opening_deg), **kwargs)


def print_self_contacts(pb, body_id, link_name_to_index):
    cps = pb.getContactPoints(bodyA=body_id, bodyB=body_id)
    rev = {v:k for k,v in link_name_to_index.items()}
    seen = set()
    for c in cps:
        a = rev.get(c[3], c[3]); b = rev.get(c[4], c[4])  # link names
        pair = tuple(sorted([a,b]))
        if pair in seen:
            continue
        seen.add(pair)
        print(f"[self-collide] {pair}  normalF={c[9]:.3f}  dist={c[8]:.5f}")
def disable_gripper_self_collision(pb, emb):
    bid  = emb.arm.embodiment_id
    L    = emb.arm.link_name_to_index

    def off(a, b):
        pb.setCollisionFilterPair(bid, bid, L[a], L[b], enableCollision=0)

    # Robotiq85 常见会互撞的对（按你的 URDF 名称）
    pairs = [
        # 左手指内部互撞
        ("left_inner_finger",  "left_outer_finger"),
        ("left_inner_knuckle", "left_outer_knuckle"),
        # 右手指内部互撞
        ("right_inner_finger",  "right_outer_finger"),
        ("right_inner_knuckle", "right_outer_knuckle"),
        # 手指与指套/传感器外壳
        ("left_inner_finger",  "left_tactip_body_link"),
        ("right_inner_finger", "right_tactip_body_link"),
        # 如有需要再加：外指与指套
        ("left_outer_finger",  "left_tactip_body_link"),
        ("right_outer_finger", "right_tactip_body_link"),
    ]
    for a,b in pairs:
        if a in L and b in L:
            off(a,b)
    # for side in ("left", "right"):
    #     for finger in ("inner_finger", "outer_finger", "inner_knuckle", "outer_knuckle"):
    #         pb.setCollisionFilterPair(
    #             emb.arm.embodiment_id, emb.arm.embodiment_id,
    #             L[f"{side}_tactip_body_link"], L[f"{side}_{finger}"], 0
    #         )
def print_robot_external_contacts(pb, robot_id, link_name_to_index, min_normF=1e-6):
    cps = pb.getContactPoints(bodyA=robot_id)  # bodyB 不设，表示与任何 B 的接触
    rev = {v:k for k,v in link_name_to_index.items()}

    # 收敛信息
    if not cps:
        print("[contacts] none")
        return

    # 汇总并仅打印力较大的若干条
    rows = []
    for c in cps:
        linkA = rev.get(c[3], c[3])   # 机器人侧link
        bodyB = c[1] if c[1] != robot_id else c[2]   # 另一侧 bodyUniqueId
        linkB = c[4]
        Fn    = float(c[9])
        dist  = float(c[8])
        if Fn < min_normF:
            continue
        rows.append((linkA, bodyB, linkB, Fn, dist, c[5], c[6]))  # 名称、力、距离、接触点等

    if not rows:
        print("[contacts] only tiny forces (< min_normF)")
        return

    # 按法向力排序打印前 N 条
    rows.sort(key=lambda r: r[3], reverse=True)
    print("[contacts] top hits (linkA, bodyB, linkB, Fn, dist):")
    for r in rows[:10]:
        print(f"  {r[0]:>24s}  vs  body{r[1]}:link{r[2]:<2d}   Fn={r[3]:.4f}  dist={r[4]:.5f}")

def debug_gripper_limits_and_opening(pb, emb):
    import numpy as np
    jmap = emb.arm.joint_name_to_index
    bid  = emb.arm.embodiment_id

    names = [
        'finger_joint',
        'left_inner_knuckle_joint', 'left_inner_finger_joint',
        'right_inner_knuckle_joint','right_inner_finger_joint',
        'right_outer_knuckle_joint',
    ]
    print("\n[Joint limits & positions]")
    for n in names:
        if n not in jmap:
            continue
        jid = jmap[n]
        info = pb.getJointInfo(bid, jid)
        lower, upper = info[8], info[9]
        pos = pb.getJointState(bid, jid)[0]
        print(f"{n:28s} pos={pos:.4f}  lim=[{lower:.4f}, {upper:.4f}]")

    # 量化“开口宽度”：两侧指尖或内指的距离
    L = emb.arm.link_name_to_index.get('left_tactip_tip_link')  or emb.arm.link_name_to_index.get('left_inner_finger')
    R = emb.arm.link_name_to_index.get('right_tactip_tip_link') or emb.arm.link_name_to_index.get('right_inner_finger')
    if L is not None and R is not None:
        pL = np.array(pb.getLinkState(bid, L)[0])
        pR = np.array(pb.getLinkState(bid, R)[0])
        width = np.linalg.norm(pL - pR)
        print(f"[Opening width] |tip_L - tip_R| = {width*1000:.1f} mm")

def set_initial_gripper_opening(pb, emb, opening_deg=35):
    """仿真加载后只设一次初始开度（不再控制）"""
    eid  = emb.arm.embodiment_id
    jidx = emb.arm.joint_name_to_index
    ang  = np.deg2rad(opening_deg)

    # 先把相关电机的速度控制关掉，避免顶回
    for n in ['finger_joint',
              'left_inner_knuckle_joint','left_inner_finger_joint',
              'right_inner_knuckle_joint','right_inner_finger_joint',
              'right_outer_knuckle_joint']:
        if n in jidx:
            pb.setJointMotorControl2(eid, jidx[n], pb.VELOCITY_CONTROL, force=0.0)

    # 按 mimic 符号把六个关节一次性写入（只这一次）
    def setj(name, val):
        if name in jidx: pb.resetJointState(eid, jidx[name], float(val))

    setj('finger_joint',              ang)
    setj('left_inner_knuckle_joint',  +ang)
    setj('left_inner_finger_joint',   -ang)
    setj('right_inner_knuckle_joint', -ang)
    setj('right_inner_finger_joint',  +ang)
    setj('right_outer_knuckle_joint', -ang)

import numpy as np
import time

def _gripper_joint_mapping(jidx):
    """
    基于 ur5.urdf（你这份）的 mimic 关系，返回 (joint_name, multiplier, offset) 列表。
    - 主关节：finger_joint
    - inner = 指尖节；outer = 靠基座节
    - 有的仓左侧没有 outer_knuckle_joint；这里做了“存在才用”的兼容。
    """
    mapping = [
        ("finger_joint",               +1.0, 0.0),   # master
        ("left_inner_knuckle_joint",   +1.0, 0.0),
        ("left_inner_finger_joint",    -1.0, 0.0),
        ("right_inner_knuckle_joint",  -1.0, 0.0),
        ("right_inner_finger_joint",   +1.0, 0.0),
        ("right_outer_knuckle_joint",  -1.0, 0.0),
        # 有些 urdf 没有 left_outer_knuckle_joint；若存在就一起驱动
        ("left_outer_knuckle_joint",   +1.0, 0.0),
        # 如果你的模型有 outer_finger_joint 也按各自符号加入：
        ("left_outer_finger_joint",    -1.0, 0.0),
        ("right_outer_finger_joint",   +1.0, 0.0),
    ]
    # 只保留 urdf 里实际存在的关节
    return [(n, m, b) for (n, m, b) in mapping if n in jidx]


def set_gripper_angle(pb, emb, angle_rad, *, use_motors=True, force=1200.0):
    """
    让两侧对称开合到 angle_rad（弧度），
    inner finger=指尖节，outer finger=靠基座节（你这份定义）。
    """
    eid  = emb.arm.embodiment_id
    jidx = emb.arm.joint_name_to_index
    mapping = _gripper_joint_mapping(jidx)

    # 关掉这些关节的速度电机，避免顶回
    for name, *_ in mapping:
        pb.setJointMotorControl2(eid, jidx[name], pb.VELOCITY_CONTROL, force=0.0)

    # 读取主关节上限并夹住（用电机模式时生效）
    lower, upper = pb.getJointInfo(eid, jidx["finger_joint"])[8:10]
    target = float(np.clip(angle_rad, lower, upper))

    if use_motors:
        # 只驱动主关节；mimic 由我们手动同步到各从关节，确保各仿真版本都稳
        pb.setJointMotorControl2(eid, jidx["finger_joint"], pb.POSITION_CONTROL,
                                 targetPosition=target, force=float(force))
        # 同步所有 mimic 关节到 m*θ + offset
        for name, mult, offs in mapping:
            if name == "finger_joint":
                continue
            pb.setJointMotorControl2(eid, jidx[name], pb.POSITION_CONTROL,
                                     targetPosition=float(mult*target + offs),
                                     force=float(force))
    else:
        # 只设一次，不再控制（调试用）
        for name, mult, offs in mapping:
            theta = target if name == "finger_joint" else (mult*target + offs)
            pb.resetJointState(eid, jidx[name], float(theta))


def current_opening_width(pb, emb):
    """
    用“指尖（inner finger）或 tactip tip”之间的距离估计当前开口宽度（米）。
    自动选择存在的 link。
    """
    L = emb.arm.link_name_to_index
    bid = emb.arm.embodiment_id
    # 优先用 tactip tip；没有就用 inner finger
    l_name = "left_tactip_tip_link"  if "left_tactip_tip_link"  in L else "left_inner_finger"
    r_name = "right_tactip_tip_link" if "right_tactip_tip_link" in L else "right_inner_finger"
    pL = np.array(pb.getLinkState(bid, L[l_name])[0])
    pR = np.array(pb.getLinkState(bid, L[r_name])[0])
    return float(np.linalg.norm(pL - pR))


def set_gripper_width(pb, emb, target_width_m, *, tol=0.001, max_iter=12):
    """
    按“目标宽度（米）”开合：二分搜索主关节角，避免几何/碰撞造成的非线性。
    需要仿真在 step() 推进时使用（每次迭代 sleep 一点点或 step 几步）。
    """
    eid  = emb.arm.embodiment_id
    jidx = emb.arm.joint_name_to_index
    lo, hi = pb.getJointInfo(eid, jidx["finger_joint"])[8:10]  # 关节上下限
    # 先把电机设置好
    set_gripper_angle(pb, emb, lo, use_motors=True)

    for _ in range(max_iter):
        mid = 0.5*(lo + hi)
        set_gripper_angle(pb, emb, mid, use_motors=True)
        # 给一点时间/步进让位姿到位
        for _ in range(5):
            pb.stepSimulation()
        w = current_opening_width(pb, emb)
        err = w - target_width_m
        if abs(err) <= tol:
            break
        if err < 0:   # 还不够宽 → 增大角度
            lo = mid
        else:         # 太宽 → 减小角度
            hi = mid
    return w

import numpy as np

def force_gripper_angle_once(pb, emb, angle_rad):
    """
    只设置一次夹爪角度 angle_rad（弧度），不保留任何电机控制。
    若你后面又调用了 open/close 或 POSITION_CONTROL，它可能会被改回去。
    """
    eid  = emb.arm.embodiment_id
    J    = emb.arm.joint_name_to_index

    # 这几个名字不存在就会被自动跳过
    names = [
        'finger_joint',
        'left_inner_knuckle_joint', 'left_inner_finger_joint',
        'right_inner_knuckle_joint','right_inner_finger_joint',
        'right_outer_knuckle_joint',    # 左侧有 outer_knuckle 就加上
        'left_outer_knuckle_joint',
        # 有些模型还有 outer_finger_joint，也一并尝试
        'left_outer_finger_joint','right_outer_finger_joint',
    ]

    # 1) 关掉这些关节的速度电机，避免被“顶回”
    for n in names:
        if n in J:
            pb.setJointMotorControl2(eid, J[n], pb.VELOCITY_CONTROL, force=0.0)

    # 2) 按你的主从关系“对称”写入一次
    def setj(n, v):
        if n in J:
            pb.resetJointState(eid, J[n], float(v))

    a = float(angle_rad)
    setj('finger_joint',              a)
    setj('left_inner_knuckle_joint',  +a)
    setj('left_inner_finger_joint',   -a)
    setj('right_inner_knuckle_joint', -a)
    setj('right_inner_finger_joint',  +a)
    setj('right_outer_knuckle_joint', -a)
    setj('left_outer_knuckle_joint',  +a)   # 若不存在会被自动跳过
    setj('left_outer_finger_joint',   -a)
    setj('right_outer_finger_joint',  +a)

    # 3) 给几步仿真让位姿稳定
    for _ in range(5):
        pb.stepSimulation()
def print_gripper_states_and_width(pb, emb):
    eid  = emb.arm.embodiment_id
    J    = emb.arm.joint_name_to_index

    check = [
        'finger_joint',
        'left_inner_knuckle_joint', 'left_inner_finger_joint',
        'right_inner_knuckle_joint','right_inner_finger_joint',
        'right_outer_knuckle_joint','left_outer_knuckle_joint',
    ]

    print("\n[Joint limits & positions]")
    for n in check:
        if n not in J:
            continue
        jid = J[n]
        info = pb.getJointInfo(eid, jid)
        lo, up = info[8], info[9]
        pos = pb.getJointState(eid, jid)[0]
        print(f"{n:28s} pos={pos:.4f}  lim=[{lo:.4f}, {up:.4f}]")

    # 计算“开口宽度”（优先用 tactip tip，没有就用 inner_finger）
    L = emb.arm.link_name_to_index
    if 'left_tactip_tip_link'  in L and 'right_tactip_tip_link' in L:
        lnm, rnm = 'left_tactip_tip_link',  'right_tactip_tip_link'
    else:
        lnm, rnm = 'left_inner_finger', 'right_inner_finger'

    pL = np.array(pb.getLinkState(eid, L[lnm])[0])
    pR = np.array(pb.getLinkState(eid, L[rnm])[0])
    width = float(np.linalg.norm(pL - pR))
    print(f"[Opening width] |{lnm} - {rnm}| = {width*1000:.1f} mm")

    return width

def _apply_angle_once(pb, emb, angle):
    eid = emb.arm.embodiment_id
    J   = emb.arm.joint_name_to_index
    names = ['finger_joint','left_inner_knuckle_joint','left_inner_finger_joint',
             'right_inner_knuckle_joint','right_inner_finger_joint','right_outer_knuckle_joint',
             'left_outer_knuckle_joint','left_outer_finger_joint','right_outer_finger_joint']
    for n in names:
        if n in J:
            pb.setJointMotorControl2(eid, J[n], pb.VELOCITY_CONTROL, force=0.0)
    def setj(n,v):
        if n in J: pb.resetJointState(eid, J[n], float(v))
    a = float(angle)
    setj('finger_joint',              a)
    setj('left_inner_knuckle_joint',  +a)
    setj('left_inner_finger_joint',   -a)
    setj('right_inner_knuckle_joint', -a)
    setj('right_inner_finger_joint',  +a)
    setj('right_outer_knuckle_joint', -a)
    setj('left_outer_knuckle_joint',  +a)
    setj('left_outer_finger_joint',   -a)
    setj('right_outer_finger_joint',  +a)
    for _ in range(5): pb.stepSimulation()

def _opening_width(pb, emb):
    import numpy as np
    bid = emb.arm.embodiment_id
    L   = emb.arm.link_name_to_index
    lnm = 'left_tactip_tip_link'  if 'left_tactip_tip_link'  in L else 'left_inner_finger'
    rnm = 'right_tactip_tip_link' if 'right_tactip_tip_link' in L else 'right_inner_finger'
    pL = np.array(pb.getLinkState(bid, L[lnm])[0])
    pR = np.array(pb.getLinkState(bid, L[rnm])[0])
    return float(np.linalg.norm(pL - pR))

def set_to_max_opening(pb, emb, samples=12):
    """在关节允许范围里采样，找到让 tip 距离最大的角度，只设置一次。"""
    eid = emb.arm.embodiment_id
    J   = emb.arm.joint_name_to_index
    lo, hi = pb.getJointInfo(eid, J['finger_joint'])[8:10]
    lo = -abs(hi)
    best_a, best_w = lo, -1.0
    for a in np.linspace(lo, hi, samples):
        _apply_angle_once(pb, emb, a)
        w = _opening_width(pb, emb)
        if w > best_w:
            best_a, best_w = a, w
    _apply_angle_once(pb, emb, best_a)   # 定在最佳角度
    return best_a, best_w

def _apply_angle_once(pb, emb, angle):
    eid = emb.arm.embodiment_id
    J   = emb.arm.joint_name_to_index
    names = ['finger_joint','left_inner_knuckle_joint','left_inner_finger_joint',
             'right_inner_knuckle_joint','right_inner_finger_joint','right_outer_knuckle_joint',
             'left_outer_knuckle_joint','left_outer_finger_joint','right_outer_finger_joint']
    for n in names:
        if n in J:
            pb.setJointMotorControl2(eid, J[n], pb.VELOCITY_CONTROL, force=0.0)
    def setj(n,v):
        if n in J: pb.resetJointState(eid, J[n], float(v))
    a = float(angle)
    setj('finger_joint',              a)
    setj('left_inner_knuckle_joint',  +a)
    setj('left_inner_finger_joint',   -a)
    setj('right_inner_knuckle_joint', -a)
    setj('right_inner_finger_joint',  +a)
    setj('right_outer_knuckle_joint', -a)
    setj('left_outer_knuckle_joint',  +a)
    setj('left_outer_finger_joint',   -a)
    setj('right_outer_finger_joint',  +a)
    for _ in range(3): pb.stepSimulation()

def _opening_width(pb, emb):
    bid = emb.arm.embodiment_id
    L   = emb.arm.link_name_to_index
    lnm = 'left_tactip_tip_link'  if 'left_tactip_tip_link'  in L else 'left_inner_finger'
    rnm = 'right_tactip_tip_link' if 'right_tactip_tip_link' in L else 'right_inner_finger'
    pL = np.array(pb.getLinkState(bid, L[lnm])[0])
    pR = np.array(pb.getLinkState(bid, L[rnm])[0])
    return float(np.linalg.norm(pL - pR))

def set_gripper_width(pb, emb, target_w_m, tol=0.0008, max_iter=16):
    """按“目标宽度（米）”开合：在 finger_joint 的限位内二分搜索角度"""
    eid = emb.arm.embodiment_id
    J   = emb.arm.joint_name_to_index
    lo, hi = pb.getJointInfo(eid, J['finger_joint'])[8:10]  # lower, upper

    # 先粗扫几点找单调区（因为你的装配可能负角更开）
    samples = np.linspace(lo, hi, 9)
    widths  = []
    for a in samples:
        _apply_angle_once(pb, emb, a); widths.append(_opening_width(pb, emb))
    # 选离目标最近的两侧作为初始搜索区间
    idx = int(np.argmin([abs(w-target_w_m) for w in widths]))
    lo_idx = max(0, idx-1); hi_idx = min(len(samples)-1, idx+1)
    lo, hi = samples[lo_idx], samples[hi_idx]

    # 二分搜索
    best_a, best_w = lo, _opening_width(pb, emb)
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        _apply_angle_once(pb, emb, mid)
        w = _opening_width(pb, emb)
        if abs(w-target_w_m) < abs(best_w-target_w_m):
            best_a, best_w = mid, w
        if abs(w - target_w_m) <= tol:
            break
        if w < target_w_m:
            # 当前不够宽 → 朝“更宽”的一侧收；根据粗扫趋势判断
            if widths[hi_idx] > widths[lo_idx]: lo = mid
            else: hi = mid
        else:
            if widths[hi_idx] > widths[lo_idx]: hi = mid
            else: lo = mid
    # 定在 best_a
    _apply_angle_once(pb, emb, best_a)
    return best_a, best_w

# 便捷封装
def gripper_open(pb, emb, open_w_mm=95):
    a, w = set_gripper_width(pb, emb, open_w_mm/1000.0)
    print(f"[gripper_open] angle={np.rad2deg(a):.1f}deg  width={w*1000:.1f}mm")
    return a, w

def gripper_close_for_cube(pb, emb, cube_mm=80, squeeze_mm=2):
    target = max(1.0, cube_mm - squeeze_mm)  # 略小于方块以夹紧
    a, w = set_gripper_width(pb, emb, target/1000.0)
    print(f"[gripper_close] angle={np.rad2deg(a):.1f}deg  width={w*1000:.1f}mm")
    return a, w
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
