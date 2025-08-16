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
        'stim_pose': (600, 0, 60, 0, 0, 0),
        'show_gui': True,
        'load_target': target_indicator
    }

    sensor_params = {
        'left': {
            'type': 'standard_tactip',
            'core': 'no_core',
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
            'core': 'no_core',
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
    return setup_embodiment(env_params, sensor_params)



def create_controllers():
    align = PIDController(
        kp=[0, 3, 0, 0, 0, 0],
        ki=[0, 0.003, 0, 0, 0, 0],
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

    pb, robot, sensor = setup_environment(target_indicator, sensor_type=args.sensor)
    embodiment = sensor.embodiment

    # —— 一次性物理设置（确保手动步进生效）——
    pb.setRealTimeSimulation(0)
    pb.setTimeStep(1.0/240.0)
    pb.setPhysicsEngineParameter(numSolverIterations=80, numSubSteps=6,
                                 fixedTimeStep=1.0/240.0, erp=0.2, contactERP=0.2,
                                 enableConeFriction=1, restitutionVelocityThreshold=0.05)

    robot.move_linear(np.array([0, 0, 0, 0, 0, 0]))
    disable_gripper_self_collision(pb, embodiment)
    gripper_open(pb, embodiment, open_w_mm=110)
    debug_gripper_limits_and_opening(pb, embodiment)
    robot.move_linear(np.array([0, -60, 130, 0, 0, 0]))

    # PB 方块 + 约束（你已替换过的函数）
    stim_info = spawn_pb_cube_between_fingers(
        pb, embodiment, edge_mm=80, z_offset=-0.01, use_tactip_tip=True
    )
    stim_id = stim_info[0]

    it = 0
    MAX_ITERS = 20000               # 迭代多一点
    SUBSTEPS_AFTER_MOVE = 12        # 每次move后额外步进的仿真步数

    try:
        while it < MAX_ITERS:
            # 1) 触觉 -> 姿态预测
            tactile_image_l, tactile_image_r = sensor.process()
            left_pre  = pose_model.predict(tactile_image_l)
            right_pre = pose_model.predict(tactile_image_r)
            ob_pose   = get_ob_pose(left_pre, right_pre)
            # print(ob_pose)  # 需要时再开

            # 2) 当前末端位姿
            current_pose = robot.pose

            # 3) 控制策略：下压 or 对齐+前进
            if current_pose[2] < 160:
                dpose = np.array([0, 0, 1, 0, 0, 0], dtype=float)  # 先抬一点，防止直接硬挤
            else:
                align_ctrl, point_ctrl = create_controllers()
                align = align_ctrl.update(
                    np.array([0, ob_pose[3], 0, 0, 0, 0], dtype=float),
                    np.zeros(6, dtype=float)
                )
                # 限幅，避免一次给太大
                dy = float(np.clip(align[1], -1.0, 1.0))
                dpose = np.array([0, dy, -2, 0, 0, 0], dtype=float)

            # 4) 执行移动
            next_pose = current_pose + dpose
            robot.move_linear(next_pose)

            # 5) 手动步进 + 放慢画面
            for _ in range(SUBSTEPS_AFTER_MOVE):
                pb.stepSimulation()
                time.sleep(1.0/240.0)

            # 6) 周期性日志与可视化
            if (it % 10) == 0:
                log_cube_state(
                    pb, "cube:loop", stim_id,
                    ee_id=embodiment.arm.embodiment_id,
                    ee_link=embodiment.arm.link_name_to_index["ee_link"]
                )
            if (it % 30) == 0:
                draw_axes(pb, stim_id, -1, length=0.04, life=0.3)
                draw_axes(pb, embodiment.arm.embodiment_id,
                          embodiment.arm.link_name_to_index["ee_link"],
                          length=0.06, life=0.3)

            it += 1

    except KeyboardInterrupt:
        print("Interrupted by user.")

    # 结束后不要立刻关GUI：按 q 退出
    print("Loop finished. Press 'q' in the GUI window to quit.")
    while pb.isConnected():
        keys = pb.getKeyboardEvents()
        if ord('q') in keys and (keys[ord('q')] & pb.KEY_WAS_TRIGGERED):
            break
        pb.stepSimulation()
        time.sleep(1.0/120.0)


def get_ob_pose(left, right):
    object_pose = np.zeros(6)
    object_pose[5] = -(left[4] - right[4]) / 2
    object_pose[1] = -(left[2] - right[2])
    object_pose[3] = (left[3] + right[3]) / 2

    return object_pose


def stiffness2cube(stim_id, rot_stiffness=10, rot_max_torque=10):
    curr_pos, curr_orn = p.getBasePositionAndOrientation(stim_id)
    target_orn = p.getQuaternionFromEuler([0.0, 1.57, 0.0])
    orn_err = p.getDifferenceQuaternion(curr_orn, target_orn)
    axis, angle = p.getAxisAngleFromQuaternion(orn_err)

    torque = -rot_stiffness * angle
    torque = np.clip(torque, -rot_max_torque, rot_max_torque)
    p.applyExternalTorque(stim_id, -1, [torque * axis[0], 0 * axis[1], 0 * axis[2]], p.WORLD_FRAME)


def spawn_stim_between_fingers(
        pb,
        embodiment,
        stim_name="cube",
        z_offset=0.0,
        use_tactip_tip=True,
        p2p_max_force=1e6,
        rot_stiffness=30,
        rot_max_torque=50,
        global_scaling=1.0,
):

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
    mid[2] += z_offset
    # mid[0] -= 0.012
    # mid[1] -= 0.015

    stim_urdf = add_assets_path(r"H:\tactile-gym-3-bowen\tg3\simulator\stimuli\cube\cube.urdf")


    mid = np.asarray(mid, dtype=float)
    stim_rpy = np.array([0.0, 1.57, 0.0], dtype=float)
    stim_pose6 = np.concatenate([mid, stim_rpy])

    stim_pose = np.array(stim_pose6)
    stim_pos, stim_rpy = stim_pose[:3], stim_pose[3:]
    stim_id = p.loadURDF(
        stim_urdf,
        stim_pos,
        pb.getQuaternionFromEuler(stim_rpy),
        useFixedBase=False,
        globalScaling=1
    )
    # 打印 PyBullet 认为的惯性参数（COM 在 local inertia frame 的位置）
    info = p.getDynamicsInfo(stim_id, -1)
    mass = info[0]
    inertia_diag = info[2]
    com_local = info[3]  # localInertiaPosition
    com_local_orn = info[4]  # localInertiaOrientation
    print(f"[dyn] mass={mass}  inertia={inertia_diag}  COM(local)={com_local}  COM_orn(local)={com_local_orn}")

    # 2) 还原到世界系，画一个小十字标 COM 位置
    base_pos, base_orn = p.getBasePositionAndOrientation(stim_id)
    com_world, _ = p.multiplyTransforms(base_pos, base_orn, com_local, com_local_orn)
    p.addUserDebugLine(com_world, (com_world[0] + 0.03, com_world[1], com_world[2]), [1, 0, 0], 2, lifeTime=1)
    p.addUserDebugLine(com_world, (com_world[0], com_world[1] + 0.03, com_world[2]), [0, 1, 0], 2, lifeTime=1)
    p.addUserDebugLine(com_world, (com_world[0], com_world[1], com_world[2] + 0.03), [0, 0, 1], 2, lifeTime=1)

    # pb.setCollisionFilterGroupMask(stim_id, -1, 0, 0)
    p.changeDynamics(stim_id, -1, mass=5,
                     lateralFriction=1,
                     rollingFriction=0.1,
                     spinningFriction=0.1,
                     angularDamping=3,
                     linearDamping=3
                     )

    parent_link_index = link_map["ee_link"]

    link_state = p.getLinkState(body_id, parent_link_index, computeForwardKinematics=True)
    link_world_pos = np.array(link_state[4])  # worldLinkFramePosition
    link_world_orn = np.array(link_state[5])
    inv_link_pos, inv_link_orn = p.invertTransform(link_world_pos.tolist(), link_world_orn.tolist())
    pivot_in_parent, _ = p.multiplyTransforms(
        inv_link_pos, inv_link_orn,
        mid.tolist(), [0, 0, 0, 1]
    )
    pivot_in_parent = list(pivot_in_parent)
    pivot_in_parent = [0, 0, 0.17]
    pivot_in_child = [0, 0, 0]


    cid = p.createConstraint(
        parentBodyUniqueId=body_id,
        parentLinkIndex=parent_link_index,
        childBodyUniqueId=stim_id,
        childLinkIndex=-1,
        jointType=p.JOINT_POINT2POINT,
        jointAxis=[0, 0, 0],
        parentFramePosition=pivot_in_parent,
        childFramePosition=pivot_in_child
    )
    # gear_cid = p.createConstraint(
    #     parentBodyUniqueId=body_id,
    #     parentLinkIndex=parent_link_index,
    #     childBodyUniqueId=stim_id,
    #     childLinkIndex=-1,
    #     jointType=p.JOINT_GEAR,
    #     jointAxis=[0, 0, 1],
    #     parentFramePosition=pivot_in_parent,
    #     childFramePosition=pivot_in_child,
    #
    # )
    # p.changeConstraint(
    #     gear_cid,
    #     gearRatio=1.0,
    #     relativePositionTarget=0.0,
    #     erp=rot_stiffness,
    #     maxForce=rot_max_torque
    # )
    p.changeConstraint(cid, maxForce=1e6)
    p.setPhysicsEngineParameter(
        numSolverIterations=60,
        numSubSteps=4,
        fixedTimeStep=1.0 / 240.0,
        contactERP=0.2,
        erp=0.2
    )
    print(f"[INFO] Spawned {stim_name} and constrained (no translation, free rotation) at {mid.tolist()}")
    return stim_id, cid



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

    L = emb.arm.link_name_to_index.get('left_tactip_tip_link')  or emb.arm.link_name_to_index.get('left_inner_finger')
    R = emb.arm.link_name_to_index.get('right_tactip_tip_link') or emb.arm.link_name_to_index.get('right_inner_finger')
    if L is not None and R is not None:
        pL = np.array(pb.getLinkState(bid, L)[0])
        pR = np.array(pb.getLinkState(bid, R)[0])
        width = np.linalg.norm(pL - pR)
        print(f"[Opening width] |tip_L - tip_R| = {width*1000:.1f} mm")
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


    samples = np.linspace(lo, hi, 9)
    widths  = []
    for a in samples:
        _apply_angle_once(pb, emb, a); widths.append(_opening_width(pb, emb))

    idx = int(np.argmin([abs(w-target_w_m) for w in widths]))
    lo_idx = max(0, idx-1); hi_idx = min(len(samples)-1, idx+1)
    lo, hi = samples[lo_idx], samples[hi_idx]


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

    _apply_angle_once(pb, emb, best_a)
    return best_a, best_w
def gripper_open(pb, emb, open_w_mm=95):
    a, w = set_gripper_width(pb, emb, open_w_mm/1000.0)
    print(f"[gripper_open] angle={np.rad2deg(a):.1f}deg  width={w*1000:.1f}mm")
    return a, w
def gripper_close_for_cube(pb, emb, cube_mm=80, squeeze_mm=2):
    target = max(1.0, cube_mm - squeeze_mm)  # 略小于方块以夹紧
    a, w = set_gripper_width(pb, emb, target/1000.0)
    print(f"[gripper_close] angle={np.rad2deg(a):.1f}deg  width={w*1000:.1f}mm")
    return a, w

def _quat_to_mat(pb, q):
    return np.array(pb.getMatrixFromQuaternion(q), dtype=float).reshape(3,3)

def log_cube_state(pb, tag, cube_id, ee_id=None, ee_link=None):
    pos, orn = pb.getBasePositionAndOrientation(cube_id)
    rpy = pb.getEulerFromQuaternion(orn)
    R = _quat_to_mat(pb, orn)
    z_world = R[:,2]
    tilt_deg = np.degrees(np.arccos(np.clip(np.dot(z_world, [0,0,1]), -1.0, 1.0)))

    rel_yaw_deg = None
    if ee_id is not None and ee_link is not None:
        ls = pb.getLinkState(ee_id, ee_link, computeForwardKinematics=True)
        ee_pos, ee_orn = ls[4], ls[5]
        inv_pos, inv_orn = pb.invertTransform(ee_pos, ee_orn)
        _, rel_orn = pb.multiplyTransforms(inv_pos, inv_orn, pos, orn)
        rel_rpy = pb.getEulerFromQuaternion(rel_orn)
        rel_yaw_deg = np.degrees(rel_rpy[2])

    cps = pb.getContactPoints(bodyA=cube_id)
    cnum = len(cps)
    mind = None; Fn_sum = 0.0
    if cnum:
        mind   = min([c[8] for c in cps])   # 负值=穿插深度
        Fn_sum = sum([c[9] for c in cps])   # 法向力合计(N)

    print(f"[{tag}] pos={np.round(pos,4).tolist()} rpy(deg)={np.round(np.degrees(rpy),2).tolist()} "
          f"tilt={tilt_deg:.2f}° contacts={cnum} minDist={mind} Fn_sum={Fn_sum:.2f} "
          + (f"relYaw(ee)={rel_yaw_deg:.2f}°" if rel_yaw_deg is not None else ""))

def draw_axes(pb, body_id, link_idx=-1, length=0.05, life=0.2):
    if link_idx == -1:
        pos, orn = pb.getBasePositionAndOrientation(body_id)
    else:
        ls = pb.getLinkState(body_id, link_idx, computeForwardKinematics=True)
        pos, orn = ls[4], ls[5]
    R = _quat_to_mat(pb, orn)
    x2 = (pos[0]+length*R[0,0], pos[1]+length*R[1,0], pos[2]+length*R[2,0])
    y2 = (pos[0]+length*R[0,1], pos[1]+length*R[1,1], pos[2]+length*R[2,1])
    z2 = (pos[0]+length*R[0,2], pos[1]+length*R[1,2], pos[2]+length*R[2,2])
    pb.addUserDebugLine(pos, x2, [1,0,0], 2, life)
    pb.addUserDebugLine(pos, y2, [0,1,0], 2, life)
    pb.addUserDebugLine(pos, z2, [0,0,1], 2, life)

def spawn_pb_cube_between_fingers(
    pb,
    embodiment,
    edge_mm=80,
    mass_kg=0.5,
    rgba=(0.2, 0.6, 0.9, 1.0),
    z_offset=0.0,
    use_tactip_tip=True,
    max_force=1e5
):
    body_id = embodiment.arm.embodiment_id
    link_map = embodiment.arm.link_name_to_index

    # 指尖选择
    left_link  = "left_tactip_tip_link" if use_tactip_tip else "left_inner_finger"
    right_link = "right_tactip_tip_link" if use_tactip_tip else "right_inner_finger"
    assert left_link in link_map and right_link in link_map, "finger link not found"

    l_pos, _, *_ = pb.getLinkState(body_id, link_map[left_link],  computeForwardKinematics=True)
    r_pos, _, *_ = pb.getLinkState(body_id, link_map[right_link], computeForwardKinematics=True)
    l_pos, r_pos = np.array(l_pos), np.array(r_pos)
    mid = (l_pos + r_pos) / 2.0
    mid[2] += z_offset

    # === 生成方块（加 3mm 安全间隙避免初始穿插） ===
    half = float(edge_mm) / 2000.0
    col_id = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[half, half, half])
    vis_id = pb.createVisualShape(   pb.GEOM_BOX, halfExtents=[half, half, half], rgbaColor=rgba)
    spawn_pos = mid.copy(); spawn_pos[2] += (half + 0.003)

    cube_id = pb.createMultiBody(
        baseMass=mass_kg,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=spawn_pos.tolist(),
        baseOrientation=[0, 0, 0, 1]
    )
    pb.changeDynamics(cube_id, -1,
                      lateralFriction=0.8, spinningFriction=0.002,
                      rollingFriction=0.0, linearDamping=0.04, angularDamping=0.04)

    # === 目标：只允许绕 ee 的 z 轴旋转 ===
    ee_idx = link_map["ee_link"]
    ls = pb.getLinkState(body_id, ee_idx, computeForwardKinematics=True)
    ee_pos, ee_orn = ls[4], ls[5]
    inv_pos, inv_orn = pb.invertTransform(ee_pos, ee_orn)
    parent_pivot, _ = pb.multiplyTransforms(inv_pos, inv_orn, spawn_pos.tolist(), [0,0,0,1])

    cid_info = {"type": None, "ids": []}
    try:
        # 首选：真正的铰链
        cid_hinge = pb.createConstraint(
            parentBodyUniqueId=body_id, parentLinkIndex=ee_idx,
            childBodyUniqueId=cube_id, childLinkIndex=-1,
            jointType=pb.JOINT_REVOLUTE,               # 某些构建不支持 → 会抛异常
            jointAxis=[0, 0, 1],
            parentFramePosition=parent_pivot,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=[0, 0, 0, 1]
        )
        pb.changeConstraint(cid_hinge, maxForce=max_force, erp=0.2)
        cid_info["type"] = "HINGE"
        cid_info["ids"]  = [cid_hinge]
        print("[INFO] Hinge constraint OK (real revolute).")
    except Exception as e:
        print(f"[WARN] Hinge not supported, fallback to P2P+GEAR. Reason: {e}")

        # 先把cube的“朝向”对齐到ee的yaw，避免一上来就有大角误差
        ls = pb.getLinkState(body_id, ee_idx, computeForwardKinematics=True)
        ee_pos, ee_orn = ls[4], ls[5]
        ex, ey, ez = pb.getEulerFromQuaternion(ee_orn)
        cx, cy, cz = pb.getEulerFromQuaternion([0, 0, 0, 1])  # cube当前为[0,0,0,1]
        new_orn = pb.getQuaternionFromEuler([cx, cy, ez])  # 只对齐yaw
        pb.resetBasePositionAndOrientation(cube_id, spawn_pos.tolist(), new_orn)

        # P2P：把cube COM挂到ee局部 parent_pivot（这是COM在ee坐标下的位置）
        cid_p2p = pb.createConstraint(
            parentBodyUniqueId=body_id, parentLinkIndex=ee_idx,
            childBodyUniqueId=cube_id, childLinkIndex=-1,
            jointType=pb.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=parent_pivot,
            childFramePosition=[0, 0, 0]
        )
        # 软一点，避免瞬间“拉拽”
        pb.changeConstraint(cid_p2p, maxForce=5e3, erp=0.1)

        # GEAR1：锁“roll”（让两者的Z轴映射到X轴去耦合）
        qZ_to_X = pb.getQuaternionFromEuler([0, np.pi / 2, 0])
        cid_gx = pb.createConstraint(
            parentBodyUniqueId=body_id, parentLinkIndex=ee_idx,
            childBodyUniqueId=cube_id, childLinkIndex=-1,
            jointType=pb.JOINT_GEAR,
            jointAxis=[0, 0, 1],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=qZ_to_X,
            childFrameOrientation=qZ_to_X
        )
        pb.changeConstraint(cid_gx, gearRatio=1.0, erp=0.15, maxForce=3e3)

        # GEAR2：锁“pitch”（让Z轴映射到Y轴）
        qZ_to_Y = pb.getQuaternionFromEuler([np.pi / 2, 0, 0])
        cid_gy = pb.createConstraint(
            parentBodyUniqueId=body_id, parentLinkIndex=ee_idx,
            childBodyUniqueId=cube_id, childLinkIndex=-1,
            jointType=pb.JOINT_GEAR,
            jointAxis=[0, 0, 1],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=qZ_to_Y,
            childFrameOrientation=qZ_to_Y
        )
        pb.changeConstraint(cid_gy, gearRatio=1.0, erp=0.15, maxForce=3e3)

        cid_info["type"] = "P2P+GEAR"
        cid_info["ids"] = [cid_p2p, cid_gx, cid_gy]

    print(f"[INFO] Spawned PB cube {edge_mm}mm @ {np.round(spawn_pos,4).tolist()} "
          f"and constrained to ee_link ({cid_info['type']}).")

    # 初始调试打印与坐标轴可视化
    log_cube_state(pb, "cube:init", cube_id, ee_id=body_id, ee_link=ee_idx)
    draw_axes(pb, cube_id, -1)
    draw_axes(pb, body_id, ee_idx)
    return cube_id, cid_info



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
