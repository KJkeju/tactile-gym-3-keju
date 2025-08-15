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


    # robot, sensor = setup_environment(target_indicator)
    pb, robot, sensor = setup_environment(target_indicator, sensor_type=args.sensor)
    embodiment = sensor.embodiment

    robot.move_linear(np.array([0, 0, 0, 0, 0, 0]))
    disable_gripper_self_collision(pb, embodiment)
    # set_initial_gripper_opening(pb, embodiment, opening_deg=40)  # 40°左右一般能放下 scale=1 的 cube
    gripper_open(pb, embodiment, open_w_mm=110)
    debug_gripper_limits_and_opening(pb, embodiment)
    robot.move_linear(np.array([0, -60, 130, 0, 0, 0]))

    stim_id, _ = spawn_stim_between_fingers(pb, embodiment, stim_name="long_edge_flat", z_offset=-0.01, use_tactip_tip=True)


    while True:

        # stiffness2cube(stim_id)
        tactile_image_l, tactile_image_r = sensor.process()
        left_pre = pose_model.predict(tactile_image_l)
        right_pre = pose_model.predict(tactile_image_r)
        ob_pose = get_ob_pose(left_pre, right_pre)
        print(ob_pose)
        current_pose = robot.pose

        if current_pose[2] < 160:
            step = np.array([0, 0, 1, 0, 0, 0])
        else:
            align_ctrl, point_ctrl = create_controllers()
            align = align_ctrl.update(np.array([0, ob_pose[3], 0, 0, 0, 0]), np.zeros(6))
            print(align)
            # for i in range(15):
            #     current_pose = robot.pose
            #     step = np.array([0, 0, -1, 0, 0, 0])
            #     robot.move_linear(current_pose + step)
            step = np.array([0, align[1], -2, 0, 0, 0])
        next_pose = current_pose + step
        robot.move_linear(next_pose)




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

    stim_urdf = add_assets_path(r"E:\ai\work\supwork\tactile-gym-3-keju\tg3\simulator\stimuli\cube\cube.urdf")


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
    args.model_dir = r"E:\ai\work\supwork\tactile-gym-3-keju\tactile_data\sim_tactip\surface_zRxy\regress_pose_zRxy\simple_cnn"
    tactile_pushing(args)
    # simple_animation_demo(args)
