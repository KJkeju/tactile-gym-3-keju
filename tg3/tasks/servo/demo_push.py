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

def transfer_pose_sensor2object(left, right):

    object_pose = np.zeros(6)
    object_pose[5] = -(left[4] - right[4])/2
    object_pose[1] = -(left[2] - right[2])
    object_pose[3] = (left[3] - right[3])/2

    pass
def tactile_pushing(args):
    pose_model = load_pose_model(args.model_dir, args.device)


    target_pose, target_indicator = generate_random_target()



    pb, robot, sensor = setup_environment(target_indicator, sensor_type=args.sensor)
    embodiment = sensor.embodiment

    pb.setPhysicsEngineParameter(
        numSolverIterations=60,
        numSubSteps=4,
        fixedTimeStep=1.0 / 240.0,
        contactERP=0.2,
        erp=0.2
    )
    robot.move_linear(np.array([0, 0, 0, 0, 0, 0]))


    robot.move_linear(np.array([0, 0, 150, 0, 0, 0]))
    spawn_stim_between_fingers(pb, embodiment, stim_name="long_edge_flat", z_offset=0.000, use_tactip_tip=True)

    # robot.move_linear(np.array([0, -100, 0, 0, 0, 0]))

    # robot.move_linear(np.array([0, 0, 150, 15, 0, 0]))


    while True:

        tactile_image_l, tactile_image_r = sensor.process()

        pred_pose_l = pose_model.predict(tactile_image_l)[:6]
        pred_pose_r = pose_model.predict(tactile_image_r)[:6]
        print(f'left pred {pred_pose_l}, \n right pred {pred_pose_r}')

        print('\n contacts')
        # print_self_contacts(pb, embodiment.arm.embodiment_id, embodiment.arm.link_name_to_index)
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
        step = np.array([0, 0, 0, 0, 0, 0])
        target_pose += step
        robot.move_linear(target_pose),



    # align_ctrl, point_ctrl = create_controllers()
    #
    # run_control_loop(robot, sensor, pose_model, target_pose, align_ctrl, point_ctrl)


def spawn_stim_between_fingers(pb, embodiment, stim_name="cube", z_offset=0.0, use_tactip_tip=True):

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
    mid[2] += z_offset  #
    mid[0] -= 0.012
    mid[1] -= 0.015

    stim_urdf = add_assets_path(r"E:\ai\work\supwork\tactile-gym-3-keju\tg3\simulator\stimuli\long_edge_flat\long_edge.urdf")


    mid = np.asarray(mid, dtype=float)
    stim_rpy = np.array([0.0, 1.57, 0.0], dtype=float)
    stim_pose6 = np.concatenate([mid, stim_rpy])


    stim_pose = np.array(stim_pose6)
    stim_pos, stim_rpy = stim_pose[:3], stim_pose[3:]
    stim_id = p.loadURDF(
        stim_urdf,
        stim_pos,
        pb.getQuaternionFromEuler(stim_rpy),
        useFixedBase=True,
        # globalScaling=0.747
        globalScaling=0.947
    )
    p.changeDynamics(stim_id, -1, mass=0.005,
                     lateralFriction=0.1,
                     rollingFriction=0.1,
                     spinningFriction=0.1
                     )
    # p.setCollisionFilterGroupMask(stim_id, -1, 0, 0)
    # load_stim(pb, stim_urdf, stim_pose6, fixed_base=False, enable_collision=True, scale=0.747)
    # print(f"[INFO] Spawned {stim_name} at", mid.tolist())



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
    cps = pb.getContactPoints(bodyA=robot_id)
    rev = {v:k for k,v in link_name_to_index.items()}


    if not cps:
        # print("[contacts] none")
        return


    rows = []
    for c in cps:
        linkA = rev.get(c[3], c[3])
        bodyB = c[1] if c[1] != robot_id else c[2]
        linkB = c[4]
        Fn    = float(c[9])
        dist  = float(c[8])
        if Fn < min_normF:
            continue
        rows.append((linkA, bodyB, linkB, Fn, dist, c[5], c[6]))

    if not rows:
        # print("[contacts] only tiny forces (< min_normF)")
        return


    rows.sort(key=lambda r: r[3], reverse=True)
    # print("[contacts] top hits (linkA, bodyB, linkB, Fn, dist):")
    for r in rows[:10]:
        # print(f"  {r[0]:>24s}  vs  body{r[1]}:link{r[2]:<2d}   Fn={r[3]:.4f}  dist={r[4]:.5f}")
        pass
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
