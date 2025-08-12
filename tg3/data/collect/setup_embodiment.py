import ipdb
from cri.robot import SyncRobot
from cri.controller import Controller, DummyController, SimController

from tg3.data.collect.simple_sensors import RealSensor, ReplaySensor, SimSensor, DummySensor
from tg3.simulator.setup_pybullet_env import setup_pybullet_env
from tg3.simulator.utils.setup_pb_utils import simple_pb_loop


def setup_real_embodiment(
    env_params,
    sensor_params,
):
    # setup real robot
    robot = SyncRobot(Controller[env_params['robot']]())
    sensor = RealSensor(sensor_params)

    robot.controller.servo_delay = env_params.get('servo_delay', 0.0)
    robot.coord_frame = env_params['work_frame']
    robot.tcp = env_params['tcp_pose']
    robot.speed = env_params.get('speed', 10)

    return robot, sensor


# def setup_sim_embodiment(
#     env_params,
#     sensor_params,
# ):
#     # setup simulated robot
#     embodiment = setup_pybullet_env(**env_params, **sensor_params)

def setup_sim_embodiment(
            env_params,
            sensor_params,
    ):
        # 组织robot_arm_params, tactile_sensor_params, visual_sensor_params
    from tg3.simulator.assets.default_rest_poses import rest_poses_dict
    import numpy as np

    robot_arm_params = {
            "type": env_params.get("arm_type", "ur5"),
            "rest_poses": rest_poses_dict.get(env_params.get("arm_type", "ur5")),
            "tcp_lims": np.column_stack([-np.inf * np.ones(6), np.inf * np.ones(6)]),
        }
        # 你根据需要自定义tactile_sensor_params, visual_sensor_params
    # tactile_sensor_params = {
    #         'left': {
    #             "type": sensor_params.get('type', 'standard_tactip'),
    #             "core": "fixed",
    #             "dynamics": {'stiffness': 100, 'damping': 0, 'friction': 0},
    #             "image_size": sensor_params.get('image_size', (128, 128)),
    #             "turn_off_border": False,
    #             "show_tactile": sensor_params.get('show_tactile', True),
    #         },
    #         'right': {
    #             "type": sensor_params.get('type', 'standard_tactip'),
    #             "core": "fixed",
    #             "dynamics": {'stiffness': 100, 'damping': 0, 'friction': 0},
    #             "image_size": sensor_params.get('image_size', (128, 128)),
    #             "turn_off_border": False,
    #             "show_tactile": sensor_params.get('show_tactile', True),
    #         }
    #     }

    tactile_sensor_params = {
            'left': {
                "type": sensor_params.get('left', {}).get('type', 'standard_tactip'),
                "core": sensor_params.get('left', {}).get('core', 'fixed'),
                "dynamics": sensor_params.get('left', {}).get('dynamics',
                                                              {'stiffness': 100, 'damping': 0, 'friction': 0}),
                "image_size": sensor_params.get('left', {}).get('image_size', (128, 128)),
                "turn_off_border": sensor_params.get('left', {}).get('turn_off_border', False),
                "show_tactile": sensor_params.get('left', {}).get('show_tactile', True),
                "body_link": sensor_params.get('left', {}).get('body_link', 'left_tactip_body_link'),
                "tip_link": sensor_params.get('left', {}).get('tip_link', 'left_tactip_tip_link'),
            },
            'right': {
                "type": sensor_params.get('right', {}).get('type', 'standard_tactip'),
                "core": sensor_params.get('right', {}).get('core', 'fixed'),
                "dynamics": sensor_params.get('right', {}).get('dynamics',
                                                               {'stiffness': 100, 'damping': 0, 'friction': 0}),
                "image_size": sensor_params.get('right', {}).get('image_size', (128, 128)),
                "turn_off_border": sensor_params.get('right', {}).get('turn_off_border', False),
                "show_tactile": sensor_params.get('right', {}).get('show_tactile', True),
                "body_link": sensor_params.get('right', {}).get('body_link', 'right_tactip_body_link'),
                "tip_link": sensor_params.get('right', {}).get('tip_link', 'right_tactip_tip_link'),
            }
        }

    print('[DEBUG] tactile_sensor_params:', tactile_sensor_params)
    visual_sensor_params = {
            'image_size': [128, 128],
            'dist': 0.25,
            'yaw': 90.0,
            'pitch': -25.0,
            'pos': [0.6, 0.0, 0.0525],
            'fov': 75.0,
            'near_val': 0.1,
            'far_val': 100.0,
            'show_vision': False
        }

        # 注意只传必须的参数
    # import pybullet as p
    # pb = p.connect(p.GUI)  # 或你的 pybullet 客户端获取方法

    pb, embodiment = setup_pybullet_env(
            # pb=pb,
            embodiment_type='tactile_arm',
            robot_arm_params=robot_arm_params,
            tactile_sensor_params=tactile_sensor_params,
            visual_sensor_params=visual_sensor_params,
            stim_name=env_params.get('stim_name', 'circle'),
            stim_path=env_params.get('stim_path', None),
            stim_pose=env_params.get('stim_pose', (600, 0, 12.5, 0, 0, 0)),
            show_gui=env_params.get('show_gui', True),
            load_target=env_params.get('load_target', None),
        )

    robot = SyncRobot(SimController(embodiment.arm))
    sensor = SimSensor(sensor_params, embodiment)

    robot.speed = env_params.get('speed', float('inf'))
    robot.controller.servo_delay = env_params.get('servo_delay', 0.0)
    robot.coord_frame = env_params['work_frame']
    robot.tcp = env_params['tcp_pose']

    return pb, robot, sensor


def setup_dummy_embodiment(
    env_params,
    sensor_params,
):
    # setup dummy robot
    robot = SyncRobot(DummyController())
    sensor = DummySensor(sensor_params)    

    return robot, sensor


def setup_embodiment(
    env_params,
    sensor_params,
):
    if 'dummy' in env_params['robot']:
        pb, robot, sensor = setup_dummy_embodiment(env_params, sensor_params)

    elif 'sim' in env_params['robot']:
        pb, robot, sensor = setup_sim_embodiment(env_params, sensor_params)
   
    else: # real robot
        pb, robot, sensor = setup_real_embodiment(env_params, sensor_params)

    # if replay overwrite sensor
    # if sensor_params['type'] == 'replay':
    #     sensor = ReplaySensor(sensor_params)
    # 替换为如下
    if (
            (isinstance(sensor_params, dict) and 'left' in sensor_params and sensor_params['left'].get(
                'type') == 'replay') or
            (isinstance(sensor_params, dict) and 'right' in sensor_params and sensor_params['right'].get(
                'type') == 'replay') or
            (isinstance(sensor_params, dict) and sensor_params.get('type') == 'replay')
    ):
        sensor = ReplaySensor(sensor_params)

    return pb, robot, sensor


if __name__ == '__main__':

    env_params = {
        'robot': 'sim_ur',
        'stim_name': 'circle',
        'speed': 50,
        'work_frame': (600, 0, 200, 0, 0, 0),
        'tcp_pose': (600, 0, 0, 0, 0, 0),
        'stim_pose': (600, 0, 0, 0, 0, 0),
        'show_gui': True
    }

    sensor_params = {
        "type": "standard_tactip",
        "image_size": (256, 256),
        "show_tactile": False
    }

    robot = setup_embodiment(
        env_params,
        sensor_params
    )

    simple_pb_loop()