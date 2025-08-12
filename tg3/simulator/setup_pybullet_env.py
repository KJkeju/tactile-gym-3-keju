import os

import ipdb
import numpy as np

from tg3.simulator.assets.default_rest_poses import rest_poses_dict
from tg3.simulator.embodiments.create_embodiment import create_embodiment
from tg3.simulator.utils.setup_pb_utils import connect_pybullet, load_standard_environment
from tg3.simulator.utils.setup_pb_utils import load_stim, set_debug_camera, simple_pb_loop, load_target_indicator


# def setup_pybullet_env(
#     embodiment_type='tactile_arm',
#     arm_type='ur5',
#     sensor_type='standard_tactip',
#     image_size=(128, 128),
#     show_tactile=False,
#     stim_name='circle',
#     stim_path=os.path.dirname(__file__)+'/stimuli',
#     stim_pose=(600, 0, 12.5, 0, 0, 0),
#     show_gui=True,
#     load_target=None,
#     **kwargs
# ):
#     timestep = 1/240.0
#     # define sensor parameters
#     robot_arm_params = {
#         "type": arm_type,
#         "rest_poses": rest_poses_dict[arm_type],
#         "tcp_lims": np.column_stack([-np.inf*np.ones(6), np.inf*np.ones(6)]),
#     }
#
#     tactile_sensor_params = {
#         "type": sensor_type,
#         # "core": "no_core",
#         "core": "fixed",
#         # "dynamics": {},  # {'stiffness': 50, 'damping': 100, 'friction': 10.0},
#         "dynamics": {'stiffness': 100, 'damping': 0, 'friction': 0},
#         "image_size": image_size,
#         "turn_off_border": False,
#         "show_tactile": show_tactile,
#     }
#
#     # set debug camera position
#     visual_sensor_params = {
#         'image_size': [128, 128],
#         'dist': 0.25,
#         'yaw': 90.0,
#         'pitch': -25.0,
#         'pos': [0.6, 0.0, 0.0525],
#         'fov': 75.0,
#         'near_val': 0.1,
#         'far_val': 100.0,
#         'show_vision': False
#     }
#
#     pb = connect_pybullet(timestep, show_gui)
#     load_standard_environment(pb)
#     stim_name = os.path.join(stim_path, stim_name, stim_name+'.urdf')
#     load_stim(pb, stim_name, np.array(stim_pose)/1e3, fixed_base=False, enable_collision=True)
#     if load_target is not None:
#         load_target_indicator(pb, load_target)
#     embodiment = create_embodiment(
#         pb,
#         embodiment_type,
#         robot_arm_params,
#         tactile_sensor_params,
#         visual_sensor_params
#     )
#     set_debug_camera(pb, visual_sensor_params)
#     return embodiment

def setup_pybullet_env(
    # pb,
    embodiment_type='tactile_arm',
    robot_arm_params=None,
    tactile_sensor_params=None,
    visual_sensor_params=None,
    stim_name='circle',
    stim_path=None,
    stim_pose=(600, 0, 12.5, 0, 0, 0),
    show_gui=True,
    load_target=None
):
    """
    pb: 已连接的pybullet client
    robot_arm_params: dict
    tactile_sensor_params: dict（如 {'left': {...}, 'right': {...}} ）
    visual_sensor_params: dict
    """
    import os
    import numpy as np
    from tg3.simulator.assets.default_rest_poses import rest_poses_dict
    from tg3.simulator.embodiments.create_embodiment import create_embodiment
    from tg3.simulator.utils.setup_pb_utils import load_standard_environment, load_stim, load_target_indicator, set_debug_camera

    timestep = 1/240.0

    # 缺省参数兜底
    if robot_arm_params is None:
        robot_arm_params = {
            "type": 'ur5',
            "rest_poses": rest_poses_dict['ur5'],
            "tcp_lims": np.column_stack([-np.inf*np.ones(6), np.inf*np.ones(6)]),
        }
    if tactile_sensor_params is None:
        tactile_sensor_params = {
            'left': {
                "type": 'standard_tactip',
                "core": "fixed",
                "dynamics": {'stiffness': 100, 'damping': 0, 'friction': 0},
                "image_size": (128, 128),
                "turn_off_border": False,
                "show_tactile": True,
            },
            'right': {
                "type": 'standard_tactip',
                "core": "fixed",
                "dynamics": {'stiffness': 100, 'damping': 0, 'friction': 0},
                "image_size": (128, 128),
                "turn_off_border": False,
                "show_tactile": True,
            }
        }
    if visual_sensor_params is None:
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

    # 加载环境
    pb = connect_pybullet(timestep, show_gui)
    load_standard_environment(pb)

    # 加载stimulus
    if stim_path is None:
        stim_path = os.path.dirname(__file__)+'/stimuli'
    stim_urdf = os.path.join(stim_path, stim_name, stim_name+'.urdf')
    load_stim(pb, stim_urdf, np.array(stim_pose)/1e3, fixed_base=False, enable_collision=True)

    # 加载target（可选）
    if load_target is not None:
        load_target_indicator(pb, load_target)

    # **只传dict，不解包**
    embodiment = create_embodiment(
        pb,
        embodiment_type,
        robot_arm_params,
        tactile_sensor_params,
        visual_sensor_params
    )
    set_debug_camera(pb, visual_sensor_params)
    return pb, embodiment

if __name__ == '__main__':
    pass
