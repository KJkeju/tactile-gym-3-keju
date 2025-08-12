import os
import numpy as np
import argparse

from tg3.utils.utils import save_json_obj

POSE_LABEL_NAMES = [
    "pose_x", "pose_y", "pose_z", "pose_Rx", "pose_Ry", "pose_Rz"
]
SHEAR_LABEL_NAMES = [
    "shear_x", "shear_y", "shear_z", "shear_Rx", "shear_Ry", "shear_Rz"
]
OBJECT_POSE_LABEL_NAMES = [
    "object_x", "object_y", "object_z", "object_Rx", "object_Ry", "object_Rz"
]
SPHERE_LABEL_NAMES = [
    '2mm', '3mm', '4mm', '5mm', '6mm', '7mm', '8mm', '9mm', '10mm'
]
MIXED_LABEL_NAMES = [
    'cone', 'cross_lines', 'curved_surface', 'cylinder', 'cylinder_shell', 'cylinder_side', 'dot_in',
    'dots', 'flat_slab', 'hexagon', 'line', 'moon', 'pacman', 'parallel_lines',
    'prism', 'random', 'sphere', 'sphere2', 'torus', 'triangle', 'wave1'
]
BRAILLE_LABEL_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'UP', 'DOWN', 'LEFT', 'RIGHT',
    'NONE', 'SPACE'
]


def setup_parse(
    path='./tactile_data',
    transfer='',
    robot='sim',
    sensor='tactip',
    experiments=['edge_yRz'],
    data_dirs=['data_train', 'data_val'],
    sample_nums=[4000, 1000]
):
    parser = argparse.ArgumentParser(description="Setup parameters for tactile data collection.")
    parser.add_argument('-p', '--path', type=str, default=path, help='Root path for data storage')
    parser.add_argument('-t', '--transfer', type=str, default=transfer, help='Transfer learning setting')
    parser.add_argument('-r', '--robot', type=str, default=robot, help='Robot type (e.g., sim, ur)')
    parser.add_argument('-s', '--sensor', type=str, default=sensor, help='Sensor type (e.g., tactip)')
    parser.add_argument('-e', '--experiments', nargs='+', default=experiments, help='List of experiments')
    parser.add_argument('-dd', '--data_dirs', nargs='+', default=data_dirs, help='Data directories')
    parser.add_argument('-n', '--sample_nums', type=int, nargs='+', default=sample_nums, help='Number of samples per directory')
    return parser.parse_args()


def setup_sensor_image_params(robot, sensor, save_dir=None):
    """
    Set up sensor image parameters based on robot and sensor type.

    Args:
        robot (str): Robot type (e.g., 'sim', 'ur').
        sensor (str): Sensor type (e.g., 'mini', 'midi').
        save_dir (str, optional): Directory to save the parameters as a JSON file.

    Returns:
        dict: Dictionary containing sensor image parameters.
    """
    bbox_dict = {
        'mini': (160, 105, 480, 425),
        'midi': (110, 0, 550, 440)
    }
    sensor_type = 'midi'  # TODO: Fix hardcoded sensor type

    if 'sim' in robot: # Parameters for simulated sensor images
        sensor_image_params = {
            "type": "standard_tactip",
            "image_size": (256, 256),
            "show_tactile": True
        }
    else: # Parameters for real sensor images  
        sensor_image_params = {
            'type': sensor_type,
            'source': 1,
            'exposure': -7,
            'gray': True,
            'bbox': bbox_dict[sensor_type]
        }

    # Optionally save sensor image parameters to file
    if save_dir:
        save_json_obj(sensor_image_params, os.path.join(save_dir, 'sensor_image_params'))

    return sensor_image_params


def setup_collect_params(robot, dataset, save_dir=None):
    """
    Set up collection parameters for tactile data collection based on robot and dataset.

    Args:
        robot (str): Robot type (e.g., 'sim', 'ur').
        dataset (str): Dataset name indicating object and pose configuration.
        save_dir (str, optional): Directory to save the parameters as a JSON file.

    Returns:
        dict: Dictionary containing collection parameters.
    """

    # Normalize robot name if it starts with 'sim_'
    if robot.split('_')[0] == 'sim':
        robot = 'sim'

    # Extract object type and pose configuration from dataset name
    object = dataset.split('_')[0]
    object_poses = '_'.join(dataset.split('_')[:2])
    
    # Define pose limits for different dataset configurations
    pose_lims_dict = {
        'edge_xRz':      [(-6, 0, 3, 0, 0, -180),     (6, 0, 5, 0, 0, 180)],
        'edge_yRz':      [(0, -6, 3, 0, 0, -180),     (0, 6, 5, 0, 0, 180)],
        'edge_xzRxyz':   [(-6, 0, 1, -15, -15, -180), (6, 0, 5, 15, 15, 180)],
        'edge_yzRxyz':   [(0, -6, 1, -15, -15, -180), (0, 6, 5, 15, 15, 180)],
        'surface_zRxy':  [(0, 0, 1, -15, -15, 0),     (0, 0, 5, 15, 15, 0)],
        'arrows_xyRz':   [(-2.5, -2.5, 3, 0, 0, -10), (2.5, 2.5, 5, 0, 0, 10)],
        'braille_xyRz':  [(-2.5, -2.5, 3, 0, 0, -10), (2.5, 2.5, 5, 0, 0, 10)],
        'mixed_xy':      [(-5, -5, 4, 0, 0, 0),       (5, 5, 5, 0, 0, 0)],
        'spheres_xy':    [(-12.5, -12.5, 4, 0, 0, 0), (12.5, 12.5, 5, 0, 0, 0)],
    }

    # Define shear limits for different dataset configurations
    shear_lims_dict = {
        'none':          [(-0, -0, 0, 0, 0, 0),       (0, 0, 0, 0, 0, 0)],
        'edge_xRz':      [(-5, -5, 0, 0, 0, -5),      (5, 5, 0, 0, 0, 5)],
        'edge_yRz':      [(-5, -5, 0, 0, 0, -5),      (5, 5, 0, 0, 0, 5)],
        'edge_xzRxyz':   [(-5, -5, 0, -5, -5, -5),    (5, 5, 0, 5, 5, 5)],
        'edge_yzRxyz':   [(-5, -5, 0, -5, -5, -5),    (5, 5, 0, 5, 5, 5)],
        'surface_zRxy':  [(-5, -5, 0, -5, -5, -5),    (5, 5, 0, 5, 5, 5)],
    }

    # Define object poses for different dataset types
    object_poses_dict = {
        'surface':       {'surface': (0, 0, 0, 0, 0, 0)},
        'edge':          {'edge':    (0, 0, 0, 0, 0, 0)},
        'braille': {      # 3x10 grid for Braille positions
            **{BRAILLE_LABEL_NAMES[10*i+j]: (-17.5*i, 17.5*j, 0, 0, 0, 0) for i, j in np.ndindex(3, 10)},
            BRAILLE_LABEL_NAMES[-2]: (-17.5*3, 17.5*8, -10, 0, 0, 0),  # Special positions for 'LEFT'
            BRAILLE_LABEL_NAMES[-1]: (-17.5*3, 17.5*3, 0, 0, 0, 0)     # Special position for 'SPACE'
        },
        'arrows': {      # just the arrow keys
            **{BRAILLE_LABEL_NAMES[26+i]: (-17.5*2, 17.5*(6+i), 0, 0, 0, 0) for (i,) in np.ndindex(4)},
        },
        'mixed': dict(    # 3x7 grid for mixed objects
            (MIXED_LABEL_NAMES[7*i+j], (25*(i-1), 25*(3-j), 0, 0, 0, 0)) for i, j in np.ndindex(3, 7)
        ),
        'spheres': dict(  # 3x3 grid for spheres
            (SPHERE_LABEL_NAMES[3*i+j], (60*(1-j), 60*(1-i), 0, 0, 0, -48)) for i, j in np.ndindex(3, 3)
        ),
    }
    
    # Build collection parameters dictionary
    collect_params = {
        'object_poses': object_poses_dict[object],
        'pose_llims': pose_lims_dict[object_poses][0],
        'pose_ulims': pose_lims_dict[object_poses][1],
        'shear_llims': shear_lims_dict.get(object_poses, shear_lims_dict['none'])[0],
        'shear_ulims': shear_lims_dict.get(object_poses, shear_lims_dict['none'])[1],
        'sample_disk': False,
        'sort': False,
        'seed': 0,
    }

    # If dataset is not a shear dataset, set shear limits to zero
    if dataset.split('_')[-1] != 'shear':
        collect_params['shear_llims'] = [0, 0, 0, 0, 0, 0]
        collect_params['shear_ulims'] = [0, 0, 0, 0, 0, 0]

    # For simulation, enable sorting to speed data collection
    if robot == 'sim':
        collect_params['sort'] = True

    # Optionally save collection parameters to file
    if save_dir:
        save_json_obj(collect_params, os.path.join(save_dir, 'collect_params'))

    return collect_params


def setup_env_params(robot, dataset, save_dir=None):
    """
    Set up environment parameters for tactile data collection based on robot and dataset.

    Args:
        robot (str): Robot type (e.g., 'sim', 'ur').
        dataset (str): Dataset name indicating object and pose configuration.
        save_dir (str, optional): Directory to save the parameters as a JSON file.

    Returns:
        dict: Dictionary containing environment parameters.
    """

    # Normalize robot name if it starts with 'sim_'
    if robot.split('_')[0] == 'sim':
        robot = 'sim'

    # Define stimuli configurations based on dataset type. Each tuple contains:
    # (condition function, stimulus name, stimulus pose, work frame dict, tcp pose dict)
    stim_configs = [
        # For surface datasets
        (lambda d: 'surface' in d, 'square', (650, 0, 12.5, 0, 0, 0),
         {'sim': (650, 0, 50, -180, 0, 90), 'ur': (0, -500, 54, -180, 0, 0)},
         {'sim': (0, 0, -85, 0, 0, 0), 'ur': (0, 0, 101, 0, 0, 0)}),
        # For edge datasets
        (lambda d: 'edge' in d, 'square', (600, 0, 12.5, 0, 0, 0),
         {'sim': (650, 0, 50, -180, 0, 90), 'ur': (0, -451, 54, -180, 0, 0), 'mg400': (374, 15, -125, 0, 0, 0)},
         {'sim': (0, 0, -85, 0, 0, 0), 'ur': (0, 0, 101, 0, 0, 0), 'mg400': (0, 0, 0, 0, 0, 0)}),
        # For braille or arrow datasets
        (lambda d: 'braille' or 'arrows' in d, 'static_keyboard', (600, 0, 0, 0, 0, 0),
         {'sim': (593, -7, 23, -180, 0, 0), 'ur': (111, -473.5, 28.6, -180, 0, -90)},
         {'sim': (0, 0, -85, 0, 0, 0), 'ur': (0, 0, 125, 0, 0, 0)}),
        # For mixed datasets
        (lambda d: 'mixed' in d, 'mixed_probes', (650, 0, 0, 0, 0, 0),
         {'sim': (650, 0, 20, -180, 0, 0)},
         {'sim': (0, 0, -85, 0, 0, 0)}),
        # For spheres datasets
        (lambda d: 'spheres' in d, 'spherical_probes', (650, 0, 0, 0, 0, 0),
         {'sim': (650, 0, 42.5, -180, 0, 90), 'ur': (-15.75, -462, 47.0, -180, 0, 0)},
         {'sim': (0, 0, -85, 0, 0, 0), 'ur': (0, 0, 88.5, 0, 0, 0)}),
     ]

    # Select the appropriate stimulus configuration for the dataset
    for cond, stim_name, stim_pose, work_frame_dict, tcp_pose_dict in stim_configs:
        if cond(dataset):
            break
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Build environment parameters dictionary
    env_params = {
        'robot': robot,
        'stim_name': stim_name,
        'speed': float('inf') if robot == 'sim' else 50,
        'work_frame': work_frame_dict[robot],
        'tcp_pose': tcp_pose_dict[robot],
        **({'stim_pose': stim_pose} if robot == 'sim' else {}) # Only include stim_pose for simulation
    }

    # Optionally save environment parameters to file
    if save_dir:
        save_json_obj(env_params, os.path.join(save_dir, 'env_params'))

    return env_params


def setup_collect_data(robot, sensor, dataset, save_dir=None):
    """
    Set up and return environment and sensor image parameters for data collection.
    """
    sensor_image_params = setup_sensor_image_params(robot, sensor, save_dir)
    env_params = setup_env_params(robot, dataset, save_dir)
    return env_params, sensor_image_params


if __name__ == '__main__':
    pass
