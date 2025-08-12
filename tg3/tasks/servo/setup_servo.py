import os
import numpy as np
import argparse

from tg3.utils.utils import save_json_obj


def setup_parse(
    path='./tactile_data',
    robot='sim',
    sensor='tactip',
    experiments=['edge_yRz'],
    predicts=['pose_yRz'],
    models=['simple_cnn'],
    objects=['circle'],
    sample_nums=[100],
    run_version=[''],
    device='cpu'
):
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', type=str, help="Default: './tactile_data'", default=path)
    parser.add_argument('-r', '--robot', type=str, help="Default: 'sim'", default=robot)
    parser.add_argument('-s', '--sensor', type=str, help="Default: 'tactip'", default=sensor)
    parser.add_argument('-e', '--experiments', nargs='+', help="Default: 'edge_yRz'. Options: ['edge_yRz', 'edge_yRz_shear']", default=experiments)
    parser.add_argument('-t', '--predicts', nargs='+', help="Default: ['pose_yRz']", default=predicts)
    parser.add_argument('-m', '--models', nargs='+', help="Default: ['simple_cnn']. Options: ['simple_cnn', 'nature_cnn', 'posenet', 'resnet', 'vit']", default=models)
    parser.add_argument('-o', '--objects', nargs='+', help="Default: ['circle']. Options: ['circle', 'square']", default=objects)
    parser.add_argument('-n', '--sample_nums', type=int, nargs='+', help="Default [100]", default=sample_nums)
    parser.add_argument('-v', '--run_version', nargs='+', help="Default: ['']", default=run_version)
    parser.add_argument('-d', '--device', type=str, help="Default: 'cuda'. Options: ['cpu', 'cuda']", default=device)
    
    return parser.parse_args()


def setup_control_params(task, save_dir=None):

    if task[:8] == 'pose_xRz':
        control_params = {
            'kp': [0.5, 1, 0, 0, 0, 0.5],
            'ki': [0.3, 0, 0, 0, 0, 0.1],
            'ei_clip': [[-5, 0, 0, 0, 0, -45], [5, 0, 0, 0, 0,  45]],
            'error': 'lambda y, r: transform_euler(r, y)',  # SE(3) error
            'ref': [0, 2, 0, 0, 0, 0]
        }

    if task[:8] == 'pose_yRz':
        control_params = {
            'kp': [1, 0.5, 0, 0, 0, 0.5],
            'ki': [0, 0.3, 0, 0, 0, 0.1],
            'ei_clip': [[0, -5, 0, 0, 0, -45], [0, 5, 0, 0, 0,  45]],
            'error': 'lambda y, r: transform_euler(r, y)',  # SE(3) error
            'ref': [2, 0, 0, 0, 0, 0]
        }

    elif task[:9] == 'pose_xzRz':
        control_params = {
            'kp': [0.5, 1, 0.5, 0, 0, 0.5],
            'ki': [0.3, 0, 0.3, 0, 0, 0.1],
            'ei_clip': [[-5, 0, -2.5, 0, 0, -45], [5, 0, 2.5, 0, 0, 45]],
            'error': 'lambda y, r: transform_euler(r, y)',  # SE(3) error
            'ref': [0, 2, 3.5, 0, 0, 0]
        }

    elif task[:13] == 'pose_xzRxRyRz':
        control_params = {
            'kp': [0.5, 1, 0.5, 0.5, 0.5, 0.5],
            'ki': [0.3, 0, 0.3, 0.1, 0.1, 0.1],
            'ei_clip': [[-5, 0, -2.5, -15, -15, -45], [5, 0, 2.5, 15, 15, 45]],
            'error': 'lambda y, r: transform_euler(r, y)',  # SE(3) error
            'ref': [0, 2, 3.5, 0, 0, 0]
        }
    
    elif task[:13] == 'pose_yzRxRyRz':
        control_params = {
            'kp': [1, 0.5, 0.5, 0.5, 0.5, 0.5],
            'ki': [0, 0.3, 0.3, 0.1, 0.1, 0.1],
            'ei_clip': [[0, -5, -2.5, -15, -15, -45], [0, 5, 2.5, 15, 15, 45]],
            'error': 'lambda y, r: transform_euler(r, y)',  # SE(3) error
            'ref': [2, 0, 3.5, 0, 0, 0]
        }

    elif task[:10] == 'pose_zRxRy':
        control_params = {
            'kp': [1, 1, 0.5, 0.5, 0.5, 1],
            'ki': [0, 0, 0.3, 0.1, 0.1, 0],
            'ei_clip': [[0, 0, 0, -30, -30, 0], [0, 0, 5, 30, 30, 0]],
            'error': 'lambda y, r: transform_euler(r, y)',  # SE(3) error
            'ref': [0, 1, 3, 0, 0, 0]
        }

    else: 
        raise ValueError(f'Incorrect task specified: {task}')

    if save_dir:
        save_json_obj(control_params, os.path.join(save_dir, 'control_params'))

    return control_params


def update_env_params(env_params, object, save_dir=None):

    wf_offset_dict = {
        'ball': (0, 0, 50, 0, 0, 0),
        'saddle':  (-10, 0, 18.5, 0, 0, 0),
        'default': (0, 0, 3.5, 0, 0, 0)
    }
    wf_offset = np.array(wf_offset_dict.get(object, wf_offset_dict['default']))

    env_params.update({
        'stim_name': object,
        'work_frame': tuple(env_params['work_frame'] - wf_offset),
        'speed': 20,
    })

    if save_dir:
        save_json_obj(env_params, os.path.join(save_dir, 'env_params'))

    return env_params


def setup_task_params(sample_num, model_dir, save_dir=None):

    task_params = {
        'num_iterations': sample_num,
        'show_plot': True,
        'show_slider': False,
        'model': model_dir
        # 'servo_delay': 0.0,
    }

    if save_dir:
        save_json_obj(task_params, os.path.join(save_dir, 'task_params'))

    return task_params


def setup_servo(sample_num, task, object, model_dir, env_params, save_dir=None):
    env_params = update_env_params(env_params, object, save_dir)
    control_params = setup_control_params(task, save_dir)
    task_params = setup_task_params(sample_num, model_dir, save_dir)

    return control_params, env_params, task_params


if __name__ == '__main__':
    pass
