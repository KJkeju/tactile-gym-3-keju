import os
import shutil
import numpy as np
import argparse

from tg3.data.setup_collect import \
    POSE_LABEL_NAMES, SHEAR_LABEL_NAMES, BRAILLE_LABEL_NAMES, MIXED_LABEL_NAMES
from tg3.utils.utils import load_json_obj, save_json_obj


def csv_row_to_label_regress_pose(row):
    return {label: np.array(row[label]) for label in POSE_LABEL_NAMES}

def csv_row_to_label_regress_shear(row):
    return {label: np.array(row[label]) for label in SHEAR_LABEL_NAMES}

def csv_row_to_label_arrows(row):
    return {'id': BRAILLE_LABEL_NAMES[26:30].index(row['object_label']), 'label': row['object_label']}

def csv_row_to_label_braille(row):
    return {'id': BRAILLE_LABEL_NAMES.index(row['object_label']), 'label': row['object_label']}

def csv_row_to_label_mixed(row):
    return {'id': MIXED_LABEL_NAMES.index(row['object_label']), 'label': row['object_label']}

csv_row_to_label = {
    'regress_pose': csv_row_to_label_regress_pose,
    'regress_shear': csv_row_to_label_regress_shear,
    'classify_arrows': csv_row_to_label_arrows,
    'classify_braille': csv_row_to_label_braille,
    'classify_mixed': csv_row_to_label_mixed,
}


def setup_parse(
    path='./tactile_data',
    robot='sim',
    sensor='tactip',
    experiments=['edge_yRz'],
    predicts=['regress_pose_yRz'],
    models=['simple_cnn'],
    train_dirs=['data_train'],
    val_dirs=['data_val'],
    device='cuda'
):
    parser = argparse.ArgumentParser(description="Setup parameters for tactile data collection.")
    parser.add_argument('-p', '--path', type=str, default=path, help='Root path to tactile data.')
    parser.add_argument('-r', '--robot', type=str, default=robot, help='Robot type (e.g., sim, real).')
    parser.add_argument('-s', '--sensor', type=str, default=sensor, help='Sensor type (e.g., tactip).')
    parser.add_argument('-e', '--experiments', nargs='+', default=experiments, help='List of experiment names.')
    parser.add_argument('-t', '--predicts', nargs='+', default=predicts, help='List of prediction targets.')
    parser.add_argument('-dt', '--train_dirs', nargs='+', default=train_dirs, help='List of training data directories.')
    parser.add_argument('-dv', '--val_dirs', nargs='+', default=val_dirs, help='List of validation data directories.')
    parser.add_argument('-m', '--models', nargs='+', default=models, help='List of model types.')
    parser.add_argument('-d', '--device', type=str, default=device, help='Device to use (e.g., cuda, cpu).')
    return parser.parse_args()


def setup_learning(model_type, save_dir=None):
    """
    Set up learning (training) parameters based on the model type.

    Args:
        model_type (str): The type of model to configure (e.g., 'simple_cnn', 'resnet', etc.).
        save_dir (str, optional): Directory to save the learning parameters as a JSON file.

    Returns:
        dict: Dictionary containing learning parameters.
    """
    learning_params = {
        'seed': 42,
        'batch_size': 16,
        'epochs': 10,
        'shuffle': True,
        'n_cpu': 1,
        'n_train_batches_per_epoch': None,
        'n_val_batches_per_epoch': None,
    }

    # Use different optimizer settings for MDN models
    if '_mdn' in model_type:
        learning_params.update({
            'cyclic_base_lr': 1e-7,
            'cyclic_max_lr': 1e-3,
            'cyclic_half_period': 5,
            'cyclic_mode': 'triangular',
        })
    else:
        learning_params.update({
            'lr': 1e-4,
            'lr_factor': 0.5,
            'lr_patience': 10,
            'adam_decay': 1e-6,
            'adam_b1': 0.9,
            'adam_b2': 0.999,
        })

    # Optionally save learning parameters to a file
    if save_dir:
        save_json_obj(learning_params, os.path.join(save_dir, 'learning_params'))

    return learning_params


def setup_model_image(save_dir=None):
    """
    Set up image processing and augmentation parameters for the model.

    Args:
        save_dir (str, optional): Directory to save the image parameters as a JSON file.

    Returns:
        dict: Dictionary containing image processing and augmentation parameters.
    """
    model_image_params = {
        'image_processing': {
            'dims': (128, 128),
            'bbox': None,
            'thresh': None,
            'stdiz': False,
            'normlz': True,
        },
        'augmentation': {
            'rshift': (0.025, 0.025),
            'rzoom': None,
            'brightlims': None,
            'noise_var': None,
        }
    }

    # Optionally save image parameters to a file
    if save_dir:
        save_json_obj(model_image_params, os.path.join(save_dir, 'model_image_params'))

    return model_image_params


def setup_model_params(model_type, save_dir=None):
    """
    Set up model parameters based on the specified model type.

    Args:
        model_type (str): The type of model to configure (e.g., 'simple_cnn', 'resnet', etc.).
        save_dir (str, optional): Directory to save the model parameters as a JSON file.

    Returns:
        dict: Dictionary containing model parameters and keyword arguments.
    """

    # Initialize model_params with the model type
    model_params = {'model_type': model_type}

    # Dictionary containing model-specific keyword arguments
    model_kwargs_dict = {
        'nature_cnn': {
            'fc_layers': [512, 512],
            'dropout': 0.0,
        },
        'posenet': { 
            'conv_layers': [256]*5,
            'conv_kernel_sizes': [3]*5,
            'fc_layers': [64],
            'activation': 'elu',
            'dropout': 0.0,
            'apply_batchnorm': True,
        },
        'resnet': {
            'layers': [2, 2, 2, 2]
        },
        'simple_cnn': {
            'conv_layers': [32]*4,
            'conv_kernel_sizes': [11, 9, 7, 5],
            'fc_layers': [512, 512],
            'activation': 'relu',
            'dropout': 0.0,
            'apply_batchnorm': True,
        },
        'vit': {
            'patch_size': 32,
            'dim': 128,
            'depth': 6,
            'heads': 8,
            'mlp_dim': 512,
            'pool': 'mean',
        }
    }
    
    # Select the appropriate model_kwargs based on the model_type prefix
    for key in model_kwargs_dict:
        if model_type.startswith(key):
            model_params['model_kwargs'] = model_kwargs_dict[key]
            break
    else:
        raise ValueError(f'Incorrect model_type specified: {model_type}')

    # Add MDN-specific keyword arguments if the model_type includes '_mdn'
    if '_mdn' in model_type:
        model_params['mdn_kwargs'] = {
            'model_out_dim': 128,
            'n_mdn_components': 1,
            'pi_dropout': 0.1,
            'mu_dropout': [0.1]*6,
            'sigma_inv_dropout': [0.1]*6,
            'mu_min': [-np.inf]*6,
            'mu_max': [np.inf]*6,
            'sigma_inv_min': [1e-6]*6,
            'sigma_inv_max': [1e6]*6,
        }
        
    # Optionally save model parameters to a file
    if save_dir:
        save_json_obj(model_params, os.path.join(save_dir, 'model_params'))

    return model_params


def setup_regress_model_labels(predict_name, data_dirs, save_dir=None):
    """
    Returns settings for regression model labelling of outputs.

    Args:
        predict_name (str): The name of the prediction task (e.g., 'regress_pose_yRz').
        data_dirs (list of str): List of directories containing data and parameter files.
        save_dir (str, optional): Directory to save the model label parameters as a JSON file.

    Returns:
        dict: Dictionary containing model label parameters, including target label names,
              label weights, label limits, and periodic label names.
    """
    # Mapping from task names to the corresponding target label names
    target_label_names_dict = {
        'pose_xy':     ['pose_x', 'pose_y'],
        'pose_xRz':    ['pose_x', 'pose_Rz'],
        'pose_yRz':    ['pose_y', 'pose_Rz'],
        'pose_xyRz':   ['pose_x', 'pose_y', 'pose_Rz'],
        'pose_zRxy':   ['pose_z', 'pose_Rx', 'pose_Ry'],
        'pose_xzRxyz': ['pose_x', 'pose_z', 'pose_Rx', 'pose_Ry', 'pose_Rz'],
        'pose_yzRxyz': ['pose_y', 'pose_z', 'pose_Rx', 'pose_Ry', 'pose_Rz'],
        'shear_xy':    ['shear_x', 'shear_y'],
        'pose_z_shear_xy': ['pose_z', 'shear_x', 'shear_y'],
        'shear_xyRz':  ['shear_x', 'shear_y', 'shear_Rz'],
    }

    # Aggregate lower and upper limits from all data directories
    llims, ulims = [], []
    for data_dir in data_dirs:
        params = load_json_obj(os.path.join(data_dir, 'collect_params'))
        llims.append(params['pose_llims'] + params['shear_llims'])
        ulims.append(params['pose_ulims'] + params['shear_ulims'])

    # Get the target label names for the specified task
    names = target_label_names_dict[predict_name.split('_', 1)[1]]
    model_label_params = {
        'llims': tuple(np.min(llims, axis=0).astype(float)),
        'ulims': tuple(np.max(ulims, axis=0).astype(float)),
        'target_label_names': names,
        'target_weights': [1.0] * len(names),
        'label_names': POSE_LABEL_NAMES + SHEAR_LABEL_NAMES,
        'periodic_label_names': ['pose_Rz']
    }

    # Optionally save parameters to a file
    if save_dir:
        save_json_obj(model_label_params, os.path.join(save_dir, 'model_label_params'))

    return model_label_params


def setup_classify_model_labels(predict_name, data_dirs, save_dir=None):

    label_names_dict = {
        'classify_arrows': BRAILLE_LABEL_NAMES[26:30],
        'classify_braille': BRAILLE_LABEL_NAMES,
        'classify_mixed': MIXED_LABEL_NAMES,
    }

    model_label_params = {
        'label_names': label_names_dict[predict_name],
        'out_dim': len(label_names_dict[predict_name]),
    }

    # save parameters
    if save_dir:
        save_json_obj(model_label_params, os.path.join(save_dir, 'model_label_params'))

    return model_label_params


def setup_training(model_type, predict, data_dirs, save_dir=None):
    """
    Set up and aggregate all training parameters for a supervised learning experiment.
    """
    learning_params = setup_learning(model_type, save_dir)
    model_params = setup_model_params(model_type, save_dir)
    model_image_params = setup_model_image(save_dir)
    if 'regress' in predict:
        model_label_params = setup_regress_model_labels(predict, data_dirs, save_dir)
    elif 'classify' in predict:
        model_label_params = setup_classify_model_labels(predict, data_dirs, save_dir)
    else:
        raise ValueError(f'Prediction must be "regress_" or "classify_": {predict}')

    # Retain data parameters for reproducibility
    is_processed = os.path.isdir(os.path.join(data_dirs[0], 'processed_images'))
    if save_dir:
        shutil.copy(os.path.join(data_dirs[0], 'env_params.json'), save_dir)
        params_file = 'processed_image_params.json' if is_processed else 'sensor_image_params.json'
        shutil.copy(os.path.join(data_dirs[0], params_file), save_dir)

    return learning_params, model_params, model_label_params, model_image_params
