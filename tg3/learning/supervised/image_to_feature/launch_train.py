"""
python launch_training.py -r sim -s tactip -e edge_yRz_shear -t pose_yRz -m simple_cnn_test
"""
import os

from tg3.learning.supervised.image_to_feature.cnn.image_generator import ImageGenerator
from tg3.learning.supervised.image_to_feature.cnn.label_encoder_regress import LabelEncoder as LabelEncoderRegress
from tg3.learning.supervised.image_to_feature.cnn.label_encoder_classify import LabelEncoder as LabelEncoderClassify
from tg3.learning.supervised.image_to_feature.cnn.setup_model import setup_model as setup_cnn_model
from tg3.learning.supervised.image_to_feature.cnn.train_model import train_model as train_cnn_model
from tg3.learning.supervised.image_to_feature.cnn.utils_plots import ClassificationPlotter, LearningPlotter, RegressionPlotter
from tg3.learning.supervised.image_to_feature.launch_evaluate import evaluate_model 
from tg3.learning.supervised.image_to_feature.mdn.setup_model import setup_model as setup_mdn_model
from tg3.learning.supervised.image_to_feature.mdn.train_model import train_model as train_mdn_model
from tg3.learning.supervised.image_to_feature.setup_train import csv_row_to_label, setup_training, setup_parse
from tg3.utils.utils import make_dir, seed_everything


def launch(args):

    output_dir = '_'.join([args.robot, args.sensor])

    for experiment, predict in zip(args.experiments, args.predicts):
        for args.model in args.models:
            # Setup directories for training, validation, and saving results
            train_dirs = [os.path.join(args.path, output_dir, experiment, d) for d in args.train_dirs]
            val_dirs = [os.path.join(args.path, output_dir, experiment, d) for d in args.val_dirs]
            save_dir = os.path.join(args.path, output_dir, experiment, predict, args.model)
            make_dir(save_dir)

            # Prepare training, model, label, and image parameters
            learning_params, model_params, label_params, image_params = setup_training(
                args.model, predict, train_dirs, save_dir
            )

            # Choose label encoder and error plotters based on prediction type
            if 'classify' in predict:
                error_plotters = [LearningPlotter(save_dir), ClassificationPlotter(save_dir, label_params['label_names'])]
                label_encoder = LabelEncoderClassify(label_params['label_names'], args.device)
            else:
                error_plotters = [LearningPlotter(save_dir), RegressionPlotter(save_dir, label_params['target_label_names'])]
                label_encoder = LabelEncoderRegress(label_params, args.device)
            predict_mode = '_'.join(predict.split('_')[:2])

            # Create image generators for training and validation
            train_gen = ImageGenerator(train_dirs, predict_mode, **image_params['image_processing'], **image_params['augmentation'])
            val_gen = ImageGenerator(val_dirs, predict_mode, **image_params['image_processing'])

            # Initialize model with given parameters and seed
            seed_everything(learning_params['seed'])
            model = setup_model(
                in_dim=image_params['image_processing']['dims'],
                in_channels=1,
                out_dim=label_encoder.out_dim,
                model_params=model_params,
                device=args.device
            )

            # run the training
            train_model(
                predict_mode=predict_mode,
                model=model,
                label_encoder=label_encoder,
                train_generator=train_gen,
                val_generator=val_gen,
                learning_params=learning_params,
                save_dir=save_dir,
                error_plotters=error_plotters,
                device=args.device
            )
            
            # run an evaluation using the last model
            evaluate_model(
                predict_mode, model, label_encoder, val_gen, learning_params, error_plotters, args.device
            )

def setup_model(**kwargs):
    return setup_mdn_model(**kwargs) if '_mdn' in args.model else setup_cnn_model(**kwargs)

def train_model(**kwargs):
    return train_mdn_model(**kwargs) if '_mdn' in args.model else train_cnn_model(**kwargs) 


if __name__ == "__main__":

    args = setup_parse(
        # path='./../tactile-data',
        robot='sim',
        sensor='tactip', 
        models=['simple_cnn'], # first part model type; optional suffix ('_mdn', '_hyp')
        experiments=['surface_zRxy'],#['edge_yzRxyz'], # order matches predicts
        predicts=['regress_pose_zRxy'],#['regress_pose_yzRxyz'],
        # experiments=['braille_xyRz'],#, 'arrows_xyRz', 'mixed_Pxy'],
        # predicts=['classify_braille'],#, 'classify_arrows', 'classify_mixed'], #['regress_pose_xyRz', 'regress_pose_xy'], 
        # train_dirs=['data_train'],
        # val_dirs=['data_val'],
        device='cpu'
    )

    launch(args)
