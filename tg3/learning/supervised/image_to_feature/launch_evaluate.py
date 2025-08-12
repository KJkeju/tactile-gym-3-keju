"""
python evaluate_model.py -r abb -m simple_cnn -t edge_2d
"""
import os
import pandas as pd
import torch

from tg3.utils.utils import load_json_obj
from tg3.learning.supervised.image_to_feature.cnn.setup_model import setup_model as setup_cnn_model
from tg3.learning.supervised.image_to_feature.mdn.setup_model import setup_model as setup_mdn_model
from tg3.learning.supervised.image_to_feature.cnn.image_generator import ImageGenerator
from tg3.learning.supervised.image_to_feature.cnn.label_encoder_regress import LabelEncoder as LabelEncoderRegress
from tg3.learning.supervised.image_to_feature.cnn.label_encoder_classify import LabelEncoder as LabelEncoderClassify
from tg3.learning.supervised.image_to_feature.cnn.utils_plots import RegressionPlotter, ClassificationPlotter
from tg3.learning.supervised.image_to_feature.setup_train import csv_row_to_label, setup_parse


def evaluate_model(
    predict_mode,
    model,
    label_encoder,
    generator,
    learning_params,
    error_plotters,
    device='cpu'
):
    """
    Evaluates the given model on data provided by the generator, decodes predictions,
    collects results, and generates error plots.
    """
    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    # Initialize empty DataFrames to store predictions and targets
    pred_df, targ_df = pd.DataFrame(), pd.DataFrame()

    # Iterate over batches from the data loader
    for batch in loader:
        inputs = batch['inputs'].float().to(device) # Move inputs to the specified device and ensure correct type
        outputs = model(inputs) # Get model outputs        
        pred_dict = label_encoder.decode_label(outputs) # Decode predictions using the label encoder
        # Convert predictions and targets to DataFrames
        batch_pred_df = pd.DataFrame.from_dict(pred_dict)
        batch_targ_df = pd.DataFrame.from_dict(batch['labels'])
        # Concatenate batch results to the main DataFrames
        pred_df = pd.concat([pred_df, batch_pred_df])
        targ_df = pd.concat([targ_df, batch_targ_df])

    # Prepare results for plotting
    pred_df = pred_df.reset_index(drop=True).fillna(0.0)
    targ_df = targ_df.reset_index(drop=True).fillna(0.0)
    metrics = label_encoder.calc_metrics(pred_df, targ_df)
        
    # Generate final plots using the provided plotters
    for plotter in error_plotters:
        if plotter and 'max_epochs' not in plotter.__dict__:

            plotter.final_plot(pred_df, targ_df, metrics)


def launch(args):

    # Set output directory based on robot and sensor
    output_dir = f"{args.robot}_{args.sensor}"

    # Loop through each experiment and prediction type
    for experiment, predict in zip(args.experiments, args.predicts):
        for args.model in args.models:
            # Set validation directories and model directory
            val_dirs = [os.path.join(args.path, output_dir, experiment, d) for d in args.val_dirs]
            model_dir = os.path.join(args.path, output_dir, experiment, predict, args.model)

            # Load parameters for learning, model, labels, and images
            learning_params = load_json_obj(os.path.join(model_dir, 'learning_params'))
            model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
            label_params = load_json_obj(os.path.join(model_dir, 'model_label_params'))
            image_params = load_json_obj(os.path.join(model_dir, 'model_image_params'))

            # Choose error plotter and label encoder based on prediction type
            if 'classify' in predict:
                error_plotters = [ClassificationPlotter(model_dir, label_params['label_names'])]
                label_encoder = LabelEncoderClassify(label_params['label_names'], args.device)
            else:
                error_plotters = [RegressionPlotter(model_dir, label_params)]
                label_encoder = LabelEncoderRegress(label_params, args.device)

            # Set prediction mode and validation data generator
            predict_mode = '_'.join(predict.split('_')[:2])
            val_gen = ImageGenerator(val_dirs, predict_mode, **image_params['image_processing'])

            # Setup model for evaluation
            model = setup_model(
                in_dim=image_params['image_processing']['dims'],
                in_channels=1,
                out_dim=label_encoder.out_dim,
                model_params=model_params,
                saved_model_dir=model_dir,
                device=args.device
            )
            model.eval()

            # Evaluate the model and plot errors
            evaluate_model(
                predict_mode, model, label_encoder, val_gen, learning_params, error_plotters, args.device
            )

def setup_model(**kwargs):
    return setup_mdn_model(**kwargs) if '_mdn' in args.model else setup_cnn_model(**kwargs)


if __name__ == "__main__":

    args = setup_parse(
        # path='./../tactile-data',
        robot='sim',
        sensor='tactip', 
        models=['simple_cnn'], # first part model type; optional suffix ('_mdn', '_hyp')
        # experiments=['edge_yRz', 'surface_zRxy', 'edge_yzRxyz'], # order matches predicts
        # predicts=['regress_pose_yRz', 'regress_pose_zRxy', 'regress_pose_yzRxyz'], 
        experiments=['braille_xyRz'],
        predicts=['classify_braille'], # ['regress_pose_xyRz'], 
        # train_dirs=['data_train'],
        # val_dirs=['data_val'],
        # device='cuda'
    )

    launch(args)
