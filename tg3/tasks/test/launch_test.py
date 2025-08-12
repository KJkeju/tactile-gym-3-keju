"""
python launch_test.py -r sim -s tactip -e edge_yRz_shear -t pose_yRz -m simple_cnn -n 100 -v test
"""
import os
import numpy as np
import pandas as pd

from tg3.data.collect.setup_embodiment import setup_embodiment
from tg3.data.collect.setup_targets import setup_targets, POSE_LABEL_NAMES, SHEAR_LABEL_NAMES
from tg3.data.setup_collect import setup_collect_params
from tg3.tasks.servo.setup_servo import setup_parse
from tg3.tasks.servo.servo_utils.labelled_model import LabelledModel
from tg3.learning.supervised.image_to_feature.cnn.label_encoder_regress import LabelEncoder
from tg3.learning.supervised.image_to_feature.cnn.setup_model import setup_model
from tg3.learning.supervised.image_to_feature.cnn.utils_plots import RegressionPlotter
from tg3.utils.utils import load_json_obj, save_json_obj, make_dir


def test(
    robot,
    sensor,
    pose_model,
    collect_params,
    label_params,
    target_df,
    image_dir,
    save_dir=None,
    error_plotter=None
):
    # start 50mm above workframe origin
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints((*robot.joint_angles[:-1], 0))

    # drop 10mm to contact object
    clearance = (0, 0, 10, 0, 0, 0)
    robot.move_linear(np.zeros(6) - clearance)
    joint_angles = robot.joint_angles

    # initialize pred_df
    pred_df = pd.DataFrame(columns=label_params['label_names'])

    # ==== data testing loop ====
    for i, row in target_df.iterrows():
        image_name = row.loc["sensor_image"]
        pose = row.loc[POSE_LABEL_NAMES].values.astype(float)
        shear = row.loc[SHEAR_LABEL_NAMES].values.astype(float)

        # report
        with np.printoptions(precision=2, suppress=True):
            print(f"\n\nCollecting for pose {i+1}: pose{pose}, shear{shear}")

        # move to above new pose (avoid changing pose in contact with object)
        robot.move_linear(pose + shear - clearance)

        # move down to offset pose
        robot.move_linear(pose + shear)

        # move to target pose inducing shear
        robot.move_linear(pose)

        # collect and process tactile image
        image_outfile = os.path.join(image_dir, image_name)
        tactile_image = sensor.process(image_outfile)
        pred_df.loc[i] = pose_model.predict(tactile_image)

        # move above the target pose
        robot.move_linear(pose - clearance)

        # if sorted, don't move to reset position
        if not collect_params['sort']:
            robot.move_joints(joint_angles)

    # save results
    if save_dir:
        pred_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
        target_df.to_csv(os.path.join(save_dir, 'targets.csv'), index=False)
    
    if error_plotter:
        error_plotter.plot_interp = False
        error_plotter.final_plot(pred_df, target_df)

    # finish 50mm above workframe origin then zero last joint
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints((*robot.joint_angles[:-1], 0))
    robot.close()


def launch(args):

    output_dir = '_'.join([args.robot, args.sensor])

    # test the trained networks
    for args.experiment, args.predict in zip(args.experiments, args.predicts):
        for args.model, args.sample_num in zip(args.models, args.sample_nums):
            
            run_dir_name = '_'.join(filter(None, ['test', args.model, *args.run_version]))

            # setup save dir
            save_dir = os.path.join(args.path, output_dir, args.experiment, 'regress_'+args.predict, run_dir_name)
            image_dir = os.path.join(save_dir, "processed_images")
            make_dir([save_dir, image_dir])

            # load model params from model directory
            model_dir = os.path.join(args.path, output_dir, args.experiment, 'regress_'+args.predict, args.model)
            model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
            model_image_params = load_json_obj(os.path.join(model_dir, 'model_image_params'))
            label_params = load_json_obj(os.path.join(model_dir, 'model_label_params'))
 
            # load/save environment and sensor params from model directory
            env_params = load_json_obj(os.path.join(model_dir, 'env_params'))
            save_json_obj(env_params, os.path.join(save_dir, 'env_params'))
            if os.path.isfile(os.path.join(model_dir, 'processed_image_params.json')):
                sensor_image_params = load_json_obj(os.path.join(model_dir, 'processed_image_params'))
            else:
                sensor_image_params = load_json_obj(os.path.join(model_dir, 'sensor_image_params'))

            # setup targets to collect and predictions
            collect_params = setup_collect_params(args.robot, args.experiment, save_dir)
            target_df = setup_targets(
                collect_params,
                args.sample_num,
                save_dir
            )

            # setup the robot and sensor
            robot, sensor = setup_embodiment(
                env_params,
                sensor_image_params
            )

            # setup any plotters
            error_plotter = RegressionPlotter(save_dir, label_params['target_label_names'], name='test.png')

            # setup the model
            label_encoder = LabelEncoder(label_params, args.device)

            model = setup_model(
                in_dim=model_image_params['image_processing']['dims'],
                in_channels=1,
                out_dim=label_encoder.out_dim,
                model_params=model_params,
                saved_model_dir=model_dir,
                device=args.device
            )
            model.eval()

            pose_model = LabelledModel(
                model,
                model_image_params['image_processing'],
                label_encoder,
                args.device
            )

            # run the test
            test(
                robot,
                sensor,
                pose_model,
                collect_params,
                label_params,
                target_df,
                image_dir,
                save_dir,
                error_plotter
            )


if __name__ == "__main__":

    args = setup_parse(
        # path='./../tactile-data', 
        robot='sim',
        sensor='tactip',
        experiments=['edge_yRz'],#, 'surface_zRxRy', 'edge_yzRxRyRz'],
        predicts=['pose_yRz'],#, 'pose_zRxRy', 'pose_yzRxRyRz'], # order must match experiments
        models=['simple_cnn'],
        sample_nums=[100],
        # run_version=['test'], # to not overwrite previous runs
        device='cpu' # 'cuda' or 'cpu'
    )

    launch(args)
