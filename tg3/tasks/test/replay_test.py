"""
python launch_test.py -r sim -s tactip -e edge_yRz_shear -t pose_yRz -m simple_cnn
"""
import os
import pandas as pd

from tg3.learning.supervised.image_to_feature.cnn.label_encoder_regress import LabelEncoder
from tg3.learning.supervised.image_to_feature.cnn.setup_model import setup_model
from tg3.learning.supervised.image_to_feature.cnn.utils_plots import RegressionPlotter
from tg3.data.collect.setup_embodiment import setup_embodiment
from tg3.tasks.servo.servo_utils.labelled_model import LabelledModel
from tg3.tasks.servo.setup_servo import setup_parse
from tg3.tasks.test.launch_test import test
from tg3.utils.utils import load_json_obj


def replay(args):        
        
    output_dir = '_'.join([args.robot.replace('dummy_',''), args.sensor])

     # test the trained networks
    for args.experiment, args.predict in zip(args.experiments, args.predicts):
        for args.model, args.sample_num in zip(args.models, args.sample_nums):

            run_dir_name = '_'.join(filter(None, ['test', args.model, *args.run_version]))

            # setup save dir
            run_dir = os.path.join(args.path, output_dir, args.experiment, 'predict_'+args.predict, run_dir_name)
            image_dir = os.path.join(run_dir, "processed_images")

            # load model and preproc parameters from model dir
            model_dir = os.path.join(args.path, output_dir, args.experiment, 'predict_'+args.predict, args.model)
            model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
            model_image_params = load_json_obj(os.path.join(model_dir, 'model_image_params'))
            label_params = load_json_obj(os.path.join(model_dir, 'model_label_params'))
            sensor_params = {'type': 'replay'}
            
            # load collect, environment params and target_df from run_dir
            collect_params = load_json_obj(os.path.join(run_dir, 'collect_params'))
            env_params = load_json_obj(os.path.join(run_dir, 'env_params'))
            target_df = pd.read_csv(os.path.join(run_dir, 'targets.csv'))
            env_params['robot'] = args.robot # for dummy

            # setup any plotters
            error_plotter = RegressionPlotter(label_params)

            # setup the robot and sensor
            robot, sensor = setup_embodiment(
                env_params,
                sensor_params
            )

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
                device=args.device
            )

            test(
                robot,
                sensor,
                pose_model,
                collect_params,
                label_params,
                target_df,
                image_dir,
                run_dir,
                error_plotter
            )


if __name__ == "__main__":

    args = setup_parse(
        # path='./../tactile-data', 
        robot='sim', # prefix with dummy_ to use dummy robot
        sensor='tactip',
        experiments=['edge_yRz_shear'],
        predicts=['pose_yRz'],
        models=['simple_cnn'],
        # run_version=['test'],
        # device='cpu' # 'cuda' or 'cpu'
    )

    replay(args)
