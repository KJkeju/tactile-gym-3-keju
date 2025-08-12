"""
python replay_servo.py -r sim -s tactip -e edge_yRz_shear -t pose_yRz -m simple_cnn -o circle
"""
import os
import itertools as it

from tg3.data.collect.setup_embodiment import setup_embodiment
from tg3.tasks.servo.launch_servo import servo
from tg3.tasks.servo.servo_utils.controller import PIDController
from tg3.tasks.servo.servo_utils.labelled_model import LabelledModel
from tg3.tasks.servo.servo_utils.utils_plots import Contour3DPlotter
from tg3.tasks.servo.setup_servo import setup_parse
from tg3.learning.supervised.image_to_feature.cnn.label_encoder_regress import LabelEncoder
from tg3.learning.supervised.image_to_feature.cnn.setup_model import setup_model
from tg3.utils.utils import load_json_obj


def replay(args):

    output_dir = '_'.join([args.robot.replace('dummy_',''), args.sensor])

    for args.experiment, args.predict, args.model, args.object in it.product(args.experiments, args.predicts, args.models, args.objects):

        run_dir_name = '_'.join(filter(None, [args.object, *args.run_version]))

        # setup save dir
        run_dir = os.path.join(args.path, output_dir, args.experiment, 'servo_'+run_dir_name)
        image_dir = os.path.join(run_dir, "processed_images")

        # load model and preproc parameters from model dir
        model_dir = os.path.join(args.path, output_dir, args.experiment, 'regress_'+args.predict, args.model)
        model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
        model_image_params = load_json_obj(os.path.join(model_dir, 'model_image_params'))
        label_params = load_json_obj(os.path.join(model_dir, 'model_label_params'))
        sensor_params = {'type': 'replay'}

        # load control, environment and task parameters from run_dir
        control_params = load_json_obj(os.path.join(run_dir, 'control_params'))
        env_params = load_json_obj(os.path.join(run_dir, 'env_params'))
        task_params = load_json_obj(os.path.join(run_dir, 'task_params'))
        env_params['robot'] = args.robot # for dummy

        # setup the robot and sensor
        robot, sensor = setup_embodiment(
            env_params,
            sensor_params
        )

        # setup the controller
        pid_controller = PIDController(**control_params)

        # setup any plotters
        plotter = Contour3DPlotter(run_dir, name='replay.png', elev=90, azim=180)
        
        # setup the model
        label_encoder = LabelEncoder(label_params, device=args.device)

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

        # run the servo control
        servo(
            robot,
            sensor,
            pose_model,
            pid_controller,
            image_dir,
            task_params,
            pose_plotter=plotter
        )


if __name__ == "__main__":

    args = setup_parse(
        # path='./../tactile-data', 
        robot='sim',
        sensor='tactip',
        experiments=['edge_yRz'],
        predicts=['pose_yRz'],
        models=['simple_cnn'],
        objects=['circle', 'square'],
        # run_version=['test'], 
        # device='cpu' # 'cuda' or 'cpu'
    )

    replay(args)
