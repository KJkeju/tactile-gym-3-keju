"""
Collect tactile data from a robot with a tactile sensor.

To run a Tactip simulation with the edge_yRz experiment:
    python launch_collect.py -r sim -s tactip -e edge_yRz -dd data -n 100
This saves data to the tactile_data/sim_tactip/edge_yRz/data/ directory, which includes:
 - sensor_images/: tactile images in (image_0.png is the reference image)
 - processed_images/: processed images; in this case, cropped
 - collect_params.json: data collection parameters
 - env_params.json: environment parameters
 - processed_image_params.json, sensor_image_params.json: parameters describing the images
 - targets.csv: table of target poses
 - targets_images.csv: table of target poses with sensor image filenames
"""
import os
import pandas as pd

from tg3.data.collect.collect import collect
from tg3.data.collect.setup_embodiment import setup_embodiment
from tg3.data.collect.setup_targets import setup_targets
from tg3.data.process.process_images import process_images, partition_data
from tg3.data.setup_collect import setup_collect_data, setup_collect_params, setup_parse
from tg3.utils.utils import make_dir, load_json_obj, save_json_obj


def launch(args):
    """
    For each experiment, collect the specified amount of tactile data and put it in
    the right data directories.

    This sets the environment and sensor up, saves the current configuration and poses
    to JSON and CSV files, and runs the data collection. The collected data is saved
    as images in the sensor_images/ directory, and the processed images are saved in
    the processed_images/ directory.

    Args:
        args: argparse CLI args. See setup_parse() for details
    """
    output_dir = '_'.join([args.robot, args.sensor])

    for experiment in args.experiments:
        for data_dir, sample_num in zip(args.data_dirs, args.sample_nums):

            # setup save dir
            save_dir = os.path.join(args.path, output_dir, experiment, data_dir)
            image_dir = os.path.join(save_dir, "sensor_images")
            make_dir([save_dir, image_dir])

            # setup environment and sensor parameters
            env_params, sensor_params = setup_collect_data(args.robot, args.sensor, experiment, save_dir)

            # setup embodiment
            robot, sensor = setup_embodiment(env_params, sensor_params)

            # either transfer collection parameters and targets or new set upfor this run
            if args.transfer:
                load_dir = os.path.join(args.path, args.transfer, experiment, data_dir)
                collect_params = load_json_obj(os.path.join(load_dir, 'collect_params'))
                target_df = pd.read_csv(os.path.join(load_dir, 'targets_images.csv'))
            else:
                collect_params = setup_collect_params(args.robot, experiment, save_dir)
                target_df = setup_targets(collect_params, sample_num, save_dir)

            # save parameters and targets to the current save_dir
            save_json_obj(collect_params, os.path.join(save_dir, 'collect_params'))
            target_df.to_csv(os.path.join(save_dir, "targets.csv"), index=False)

            # run the collection
            collect(
                robot,
                sensor,
                target_df,
                image_dir,
                collect_params
            )


def process(args, image_params, split=None):
    """
    For each experiment, process the images in the right data directories
    and save them along with a JSON dump of the processing parameters
    and a CSV mapping processed images to target poses.

    Args:
        args: argparse CLI args. See setup_parse() for details
        image_params: dict of options for processing images; see
                      tg3.data.process.transform_image.transform_image()
                      for possible options
        split: tells us which directories to process. Pass None to process
               all directories; pass a float to split the data into two
               chunks and save the processed images in separate directories.
    """
    output_dir = '_'.join([args.robot, args.sensor])

    for experiment in args.experiments:
        path = os.path.join(args.path, output_dir, experiment)
        dir_names = partition_data(path, args.data_dirs, split)
        process_images(path, dir_names, image_params)


if __name__ == "__main__":

    # Set default CLI args and parse the args passed in
    args = setup_parse(
        # path='./../tactile-data', 
        # transfer='ur_tactip',
        robot='sim',
        sensor='tactip',
        # experiments=['edge_yRz', 'surface_zRxRy', 'edge_yzRxRyRz'], 
        experiments=['surface_zRxy'],
        data_dirs=['data_train', 'data_val'],
        sample_nums=[80, 20]
    )

    # Create raw tactile data
    launch(args)

    # Process the data for ML
    # In this case, just crop to this bounding box
    image_params = {
        "bbox": (12, 12, 240, 240)  
    }
    process(args, image_params)# , split=0.8)
