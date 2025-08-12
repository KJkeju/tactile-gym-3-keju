"""
python launch_training.py -i ur_tactip -o sim_tactip -t edge_2d -m pix2pix_128 -v tap
"""
import os
import itertools as it

from tg3.learning.supervised.image_to_image.pix2pix.image_generator import Image2ImageGenerator as Image2ImageGenerator_pix
from tg3.learning.supervised.image_to_image.pix2pix.setup_model import setup_model as setup_pix_model
from tg3.learning.supervised.image_to_image.pix2pix.train_model import train_model as train_pix_model
from tg3.learning.supervised.image_to_image.setup_training import setup_training, setup_parse
from tg3.learning.supervised.image_to_image.shpix2pix.image_generator import Image2ImageGenerator as Image2ImageGenerator_sh
from tg3.learning.supervised.image_to_image.shpix2pix.setup_model import setup_model as setup_sh_model
from tg3.learning.supervised.image_to_image.shpix2pix.train_model import train_model as train_sh_model
from tg3.utils.utils import make_dir, seed_everything


def launch(args):

    input_train_dirs = [
        os.path.join(args.path, args.input, *i) for i in it.product(args.experiments, args.train_dirs)
    ]
    target_train_dirs = [
        os.path.join(args.path, args.target, *i) for i in it.product(args.experiments, args.train_dirs)
    ]
    input_val_dirs = [
        os.path.join(args.path, args.input, *i) for i in it.product(args.experiments, args.val_dirs)
    ]
    target_val_dirs = [
        os.path.join(args.path, args.target, *i) for i in it.product(args.experiments, args.val_dirs)
    ]

    for args.model in args.models:

        experiment_dir = '_'.join(args.experiments)

        # setup save dir
        save_dir = os.path.join(args.path, args.input, experiment_dir, 'target_'+args.target, args.model)
        make_dir(save_dir)

        # setup parameters
        learning_params, model_params, image_params = setup_training(
            args.model,
            input_train_dirs,
            save_dir
        )

        # configure dataloaders
        train_generator = Image2ImageGenerator(
            input_data_dirs=input_train_dirs,
            target_data_dirs=target_train_dirs,
            **{**image_params['image_processing'], **image_params['augmentation']}
        )
        val_generator = Image2ImageGenerator(
            input_data_dirs=input_val_dirs,
            target_data_dirs=target_val_dirs,
            **image_params['image_processing']
        )

        # create the model
        seed_everything(learning_params['seed'])
        generator, discriminator = setup_model(
            in_dim=image_params['image_processing']['dims'],
            model_params=model_params,
            device=args.device
        )
  
        # run training
        train_model(
            generator=generator,
            discriminator=discriminator,
            train_generator=train_generator,
            val_generator=val_generator,
            learning_params=learning_params,
            image_processing_params=image_params['image_processing'],
            save_dir=save_dir,
            device=args.device,
            debug=True
        )
        

def Image2ImageGenerator(**kwargs):
    if "sh" not in args.model:
        generator = Image2ImageGenerator_pix(**kwargs)
    else:
        generator = Image2ImageGenerator_sh(**kwargs)
    return generator

def setup_model(**kwargs):
    if "sh" not in args.model:
        model = setup_pix_model(**kwargs)
    else:
        model = setup_sh_model(**kwargs)
    return model

def train_model(**kwargs):
    if "sh" not in args.model:
        train_pix_model(**kwargs)
    else:
        train_sh_model(**kwargs)


if __name__ == "__main__":

    args = setup_parse(
        path='./../tactile-data',
        input='sim_ur_tactip',
        target='ur_tactip',
        experiments=['edge_yRz_shear'],
        models=['shpix2pix_128'],
        # train_dirs=['data_train'],
        # val_dirs=['data_val'],
        # device='cuda'
    )

    launch(args)
