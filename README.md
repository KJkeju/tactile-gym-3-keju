# Tactile-Gym: Robot learning suite for tactile robotics 
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

<!-- [Project Website](https://sites.google.com/my.bristol.ac.uk/tactile-gym-sim2real/home) &nbsp;&nbsp;• -->
**Tactile Gym 2.0**: [Project Website](https://sites.google.com/my.bristol.ac.uk/tactilegym2/home) &nbsp;&nbsp;•&nbsp;&nbsp;[Paper](https://ieeexplore.ieee.org/abstract/document/9847020)

**Tactile Gym 1.0**: [Project Website](https://sites.google.com/my.bristol.ac.uk/tactile-gym-sim2real/home) &nbsp;&nbsp;•&nbsp;&nbsp;[Paper](http://arxiv.org/abs/2106.08796)

### Installation ###
This repo has only been developed and tested with Ubuntu 22.04 and python 3.10.
We use `uv` to manage the python environment.

Clone the repository:
```console
git clone https://github.com/robot-dexterity/tactile_gym_3
cd tactile_gym_3
```

Check if you have `uv` installed:
```sh
which uv
```
if you don't see output like: `/home/user/.local/bin/uv`, then [install `uv`](https://docs.astral.sh/uv/getting-started/installation/):
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Set the environment up with:
```sh
uv sync
```

Temporary Hack #1 (Linux only): Add a line setting [PYTHONPATH in '~/.bashrc'](https://stackoverflow.com/questions/3402168/permanently-add-a-directory-to-pythonpath): 

export PYTHONPATH="${PYTHONPATH}:/my/other/path" 

where '/my/other/path' points to the 'tactile_gym_3' directory. Then run
```sh
source ~/.bashrc
```

### Where to start ###

Windows/Linux: You can also run these from VS Code. Adjust by modifying arguments in setup parse at end of scripts.

Linux:
```sh
source .venv/bin/activate
```

Collect some tactile data and save in tactile_data as 'data':
```sh
python ./tg3/data/launch_collect.py -r sim -s tactip -e edge_yRz -dd data -n 100
```

Train a model on some data already saved in tactile_data:
```sh
python ./tg3/learning/supervised/image_to_feature/launch_training.py -r sim -s tactip -e edge_yRz_shear -t pose_yRz -m simple_cnn_test
```

Test that a model works in real time:
```sh
python ./tg3/tasks/test/launch_test.py -r sim -s tactip -e edge_yRz_shear -t pose_yRz -m simple_cnn -n 100 -v test
```

Run servo control in real time: 
```sh
python ./tg3/tasks/servo/launch_servo.py -r sim -s tactip -e edge_yRz_shear -t pose_yRz -m simple_cnn -o circle -n 160 -v test
```
