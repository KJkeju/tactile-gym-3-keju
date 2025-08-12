import os
import sys

from stable_baselines3.common.callbacks import EvalCallback, EveryNTimesteps
from stable_baselines3 import PPO, SAC

import tg3.learning.reinforcement.envs_pb
from tg3.learning.reinforcement.launch_eval_agent import final_evaluation
from tg3.learning.reinforcement.rl_sb3.custom.custom_callbacks import FullPlottingCallback, ProgressBarManager
from tg3.learning.reinforcement.rl_sb3.params import import_parameters
from tg3.learning.reinforcement.rl_sb3.rl_utils import make_training_envs, make_eval_env
from tg3.utils.utils import save_json_obj, convert_json, make_dir


def train_agent(
    algo_name='ppo', env_id='edge_follow-v0', env_args={}, rl_params={}, algo_params={}, path='./tactile_data/rl', 
):
    """
    Trains a reinforcement learning agent using the specified algorithm and environment.

    Args:
        algo_name (str): Name of the RL algorithm to use ('ppo' or 'sac').
        env_id (str): Gym environment ID.
        env_args (dict): Arguments for environment creation.
        rl_params (dict): RL-specific parameters (e.g., seed, eval_freq, total_timesteps).
        algo_params (dict): Algorithm-specific parameters.
    """
    # Setup save directory
    save_dir = os.path.join(
        path, rl_params['env_id'], algo_name,
        f"s{rl_params['seed']}_{env_args['env_params']['observation_mode']}"
    )
    make_dir(save_dir)

    # Save parameters for reproducibility
    for name, params in [('rl_params', rl_params), ('algo_params', algo_params), ('env_args', env_args)]:
        save_json_obj(convert_json(params), os.path.join(save_dir, name))

    # Create training and evaluation environments
    env = make_training_envs(env_id, env_args, rl_params, save_dir)
    eval_env = make_eval_env(env_id, env_args, rl_params)

    # Setup evaluation and plotting callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, 'trained_models'),
        log_path=os.path.join(save_dir, 'trained_models'),
        eval_freq=rl_params['eval_freq'],
        n_eval_episodes=rl_params['n_eval_episodes'],
        deterministic=True,
        render=False,
        verbose=1,
    )
    plotting_callback = FullPlottingCallback(log_dir=save_dir, total_timesteps=rl_params['total_timesteps'])
    event_plotting_callback = EveryNTimesteps(n_steps=rl_params['eval_freq'] * rl_params['n_envs'], callback=plotting_callback)

    # Select RL algorithm
    ModelClass = PPO if algo_name == 'ppo' else SAC if algo_name == 'sac' else None
    if ModelClass is None:
        sys.exit(f'Incorrect algorithm specified: {algo_name}.')
    model = ModelClass(rl_params['policy'], env, **algo_params, verbose=1)

    # Train the agent with progress bar and callbacks
    with ProgressBarManager(rl_params['total_timesteps']) as progress_bar_callback:
        model.learn(
            total_timesteps=rl_params['total_timesteps'],
            callback=[progress_bar_callback, eval_callback, event_plotting_callback],
        )

    # Save the final trained model
    model.save(os.path.join(save_dir, 'trained_models', 'final_model'))
    env.close()
    eval_env.close()

    # Run final evaluation and save results
    final_evaluation(
        saved_model_dir=save_dir,
        n_eval_episodes=10,
        seed=1,
        deterministic=True,
        show_gui=False,
        show_tactile=False,
        show_vision=False,
        render=True,
        save_vid=True,
        take_snapshot=False,
    )


if __name__ == '__main__':

    # Choose which RL algorithm to use ('ppo' or 'sac')
    algo_name = 'ppo'

    # Select environment ID ('edge_follow-v0', 'surface_follow-v0', 'surface_follow-v1', 'object_roll-v0', 'object_push-v0', 'object_balance-v0')
    env_id = 'object_push-v0'

    # Import parameters and train the agent
    train_agent(algo_name, env_id, *import_parameters(env_id, algo_name))
