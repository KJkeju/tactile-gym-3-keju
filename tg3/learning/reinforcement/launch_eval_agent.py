import os
import sys
import numpy as np
import cv2

import stable_baselines3 as sb3

import tg3.learning.reinforcement.envs_pb
from tg3.learning.reinforcement.rl_sb3.rl_utils import make_eval_env
from tg3.utils.utils import load_json_obj


def eval_and_save_vid(
    model, env, saved_model_dir, n_eval_episodes=10, deterministic=True, 
    render=False, save_vid=False, take_snapshot=False
):
    """
    Evaluates a trained RL agent in the given environment for a specified number of episodes.
    Optionally saves a video and/or a snapshot of the evaluation.

    Args:
        model: The trained RL model to evaluate.
        env: The environment to evaluate the agent in.
        saved_model_dir (str): Directory to save video/snapshot files.
        n_eval_episodes (int): Number of episodes to evaluate.
        deterministic (bool): Whether to use deterministic actions.
        render (bool): Whether to render the environment during evaluation.
        save_vid (bool): Whether to save a video of the evaluation.
        take_snapshot (bool): Whether to save a snapshot of the environment.

    Returns:
        tuple: Lists of episode rewards and episode lengths.
    """

    # Lists to store rewards and lengths for each episode
    episode_rewards, episode_lengths = [], []

    video_writer = None
    if save_vid:
        # Initialize video writer if saving video
        render_img = env.render(mode='rgb_array')
        size = (render_img.shape[1], render_img.shape[0])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            os.path.join(saved_model_dir, 'evaluated_policy.mp4'), fourcc, 24.0, size
        )

    for _ in range(n_eval_episodes):
        obs = env.reset()
        done, state = False, None
        ep_reward, ep_length = 0.0, 0

        while not done:
            # Get action from model
            action, state = model.predict(obs, state, deterministic)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
            ep_length += 1

            # Render environment if needed
            if render or save_vid or take_snapshot:
                render_img = env.render(mode='rgb_array')

            # Write frame to video if saving
            if save_vid and video_writer:
                video_writer.write(cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB))

            # Save snapshot if requested
            if take_snapshot:
                cv2.imwrite(os.path.join(saved_model_dir, 'env_snapshot.png'),
                            cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB))

        # Store episode results
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

    # Release video writer if used
    if video_writer:
        video_writer.release()

    return episode_rewards, episode_lengths


def final_evaluation(
    saved_model_dir, n_eval_episodes, seed=None, deterministic=True,
    show_gui=True, show_tactile=True, show_vision=True, render=False,
    save_vid=False, take_snapshot=False,
):
    """
    Loads a trained RL agent and evaluates it in the specified environment.

    Args:
        saved_model_dir (str): Directory where the trained model and config files are stored.
        n_eval_episodes (int): Number of evaluation episodes to run.
        seed (int, optional): Random seed for reproducibility.
        deterministic (bool): Whether to use deterministic actions.
        show_gui (bool): Whether to display the environment GUI.
        show_tactile (bool): Whether to display tactile sensor output.
        show_vision (bool): Whether to display vision sensor output.
        render (bool): Whether to render the environment during evaluation.
        save_vid (bool): Whether to save a video of the evaluation.
        take_snapshot (bool): Whether to save a snapshot of the environment.
    """

    # Load RL parameters and environment arguments from saved files
    rl_params = load_json_obj(os.path.join(saved_model_dir, 'rl_params'))
    env_args = load_json_obj(os.path.join(saved_model_dir, 'env_args'))

    # Update display parameters based on function arguments
    env_args['env_params']['show_gui'] = show_gui
    env_args['tactile_sensor_params']['show_tactile'] = show_tactile
    env_args['visual_sensor_params']['show_vision'] = show_vision

    # Create the evaluation environment
    eval_env = make_eval_env(rl_params['env_id'], env_args, rl_params)

    # Load the trained RL model
    model_path = os.path.join(saved_model_dir, 'trained_models', 'best_model.zip')
    model_cls = {'ppo': sb3.PPO, 'sac': sb3.SAC}.get(rl_params['algo_name'])
    if not model_cls:
        sys.exit(f"Incorrect algorithm specified: {rl_params['algo_name']}.")
    model = model_cls.load(model_path)

    # Seed the environment if a seed is provided
    if seed is not None:
        eval_env.reset()
        eval_env.seed(seed)

    # Evaluate the agent and optionally save video/snapshot
    rewards, lengths = eval_and_save_vid(
        model, eval_env, saved_model_dir, n_eval_episodes, deterministic,
        render, save_vid, take_snapshot
    )

    # Print average episode reward and length
    print(f'Avg Ep Rew: {np.mean(rewards)}, Avg Ep Len: {np.mean(lengths)}')
    eval_env.close()


if __name__ == '__main__':

    # Choose which RL algorithm to use ('ppo' or 'sac')
    algo_name = 'ppo'

    # Select environment ID ('edge_follow-v0', 'surface_follow-v0', 'surface_follow-v1', 'object_roll-v0', 'object_push-v0', 'object_balance-v0')
    env_id = 'object_push-v0'

    # Observation type options ('s1_oracle', 's1_tactile', 's1_tactile_and_feature', 'visual', 'visuotactile')
    obs_type = 's1_tactile_and_feature'  

    # Set the directory where the trained model is saved
    saved_model_dir = os.path.join(
        './tactile_data/rl', env_id, algo_name, obs_type
    )

    # Run the final evaluation of the trained agent
    final_evaluation(
        saved_model_dir,
        n_eval_episodes=10,
        seed=1,
        deterministic=True,
        show_gui=True,
        show_tactile=True,
        show_vision=False,
        render=False,
        save_vid=True,
        take_snapshot=False,
    )
