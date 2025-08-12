from tg3.learning.reinforcement.rl_sb3.params.default_params import (
    env_args,
    rl_params_ppo,
    ppo_params, 
    rl_params_sac,
    sac_params
)


env_args["env_params"]["max_steps"] = 200
# env_args["env_params"]["observation_mode"] = "oracle"
env_args["env_params"]["observation_mode"] = "tactile"
# env_args["env_params"]["observation_mode"] = "visual"
# env_args["env_params"]["observation_mode"] = "visuotactile"

env_args["robot_arm_params"]["control_mode"] = "tcp_velocity_control"
env_args["robot_arm_params"]["control_dofs"] = ["z", "Rx", "Ry"]

env_args["tactile_sensor_params"]["type"] = "standard_tactip"
# env_args["tactile_sensor_params"]["type"] = "standard_digit"
# env_args["tactile_sensor_params"]["type"] = "standard_digitac"

rl_params_ppo["env_id"] = "surface_follow-v0"
rl_params_ppo["total_timesteps"] = int(1e6)
ppo_params["learning_rate"] = 3e-4

rl_params_sac["env_id"] = "surface_follow-v0"
rl_params_sac["total_timesteps"] = int(1e6)
sac_params["learning_rate"] = 3e-4
