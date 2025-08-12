from tg3.learning.reinforcement.rl_sb3.params.default_params import (
    env_args,
    rl_params_ppo,
    ppo_params, 
    rl_params_sac,
    sac_params
)


env_args["env_params"]["max_steps"] = 250
# env_args["env_params"]["observation_mode"] = "oracle"
env_args["env_params"]["observation_mode"] = "tactile_and_feature"
# env_args["env_params"]["observation_mode"] = "visual_and_feature"
# env_args["env_params"]["observation_mode"] = "visuotactile_and_feature"

env_args["robot_arm_params"]["control_mode"] = "tcp_velocity_control"
env_args["robot_arm_params"]["control_dofs"] = ["x", "y"]

env_args["tactile_sensor_params"]["type"] = "flat_tactip"

rl_params_ppo["env_id"] = "object_roll-v0"
rl_params_ppo["total_timesteps"] = int(2e6)
ppo_params["learning_rate"] = 3e-4

rl_params_sac["env_id"] = "object_roll-v0"
rl_params_sac["total_timesteps"] = int(2e6)
sac_params["learning_rate"] = 3e-4
