import gymnasium as gym
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


##
# Register Gym environments.
##

gym.register(
    id="G1-Tracking-v0",
    entry_point="whole_body_tracking.tasks.tracking.envs.tt_env:TableTennisEnv", 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TrackingEnvCfg,  
        "rsl_rl_cfg_entry_point": G1FlatPPORunnerCfg,
    },
)