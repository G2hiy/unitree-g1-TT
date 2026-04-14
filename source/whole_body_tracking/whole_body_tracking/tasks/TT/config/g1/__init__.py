import gymnasium as gym


from whole_body_tracking.tasks.TT.g1_tt_env_cfg import TableTennisEnvCfg
from whole_body_tracking.tasks.TT.config.g1.agents.g1_tt_ppo_cfg import G1TTPPORunnerCfg

gym.register(
    id="G1-TableTennis-v0",
    entry_point="whole_body_tracking.tasks.TT.envs.tt_env:TableTennisEnv", 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": TableTennisEnvCfg,  
        "rsl_rl_cfg_entry_point": G1TTPPORunnerCfg,
    },
)