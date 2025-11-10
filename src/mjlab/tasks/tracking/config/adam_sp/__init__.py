import gymnasium as gym

gym.register(
  id="Mjlab-Tracking-Flat-Adam-SP",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AdamSpFlatEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:AdamSpFlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Adam-SP-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AdamSpFlatEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:AdamSpFlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Adam-SP-Demo",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AdamSpFlatEnvCfg_DEMO",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:AdamSpFlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Adam-SP-No-State-Estimation",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AdamSpFlatNoStateEstimationEnvCfg",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:AdamSpFlatPPORunnerCfg",
  },
)

gym.register(
  id="Mjlab-Tracking-Flat-Adam-SP-No-State-Estimation-Play",
  entry_point="mjlab.envs:ManagerBasedRlEnv",
  disable_env_checker=True,
  kwargs={
    "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AdamSpFlatNoStateEstimationEnvCfg_PLAY",
    "rl_cfg_entry_point": f"{__name__}.rl_cfg:AdamSpFlatPPORunnerCfg",
  },
)

