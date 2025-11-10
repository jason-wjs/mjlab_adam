from dataclasses import dataclass, replace

from mjlab.asset_zoo.robots.adam_sp.adam_sp_constants import (
  ADAM_SP_ACTION_SCALE,
  ADAM_SP_ROBOT_CFG,
)
from mjlab.managers.manager_term_config import ObservationTermCfg as ObsTerm
from mjlab.managers.manager_term_config import term
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


@dataclass
class AdamSpFlatEnvCfg(TrackingEnvCfg):
  def __post_init__(self):
    # Scene and robot entity.
    self.scene.entities = {"robot": replace(ADAM_SP_ROBOT_CFG)}

    # Self-collision sensor on pelvis subtree (like G1 pattern).
    self_collision_cfg = ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    )
    self.scene.sensors = (self_collision_cfg,)

    # Action scaling from robot constants.
    self.actions.joint_pos.scale = ADAM_SP_ACTION_SCALE

    # Motion command anchor/body mapping for Adam-SP.
    # Use pelvis as anchor; viewer focuses on torso for clarity.
    self.commands.motion.anchor_body_name = "pelvis"
    self.commands.motion.body_names = [
      # Lower body
      "pelvis",
      "hipPitchLeft",
      "thighLeft",
      "anklePitchLeft",
      "hipPitchRight",
      "thighRight",
      "anklePitchRight",
      # Torso and upper body
      "torso",
      "shoulderRollLeft",
      "elbowLeft",
      "wristYawLeft",
      "shoulderRollRight",
      "elbowRight",
      "wristYawRight",
    ]

    # Use base velocities directly from entity state (no built-in IMU in adam_sp.xml).
    self.observations.policy.base_lin_vel = ObsTerm(
      func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5)
    )
    self.observations.policy.base_ang_vel = ObsTerm(
      func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
    )
    self.observations.critic.base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    self.observations.critic.base_ang_vel = ObsTerm(func=mdp.base_ang_vel)

    # Domain randomization hooks.
    # Randomize only on foot geoms; Adam uses toe_left/toe_right naming.
    self.events.foot_friction.params["asset_cfg"].geom_names = [
      "toe_left",
      "toe_right",
    ]
    # Slight COM randomization relative to torso.
    self.events.base_com.params["asset_cfg"].body_names = "torso"

    # Termination: track end-effectors: feet toes and wrists.
    self.terminations.ee_body_pos.params["body_names"] = [
      "toe_left",
      "toe_right",
      "wristYawLeft",
      "wristYawRight",
    ]

    # Viewer.
    self.viewer.body_name = "torso"


@dataclass
class AdamSpFlatNoStateEstimationEnvCfg(AdamSpFlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Remove proprioceptive terms that would require state estimation.
    self.observations.policy.motion_anchor_pos_b = None
    self.observations.policy.base_lin_vel = None


@dataclass
class AdamSpFlatEnvCfg_PLAY(AdamSpFlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Deterministic play: disable observation corruption and disturbances.
    self.observations.policy.enable_corruption = False
    self.events.push_robot = None

    # Disable RSI randomization.
    self.commands.motion.pose_range = {}
    self.commands.motion.velocity_range = {}

    self.commands.motion.sampling_mode = "start"

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)


@dataclass
class AdamSpFlatEnvCfg_DEMO(AdamSpFlatEnvCfg_PLAY):
  def __post_init__(self):
    super().__post_init__()
    # Demo uses uniform sampling for more diverse motion coverage with num_envs > 1.
    self.commands.motion.sampling_mode = "uniform"


@dataclass
class AdamSpFlatNoStateEstimationEnvCfg_PLAY(AdamSpFlatNoStateEstimationEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.enable_corruption = False
    self.events.push_robot = None

    # Disable RSI randomization.
    self.commands.motion.pose_range = {}
    self.commands.motion.velocity_range = {}

    self.commands.motion.sampling_mode = "start"

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)
