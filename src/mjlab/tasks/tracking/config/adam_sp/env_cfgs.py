"""Adam_SP flat terrain tracking configuration.

This module provides factory functions that create complete ManagerBasedRlEnvCfg
instances for the Adam_SP robot tracking task on flat terrain.
"""

from copy import deepcopy

from mjlab.asset_zoo.robots.adam_sp.adam_sp_constants import (
  ADAM_SP_ACTION_SCALE,
  get_adam_sp_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.tracking_env_cfg import create_tracking_env_cfg
from mjlab.utils.retval import retval


@retval
def ADAM_SP_FLAT_TRACKING_ENV_CFG() -> ManagerBasedRlEnvCfg:
  """Create Adam_SP flat terrain tracking configuration."""
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  return create_tracking_env_cfg(
    robot_cfg=get_adam_sp_robot_cfg(),
    action_scale=ADAM_SP_ACTION_SCALE,
    viewer_body_name="torso",
    motion_file="",
    anchor_body_name="pelvis",
    body_names=(
      "pelvis",
      "hipPitchLeft",
      "thighLeft",
      "anklePitchLeft",
      "hipPitchRight",
      "thighRight",
      "anklePitchRight",
      "torso",
      "shoulderRollLeft",
      "elbowLeft",
      "wristYawLeft",
      "shoulderRollRight",
      "elbowRight",
      "wristYawRight",
    ),
    foot_friction_geom_names=(r"toe_(left|right)",),
    ee_body_names=(
      "anklePitchLeft",
      "anklePitchRight",
      "wristYawLeft",
      "wristYawRight",
    ),
    base_com_body_name="torso",
    sensors=(self_collision_cfg,),
    pose_range={
      "x": (-0.05, 0.05),
      "y": (-0.05, 0.05),
      "z": (-0.01, 0.01),
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
      "yaw": (-0.2, 0.2),
    },
    velocity_range={
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "z": (-0.2, 0.2),
      "roll": (-0.52, 0.52),
      "pitch": (-0.52, 0.52),
      "yaw": (-0.78, 0.78),
    },
    joint_position_range=(-0.1, 0.1),
  )


@retval
def ADAM_SP_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG() -> ManagerBasedRlEnvCfg:
  """Create Adam_SP flat terrain tracking config without state estimation.

  This variant disables motion_anchor_pos_b and base_lin_vel observations,
  simulating the lack of state estimation.
  """
  cfg = deepcopy(ADAM_SP_FLAT_TRACKING_ENV_CFG)
  assert "policy" in cfg.observations
  cfg.observations["policy"].terms.pop("motion_anchor_pos_b")
  cfg.observations["policy"].terms.pop("base_lin_vel")
  return cfg
