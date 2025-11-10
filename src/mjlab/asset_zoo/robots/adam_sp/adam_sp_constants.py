"""Adam_SP robot configuration for mjlab."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import ActuatorCfg, CollisionCfg


##
# MJCF and assets.
##

ADAM_SP_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "adam_sp" / "xmls" / "adam_sp.xml" # no hand 
)
assert ADAM_SP_XML.exists()


def _get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, MJLAB_SRC_PATH / "asset_zoo" / "robots" / "adam_sp" / "xmls" / "meshes_stl_0.25" / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(ADAM_SP_XML))
  spec.assets = _get_assets(spec.meshdir)
  # The XML already defines actuators (motors). We manage actuators via
  # ADAM_SP_ARTICULATION below to ensure consistent gains/limits. Remove
  # existing actuators from the spec to avoid duplicate names on attach.
  try:
    spec.actuators.clear()  # type: ignore[attr-defined]
  except Exception:
    # Fallback: attempt reassignment if .clear() not available.
    try:
      spec.actuators = []  # type: ignore[attr-defined]
    except Exception:
      pass
  return spec


##
# Actuator config.
##

NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0


def _kd_from_armature(armature: float, w_n: float = NATURAL_FREQ, z: float = DAMPING_RATIO) -> tuple[float, float]:
  stiffness = armature * (w_n**2)
  damping = 2.0 * z * armature * w_n
  return stiffness, damping


ARMATURE_PND130_92 = 0.13426
ARMATURE_PND80_20 = 0.281573
ARMATURE_PND60_17 = 0.23409
ARMATURE_PND50_14 = 0.1578807
ARMATURE_PND50_52 = 0.0549
ARMATURE_PND30_14 = 0.0423963

STIFF_PND130_92, DAMP_PND130_92 = _kd_from_armature(ARMATURE_PND130_92)
STIFF_PND80_20, DAMP_PND80_20 = _kd_from_armature(ARMATURE_PND80_20)
STIFF_PND60_17, DAMP_PND60_17 = _kd_from_armature(ARMATURE_PND60_17)
STIFF_PND50_14, DAMP_PND50_14 = _kd_from_armature(ARMATURE_PND50_14)
STIFF_PND50_52, DAMP_PND50_52 = _kd_from_armature(ARMATURE_PND50_52)
STIFF_PND30_14, DAMP_PND30_14 = _kd_from_armature(ARMATURE_PND30_14)

HIP_PITCH_ACTUATORS = ActuatorCfg(
  joint_names_expr=["hipPitch_Left", "hipPitch_Right"],
  effort_limit=230.0,
  stiffness=STIFF_PND130_92,
  damping=DAMP_PND130_92,
  armature=ARMATURE_PND130_92,
)

HIP_ROLL_ACTUATORS = ActuatorCfg(
  joint_names_expr=["hipRoll_Left", "hipRoll_Right"],
  effort_limit=160.0,
  stiffness=STIFF_PND80_20,
  damping=DAMP_PND80_20,
  armature=ARMATURE_PND80_20,
)

HIP_YAW_ACTUATORS = ActuatorCfg(
  joint_names_expr=["hipYaw_Left", "hipYaw_Right"],
  effort_limit=105.0,
  stiffness=STIFF_PND60_17,
  damping=DAMP_PND60_17,
  armature=ARMATURE_PND60_17,
)

KNEE_ACTUATORS = ActuatorCfg(
  joint_names_expr=["kneePitch_Left", "kneePitch_Right"],
  effort_limit=230.0,
  stiffness=STIFF_PND130_92,
  damping=DAMP_PND130_92,
  armature=ARMATURE_PND130_92,
)

ANKLE_PITCH_ACTUATORS = ActuatorCfg(
  joint_names_expr=["anklePitch_Left", "anklePitch_Right"],
  effort_limit=40.0,
  stiffness=STIFF_PND50_52,
  damping=DAMP_PND50_52,
  armature=ARMATURE_PND50_52,
)

ANKLE_ROLL_ACTUATORS = ActuatorCfg(
  joint_names_expr=["ankleRoll_Left", "ankleRoll_Right"],
  effort_limit=12.0,
  stiffness=STIFF_PND50_52,
  damping=DAMP_PND50_52,
  armature=ARMATURE_PND50_52,
)

WAIST_ACTUATORS = ActuatorCfg(
  joint_names_expr=["waistRoll", "waistPitch", "waistYaw"],
  effort_limit=110.0,
  stiffness=STIFF_PND60_17,
  damping=DAMP_PND60_17,
  armature=ARMATURE_PND60_17,
)

SHOULDER_PITCH_ACTUATORS = ActuatorCfg(
  joint_names_expr=["shoulderPitch_Left", "shoulderPitch_Right"],
  effort_limit=65.0,
  stiffness=STIFF_PND50_14,
  damping=DAMP_PND50_14,
  armature=ARMATURE_PND50_14,
)

SHOULDER_ROLL_ACTUATORS = ActuatorCfg(
  joint_names_expr=["shoulderRoll_Left", "shoulderRoll_Right"],
  effort_limit=65.0,
  stiffness=STIFF_PND50_14,
  damping=DAMP_PND50_14,
  armature=ARMATURE_PND50_14,
)

SHOULDER_YAW_ACTUATORS = ActuatorCfg(
  joint_names_expr=["shoulderYaw_Left", "shoulderYaw_Right"],
  effort_limit=65.0,
  stiffness=STIFF_PND30_14,
  damping=DAMP_PND30_14,
  armature=ARMATURE_PND30_14,
)

ELBOW_ACTUATORS = ActuatorCfg(
  joint_names_expr=["elbow_Left", "elbow_Right"],
  effort_limit=30.0,
  stiffness=STIFF_PND30_14,
  damping=DAMP_PND30_14,
  armature=ARMATURE_PND30_14,
)


##
# Initial state and collisions.
##

ADAM_SP_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.95),
  joint_pos={
    "hipPitch_Left": -0.334,
    "hipRoll_Left": 0.045,
    "hipYaw_Left": -0.023,
    "kneePitch_Left": 0.766,
    "anklePitch_Left": -0.490,
    "ankleRoll_Left": -0.033,
    "hipPitch_Right": -0.334,
    "hipRoll_Right": -0.045,
    "hipYaw_Right": 0.023,
    "kneePitch_Right": 0.766,
    "anklePitch_Right": -0.490,
    "ankleRoll_Right": 0.033,
    "waistRoll": 0.0,
    "waistPitch": 0.0,
    "waistYaw": 0.0,
    "shoulderPitch_Left": 0.0,
    "shoulderRoll_Left": 0.0,
    "shoulderYaw_Left": 0.0,
    "elbow_Left": -0.3,
    "shoulderPitch_Right": 0.0,
    "shoulderRoll_Right": -0.0,
    "shoulderYaw_Right": 0.0,
    "elbow_Right": -0.3,
  },
  joint_vel={".*": 0.0},
)

FOOT_COLLISION = CollisionCfg(
  geom_names_expr=["toe_left", "toe_right"],
  condim=3,
  friction=(0.6,),
  disable_other_geoms=False,
)


##
# Final config.
##

ADAM_SP_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    HIP_PITCH_ACTUATORS,
    HIP_ROLL_ACTUATORS,
    HIP_YAW_ACTUATORS,
    KNEE_ACTUATORS,
    ANKLE_PITCH_ACTUATORS,
    ANKLE_ROLL_ACTUATORS,
    WAIST_ACTUATORS,
    SHOULDER_PITCH_ACTUATORS,
    SHOULDER_ROLL_ACTUATORS,
    SHOULDER_YAW_ACTUATORS,
    ELBOW_ACTUATORS,
  ),
  soft_joint_pos_limit_factor=0.9,
)

ADAM_SP_ROBOT_CFG = EntityCfg(
  init_state=ADAM_SP_INIT_STATE,
  collisions=(FOOT_COLLISION,),
  spec_fn=get_spec,
  articulation=ADAM_SP_ARTICULATION,
)

ADAM_SP_ACTION_SCALE: dict[str, float] = {}
for actuator_cfg in ADAM_SP_ARTICULATION.actuators:
  effort = actuator_cfg.effort_limit
  stiffness = actuator_cfg.stiffness
  joint_patterns = actuator_cfg.joint_names_expr
  effort_map = {name: effort for name in joint_patterns}
  stiffness_map = {name: stiffness for name in joint_patterns}
  for joint_pattern in joint_patterns:
    stiffness_value = stiffness_map[joint_pattern]
    if stiffness_value:
      ADAM_SP_ACTION_SCALE[joint_pattern] = 0.25 * effort_map[joint_pattern] / stiffness_value


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  entity = Entity(ADAM_SP_ROBOT_CFG)
  viewer.launch(entity.spec.compile())

