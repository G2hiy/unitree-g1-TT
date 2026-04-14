import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

from whole_body_tracking.assets import ASSET_DIR

cfg = UrdfConverterCfg(
    asset_path=f"{ASSET_DIR}/unitree_description/urdf/g1/main.urdf",
    fix_base=False,
    merge_fixed_joints=False,
    replace_cylinders_with_capsules=True,
    self_collision=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    ),
)

converter = UrdfConverter(cfg)
print("USD saved to:", converter.usd_path)
