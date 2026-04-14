from __future__ import annotations

from dataclasses import MISSING

import math
import torch
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.utils.noise import GaussianNoiseCfg
import isaaclab.sim as sim_utils
from dataclasses import MISSING

import isaaclab.envs.mdp as isaac_mdp

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommandCfg
import whole_body_tracking.tasks.tracking.mdp.observations as obs_mdp
import whole_body_tracking.tasks.tracking.mdp.rewards as rew_mdp
import whole_body_tracking.tasks.tracking.mdp.events as event_mdp
import whole_body_tracking.tasks.tracking.mdp.terminations as term_mdp

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG 
from whole_body_tracking.assets.table_tennis.table import TABLE_CFG
from whole_body_tracking.assets.table_tennis.ball import BALL_CFG

##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""
    env_spacing=5.0
    num_envs: int = 4096
    
    robot: ArticulationCfg = MISSING
    table: RigidObjectCfg = MISSING
    ball: RigidObjectCfg = MISSING
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    ball_future: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BallFuture",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.0, roughness=0.5),
        ),
    )
    ball_pred: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BallPred",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.7, 0.0), metallic=0.0, roughness=0.5),
        ),
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=0.0,
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    motion_forehand = MotionCommandCfg(
        asset_name="robot",
        motion_file="/home/u20-1/gzy/whole_body_tracking/artifacts/fore_hand:v2/motion.npz",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        body_names=[
            "pelvis",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
    )
    


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = isaac_mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel, noise=GaussianNoiseCfg(mean=0.0, std=0.1))
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity, noise=GaussianNoiseCfg(mean=0.0, std=0.05))
        base_heading = ObsTerm(func=obs_mdp.get_robot_heading_vector)
        base_target_xy = ObsTerm(func=obs_mdp.get_base_target_xy)
        racket_target_pos = ObsTerm(func=obs_mdp.get_racket_target_pos)
        racket_target_vel = ObsTerm(func=obs_mdp.get_racket_target_vel)
        t_strike = ObsTerm(func=obs_mdp.get_t_strike)
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel, noise=GaussianNoiseCfg(mean=0.0, std=0.01))
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel, noise=GaussianNoiseCfg(mean=0.0, std=0.1))
        actions = ObsTerm(func=isaac_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        base_ang_vel = ObsTerm(func=isaac_mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=isaac_mdp.projected_gravity)
        base_heading = ObsTerm(func=obs_mdp.get_robot_heading_vector)
        base_target_xy = ObsTerm(func=obs_mdp.get_base_target_xy)
        racket_target_pos = ObsTerm(func=obs_mdp.get_racket_target_pos)
        racket_target_vel = ObsTerm(func=obs_mdp.get_racket_target_vel)
        t_strike = ObsTerm(func=obs_mdp.get_t_strike)
        joint_pos = ObsTerm(func=isaac_mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=isaac_mdp.joint_vel_rel)
        actions = ObsTerm(func=isaac_mdp.last_action)
        base_lin_vel = ObsTerm(func=isaac_mdp.base_lin_vel)
        upper_body_poses = ObsTerm(
            func=obs_mdp.get_upper_body_poses,
            params={
                "body_names": [
                    "pelvis",
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ]
            },
        )
        t_left = ObsTerm(func=obs_mdp.get_time_left)
        ref_q_qd = ObsTerm(func=obs_mdp.get_reference_joint_pos_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PredictorCfg(ObservationGroupCfg):
        ball_history = ObsTerm(func=obs_mdp.get_ball_trajectory_history)
        base_heading = ObsTerm(func=obs_mdp.get_robot_heading_vector)
        base_pos_xy = ObsTerm(func=obs_mdp.get_base_pos_xy)
        ball_linvel = ObsTerm(func=obs_mdp.get_ball_linvel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    predictor: PredictorCfg = PredictorCfg()


@configclass
class EventsCfg:
    physics_material = EventTerm(
        func=isaac_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.5, 0.8),
            "restitution_range": (0.0, 0.005),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=isaac_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["pelvis"]), "mass_distribution_params": (-0.5, 0.5), "operation": "add"},
    )
    reset_base = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"y": (0.0, 0.0),  "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "roll": (0.0, 0.0),  
                "pitch": (0.0, 0.0), 
                "yaw": (-0.1, 0.1),
            },
        },
    )
    reset_all_joints = EventTerm(
        func=isaac_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0), 
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), 
        },
    )


@configclass
class RewardsTaskCfg:
    
    contact = RewTerm(
        func=rew_mdp.reward_contact, 
        weight=100.0  # 100.0
    )
    

    hit_forward = RewTerm(
        func=rew_mdp.reward_hit_forward, 
        weight=10.0 # 10.0
    )
    

    # future_pass_net = RewTerm(
    #     func=rew_mdp.reward_future_pass_net, 
    #     weight=20.0,    # 20.0
    #     params={"std_h": 0.06, "z_target": 1.0}
    # )
    

    table_success = RewTerm(
        func=rew_mdp.reward_table_success, 
        weight=200.0    # 200.0
    )
    
    track_base_target = RewTerm(
        func=rew_mdp.reward_track_base_target,
        weight=20.0,    # 20.0
        params={"std": 0.3, "t_min": 0.12, "t_max": 0.6, "center": 0.32, "width": 0.16},
    )
    
    track_racket_pos = RewTerm(
        func=rew_mdp.reward_track_racket_pos,
        weight=40.0,
        params={"std": 0.1, "center": 0.0, "cutoff": 0.02}, # 
    )
    track_racket_vel = RewTerm(
        func=rew_mdp.reward_track_racket_vel,
        weight=45.0, 
        params={"std": 0.5, "center": 0.0, "cutoff": 0.10},
    )
    track_racket_ori = RewTerm(
        func=rew_mdp.reward_track_racket_ori,
        weight=30.0,
        params={"std": 0.2, "center": 0.0, "cutoff": 0.10},
    )
    
    track_racket_prepose = RewTerm(
        func=rew_mdp.reward_track_racket_prepose,
        weight=15.0,    # 15.0
        params={"std": 0.25, "t_min": 0.10, "t_max": 0.36, "center": 0.20, "width": 0.07},
    )

    motion_global_anchor_pos = RewTerm(
        func=rew_mdp.motion_global_anchor_position_error_exp,
        weight=0.1,
        params={"command_name": "motion_forehand", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=rew_mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion_forehand", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=rew_mdp.motion_relative_body_position_error_exp,
        weight=0.5,
        params={"command_name": "motion_forehand", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=rew_mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion_forehand", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=rew_mdp.motion_global_body_linear_velocity_error_exp,
        weight=0.5,
        params={"command_name": "motion_forehand", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=rew_mdp.motion_global_body_angular_velocity_error_exp,
        weight=0.5,
        params={"command_name": "motion_forehand", "std": 3.14},
    )


    action_rate_l2 = RewTerm(func=isaac_mdp.action_rate_l2, weight=-2e-4)
    
    joint_limit = RewTerm(
        func=isaac_mdp.joint_pos_limits,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=isaac_mdp.undesired_contacts,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!right_wrist_yaw_link$).+$",
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    
    anchor_pos = DoneTerm(
        func=term_mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion_forehand", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=term_mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion_forehand", "threshold": 0.8},
    )
    ee_body_pos = None


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsTaskCfg = RewardsTaskCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    planner_use_prediction = True

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.asset_name = "robot"
        
        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.table = TABLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Table")
        self.scene.ball = BALL_CFG.replace(prim_path="{ENV_REGEX_NS}/Ball")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion_forehand.anchor_body_name = "pelvis"
        self.commands.motion_forehand.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ]

