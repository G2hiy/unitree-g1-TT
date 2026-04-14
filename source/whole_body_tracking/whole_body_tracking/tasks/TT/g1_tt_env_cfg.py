import isaaclab.envs.mdp as isaac_mdp
import isaaclab.sim as sim_utils

import whole_body_tracking.tasks.TT.mdp.events as event_mdp
import whole_body_tracking.tasks.TT.mdp.observations as obs_mdp
import whole_body_tracking.tasks.TT.mdp.rewards as rew_mdp
import whole_body_tracking.tasks.TT.mdp.terminations as term_mdp

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from whole_body_tracking.assets.table_tennis.ball import BALL_CFG
from whole_body_tracking.assets.table_tennis.table import TABLE_CFG
from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.TT.mdp.commands import MotionCommandCfg


@configclass
class TTSceneCfg(InteractiveSceneCfg):
    num_envs: int = 4096
    env_spacing: float = 5.0

    # 在class TTObservationsCfg里面进行注册
    robot: ArticulationCfg = MISSING
    table: RigidObjectCfg = MISSING
    ball: RigidObjectCfg = MISSING

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    # 地面配置
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

    # 小球的mark
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
    paddle_offset: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PaddleOffset",
        spawn=sim_utils.SphereCfg(
            radius=0.025,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 1.0), metallic=0.0, roughness=0.5),
        ),
    )

    # 接触传感器
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=0.0,
    )


@configclass
class TTCommandsCfg:

    # 正手参考动作
    motion_forehand: MotionCommandCfg = MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.0, 0.0),
        motion_file="/home/u20-1/gzy/whole_body_tracking/artifacts/fore_hand:v2/motion.npz",
        anchor_body_name="pelvis",
        body_names=[
            "pelvis",
            "torso_link",
            # "left_hip_roll_link",
            # "left_knee_link",
            # "left_ankle_roll_link",
            # "right_hip_roll_link",
            # "right_knee_link",
            # "right_ankle_roll_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
        start_frame=0,
        num_frames=94,  # 模仿视频一共有94帧
        phase_from_env=True,
        enable_resample=False,
        debug_swing_value=0,
        debug_vis_tag="forehand",
        debug_vis=True,
    )

    # 反手参考动作
    motion_backhand: MotionCommandCfg = MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.0, 0.0),
        motion_file="/home/u20-1/gzy/whole_body_tracking/artifacts/back_hand:v2/motion.npz",
        anchor_body_name="pelvis",
        body_names=[
            "pelvis",
            "torso_link",
            # "left_hip_roll_link",
            # "left_knee_link",
            # "left_ankle_roll_link",
            # "right_hip_roll_link",
            # "right_knee_link",
            # "right_ankle_roll_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_yaw_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_yaw_link",
        ],
        start_frame=0,
        num_frames=94,
        phase_from_env=True,
        enable_resample=False,
        debug_swing_value=1,
        debug_vis_tag="backhand",
        debug_vis=True,
    )


@configclass
class TTObservationsCfg:

    # 这里的观测设计是不是和原文一样的？
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
        # base_heading = ObsTerm(func=obs_mdp.get_robot_heading_vector)
        # base_pos_xy = ObsTerm(func=obs_mdp.get_base_pos_xy)
        ball_linvel = ObsTerm(func=obs_mdp.get_ball_linvel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    predictor: PredictorCfg = PredictorCfg()

@configclass
class TTRewardsTaskCfg:
    # ---------------- r_g：稀疏但高权重的目标追踪（HITTER Eq.7） ----------------
    # Phase 1：训练阶段不依赖真球，删除 contact / hit_forward / future_pass_net /
    # table_success；sparse 奖励由下方 strike_terminal_* 三项在 t_strike→0 瞬间给出。

    track_base_target = RewTerm(
        func=rew_mdp.reward_track_base_target,
        weight=20.0,
        params={"std": 0.3, "t_min": 0.12, "t_max": 0.6, "center": 0.32, "width": 0.16},
    )

    # 挥拍窗口内 racket 追踪（dense，但 cutoff 放宽到 0.04s）
    track_racket_pos = RewTerm(
        func=rew_mdp.reward_track_racket_pos,
        weight=40.0,
        params={"std": 0.1, "center": 0.0, "cutoff": 0.04},
    )
    track_racket_vel = RewTerm(
        func=rew_mdp.reward_track_racket_vel,
        weight=45.0,
        params={"std": 0.5, "center": 0.0, "width": 0.04, "cutoff": 0.04},
    )
    track_racket_ori = RewTerm(
        func=rew_mdp.reward_track_racket_ori,
        weight=30.0,
        params={"std": 0.2, "center": 0.0, "width": 0.04, "cutoff": 0.04},
    )

    track_racket_prepose = RewTerm(
        func=rew_mdp.reward_track_racket_prepose,
        weight=25.0,
        params={"std": 0.25, "t_min": 0.12, "t_max": 0.40, "center": 0.25, "width": 0.10},
    )

    # t_strike→0 瞬间的 sparse 大奖励（一步有效）
    strike_terminal_pos = RewTerm(func=rew_mdp.reward_strike_terminal_pos, weight=80.0, params={"std": 0.10})
    strike_terminal_vel = RewTerm(func=rew_mdp.reward_strike_terminal_vel, weight=80.0, params={"std": 1.0})
    strike_terminal_ori = RewTerm(func=rew_mdp.reward_strike_terminal_ori, weight=40.0, params={"std": 0.4})


    motion_global_anchor_pos = RewTerm(
        func=rew_mdp.motion_global_anchor_position_error_exp,
        weight=2.0, 
        params={"std": 0.4, "t_min": 0.35, "t_max": 2.0, "center": 0.6, "width": 0.3}, # t_min 提高到 0.35
    )
    motion_global_anchor_ori = RewTerm(
        func=rew_mdp.motion_global_anchor_orientation_error_exp,
        weight=2.0, 
        params={"std": 0.5, "t_min": 0.35, "t_max": 2.0, "center": 0.6, "width": 0.3},
    )
    motion_body_pos = RewTerm(
        func=rew_mdp.motion_relative_body_position_error_exp,
        weight=6.0, 
        params={
            "std": 0.35,
            "body_names": ["torso_link", "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"],
            "t_min": 0.35, "t_max": 2.0, "center": 0.6, "width": 0.3,
        },
    )
    motion_body_ori = RewTerm(
        func=rew_mdp.motion_relative_body_orientation_error_exp,
        weight=3.0, 
        params={
            "std": 0.45,
            "body_names": ["torso_link", "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"],
            "t_min": 0.35, "t_max": 2.0, "center": 0.6, "width": 0.3,
        },
    )
    motion_body_lin_vel = RewTerm(
        func=rew_mdp.motion_global_body_linear_velocity_error_exp,
        weight=4.0, 
        params={
            "std": 1.2,
            "body_names": ["right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"],
            "t_min": 0.35, "t_max": 2.0, "center": 0.6, "width": 0.3,
        },
    )
    motion_body_ang_vel = RewTerm(
        func=rew_mdp.motion_global_body_angular_velocity_error_exp,
        weight=2.0, 
        params={
            "std": 3.5,
            "body_names": ["right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"],
            "t_min": 0.35, "t_max": 2.0, "center": 0.6, "width": 0.3,
        },
    )

    lin_vel_z_l2 = RewTerm(func=rew_mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=rew_mdp.ang_vel_xy_l2, weight=-0.05)
    ang_vel_z_l2 = RewTerm(func=rew_mdp.ang_vel_z_l2, weight=-0.02)
    action_rate_l2 = RewTerm(func=rew_mdp.action_rate_l2, weight=-2e-4)
    dof_pos_limits = RewTerm(func=isaac_mdp.joint_pos_limits, weight=-1.0)
    joint_limit = RewTerm(
        func=isaac_mdp.joint_pos_limits,
        weight=-4.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=rew_mdp.undesired_contacts,
        weight=-2.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!right_wrist_yaw_link$).+$",
            ),
            "threshold": 1.0,
        },
    )
    flat_orientation_l2 = RewTerm(func=rew_mdp.flat_orientation_l2, weight=-2.0)
    termination_penalty = RewTerm(func=rew_mdp.is_terminated, weight=-5.0)


@configclass
class TTEventsCfg:
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
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["pelvis"]), 
                "mass_distribution_params": (-0.5, 0.5), 
                "operation": "add"
        },
    )
    reset_base = EventTerm(
        func=isaac_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (-0.1, 0.1), 
                "y": (-0.1, 0.1), 
                "yaw": (-0.1, 0.1),
            },
        },
    )
    reset_all_joints = EventTerm(
        func=isaac_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0), 
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), 
        },
    )


@configclass
class TTTerminationsCfg:
    time_out = DoneTerm(func=isaac_mdp.time_out, time_out=True)
    
    fallen_tilt = DoneTerm(
        func=term_mdp.bad_upright_orientation,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.9},
    )
    
    root_height = DoneTerm(
        func=isaac_mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.4},
    )


@configclass
class TTActionsCfg:
    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        use_default_offset=True,
    )


@configclass
class TableTennisEnvCfg(ManagerBasedRLEnvCfg):
    scene: TTSceneCfg = TTSceneCfg(num_envs=4096, env_spacing=5.0)
    actions: TTActionsCfg = TTActionsCfg()
    observations: TTObservationsCfg = TTObservationsCfg()
    rewards: TTRewardsTaskCfg = TTRewardsTaskCfg()
    events: TTEventsCfg = TTEventsCfg()
    terminations: TTTerminationsCfg = TTTerminationsCfg()
    commands: TTCommandsCfg = TTCommandsCfg()
    planner_use_prediction: bool = True
    swing_decision_margin: float = 0.04
    swing_random_prob: float = 0.15
    swing_lock_t_min: float = 0.05
    swing_lock_t_max: float = 0.75
    planner_command_alpha: float = 0.35
    planner_time_alpha: float = 0.6
    motion_clip_strike_time: float = 0.86
    motion_phase_alpha: float = 0.2
    motion_phase_max_step: float = 0.04
    motion_phase_max_time: float = 1.88

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 0.005
        self.episode_length_s = 10.0
        # self.decimation = 10         
        # self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.asset_name = "robot"
        self.sim.physx.enable_ccd = True
        self.sim.physx.bounce_threshold_velocity = 0.2

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.table = TABLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Table")
        self.scene.ball = BALL_CFG.replace(prim_path="{ENV_REGEX_NS}/Ball")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
