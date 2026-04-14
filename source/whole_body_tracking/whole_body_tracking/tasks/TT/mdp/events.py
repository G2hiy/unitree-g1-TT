from isaaclab.actuators.actuator_cfg import Literal
from isaaclab.assets import Articulation, RigidObject
import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from whole_body_tracking.tasks.TT.envs.tt_env import TableTennisEnv
    
def reset_ball_serve(
    env: "TableTennisEnv",
    env_ids: torch.Tensor,
    speed_x_range: tuple[float, float],
    speed_y_range: tuple[float, float],
    speed_z_range: tuple[float, float],
    pos_y_range: tuple[float, float],
    asset_cfg: SceneEntityCfg | RigidObject = SceneEntityCfg("ball"),
):
    """
    在环境重置或触发异步发球时，随机初始化乒乓球的起始位置和初速度。
    完全复刻原项目 tt_env.py 中的随机发球逻辑。
    """
    # 1. 提取资产引用
    if isinstance(asset_cfg, SceneEntityCfg):
        asset = env.scene[asset_cfg.name]
    else:
        asset = asset_cfg
    
    # 2. 拷贝球的默认根状态 (这包含了在资产配置中设定的初始相对位置和四元数姿态)
    ball_state = asset.data.default_root_state[env_ids].clone()
    
    # 3. 转换到世界坐标系 (加上每个并行环境各自的原点偏移)
    ball_state[:, :3] += env.scene.env_origins[env_ids]
    
    # 4. 随机化位置 (仅对 Y 轴施加扰动，X和Z由默认状态保证)
    pos_y_noise = torch.empty(len(env_ids), device=env.device).uniform_(*pos_y_range)
    ball_state[:, 1] += pos_y_noise
    
    # 5. 随机化线速度 (X, Y, Z 三个方向独立采样)
    v_x = torch.empty(len(env_ids), device=env.device).uniform_(*speed_x_range)
    v_y = torch.empty(len(env_ids), device=env.device).uniform_(*speed_y_range)
    v_z = torch.empty(len(env_ids), device=env.device).uniform_(*speed_z_range)
    
    # 根状态(Root State)的 13 维结构为: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
    # 更新线速度 (索引 7, 8, 9)
    ball_state[:, 7] = v_x
    ball_state[:, 8] = v_y
    ball_state[:, 9] = v_z
    
    # 发球初始设定为不带旋转（角速度归零）
    ball_state[:, 10:13] = 0.0

    # 6. 将全新的状态下发回 Isaac Sim 仿真引擎
    asset.write_root_state_to_sim(ball_state, env_ids)
    
# def reset_ball_serve(
#     env: "TableTennisEnv", 
#     env_ids: torch.Tensor, 
#     speed_x_range: tuple[float, float], 
#     speed_y_range: tuple[float, float], 
#     speed_z_range: tuple[float, float], 
#     pos_y_range: tuple[float, float], 
#     asset_cfg: SceneEntityCfg | RigidObject = SceneEntityCfg("ball"),
# ):
    
#     if isinstance(asset_cfg, SceneEntityCfg):
#         asset = env.scene[asset_cfg.name]
#     else:
#         asset = asset_cfg
    
#     # 提取默认的初始状态
#     root_states = asset.data.default_root_state[env_ids].clone()
    
#     # 获取刚刚分配的挥拍类型 (0: 正手, 1: 反手)
#     swing_type = env.serve_swing_type[env_ids]
    
#     # 初始化 Y 轴落点和速度数组
#     pos_y = torch.zeros(len(env_ids), device=env.device)
#     vel_y = torch.zeros(len(env_ids), device=env.device)
    
#     # ==========================================
#     # 🌟 互不重叠的发球分区 (Non-overlapping Zones)
#     # ==========================================
#     idx_fh = (swing_type == 0) # 正手 (通常接身体右侧球, Y < 0)
#     if idx_fh.any():
#         num_fh = idx_fh.sum()
#         # 喂给机器人的偏右侧 (-0.4 到 -0.1)
#         pos_y[idx_fh] = torch.empty(num_fh, device=env.device).uniform_(-0.4, -0.1)
#         vel_y[idx_fh] = torch.empty(num_fh, device=env.device).uniform_(-0.5, 0.0)
        
#     idx_bh = (swing_type == 1) # 反手 (通常接身体左侧球, Y > 0)
#     if idx_bh.any():
#         num_bh = idx_bh.sum()
#         # 喂给机器人的偏左侧 (0.1 到 0.4)
#         pos_y[idx_bh] = torch.empty(num_bh, device=env.device).uniform_(0.1, 0.4)
#         vel_y[idx_bh] = torch.empty(num_bh, device=env.device).uniform_(0.0, 0.5)

#     # 写入位置
#     root_states[:, 0] = 1.37  # 对方球台边缘发球
#     root_states[:, 1] = pos_y
#     root_states[:, 2] = 0.90  # 发球高度
#     root_states[:, :3] += env.scene.env_origins[env_ids]
    
#     # 写入速度
#     root_states[:, 7] = torch.empty(len(env_ids), device=env.device).uniform_(speed_x_range[0], speed_x_range[1])
#     root_states[:, 8] = vel_y
#     root_states[:, 9] = torch.empty(len(env_ids), device=env.device).uniform_(speed_z_range[0], speed_z_range[1])

#     root_states[:, 10:13] = 0.0
    
#     # 将状态应用到仿真器
#     asset.write_root_pose_to_sim(root_states[:, :7], env_ids)
#     asset.write_root_velocity_to_sim(root_states[:, 7:], env_ids)


def randomize_joint_default_pos(
    env: "TableTennisEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos
        
def randomize_rigid_body_com(
    env: "TableTennisEnv",
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)