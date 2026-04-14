from isaaclab.assets import RigidObject
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from whole_body_tracking.tasks.tracking.envs.tt_env import TableTennisEnv
    
def reset_ball_serve(
    env: "TableTennisEnv", 
    env_ids: torch.Tensor, 
    speed_x_range: tuple[float, float], 
    speed_y_range: tuple[float, float], 
    speed_z_range: tuple[float, float], 
    pos_y_range: tuple[float, float], 
    asset_cfg: SceneEntityCfg | RigidObject = SceneEntityCfg("ball"),
):
    
    if isinstance(asset_cfg, SceneEntityCfg):
        asset = env.scene[asset_cfg.name]
    else:
        asset = asset_cfg
    
    # 提取默认的初始状态
    root_states = asset.data.default_root_state[env_ids].clone()
    
    # 获取刚刚分配的挥拍类型 (0: 正手, 1: 反手)
    swing_type = env.swing_type[env_ids]
    
    # 初始化 Y 轴落点和速度数组
    pos_y = torch.zeros(len(env_ids), device=env.device)
    vel_y = torch.zeros(len(env_ids), device=env.device)
    
    # ==========================================
    # 🌟 互不重叠的发球分区 (Non-overlapping Zones)
    # ==========================================
    idx_fh = (swing_type == 0) # 正手 (通常接身体右侧球, Y < 0)
    if idx_fh.any():
        num_fh = idx_fh.sum()
        # 喂给机器人的偏右侧 (-0.4 到 -0.1)
        pos_y[idx_fh] = torch.empty(num_fh, device=env.device).uniform_(-0.4, -0.1)
        vel_y[idx_fh] = torch.empty(num_fh, device=env.device).uniform_(-0.5, 0.0)
        
    idx_bh = (swing_type == 1) # 反手 (通常接身体左侧球, Y > 0)
    if idx_bh.any():
        num_bh = idx_bh.sum()
        # 喂给机器人的偏左侧 (0.1 到 0.4)
        pos_y[idx_bh] = torch.empty(num_bh, device=env.device).uniform_(0.1, 0.4)
        vel_y[idx_bh] = torch.empty(num_bh, device=env.device).uniform_(0.0, 0.5)

    # 写入位置
    root_states[:, 0] = 1.37  # 对方球台边缘发球
    root_states[:, 1] = pos_y
    root_states[:, 2] = 0.90  # 发球高度
    root_states[:, :3] += env.scene.env_origins[env_ids]
    
    # 写入速度
    root_states[:, 7] = torch.empty(len(env_ids), device=env.device).uniform_(speed_x_range[0], speed_x_range[1])
    root_states[:, 8] = vel_y
    root_states[:, 9] = torch.empty(len(env_ids), device=env.device).uniform_(speed_z_range[0], speed_z_range[1])

    root_states[:, 10:13] = 0.0
    
    # 将状态应用到仿真器
    asset.write_root_pose_to_sim(root_states[:, :7], env_ids)
    asset.write_root_velocity_to_sim(root_states[:, 7:], env_ids)
    
    
    
    
    