# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple
import math

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from whole_body_tracking.tasks.TT.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from whole_body_tracking.tasks.TT.envs.tt_env import TableTennisEnv
    
    
def _time_window_weight(t_strike: torch.Tensor, center: float, width: float) -> torch.Tensor:
    width = max(float(width), 1e-6)
    return torch.exp(-((t_strike - float(center)) ** 2) / (2.0 * width * width))


def _time_interval_mask(t_strike: torch.Tensor, t_min: float, t_max: float) -> torch.Tensor:
    return (t_strike > float(t_min)) & (t_strike < float(t_max))

def reward_track_racket_ori(
    env: "TableTennisEnv", 
    std: float = 0.5,  
    center: float = 0.0, 
    width: float = 0.06, 
    cutoff: float = 0.12
) -> torch.Tensor:
    """基于时间窗口的球拍姿态追踪奖励"""
    current_quat = env.paddle_quat  
    target_quat = env.racket_cmd[:, 3:7]
    
    from isaaclab.utils.math import quat_error_magnitude
    err_mag = quat_error_magnitude(current_quat, target_quat)

    reward = torch.exp(-err_mag / std)

    t_strike = env.t_strike.squeeze(-1)
    mask = torch.abs(t_strike - float(center)) < float(cutoff)
    
    return torch.where(mask, reward, torch.zeros_like(reward))

def reward_track_racket_pos(
    env: "TableTennisEnv",
    std: float = 0.4,
    center: float = 0.0,
    cutoff: float = 0.12,
) -> torch.Tensor:
    """Track racket position tightly around strike."""
    target = env.racket_cmd[:, 0:3]
    pos = env.paddle_pos
    
    dist = torch.linalg.norm(pos - target, dim=1)

    reward = torch.exp(-dist / std) 

    t_strike = env.t_strike.squeeze(-1)
    mask = torch.abs(t_strike - float(center)) <= float(cutoff)
    
    return torch.where(mask, reward, torch.zeros_like(reward))

def reward_track_racket_vel(
    env: "TableTennisEnv",
    std: float = 0.4,
    center: float = 0.0,
    width: float = 0.06,
    cutoff: float = 0.12,
) -> torch.Tensor:
    """Track racket velocity tightly around strike."""

    target = env.racket_cmd[:, 7:10] 
    vel = env.paddle_linvel
    
    dist = torch.linalg.norm(vel - target, dim=1)
    
    reward = torch.exp(-dist / std)
    
    t_strike = env.t_strike.squeeze(-1)
    mask = torch.abs(t_strike - float(center)) <= float(cutoff)
    
    return torch.where(mask, reward, torch.zeros_like(reward))


def _imitation_time_weight(
    env: "TableTennisEnv", t_min: float | None = None, t_max: float | None = None, center: float | None = None, width: float | None = None
) -> torch.Tensor:
    t_strike = env.t_strike.squeeze(-1)
    weight = torch.ones_like(t_strike)
    if t_min is not None and t_max is not None:
        weight = weight * _time_interval_mask(t_strike, t_min=t_min, t_max=t_max).float()
    if center is not None and width is not None:
        weight = weight * _time_window_weight(t_strike, center=center, width=width)
    return weight


def motion_global_anchor_position_error_exp(
    env: "TableTennisEnv",
    std: float,
    t_min: float | None = None,
    t_max: float | None = None,
    center: float | None = None,
    width: float | None = None,
) -> torch.Tensor:
    cmd_fh, cmd_bh = _dual_mode_commands(env)
    err_fh = torch.sum(torch.square(cmd_fh.anchor_pos_w - cmd_fh.robot_anchor_pos_w), dim=-1)
    err_bh = torch.sum(torch.square(cmd_bh.anchor_pos_w - cmd_bh.robot_anchor_pos_w), dim=-1)
    error = _select_by_swing(env, err_fh, err_bh)
    reward = 1.0 / (1.0 + error / (std**2)) 
    return reward * _imitation_time_weight(env, t_min=t_min, t_max=t_max, center=center, width=width)
  

def motion_global_anchor_orientation_error_exp(
    env: "TableTennisEnv",
    std: float,
    t_min: float | None = None,
    t_max: float | None = None,
    center: float | None = None,
    width: float | None = None,
) -> torch.Tensor:
    cmd_fh, cmd_bh = _dual_mode_commands(env)
    err_fh = math_utils.quat_error_magnitude(cmd_fh.anchor_quat_w, cmd_fh.robot_anchor_quat_w) ** 2
    err_bh = math_utils.quat_error_magnitude(cmd_bh.anchor_quat_w, cmd_bh.robot_anchor_quat_w) ** 2
    error = _select_by_swing(env, err_fh, err_bh)
    reward = 1.0 / (1.0 + error / (std**2))
    return reward * _imitation_time_weight(env, t_min=t_min, t_max=t_max, center=center, width=width)


def motion_relative_body_position_error_exp(
    env: "TableTennisEnv",
    std: float,
    body_names: list[str] | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    center: float | None = None,
    width: float | None = None,
) -> torch.Tensor:
    cmd_fh, cmd_bh = _dual_mode_commands(env)
    idx_fh = _get_body_indexes(cmd_fh, body_names)
    idx_bh = _get_body_indexes(cmd_bh, body_names)
    err_fh = torch.sum(
        torch.square(cmd_fh.body_pos_relative_w[:, idx_fh] - cmd_fh.robot_body_pos_w[:, idx_fh]), dim=-1
    )
    err_bh = torch.sum(
        torch.square(cmd_bh.body_pos_relative_w[:, idx_bh] - cmd_bh.robot_body_pos_w[:, idx_bh]), dim=-1
    )
    error = _select_by_swing(env, err_fh.mean(-1), err_bh.mean(-1))
    reward = 1.0 / (1.0 + error / (std**2))
    return reward * _imitation_time_weight(env, t_min=t_min, t_max=t_max, center=center, width=width)


def motion_relative_body_orientation_error_exp(
    env: "TableTennisEnv",
    std: float,
    body_names: list[str] | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    center: float | None = None,
    width: float | None = None,
) -> torch.Tensor:
    cmd_fh, cmd_bh = _dual_mode_commands(env)
    idx_fh = _get_body_indexes(cmd_fh, body_names)
    idx_bh = _get_body_indexes(cmd_bh, body_names)
    err_fh = (
        math_utils.quat_error_magnitude(cmd_fh.body_quat_relative_w[:, idx_fh], cmd_fh.robot_body_quat_w[:, idx_fh])
        ** 2
    )
    err_bh = (
        math_utils.quat_error_magnitude(cmd_bh.body_quat_relative_w[:, idx_bh], cmd_bh.robot_body_quat_w[:, idx_bh])
        ** 2
    )
    error = _select_by_swing(env, err_fh.mean(-1), err_bh.mean(-1))
    reward = 1.0 / (1.0 + error / (std**2))
    return reward * _imitation_time_weight(env, t_min=t_min, t_max=t_max, center=center, width=width)


def motion_global_body_linear_velocity_error_exp(
    env: "TableTennisEnv",
    std: float,
    body_names: list[str] | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    center: float | None = None,
    width: float | None = None,
) -> torch.Tensor:
    cmd_fh, cmd_bh = _dual_mode_commands(env)
    idx_fh = _get_body_indexes(cmd_fh, body_names)
    idx_bh = _get_body_indexes(cmd_bh, body_names)
    err_fh = torch.sum(
        torch.square(cmd_fh.body_lin_vel_w[:, idx_fh] - cmd_fh.robot_body_lin_vel_w[:, idx_fh]), dim=-1
    )
    err_bh = torch.sum(
        torch.square(cmd_bh.body_lin_vel_w[:, idx_bh] - cmd_bh.robot_body_lin_vel_w[:, idx_bh]), dim=-1
    )
    error = _select_by_swing(env, err_fh.mean(-1), err_bh.mean(-1))
    reward = 1.0 / (1.0 + error / (std**2))
    return reward * _imitation_time_weight(env, t_min=t_min, t_max=t_max, center=center, width=width)


def motion_global_body_angular_velocity_error_exp(
    env: "TableTennisEnv",
    std: float,
    body_names: list[str] | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    center: float | None = None,
    width: float | None = None,
) -> torch.Tensor:
    cmd_fh, cmd_bh = _dual_mode_commands(env)
    idx_fh = _get_body_indexes(cmd_fh, body_names)
    idx_bh = _get_body_indexes(cmd_bh, body_names)
    err_fh = torch.sum(
        torch.square(cmd_fh.body_ang_vel_w[:, idx_fh] - cmd_fh.robot_body_ang_vel_w[:, idx_fh]), dim=-1
    )
    err_bh = torch.sum(
        torch.square(cmd_bh.body_ang_vel_w[:, idx_bh] - cmd_bh.robot_body_ang_vel_w[:, idx_bh]), dim=-1
    )
    error = _select_by_swing(env, err_fh.mean(-1), err_bh.mean(-1))
    reward = 1.0 / (1.0 + error / (std**2))
    return reward * _imitation_time_weight(env, t_min=t_min, t_max=t_max, center=center, width=width)  

def lin_vel_z_l2(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])

def ang_vel_xy_l2(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

def ang_vel_z_l2(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])

# =========================================================
# 1. 速度与指令追踪 (Velocity & Command Tracking)
# =========================================================

def track_lin_vel_xy_yaw_frame_exp(
    env: "TableTennisEnv", std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)

def track_ang_vel_z_world_exp(
    env: "TableTennisEnv", std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def lin_vel_x_l2(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 0])

def lin_vel_y_l2(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])


# =========================================================
# 2. 动作与关节惩罚 (Action & Joint Penalties)
# =========================================================

def energy(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(
        torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]), 
        dim=-1
    )
    return reward

def joint_acc_l2(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)

def action_rate_l2(env: "TableTennisEnv") -> torch.Tensor:
    buffer = env.action_buffer._circular_buffer.buffer
    if buffer.shape[1] < 2:
        return torch.zeros(buffer.shape[0], device=buffer.device, dtype=buffer.dtype)
    return torch.sum(torch.square(buffer[:, -1, :] - buffer[:, -2, :]), dim=1)

def joint_deviation_l1(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)

def body_orientation_l2(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)

# =========================================================
# 3. 接触与安全约束 (Contact & Safety Constraints)
# =========================================================

def undesired_contacts(env: "TableTennisEnv", threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)

def fly(env: "TableTennisEnv", threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5

def feet_slide(env: "TableTennisEnv", sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def feet_stumble(env: "TableTennisEnv", sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )

def body_force(env: "TableTennisEnv", sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward

def feet_too_near_humanoid(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)

def paddle_too_near_humanoid(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 1
    asset: Articulation = env.scene[asset_cfg.name]
    link_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(link_pos[:, 0] - env.paddle_touch_point, dim=-1)
    return (threshold - distance).clamp(min=0)

def feet_too_high(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    excess_height = torch.clamp(feet_z - threshold, min=0.0)
    penalty = torch.sum(excess_height, dim=1) 
    return penalty

# =========================================================
# 4. 姿态与身体朝向惩罚 (Orientation & Posture)
# =========================================================

def flat_orientation_l2(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

def robot_px_l2(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0] - asset.data.default_root_state[:, 0])

def robot_heading_quad(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_heading: float=0.0) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.heading_w - target_heading)

def body_heading_quad(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_heading: float=0.0) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(body_quat)
    return torch.square(yaw - target_heading)

def body_heading_exp(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_heading: float=0.0, std: float=0.4) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(body_quat)
    return torch.exp(-torch.abs(yaw - target_heading) / std)

def body_pitch_exp(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target: float=0.0, std: float=0.4) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(body_quat)
    return torch.exp(-torch.abs(pitch - target) / std)

def is_terminated(env: "TableTennisEnv") -> torch.Tensor:
        """Penalize terminated episodes that don't correspond to episodic timeouts."""
        return env.reset_terminated.float()

# =========================================================
# 5. 步态辅助函数 (Gait Helper JIT Scripts)
# =========================================================

@torch.jit.script
def create_stance_mask(phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(-1).repeat(1, 2)
    stance_mask = torch.where(sin_pos >= 0, 1, 0)
    stance_mask[:, 1] = 1 - stance_mask[:, 1]
    stance_mask[torch.abs(sin_pos) < 0.1] = 1
    mask_2 = 1 - stance_mask
    mask_2[torch.abs(sin_pos) < 0.1] = 1
    return stance_mask, mask_2

@torch.jit.script
def compute_reward_reward_feet_contact_number(contacts: torch.Tensor, phase: torch.Tensor, pos_rw: float, neg_rw: float, command: torch.Tensor):
    stance_mask, mask_2 = create_stance_mask(phase)
    reward = torch.where(contacts == stance_mask, pos_rw, neg_rw)
    reward = torch.mean(reward, dim=1)
    reward *= torch.norm(command, dim=1) > 0.1
    return reward

@torch.jit.script
def compute_reward_foot_clearance_reward(com_z: torch.Tensor, standing_position_com_z: torch.Tensor, current_foot_z: torch.Tensor, target_height: float, std: float, tanh_mult: float, body_lin_vel_w: torch.Tensor, command: torch.Tensor):
    standing_height = com_z - standing_position_com_z
    standing_position_toe_roll_z = 0.0626 
    offset = (standing_height + standing_position_toe_roll_z).unsqueeze(-1)
    foot_z_target_error = torch.square((current_foot_z - (target_height + offset).repeat(1, 2)).clip(max=0.0))
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(body_lin_vel_w, dim=2))
    reward = foot_velocity_tanh * foot_z_target_error
    reward = torch.exp(-torch.sum(reward, dim=1) / std)
    reward *= torch.norm(command, dim=1) > 0.1
    return reward

@torch.jit.script
def height_target(t: torch.Tensor):
    a5, a4, a3, a2, a1, a0 = [9.6, 12.0, -18.8, 5.0, 0.1, 0.0]
    return a5 * t**5 + a4 * t**4 + a3 * t**3 + a2 * t**2 + a1 * t + a0

@torch.jit.script
def compute_reward_track_foot_height(com_z: torch.Tensor, standing_position_com_z: torch.Tensor, phase: torch.Tensor, foot_z: torch.Tensor, standing_position_toe_roll_z: float, std: float, command: torch.Tensor):
    standing_height = com_z - standing_position_com_z
    offset = standing_height + standing_position_toe_roll_z
    stance_mask, mask_2 = create_stance_mask(phase)
    swing_mask = 1 - stance_mask
    filt_foot = torch.where(swing_mask == 1, foot_z, torch.zeros_like(foot_z))
    phase_mod = torch.fmod(phase, 0.5)
    feet_z_target = height_target(phase_mod) + offset
    feet_z_value = torch.sum(filt_foot, dim=1)
    error = torch.square(feet_z_value - feet_z_target)
    reward = torch.exp(-error / std**2)
    reward *= torch.norm(command, dim=1) > 0.1
    return reward

@torch.compile
def bezier_curve(control_points, t):
    n = len(control_points) - 1 
    dim = control_points.shape[1] 
    curve_points = torch.zeros((t.shape[0], dim), dtype=control_points.dtype, device=t.device)
    for k in range(n + 1):
        binomial_coeff = math.comb(n, k)
        bernstein_poly = binomial_coeff * (t**k) * ((1 - t) ** (n - k))
        curve_points += bernstein_poly.unsqueeze(1) * control_points[k]
    return curve_points

@torch.compile
def desired_height(phase, starting_foot):
    n_envs = phase.shape[0]
    desired_heights = torch.zeros((n_envs, 2), dtype=phase.dtype, device=phase.device)
    L = 0.5 
    H = 0.15 
    control_points_swing = torch.tensor([[0.0, 0.0], [0.3 * L, 0.1 * H], [0.6 * L, H], [L, 0.0]], dtype=torch.float32, device=phase.device)

    for leg in [0, 1]:
        is_starting_leg = starting_foot == leg
        is_other_leg = ~is_starting_leg
        swing_mask_starting_leg = is_starting_leg & (phase >= 0.02) & (phase < 0.5)
        t_swing_starting = (phase[swing_mask_starting_leg] - 0.02) / 0.48
        swing_mask_other_leg = is_other_leg & (phase >= 0.52) & (phase < 1.0)
        t_swing_other = (phase[swing_mask_other_leg] - 0.52) / 0.48
        swing_mask_leg = swing_mask_starting_leg | swing_mask_other_leg

        t_swing = torch.zeros(n_envs, dtype=phase.dtype, device=phase.device)
        t_swing[swing_mask_starting_leg] = t_swing_starting
        t_swing[swing_mask_other_leg] = t_swing_other

        if swing_mask_leg.any():
            t_swing_leg = t_swing[swing_mask_leg]
            swing_heights = bezier_curve(control_points_swing, t_swing_leg)
            desired_heights[swing_mask_leg, leg] = swing_heights[:, 1]
    return desired_heights

# =========================================================
# 6. 高级支撑与接触要求 (Advanced Support & Clearances)
# =========================================================

def feet_air_time_positive_biped(env: "TableTennisEnv", threshold: float, vel_ref: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= (torch.norm(env.robot_future_vel[:, :2], dim=1)) > vel_ref
    return reward

def feet_air_time_negative_biped(env: "TableTennisEnv", threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward-threshold, min=0.0)
    return reward

def reward_feet_contact_number(env: "TableTennisEnv", sensor_cfg: SceneEntityCfg, pos_rw: float, neg_rw: float, command_name: str = "base_velocity") -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0)
    phase = getattr(env, "get_phase", lambda: torch.zeros_like(env.ball_pos[:, 0]))() # 安全调用
    command = env.command_generator.command[:, :2]
    return compute_reward_reward_feet_contact_number(contacts, phase, pos_rw, neg_rw, command)

def foot_clearance_reward(env: "TableTennisEnv", target_height: float, std: float, tanh_mult: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "base_velocity") -> torch.Tensor:
    com_z = env.robot.data.root_pos_w[:, 2]
    current_foot_z = env.robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    standing_position_com_z = env.robot.data.default_root_state[:, 2]
    body_lin_vel_w = env.robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    command = env.command_generator.command[:, :2]
    return compute_reward_foot_clearance_reward(com_z, standing_position_com_z, current_foot_z, target_height, std, tanh_mult, body_lin_vel_w, command)

def track_foot_height(env: "TableTennisEnv", asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, std: float, command_name: str = "base_velocity") -> torch.Tensor:
    foot_z = env.robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    command = env.command_generator.command[:, :2]
    com_z = env.robot.data.root_pos_w[:, 2]
    standing_position_com_z =env.robot.data.default_root_state[:, 2]
    phase = getattr(env, "get_phase", lambda: torch.zeros_like(env.ball_pos[:, 0]))()
    return compute_reward_track_foot_height(com_z, standing_position_com_z, phase, foot_z, 0.0626, std, command)

# =========================================================
# 7. 乒乓球专项核心奖励 (Table Tennis Specific)
# =========================================================

def reward_track_base_target(
    env: "TableTennisEnv",
    std: float = 0.3,
    t_min: float = 0.12,
    t_max: float = 0.6,
    center: float = 0.32,
    width: float = 0.16,
) -> torch.Tensor:
    """Track base target during the pre-strike approach window."""
    base_pos_xy = env.robot_pos[:, :2]
    target_xy = env.base_target_xy
    dist = torch.linalg.norm(base_pos_xy - target_xy, dim=1)
    reward = 1.0 / (1.0 + (dist / std) ** 2)
    t_strike = env.t_strike.squeeze(-1)
    mask = _time_interval_mask(t_strike, t_min=t_min, t_max=t_max)
    phase_weight = _time_window_weight(t_strike, center=center, width=width)
    return torch.where(mask, reward * phase_weight, torch.zeros_like(reward))

def reward_track_racket_prepose(
    env: "TableTennisEnv",
    std: float = 0.25,
    t_min: float = 0.10,
    t_max: float = 0.36,
    center: float = 0.20,
    width: float = 0.07,
) -> torch.Tensor:
    """
    Track the racket pre-pose before entering the strike window.
    Forces the robot to pull the arm backward into a wind-up position.
    """
    # 1. Get the final predicted strike position
    hit_pos = env.racket_cmd[:, 0:3].clone()
    
    # 2. Determine if we are planning a forehand (0) or backhand (1)
    forehand = (env.swing_type == 0).unsqueeze(-1)
    
    # 3. Define the physical pullback offsets (relative to the strike point)
    # Forehand: Pull back 30cm, shift right 25cm, drop 15cm
    fh_offset = hit_pos.new_tensor([-0.30, -0.25, -0.15])
    # Backhand: Pull back 20cm, shift left (across chest) 10cm, drop 10cm
    bh_offset = hit_pos.new_tensor([-0.20, 0.10, -0.10])
    
    # 4. Apply the correct offset based on swing type
    prepose_offset = torch.where(forehand, fh_offset, bh_offset)
    target = hit_pos + prepose_offset
    
    # 5. Calculate distance between current paddle and the WIND-UP target
    pos = env.paddle_pos
    dist = torch.linalg.norm(pos - target, dim=1)
    
    # 6. Apply Inverse Quadratic reward
    reward = 1.0 / (1.0 + (dist / std) ** 2)
    
    # 7. Apply time window masks
    t_strike = env.t_strike.squeeze(-1)
    mask = _time_interval_mask(t_strike, t_min=t_min, t_max=t_max)
    phase_weight = _time_window_weight(t_strike, center=center, width=width)
    
    return torch.where(mask, reward * phase_weight, torch.zeros_like(reward))



def reward_hit_forward(env: "TableTennisEnv") -> torch.Tensor:
    """奖励：击球瞬间，球具有向前的 X 轴速度"""
    mask = env.ball_landing_dis_rew 
    vx = env.ball_linvel[:, 0]
    # 只要往前飞就给分，速度越快分越高
    reward = torch.clamp(vx, min=0.0) 
    return torch.where(mask, reward, torch.zeros_like(reward))

def reward_contact(
    env: "TableTennisEnv",
    center: float = 0.0,
    width: float = 0.05,
    cutoff: float = 0.14,
) -> torch.Tensor:
    reward = env.ball_contact_rew.float()
    t_strike = env.t_strike.squeeze(-1)
    mask = torch.abs(t_strike - float(center)) < float(cutoff)
    phase_weight = _time_window_weight(t_strike, center=center, width=width)
    return torch.where(mask, reward * phase_weight, torch.zeros_like(reward))

def reward_table_success(env: "TableTennisEnv") -> torch.Tensor:
    return (env.has_touch_paddle.float() * env.has_touch_opponent_table_just_now.float())

def reward_paddle_distance_terminal(env: "TableTennisEnv", coeff: float=100.0) -> torch.Tensor:
    distance = torch.norm(env.ball_global_pos - env.paddle_touch_point, dim=1) - 0.02
    reward = 1 / (1 + coeff * distance**2)**2 
    return torch.where(env.mask_terminal, torch.zeros_like(reward), reward)

def reward_paddle_distance_terminal_weighted(env: "TableTennisEnv", coeff_x: float=100.0, coeff_y: float=100.0, coeff_z: float=100.0, weight_x: float=1.0, weight_y: float=1.0, weight_z: float=1.0) -> torch.Tensor:
    d_x = torch.abs(env.ball_global_pos[:, 0] - env.paddle_touch_point[:, 0])
    reward_x = weight_x / (1 + coeff_x * d_x**2)**2
    d_y = torch.abs(env.ball_global_pos[:, 1] - env.paddle_touch_point[:, 1])
    reward_y = weight_y / (1 + coeff_y * d_y**2)**2
    d_z = torch.abs(env.ball_global_pos[:, 2] - env.paddle_touch_point[:, 2])
    reward_z = weight_z / (1 + coeff_z * d_z**2)**2
    reward = reward_x + reward_y + reward_z
    return torch.where(env.mask_terminal, torch.zeros_like(reward), reward)

def reward_future_ee_target(env: "TableTennisEnv", std_ee: float = 0.4, threshold: float = 0.01) -> torch.Tensor:
    dist_ee = torch.linalg.norm(env.ball_future_pose - env.paddle_pos, dim=1)
    denom_ee = std_ee * std_ee + 1e-12
    reward_ee = torch.exp(-torch.clamp(dist_ee, min=threshold) / denom_ee)
    reward_ee = torch.where(env.mask_invalid, torch.zeros_like(reward_ee), reward_ee)
    return torch.nan_to_num(reward_ee, nan=0.0, posinf=0.0, neginf=0.0)

def paddle_ball_distance(env: "TableTennisEnv", std_ee: float = 0.3, threshold: float = 0.01) -> torch.Tensor:
    denom_ee = std_ee * std_ee + 1e-12
    rew_ee = torch.exp(-torch.clamp(env.paddle_ball_distance, min=threshold) / denom_ee)
    mask_invalid_ee = (env.ball_pos[:, 0] < -1.35) | (env.ball_pos[:, 2] < 0.75)  | env.has_touch_paddle
    reward_ee = torch.where(mask_invalid_ee, torch.zeros_like(rew_ee), rew_ee)
    return torch.nan_to_num(reward_ee, nan=0.0, posinf=0.0, neginf=0.0)

def reward_future_body_target(env: "TableTennisEnv", std_ro: float = 0.5, threshold: float = 0.05) -> torch.Tensor:
    dist_ro = torch.linalg.norm(env.robot_future_pos[:, 0:2] - env.robot_pos[:, 0:2], dim=1)
    denom_ro = std_ro * std_ro + 1e-12
    rew_ro = torch.exp(-torch.clamp(dist_ro, min=threshold) / denom_ro)
    reward = torch.where(env.mask_invalid, torch.zeros_like(rew_ro), rew_ro)
    return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

def reward_future_vel_target(env: "TableTennisEnv", threshold: float = 0.03, vel_std: float= 1.41) -> torch.Tensor:
    denom_vel=vel_std * vel_std + 1e-12
    vdiff_ro = torch.linalg.norm(env.robot_future_vel[:, 0:2]-env.robot_linvel[:, 0:2],dim=1)
    reward_ro = torch.exp(-torch.clamp(vdiff_ro, min=threshold) / denom_vel)
    reward_ro = torch.where(env.mask_invalid, torch.zeros_like(reward_ro), reward_ro)
    return torch.nan_to_num(reward_ro, nan=0.0, posinf=0.0, neginf=0.0)

def reward_future_landing_dis(env: "TableTennisEnv", threshold: float= 2.0) -> torch.Tensor:
    target_x = 1.15
    target_y = 0.0
    pred_land = torch.stack([env.predict_x_land, env.predict_y_land], dim=1)
    target_land = torch.tensor([target_x, target_y], device=pred_land.device)
    dist_ball_land = torch.linalg.norm(pred_land - target_land, dim=1)
    reward = (threshold - dist_ball_land )
    mask = env.ball_landing_dis_rew
    return torch.where(mask, reward, torch.zeros_like(reward))

def reward_ball_z_nearnet(env: "TableTennisEnv", z_threshold: float = 0.94, x_tolerance: float = 0.03) -> torch.Tensor:
    x = env.ball_pos[:, 0]
    z = env.ball_pos[:, 2]
    vx = env.ball_linvel[:, 0]
    at_net_plane = torch.abs(x) < x_tolerance
    moving_forward = vx > 0.0
    hit = env.has_touch_paddle
    high_enough = z > z_threshold
    return (hit & at_net_plane & moving_forward & high_enough).float()

def reward_future_pass_net(env: "TableTennisEnv", std_h : float = 0.06, z_target : float = 1.0) -> torch.Tensor:
    x = env.ball_pos[:, 0]
    z = env.ball_pos[:, 2]
    vx = env.ball_linvel[:, 0]
    vz = env.ball_linvel[:, 2]

    moving_forward = vx > 0.0
    dx_to_net = torch.clamp(0.0 - x, min=0.0)

    g = 9.81
    eps = torch.tensor(1e-6, device=env.device, dtype=vx.dtype)
    vx_safe = torch.clamp(vx, min=eps)
    t_net = dx_to_net / vx_safe
    t_net = torch.clamp(t_net, min=0.0)
    z_at_net = z + vz * t_net - 0.5 * g * (t_net * t_net)

    height_err = torch.abs(z_at_net - z_target)
    reward = torch.exp(-height_err / (std_h + 1e-12))
    reward = torch.where(moving_forward, reward, torch.zeros_like(reward)) 
    mask = env.ball_landing_dis_rew
    return torch.where(mask, reward, torch.zeros_like(reward))

def penalty_robot_table_proximity_x(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min_distance: float = 0.10, std: float = 0.1) -> torch.Tensor:
    table_half_length = -1.37 - min_distance 
    robot_pos_x = env.robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    denom = std * std + 1e-12
    return torch.exp(-torch.clamp(torch.abs(robot_pos_x - table_half_length ), min=1e-6) / denom)

def late_serve_unstable_support(env: "TableTennisEnv", sensor_cfg: SceneEntityCfg, min_fraction: float, max_fraction: float, force_threshold: float = 0.1) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces_hist = contact_sensor.data.net_forces_w_history 
    in_contact = (net_forces_hist[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > force_threshold)
    n_contacts = torch.sum(in_contact.int(), dim=1)
    single_stance = n_contacts == 1
    both_air = n_contacts == 0
    unstable = single_stance | both_air 
    progress = env.ball_episode_length_buf.float() / float(env.max_ball_episode_length)
    late_mask = (progress > min_fraction) & (progress < max_fraction)
    return (unstable & late_mask).float()

def hit_unstable_support(env: "TableTennisEnv", sensor_cfg: SceneEntityCfg, force_threshold: float = 0.1) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces_hist = contact_sensor.data.net_forces_w_history
    in_contact = (net_forces_hist[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > force_threshold)
    n_contacts = torch.sum(in_contact.int(), dim=1)
    single_stance = n_contacts == 1
    both_air = n_contacts == 0
    unstable = single_stance | both_air 
    return unstable.float() * env.ball_contact_rew

def body_pitch_contact_exp(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target: float=0.0, std: float=0.4) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(body_quat)
    reward = torch.exp(-torch.abs(pitch - target) / std)
    return (env.ball_contact_rew > 0).float() * reward

def penalty_stand_still(env: "TableTennisEnv", sensor_cfg: SceneEntityCfg, force_threshold: float = 0.1, move_threshold: float = 0.1) -> torch.Tensor:  
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contacts = (net_contact_forces[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > force_threshold)
    both_feet_in_contact = torch.all(contacts, dim=1) 
    position_diff = torch.norm(env.robot_future_pos - env.robot_pos, dim=-1) 
    penalty = both_feet_in_contact & (position_diff > move_threshold)
    return penalty.float()

##### NEW #####
# =========================================================
# 8. 双模态特权模仿奖励 (Dual-Mode Action Correction)
# =========================================================

def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _dual_mode_commands(env: "TableTennisEnv") -> tuple[MotionCommand, MotionCommand]:
    cmd_fh = env.command_manager.get_term("motion_forehand")
    cmd_bh = env.command_manager.get_term("motion_backhand")
    return cmd_fh, cmd_bh


def _select_by_swing(env: "TableTennisEnv", a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where(env.swing_type == 0, a, b)


def feet_contact_time(env: "TableTennisEnv", sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


# =========================================================
# Phase 1：strike 瞬间的 sparse terminal reward
# =========================================================
# 由 TTCommandSampler 在 phase 跨越 0.86s 时置位 env.command_sampler.strike_triggered，
# 一个挥拍周期内仅一步为 True；此步 sparse 奖励 racket pose/vel 匹配度。


def _strike_trigger_mask(env: "TableTennisEnv") -> torch.Tensor:
    sampler = getattr(env, "command_sampler", None)
    if sampler is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return sampler.strike_triggered


def reward_strike_terminal_pos(env: "TableTennisEnv", std: float = 0.10) -> torch.Tensor:
    """t_strike→0 瞬间 racket 中心到 target 的距离 sparse 奖励。"""
    mask = _strike_trigger_mask(env)
    target_table = env.racket_cmd[:, 0:3]
    paddle_table = env.paddle_pos  # == paddle_touch_point - table_pos_w，见 compute_paddle_touch
    dist = torch.linalg.norm(paddle_table - target_table, dim=1)
    reward = torch.exp(-dist / std)
    return torch.where(mask, reward, torch.zeros_like(reward))


def reward_strike_terminal_vel(env: "TableTennisEnv", std: float = 1.0) -> torch.Tensor:
    """t_strike→0 瞬间 racket velocity 与 target velocity 的匹配度 sparse 奖励。"""
    mask = _strike_trigger_mask(env)
    target_vel = env.racket_cmd[:, 7:10]
    err = torch.linalg.norm(env.paddle_linvel - target_vel, dim=1)
    reward = torch.exp(-err / std)
    return torch.where(mask, reward, torch.zeros_like(reward))


def reward_strike_terminal_ori(env: "TableTennisEnv", std: float = 0.4) -> torch.Tensor:
    """t_strike→0 瞬间 racket 姿态匹配度（四元数误差）。"""
    mask = _strike_trigger_mask(env)
    target_quat = env.racket_cmd[:, 3:7]
    err = math_utils.quat_error_magnitude(env.paddle_quat, target_quat)
    reward = torch.exp(-err / std)
    return torch.where(mask, reward, torch.zeros_like(reward))
