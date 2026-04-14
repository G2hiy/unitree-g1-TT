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
from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from whole_body_tracking.tasks.TT.envs.tt_env import TableTennisEnv
    
    
def _time_window_weight(t_strike: torch.Tensor, center: float, width: float) -> torch.Tensor:
    width = max(float(width), 1e-6)
    return torch.exp(-((t_strike - float(center)) ** 2) / (2.0 * width * width))

def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]

def _time_interval_mask(t_strike: torch.Tensor, t_min: float, t_max: float) -> torch.Tensor:
    return (t_strike > float(t_min)) & (t_strike < float(t_max))

def reward_track_racket_prepose(
    env: "TableTennisEnv",
    std: float = 0.25,
    t_min: float = 0.10,
    t_max: float = 0.36,
    center: float = 0.20,
    width: float = 0.07,
) -> torch.Tensor:
    """Track the racket pre-pose before entering the strike window."""
    target = env.racket_cmd[:, 0:3]
    pos = env.paddle_pos
    dist = torch.linalg.norm(pos - target, dim=1)
    reward = 1.0 / (1.0 + (dist / std) ** 2)
    t_strike = env.t_strike.squeeze(-1)
    mask = _time_interval_mask(t_strike, t_min=t_min, t_max=t_max)
    phase_weight = _time_window_weight(t_strike, center=center, width=width)
    return torch.where(mask, reward * phase_weight, torch.zeros_like(reward))

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
    
    reward = 1.0 / (1.0 + (err_mag / std) ** 2)
    reward = torch.exp(err_mag / std) 
    
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


def motion_global_anchor_position_error_exp(env: "TableTennisEnv", command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: "TableTennisEnv", command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = math_utils.quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: "TableTennisEnv", command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: "TableTennisEnv", command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        math_utils.quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: "TableTennisEnv", command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: "TableTennisEnv", command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: "TableTennisEnv", sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def lin_vel_z_l2(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])

def ang_vel_xy_l2(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

def ang_vel_z_l2(env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])

def reward_contact(env: "TableTennisEnv") -> torch.Tensor:
    return env.ball_contact_rew.float()

def reward_hit_forward(env: "TableTennisEnv") -> torch.Tensor:
    """奖励：击球瞬间，球具有向前的 X 轴速度"""
    mask = env.ball_landing_dis_rew 
    vx = env.ball_linvel[:, 0]
    # 只要往前飞就给分，速度越快分越高
    reward = torch.clamp(vx, min=0.0) 
    return torch.where(mask, reward, torch.zeros_like(reward))

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

def reward_table_success(env: "TableTennisEnv") -> torch.Tensor:
    return (env.has_touch_paddle.float() * env.has_touch_opponent_table_just_now.float())