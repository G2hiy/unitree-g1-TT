from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from whole_body_tracking.tasks.tracking.envs.tt_env import TableTennisEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes


# def _dual_mode_commands(env: ManagerBasedRLEnv):
#     cmd_fh = env.command_manager.get_term("motion_forehand")
#     cmd_bh = env.command_manager.get_term("motion_backhand")
#     return cmd_fh, cmd_bh


def _select_by_swing(env: ManagerBasedRLEnv, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.where(env.swing_type == 0, a, b)


# def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
#     cmd_fh, cmd_bh = _dual_mode_commands(env)
#     err_fh = torch.norm(cmd_fh.anchor_pos_w - cmd_fh.robot_anchor_pos_w, dim=1)
#     err_bh = torch.norm(cmd_bh.anchor_pos_w - cmd_bh.robot_anchor_pos_w, dim=1)
#     err = _select_by_swing(env, err_fh, err_bh)
#     return err > threshold


# def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
#     cmd_fh, cmd_bh = _dual_mode_commands(env)
#     err_fh = torch.abs(cmd_fh.anchor_pos_w[:, -1] - cmd_fh.robot_anchor_pos_w[:, -1])
#     err_bh = torch.abs(cmd_bh.anchor_pos_w[:, -1] - cmd_bh.robot_anchor_pos_w[:, -1])
#     err = _select_by_swing(env, err_fh, err_bh)
#     return err > threshold


# def bad_anchor_ori(
#     env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
# ) -> torch.Tensor:
#     asset: RigidObject | Articulation = env.scene[asset_cfg.name]

#     cmd_fh, cmd_bh = _dual_mode_commands(env)

#     motion_pg_fh = math_utils.quat_rotate_inverse(cmd_fh.anchor_quat_w, asset.data.GRAVITY_VEC_W)
#     robot_pg_fh = math_utils.quat_rotate_inverse(cmd_fh.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

#     motion_pg_bh = math_utils.quat_rotate_inverse(cmd_bh.anchor_quat_w, asset.data.GRAVITY_VEC_W)
#     robot_pg_bh = math_utils.quat_rotate_inverse(cmd_bh.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

#     err_fh = (motion_pg_fh[:, 2] - robot_pg_fh[:, 2]).abs()
#     err_bh = (motion_pg_bh[:, 2] - robot_pg_bh[:, 2]).abs()

#     err = _select_by_swing(env, err_fh, err_bh)
#     return err > threshold


# def bad_motion_body_pos(
#     env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
# ) -> torch.Tensor:
#     cmd_fh, cmd_bh = _dual_mode_commands(env)

#     idx_fh = _get_body_indexes(cmd_fh, body_names)
#     idx_bh = _get_body_indexes(cmd_bh, body_names)

#     err_fh = torch.norm(cmd_fh.body_pos_relative_w[:, idx_fh] - cmd_fh.robot_body_pos_w[:, idx_fh], dim=-1)
#     err_bh = torch.norm(cmd_bh.body_pos_relative_w[:, idx_bh] - cmd_bh.robot_body_pos_w[:, idx_bh], dim=-1)

#     bad_fh = torch.any(err_fh > threshold, dim=-1)
#     bad_bh = torch.any(err_bh > threshold, dim=-1)

#     return torch.where(env.swing_type == 0, bad_fh, bad_bh)


# def bad_motion_body_pos_z_only(
#     env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
# ) -> torch.Tensor:
#     cmd_fh, cmd_bh = _dual_mode_commands(env)

#     idx_fh = _get_body_indexes(cmd_fh, body_names)
#     idx_bh = _get_body_indexes(cmd_bh, body_names)

#     err_fh = torch.abs(cmd_fh.body_pos_relative_w[:, idx_fh, -1] - cmd_fh.robot_body_pos_w[:, idx_fh, -1])
#     err_bh = torch.abs(cmd_bh.body_pos_relative_w[:, idx_bh, -1] - cmd_bh.robot_body_pos_w[:, idx_bh, -1])

#     bad_fh = torch.any(err_fh > threshold, dim=-1)
#     bad_bh = torch.any(err_bh > threshold, dim=-1)

#     return torch.where(env.swing_type == 0, bad_fh, bad_bh)


def bad_anchor_pos(env: "TableTennisEnv", command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: "TableTennisEnv", command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: "TableTennisEnv", asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_rotate_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_rotate_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: "TableTennisEnv", command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: "TableTennisEnv", command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


def base_height_below_minimum(
    env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.38
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < threshold


def bad_upright_orientation(
    env: "TableTennisEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.85
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    tilt = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
    return tilt > threshold

