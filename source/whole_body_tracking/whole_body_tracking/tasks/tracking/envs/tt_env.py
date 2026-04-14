import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.buffers import DelayBuffer

from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg

try:
    from whole_body_tracking.tasks.tracking.physics.aerodynamics import AeroForceField
except ImportError:
    AeroForceField = None
    print("[WARNING] aerodynamics.py not found; aerodynamics disabled.")

def dir_to_quat(directions: torch.Tensor, forward_axis: int = 0) -> torch.Tensor:
    """将目标法向方向转换为四元数 (w, x, y, z)"""
    v1 = torch.zeros_like(directions)
    v1[:, forward_axis] = 1.0
    dot = torch.sum(v1 * directions, dim=1, keepdim=True)
    cross = torch.cross(v1, directions, dim=1)
    
    q = torch.empty((directions.shape[0], 4), device=directions.device)
    q[:, 0] = 1.0 + dot.squeeze(-1)
    q[:, 1:4] = cross
    return torch.nn.functional.normalize(q, p=2, dim=1)


class TableTennisEnv(ManagerBasedRLEnv):
    cfg: TrackingEnvCfg

    def __init__(self, cfg: TrackingEnvCfg, render_mode: str | None = None, **kwargs):
        num_envs = cfg.scene.num_envs
        device = cfg.sim.device

        # --- 物理与空气动力学初始化 ---
        if AeroForceField is not None:
            self.aero = AeroForceField(device=str(device), radius_m=0.020, air_density=1.225, drag_coeff=0.4378)
        else:
            self.aero = None

        try:
            rho, cd, radius, mass = 1.225, 0.47, 0.02, 0.0027
            area = float(math.pi) * (radius**2)
            self.ball_drag_k = float(0.5 * rho * cd * area / max(1e-6, mass))
        except Exception:
            self.ball_drag_k = 0.13

        # --- 核心状态 Buffer 分配 ---
        self.ball_episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.ball_reset_counter = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.ball_reset_ids = torch.empty(0, device=device, dtype=torch.long)

        # 交互标志位
        self.has_touch_paddle = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.has_touch_paddle_rew = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.ball_landing_dis_rew = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.ball_contact_rew = torch.zeros(num_envs, device=device, dtype=torch.float32)
        self.max_contact_score = torch.zeros(num_envs, device=device, dtype=torch.float32)

        self.has_touch_opponent_table_just_now = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.has_touch_own_table_just_now = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.has_touch_own_table = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.has_touch_own_table_prev = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.has_touch_opo_table_prev = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.has_return_own_table2_prev = torch.zeros(num_envs, device=device, dtype=torch.bool)

        self.mask_invalid = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.mask_terminal = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.mask_before = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.mask_after = torch.zeros(num_envs, device=device, dtype=torch.bool)

        # 物理实体位置/速度
        self.ball_global_pos = torch.zeros(num_envs, 3, device=device)
        self.ball_pos = torch.zeros(num_envs, 3, device=device)
        self.ball_linvel = torch.zeros(num_envs, 3, device=device)
        self.paddle_touch_point = torch.zeros(num_envs, 3, device=device)
        self.paddle_pos = torch.zeros(num_envs, 3, device=device)
        self.paddle_linvel = torch.zeros(num_envs, 3, device=device)
        self.paddle_quat = torch.zeros(num_envs, 4, device=device)
        self.paddle_quat[:, 0] = 1.0 
        self.robot_pos = torch.zeros(num_envs, 3, device=device)
        self.robot_linvel = torch.zeros(num_envs, 3, device=device)

        # 预测与规划目标
        self.ball_prediction = torch.zeros(num_envs, 3, device=device)
        self.ball_prediction_vis = torch.zeros(num_envs, 3, device=device)
        self.ball_hit_prediction = torch.zeros(num_envs, 7, device=device)
        self.ball_hit_state_gt = torch.zeros(num_envs, 7, device=device)
        self.ball_future_pose = torch.zeros(num_envs, 3, device=device)
        self.ball_future_pose_vis = torch.zeros(num_envs, 3, device=device)
        self.robot_future_pos = torch.zeros(num_envs, 3, device=device)
        self.robot_future_vel = torch.zeros(num_envs, 3, device=device)
        self.ball_future_t = torch.zeros(num_envs, 1, device=device)

        self.racket_cmd = torch.zeros(num_envs, 11, device=device)
        self.racket_cmd[:, 3] = 1.0  # 初始化合法的默认四元数
        self.racket_cmd_gt = torch.zeros(num_envs, 7, device=device)
        self.base_target_xy = torch.zeros(num_envs, 2, device=device)
        self.t_strike = torch.zeros(num_envs, 1, device=device)
        self.planner_hit_state = torch.zeros(num_envs, 7, device=device)
        self.planner_base_target_raw = torch.zeros(num_envs, 2, device=device)
        self.planner_racket_cmd_raw = torch.zeros(num_envs, 11, device=device)
        self.planner_racket_cmd_raw[:, 3] = 1.0

        # 相位与击球决策
        # self.swing_type = torch.zeros(num_envs, device=device, dtype=torch.long)
        # self.swing_type_locked = torch.zeros(num_envs, device=device, dtype=torch.bool)
        # self.serve_swing_type = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.motion_phase_time = torch.zeros(num_envs, device=device)
        self.motion_phase_target_time = torch.zeros(num_envs, device=device)

        # 延迟 Buffer
        self._action_dim = 1
        self.action_buffer = DelayBuffer(1, num_envs, device=device)
        self.action_buffer.compute(torch.zeros(num_envs, self._action_dim, dtype=torch.float32, device=device))

        self.num_perception = 6
        self.perception_buffer = DelayBuffer(5, num_envs, device=device)
        self.perception_buffer.compute(torch.zeros(num_envs, self.num_perception, dtype=torch.float32, device=device))
        self.delayed_perception = torch.zeros(num_envs, self.num_perception, device=device)

        self.ball_hist_len = 10
        self.ball_history = torch.zeros(num_envs, self.ball_hist_len, 3, device=device)

        super().__init__(cfg, render_mode, **kwargs)

        # --- 场景与资产引用 ---
        self.robot = self.scene["robot"]
        self.ball = self.scene["ball"]
        self.table = self.scene["table"]
        self.ball_future_visual = self.scene.rigid_objects.get("ball_future")
        self.ball_pred_visual = self.scene.rigid_objects.get("ball_pred")
        self.paddle_offset_visual = self.scene.rigid_objects.get("paddle_offset")

        # 绑定球拍物理偏置
        paddle_body_name = "right_wrist_yaw_link"
        found_ids, found_names = self.robot.find_bodies([paddle_body_name], preserve_order=True)
        if len(found_ids) == 0:
            raise RuntimeError(f"Failed to find paddle carrier body: {paddle_body_name}")
        self.paddle_index = found_ids[0]

        self.racket_offset_pos = torch.tensor([0.12, 0.0, 0.0], dtype=torch.float32, device=device)
        racket_offset_rpy = torch.tensor([0.0, 1.5, -1.5], dtype=torch.float32, device=device)
        self.racket_offset_quat = math_utils.quat_from_euler_xyz(
            racket_offset_rpy[0], racket_offset_rpy[1], racket_offset_rpy[2]
        ).unsqueeze(0).repeat(num_envs, 1)


    def _ensure_action_buffer(self, action_dim: int):
        action_dim = int(action_dim)
        if action_dim <= 0: raise RuntimeError(f"Invalid action dimension: {action_dim}")
        if action_dim == self._action_dim: return
        
        self._action_dim = action_dim
        self.action_buffer = DelayBuffer(1, self.num_envs, device=self.device)
        self.action_buffer.compute(torch.zeros(self.num_envs, self._action_dim, dtype=torch.float32, device=self.device))

    def update_hit_state_prediction(self, preds: torch.Tensor):
        if preds is None or preds.shape[0] != self.num_envs: return
        if not isinstance(preds, torch.Tensor): preds = torch.as_tensor(preds, dtype=torch.float32)
        preds = preds.to(self.device)

        if preds.shape[1] >= 7:
            self.ball_hit_prediction[:] = preds[:, :7]
        elif preds.shape[1] >= 3:
            self.ball_hit_prediction.zero_()
            self.ball_prediction[:] = preds[:, :3]
            self.ball_prediction_vis = self.ball_prediction + self.scene.env_origins
            return

        self.ball_prediction = self.ball_hit_prediction[:, :3]
        self.ball_prediction_vis = self.ball_prediction + self.scene.env_origins

    def update_prediction(self, preds: torch.Tensor):
        self.update_hit_state_prediction(preds)

    def update_racket_command(self, preds: torch.Tensor | None):
        self.update_hit_state_prediction(preds)

    def _has_valid_prediction(self) -> torch.Tensor:
        if not getattr(self.cfg, "planner_use_prediction", False):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        pred = self.ball_hit_prediction
        return torch.isfinite(pred).all(dim=1) & (pred[:, 6] > 0.0) & (pred[:, 0] > -1.9)

    # def _select_swing_type_from_hit_state(self, hit_pos: torch.Tensor) -> torch.Tensor:
    #     lateral = hit_pos[:, 1] - self.robot_pos[:, 1]
    #     forehand_cost = torch.abs(lateral + 0.22)
    #     backhand_cost = torch.abs(lateral - 0.22)
    #     comfortable = torch.abs(lateral) < 0.75
    #     margin = float(getattr(self.cfg, "swing_decision_margin", 0.04))
        
    #     choose_backhand = (backhand_cost + margin < forehand_cost) & comfortable
    #     choose_forehand = (forehand_cost + margin <= backhand_cost) | (~comfortable)

    #     planned = self.swing_type.clone()
    #     planned = torch.where(choose_backhand, torch.ones_like(planned), planned)
    #     return torch.where(choose_forehand, torch.zeros_like(planned), planned)

    def _plan_base_target(self, hit_pos: torch.Tensor) -> torch.Tensor:
        target = torch.zeros(self.num_envs, 2, device=self.device, dtype=hit_pos.dtype)
        target[:, 0] = torch.clamp(hit_pos[:, 0] - 0.45, min=-1.85, max=-0.75)
        
        # 移除 forehand 判断，全部固定为正手的 -0.22 偏移
        y_offset = torch.full_like(hit_pos[:, 1], -0.22)
        target[:, 1] = torch.clamp(hit_pos[:, 1] + y_offset, min=-0.45, max=0.45)
        return target

    def _plan_racket_command(self, hit_state: torch.Tensor) -> torch.Tensor:
        cmd = torch.zeros(self.num_envs, 11, device=self.device) 
        hit_pos, v_i, t_hit = hit_state[:, :3], hit_state[:, 3:6], torch.clamp(hit_state[:, 6:7], min=0.0)

        # 物理反推参数
        p_l = torch.zeros_like(hit_pos)
        p_l[:, 0], p_l[:, 1], p_l[:, 2] = 0.6, -hit_pos[:, 1] * 0.5, 0.76 
        
        delta_t, g = 0.45, 9.81
        v_o = (p_l - hit_pos) / delta_t
        v_o[:, 2] += 0.5 * g * delta_t 

        delta_v = v_o - v_i
        n = delta_v / torch.clamp(torch.norm(delta_v, dim=1, keepdim=True), min=1e-6)

        C_r = 0.8  
        v_o_n = torch.sum(v_o * n, dim=1, keepdim=True)
        v_i_n = torch.sum(v_i * n, dim=1, keepdim=True)
        v_r = ((v_o_n + C_r * v_i_n) / (1.0 + C_r)) * n

        q_r = dir_to_quat(n, forward_axis=0)

        # 去掉 forehand 判断，直接使用正手 lateral_offset
        lateral_offset = hit_pos.new_tensor([0.00, -0.08, 0.02])
        
        cmd[:, 0:3] = hit_pos + lateral_offset
        cmd[:, 3:7] = q_r
        cmd[:, 7:10] = v_r
        cmd[:, 10:11] = t_hit
        return cmd

    def update_planned_commands(self):
        valid_pred = self._has_valid_prediction()
        self._curriculum_steps = getattr(self, "_curriculum_steps", 0) + 1
        progress = min(1.0, self._curriculum_steps / 240000.0) 
        
        oracle_weight = 1.0 - torch.clamp((torch.tensor(progress, device=self.device) - 0.3) / 0.5, min=0.0, max=1.0)
        blended_hit_state = (1.0 - oracle_weight) * self.ball_hit_prediction + oracle_weight * self.ball_hit_state_gt
        hit_state = torch.where(valid_pred.unsqueeze(-1), blended_hit_state, self.ball_hit_state_gt)
        hit_state[:, 6:7] = torch.clamp(hit_state[:, 6:7], min=0.0)
        self.planner_hit_state[:] = hit_state

        # 【删除了在这里判断 swing_type 及锁定的逻辑】

        self.planner_base_target_raw[:] = self._plan_base_target(hit_state[:, :3])
        self.planner_racket_cmd_raw[:] = self._plan_racket_command(hit_state)

        cmd_alpha, time_alpha = float(getattr(self.cfg, "planner_command_alpha", 0.35)), float(getattr(self.cfg, "planner_time_alpha", 0.6))
        valid_mask = (~self.mask_invalid).unsqueeze(-1)

        default_target_xy = torch.zeros_like(self.base_target_xy)
        default_target_xy[:, 0] = -2.10
        self.base_target_xy[:] = torch.where(
            valid_mask, 
            torch.lerp(self.base_target_xy, self.planner_base_target_raw, cmd_alpha), 
            torch.lerp(self.base_target_xy, default_target_xy, cmd_alpha)  # 👈 修复点
        )
        
        smoothed_racket = torch.lerp(self.racket_cmd, self.planner_racket_cmd_raw, cmd_alpha)
        smoothed_racket[:, 3:7] = torch.nn.functional.normalize(smoothed_racket[:, 3:7], p=2, dim=1) 
        self.racket_cmd[:] = torch.where(valid_mask, smoothed_racket, self.racket_cmd)  
        
        self.t_strike[:] = torch.where(
            valid_mask, 
            torch.lerp(self.t_strike, self.planner_racket_cmd_raw[:, 10:11], time_alpha), 
            self.t_strike 
        )
        self.racket_cmd[:, 10:11] = self.t_strike

    def _update_motion_phase_from_planner(self):
        if not hasattr(self, "command_manager"): return

        max_phase_step = float(getattr(self.cfg, "motion_phase_max_step", 0.04))
        max_phase_time = float(getattr(self.cfg, "motion_phase_max_time", 1.88))

        t_strike = torch.where(self.planner_hit_state[:, 6] > 0.0, self.planner_hit_state[:, 6], self.ball_future_t.squeeze(-1))
        valid_ball = ~(self.mask_invalid.squeeze(-1) if self.mask_invalid.ndim > 1 else self.mask_invalid)
        
        # 【删除了 release_now 解除 swing_type_locked 的代码】

        target_time = float(getattr(self.cfg, "motion_clip_strike_time", 0.86)) - t_strike
        target_time = torch.clamp(torch.nan_to_num(torch.where(valid_ball, target_time, torch.zeros_like(target_time))), min=0.0, max=max_phase_time)
        self.motion_phase_target_time[:] = target_time

        phase_delta = (self.motion_phase_target_time - self.motion_phase_time).clamp(min=-max_phase_step, max=max_phase_step)
        self.motion_phase_time = torch.where(valid_ball, self.motion_phase_time + float(getattr(self.cfg, "motion_phase_alpha", 0.2)) * phase_delta, torch.zeros_like(self.motion_phase_time))

        try:
            self.command_manager.get_term("motion_forehand").set_motion_phase(self.motion_phase_time.clone())
            # 【删除了 motion_backhand 的更新，如果你配置文件里也把反手删了，这里会防报错】
        except Exception: pass

    def _compute_racket_pose_from_wrist(self):
        wrist_pos = self.robot.data.body_pos_w[:, self.paddle_index, :]
        wrist_quat = self.robot.data.body_quat_w[:, self.paddle_index, :]
        wrist_linvel = self.robot.data.body_lin_vel_w[:, self.paddle_index, :]
        wrist_angvel = self.robot.data.body_ang_vel_w[:, self.paddle_index, :]

        offset_pos_w = math_utils.quat_apply(wrist_quat, self.racket_offset_pos.unsqueeze(0).expand(self.num_envs, -1))
        return wrist_pos + offset_pos_w, math_utils.quat_mul(wrist_quat, self.racket_offset_quat), wrist_linvel + torch.cross(wrist_angvel, offset_pos_w, dim=1)

    def step(self, action: torch.Tensor):
        self._ensure_action_buffer(action.shape[-1])
        delayed_action = self.action_buffer.compute(action)

        if self.aero is not None: self.aero.apply_to_rigid_object(self.ball)

        obs, rew, terminated, truncated, info = super().step(delayed_action)

        self.compute_perception()
        self.compute_paddle_touch()
        self.compute_intermediate_values()
        self.update_planned_commands()
        self._update_motion_phase_from_planner()
        self.ball_episode_length_buf += 1

        # self._update_visualizations()

        ball_on_floor = self.ball.data.root_pos_w[:, 2] < 0.1
        ball_timeout = self.ball_episode_length_buf >= 150
        self.ball_reset_ids = (ball_on_floor | ball_timeout).nonzero(as_tuple=False).flatten()
        if len(self.ball_reset_ids) > 0:
            self._reset_ball_async(self.ball_reset_ids)

        return obs, rew, terminated, truncated, info

    def compute_perception(self):
        self.ball_pos = self.ball.data.root_pos_w - self.table.data.root_pos_w
        self.robot_pos = self.robot.data.root_pos_w - self.table.data.root_pos_w
        self.delayed_perception = self.perception_buffer.compute(torch.cat([self.ball_pos, self.robot_pos], dim=-1))
        self.ball_history = torch.roll(self.ball_history, shifts=1, dims=1)
        self.ball_history[:, 0, :] = self.ball_pos

    def compute_paddle_touch(self):
        self.ball_global_pos = self.ball.data.root_pos_w
        racket_pos_w, racket_quat_w, racket_linvel_w = self._compute_racket_pose_from_wrist()
        self.paddle_touch_point, self.paddle_quat, self.paddle_linvel = racket_pos_w, racket_quat_w, racket_linvel_w
        self.paddle_pos = self.paddle_touch_point - self.table.data.root_pos_w

        distance = torch.norm(self.ball_global_pos - self.paddle_touch_point, dim=1) - 0.02
        self.ball_contact = torch.clamp((0.14 - distance) / 0.14, min=0.0, max=1.0)
        self.max_contact_score = torch.maximum(self.max_contact_score, self.ball_contact)

        new_hits = (self.ball_contact > 0) & (self.ball_contact < self.max_contact_score)
        still_false = ~self.has_touch_paddle
        self.has_touch_paddle[still_false] = new_hits[still_false]
        self.ball_contact_rew = torch.maximum(self.ball_contact_rew, self.max_contact_score)

    def compute_intermediate_values(self):
        self.ball_linvel, self.robot_linvel = self.ball.data.root_lin_vel_w, self.robot.data.root_lin_vel_w
        self.ball_contact_rew *= (self.has_touch_paddle * ~self.has_touch_paddle_rew)
        self.ball_landing_dis_rew = self.has_touch_paddle & ~self.has_touch_paddle_rew

        bx, by, bz = self.ball_pos[:, 0], self.ball_pos[:, 1], self.ball_pos[:, 2]
        
        self.has_touch_opponent_table_just_now = (bx >= 0.0) & (bx <= 1.35) & (by >= -0.7625) & (by <= 0.7625) & (bz >= 0.76) & (bz <= 0.83)
        self.has_touch_own_table_just_now = (bx >= -1.37) & (bx <= 0.0) & (by >= -0.7625) & (by <= 0.7625) & (bz >= 0.76) & (bz <= 0.83)
        
        self.has_touch_own_table = self.has_touch_own_table_just_now.clone()
        self.has_touch_own_table_prev |= self.has_touch_own_table_just_now
        self.has_touch_opo_table_prev |= self.has_touch_opponent_table_just_now
        self.has_return_own_table2_prev |= (self.has_touch_own_table_prev & self.has_touch_paddle & self.has_touch_own_table_just_now)

        vz, z, x, y, vx, vy = self.ball_linvel[:, 2], self.ball_pos[:, 2], self.ball_pos[:, 0], self.ball_pos[:, 1], self.ball_linvel[:, 0], self.ball_linvel[:, 1]
        self.mask_before, self.mask_after = ~self.has_touch_own_table_prev, self.has_touch_own_table_prev
        
        g, h, body_height = 9.81, 0.78, 0.69
        sqrt_d = torch.sqrt(torch.clamp(vz.pow(2) + 2.0 * g * (z - h), min=0.0))
        t1 = torch.where((vz.pow(2) + 2.0 * g * (z - h) >= 0.0) & self.mask_before, torch.clamp((vz + sqrt_d) / g, min=0.0), torch.zeros_like(vz))

        k_s, g_t = torch.tensor(max(1e-8, getattr(self, "ball_drag_k", 0.13)), device=self.device), torch.tensor(g, device=self.device)
        sqrt_gk, sqrt_kg = torch.sqrt(torch.clamp(g_t * k_s, min=1e-12)), torch.sqrt(torch.clamp(k_s / g_t, min=0.0))

        vz_up = torch.clamp(-0.9 * (vz - g * t1), min=0.0)
        t2_drag = torch.atan(torch.clamp(vz_up * sqrt_kg, min=0.0)) / torch.clamp(sqrt_gk, min=1e-12)
        delta_z_drag = (0.5 / k_s) * torch.log1p(torch.clamp((k_s / g_t) * vz_up.pow(2), min=0.0))
        
        vz_up_a = torch.clamp(vz, min=0.0)
        t_after_drag = torch.atan(torch.clamp(vz_up_a * sqrt_kg, min=0.0)) / torch.clamp(sqrt_gk, min=1e-12)
        delta_z_drag_a = (0.5 / k_s) * torch.log1p(torch.clamp((k_s / g_t) * vz_up_a.pow(2), min=0.0))

        def horiz_disp(v0, horizon): return (1.0 / k_s) * torch.sign(v0) * torch.log1p(torch.clamp(k_s * torch.abs(v0) * horizon, min=0.0))

        self.ball_future_pose = torch.where(
            self.mask_before.unsqueeze(-1), 
            torch.stack([torch.clamp(x + horiz_disp(vx, t1) + horiz_disp(0.7 * vx, t2_drag), max=-1.6), y + horiz_disp(vy, t1) + horiz_disp(0.7 * vy, t2_drag), delta_z_drag + h], dim=-1),
            torch.stack([torch.clamp(x + horiz_disp(vx, t_after_drag), max=-1.6), y + horiz_disp(vy, t_after_drag), z + delta_z_drag_a], dim=-1)
        )

        self.mask_invalid = (x < -1.9) | (vx > 0) | (z < 0.7) | ((x < -1.35) & (vz < 0)) | self.has_touch_paddle
        self.mask_terminal = (x > -1.5) | (x < -1.9) | self.has_touch_paddle.clone() | (vz < 0.0) | (z < 0.6)
        self.has_touch_paddle_rew = self.has_touch_paddle.clone()
        
        modified_pos = torch.clone(self.robot_pos)
        modified_pos[:, 1] -= 0.60
        modified_pos[:, 2] = body_height + 0.2
        self.ball_future_pose = torch.where(self.mask_invalid.unsqueeze(-1).expand_as(self.ball_future_pose), modified_pos, self.ball_future_pose)
        self.ball_future_pose_vis = torch.where((self.has_touch_paddle & ~(self.has_return_own_table2_prev | self.has_touch_opo_table_prev)).unsqueeze(-1), torch.stack([x + horiz_disp(vx, t1), y + horiz_disp(vy, t1), torch.ones_like(x) * h], dim=-1), self.ball_future_pose) + self.scene.env_origins

        self.ball_future_t = torch.where(self.mask_invalid.unsqueeze(-1), torch.zeros_like(t1.unsqueeze(-1)), torch.where(self.mask_before.unsqueeze(-1), (t1 + t2_drag).unsqueeze(-1), t_after_drag.unsqueeze(-1)))
        
        future_vel = self.ball_linvel.clone()
        future_vel[:, 2] -= 9.81 * self.ball_future_t.squeeze(-1)
        self.ball_hit_state_gt[:, 0:3], self.ball_hit_state_gt[:, 3:6], self.ball_hit_state_gt[:, 6:7] = self.ball_future_pose, future_vel, self.ball_future_t
        self.racket_cmd_gt.copy_(self.ball_hit_state_gt)

        invalid_pred = ~self._has_valid_prediction()
        if torch.any(invalid_pred):
            self.ball_hit_prediction[invalid_pred] = self.ball_hit_state_gt[invalid_pred]
            self.ball_prediction[invalid_pred] = self.ball_hit_prediction[invalid_pred, 0:3]
            self.ball_prediction_vis[invalid_pred] = self.ball_prediction[invalid_pred] + self.scene.env_origins[invalid_pred]

    def _update_visualizations(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        if self.ball_future_visual is not None:
            pose = torch.zeros((self.num_envs, 7), device=self.device)
            pose[:, :3], pose[:, 3] = self.ball_future_pose_vis, 1.0
            self.ball_future_visual.write_root_pose_to_sim(pose, env_ids)

        if self.ball_pred_visual is not None:
            pose = torch.zeros((self.num_envs, 7), device=self.device)
            pose[:, :3], pose[:, 3] = self.ball_prediction_vis, 1.0
            self.ball_pred_visual.write_root_pose_to_sim(pose, env_ids)
            
        if self.paddle_offset_visual is not None:
            pose = torch.zeros((self.num_envs, 7), device=self.device)
            pose[:, :3], pose[:, 3] = self.paddle_touch_point, 1.0
            self.paddle_offset_visual.write_root_pose_to_sim(pose, env_ids)

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        self.action_buffer.reset(env_ids)
        self.perception_buffer.reset(env_ids)
        self._reset_ball_async(env_ids)

    def _reset_ball_async(self, env_ids):
        self.ball_reset_ids = torch.empty(0, device=self.device, dtype=torch.long)
        self.ball_episode_length_buf[env_ids] = 0
        self.ball_reset_counter[env_ids] += 1

        # 重置交互与历史标志位
        self.has_touch_paddle[env_ids] = False
        self.has_touch_paddle_rew[env_ids] = False
        self.ball_landing_dis_rew[env_ids] = False
        self.ball_contact_rew[env_ids] = 0.0
        self.max_contact_score[env_ids] = 0.0
        self.has_touch_own_table[env_ids] = False
        self.has_touch_own_table_prev[env_ids] = False
        self.has_touch_opo_table_prev[env_ids] = False
        self.has_return_own_table2_prev[env_ids] = False

        # 预测与控制标志位赋默认值 
        self.planner_base_target_raw[env_ids, 0] = -2.10
        self.planner_base_target_raw[env_ids, 1] = 0.0
        self.base_target_xy[env_ids, 0] = -2.10
        self.base_target_xy[env_ids, 1] = 0.0
        
        self.planner_racket_cmd_raw[env_ids] = 0.0
        self.planner_racket_cmd_raw[env_ids, 3] = 1.0  
        self.racket_cmd[env_ids] = 0.0
        self.racket_cmd[env_ids, 3] = 1.0              
        self.t_strike[env_ids] = 0.0

        # self.swing_type[env_ids] = 0
        # self.swing_type_locked[env_ids] = False
        # self.serve_swing_type[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device, dtype=torch.long)

        from whole_body_tracking.tasks.TT.mdp.events import reset_ball_serve
        reset_ball_serve(
            self,
            env_ids,
            speed_x_range=(-6.5, -5.0),
            speed_y_range=(-0.8, 0.4),
            speed_z_range=(1.5, 2.0),
            pos_y_range=(-0.1, 0.1),
            asset_cfg=self.scene["ball"],
        )