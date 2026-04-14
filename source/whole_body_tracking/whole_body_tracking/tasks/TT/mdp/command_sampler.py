"""Phase 1 command sampler for HITTER-style WBC training.

训练阶段完全解耦真实球物理：以确定性的相位倒计时驱动 racket/base 目标采样，
让 policy 只学 "在 t_strike 倒计时到 0 的瞬间把 racket 送到指定 pose+velocity
并让 base 到达指定位置"，对应论文 Sec V.B.1。

Paper 参考：
  - Episode 10 s；每个 swing clip 1.88 s，strike 在 0.86 s；follow-through 到 1.88 s。
  - Forehand / backhand 区域不重叠；strike plane 在 base 前方 0.40 m。
  - Base orientation 一直朝前。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from whole_body_tracking.tasks.TT.envs.tt_env import TableTennisEnv


@dataclass
class TTCommandSamplerCfg:
    """论文 V.B.1 的采样分布。数值与 CLAUDE.md Phase 1 设计表一致。"""

    clip_duration_s: float = 1.88           # 参考动作完整时长
    strike_time_s: float = 0.86             # 参考动作 43 帧处为击球瞬间
    base_delta_x_range: tuple = (-0.15, 0.15)
    base_delta_y_range: tuple = (-0.60, 0.60)
    base_target_x_clamp: tuple = (-2.30, -1.70)
    base_target_y_clamp: tuple = (-0.60, 0.60)

    # racket target 局部偏移（相对 commanded base）
    racket_local_x: float = 0.40
    racket_local_y_forehand: tuple = (-0.40, -0.10)
    racket_local_y_backhand: tuple = (0.10, 0.40)
    racket_world_z_range: tuple = (0.90, 1.25)  # world frame z

    # racket velocity (world frame; base yaw ≈ 0)
    racket_vx_range: tuple = (1.5, 3.5)
    racket_vy_forehand: tuple = (0.5, 2.0)
    racket_vy_backhand: tuple = (-2.0, -0.5)
    racket_vz_range: tuple = (0.3, 1.5)

    # 表面 z（world）→ 目标转到 table frame 时要减掉
    table_top_world_z: float = 0.76


class TTCommandSampler:
    """管理每个环境的挥拍相位、命令目标、strike 瞬间触发信号。

    不持有独立的命令张量；直接写入 env.racket_cmd / env.base_target_xy /
    env.t_strike / env.swing_type / env.motion_phase_time，让下游 obs/reward
    透明消费（无需改 observations.py）。
    """

    def __init__(self, env: "TableTennisEnv", cfg: TTCommandSamplerCfg | None = None):
        self.env = env
        self.cfg = cfg or TTCommandSamplerCfg()
        n = env.num_envs
        d = env.device

        self.phase_time = torch.zeros(n, device=d)           # 0..clip_duration
        self.strike_triggered = torch.zeros(n, device=d, dtype=torch.bool)
        self.strike_count = torch.zeros(n, device=d, dtype=torch.long)

    # ------------------------------------------------------------------
    # 外部接口
    # ------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor):
        """env _reset_idx 回调。清相位 + 立刻采一组新命令。"""
        if env_ids.numel() == 0:
            return
        self.phase_time[env_ids] = 0.0
        self.strike_triggered[env_ids] = False
        self.strike_count[env_ids] = 0
        self._sample_new_command(env_ids)

    def tick(self, step_dt: float):
        """每个 env.step() 末尾调用一次。

        流程：
          1. phase_time += dt (上限为 clip_duration)
          2. 识别本步刚跨越 strike_time 的 envs（strike_triggered 标记供 reward 使用）
          3. 识别 phase 跑完 clip 的 envs → 重采命令、phase 归零
          4. 更新 env.t_strike / env.motion_phase_time / 同步 MotionCommand 相位
        """
        cfg = self.cfg
        prev_phase = self.phase_time.clone()
        self.phase_time = torch.clamp(self.phase_time + float(step_dt), max=cfg.clip_duration_s)

        # 本步是否刚跨越 strike 时刻（给 strike_terminal reward 使用，一步有效）
        self.strike_triggered = (prev_phase < cfg.strike_time_s) & (self.phase_time >= cfg.strike_time_s)
        self.strike_count = self.strike_count + self.strike_triggered.long()

        # 相位跑完一整个 clip → 下一轮挥拍
        done_ids = (self.phase_time >= cfg.clip_duration_s - 1e-6).nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            self.phase_time[done_ids] = 0.0
            self._sample_new_command(done_ids)

        # 倒计时到 strike 的剩余时间（>=0）
        t_strike = torch.clamp(cfg.strike_time_s - self.phase_time, min=0.0)
        self.env.t_strike[:, 0] = t_strike
        self.env.racket_cmd[:, 10] = t_strike
        self.env.motion_phase_time[:] = self.phase_time

        # 同步参考动作相位给两个 MotionCommand
        try:
            self.env.command_manager.get_term("motion_forehand").set_motion_phase(self.phase_time.clone())
            self.env.command_manager.get_term("motion_backhand").set_motion_phase(self.phase_time.clone())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 内部：采样
    # ------------------------------------------------------------------
    def _sample_new_command(self, env_ids: torch.Tensor):
        """为 env_ids 采样一组新 (swing_type, base_target, racket_target, racket_vel)。"""
        cfg = self.cfg
        n = env_ids.numel()
        d = self.env.device

        swing = torch.randint(0, 2, (n,), device=d, dtype=torch.long)   # 0=forehand, 1=backhand
        forehand = (swing == 0)

        # ---- base target：从当前 robot xy（table frame）采一个偏移 ----
        robot_xy_table = (self.env.robot.data.root_pos_w[env_ids, :2]
                          - self.env.scene["table"].data.root_pos_w[env_ids, :2])
        dx = torch.empty(n, device=d).uniform_(*cfg.base_delta_x_range)
        dy = torch.empty(n, device=d).uniform_(*cfg.base_delta_y_range)
        base_target_table = torch.stack([
            torch.clamp(robot_xy_table[:, 0] + dx, *cfg.base_target_x_clamp),
            torch.clamp(robot_xy_table[:, 1] + dy, *cfg.base_target_y_clamp),
        ], dim=-1)

        # ---- racket target：base 前方 0.40m，local y/z 由 swing_type 决定 ----
        y_fh = torch.empty(n, device=d).uniform_(*cfg.racket_local_y_forehand)
        y_bh = torch.empty(n, device=d).uniform_(*cfg.racket_local_y_backhand)
        y_local = torch.where(forehand, y_fh, y_bh)

        z_world = torch.empty(n, device=d).uniform_(*cfg.racket_world_z_range)
        z_table = z_world - cfg.table_top_world_z

        racket_target_table = torch.stack([
            base_target_table[:, 0] + cfg.racket_local_x,
            base_target_table[:, 1] + y_local,
            z_table,
        ], dim=-1)

        # ---- racket velocity (world frame)：前/上 + 依 swing 的侧向分量 ----
        vx = torch.empty(n, device=d).uniform_(*cfg.racket_vx_range)
        vy_fh = torch.empty(n, device=d).uniform_(*cfg.racket_vy_forehand)
        vy_bh = torch.empty(n, device=d).uniform_(*cfg.racket_vy_backhand)
        vy = torch.where(forehand, vy_fh, vy_bh)
        vz = torch.empty(n, device=d).uniform_(*cfg.racket_vz_range)
        racket_vel = torch.stack([vx, vy, vz], dim=-1)

        # ---- racket 姿态：让 racket 平面 ⊥ velocity（paper Sec IV.C） ----
        quat = self._quat_from_forward(racket_vel)

        # 写回 env 张量
        self.env.swing_type[env_ids] = swing
        self.env.base_target_xy[env_ids] = base_target_table
        self.env.racket_cmd[env_ids, 0:3] = racket_target_table
        self.env.racket_cmd[env_ids, 3:7] = quat
        self.env.racket_cmd[env_ids, 7:10] = racket_vel
        self.env.racket_cmd[env_ids, 10] = cfg.strike_time_s  # 将被 tick() 覆盖为当步的倒计时

    @staticmethod
    def _quat_from_forward(v: torch.Tensor) -> torch.Tensor:
        """构造一个使 racket 法向 ≈ v/|v| 的单位四元数 (w,x,y,z)。"""
        n = v.shape[0]
        forward = torch.zeros_like(v)
        forward[:, 0] = 1.0
        v_unit = torch.nn.functional.normalize(v, p=2, dim=1, eps=1e-8)
        dot = (forward * v_unit).sum(dim=1, keepdim=True).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        cross = torch.cross(forward, v_unit, dim=1)
        q = torch.empty((n, 4), device=v.device, dtype=v.dtype)
        q[:, 0] = 1.0 + dot.squeeze(-1)
        q[:, 1:4] = cross
        return torch.nn.functional.normalize(q, p=2, dim=1)
