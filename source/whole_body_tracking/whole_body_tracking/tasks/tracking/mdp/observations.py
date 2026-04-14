import torch
import isaaclab.utils.math as math_utils
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from whole_body_tracking.tasks.tracking.envs.tt_env import TableTennisEnv

# =========================================================
# 1. 共享观测 (Actor 与 Critic 均需要)
# =========================================================
def get_robot_pos_rel_table(env: "TableTennisEnv"):
    """Base position (p_base)"""
    return env.scene["robot"].data.root_pos_w - env.scene["table"].data.root_pos_w

def get_base_pos_xy(env: "TableTennisEnv"):
    """Base position xy (p_base,xy)"""
    return get_robot_pos_rel_table(env)[:, :2]

def get_ball_pos_rel_table(env: "TableTennisEnv"):
    """Ball position (p_ball)"""
    return env.scene["ball"].data.root_pos_w - env.scene["table"].data.root_pos_w

def get_robot_heading_vector(env: "TableTennisEnv"):
    """Robot heading vector (e_xy) -> [cos(yaw), sin(yaw)]"""
    quat = env.scene["robot"].data.root_quat_w
    _, _, yaw = math_utils.euler_xyz_from_quat(quat)
    return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)

# =========================================================
# 2. Actor 独有观测 (基于预测器)
# =========================================================
def get_ball_prediction(env: "TableTennisEnv"):
    """Predictor output (p_ball_tilde)"""
    return env.ball_prediction

def get_rel_target_xy(env: "TableTennisEnv"):
    """Predictor-based robot shift (Delta p_base,xy_tilde)"""
    robot_pos = env.scene["robot"].data.root_pos_w - env.scene["table"].data.root_pos_w
    ball_target_xy = torch.stack([
        env.ball_prediction[:, 0] - 0.1,
        env.ball_prediction[:, 1] + 0.6,
    ], dim=1)
    return ball_target_xy - robot_pos[:, :2]

# =========================================================
# 2. Policy commands (base/racket targets)
# =========================================================
def get_base_target_xy(env: "TableTennisEnv"):
    """Target base position xy in table frame."""
    return env.base_target_xy

def get_racket_target_pos(env: "TableTennisEnv"):
    """Target racket position in table frame."""
    return env.racket_cmd[:, 0:3]

def get_racket_target_vel(env: "TableTennisEnv"):
    """Target racket velocity in table frame."""
    return env.racket_cmd[:, 3:6]

def get_t_strike(env: "TableTennisEnv"):
    """Time to strike in seconds."""
    return env.t_strike

def get_ball_trajectory_history(env: "TableTennisEnv"):
    """Flattened ball position history in table frame."""
    return env.ball_history.reshape(env.scene.num_envs, -1)

# =========================================================
# 3. Critic 独有特权观测 (基于物理真值与环境状态)
# =========================================================
def get_ground_truth_future_pose(env: "TableTennisEnv"):
    """Physics-based prediction (p_ball_hat)"""
    return env.ball_future_pose

def get_physics_rel_target_xy(env: "TableTennisEnv"):
    """Physics-based robot shift (Delta p_base,xy_hat)"""
    robot_pos = env.scene["robot"].data.root_pos_w - env.scene["table"].data.root_pos_w
    ball_target_xy = torch.stack([
        env.ball_future_pose[:, 0] - 0.1,
        env.ball_future_pose[:, 1] + 0.6,
    ], dim=1)
    return ball_target_xy - robot_pos[:, :2]

def get_ball_linvel(env: "TableTennisEnv"):
    """Ball linear velocity (v_ball)"""
    return env.ball_linvel

def get_paddle_touch_point(env: "TableTennisEnv"):
    """End-effector position (p_ee)"""
    return env.paddle_touch_point

def get_ball_future_t(env: "TableTennisEnv"):
    """Time for ball to arrive (t_arrive)"""
    return env.ball_future_t

def get_time_left(env: "TableTennisEnv"):
    """Time left in episode (t_left)"""
    if not hasattr(env, "episode_length_buf") or not hasattr(env, "max_episode_length"):
        return torch.zeros((env.scene.num_envs, 1), device=env.device, dtype=torch.float32)
    remaining = env.max_episode_length - env.episode_length_buf.float()
    step_dt = getattr(env, "step_dt", None)
    if step_dt is None:
        step_dt = float(getattr(env.cfg.sim, "dt", 0.01)) * float(getattr(env.cfg, "decimation", 1))
    return (remaining * float(step_dt)).unsqueeze(-1)

def get_upper_body_poses(env: "TableTennisEnv", body_names: list[str]):
    """Upper body poses T_B = [pos(3), quat(4)] for specified bodies."""
    body_ids = env.scene["robot"].find_bodies(body_names, preserve_order=True)[0]
    pos = env.scene["robot"].data.body_pos_w[:, body_ids]
    quat = env.scene["robot"].data.body_quat_w[:, body_ids]
    return torch.cat([pos, quat], dim=-1).reshape(env.scene.num_envs, -1)

def get_reference_joint_pos_vel(env: "TableTennisEnv"):
    """Reference joint positions and velocities [q_hat, qdot_hat]"""
    try:
        cmd_fh = env.command_manager.get_term("motion_forehand")
        q_fh = cmd_fh.joint_pos
        qd_fh = cmd_fh.joint_vel
        q = q_fh
        qd = qd_fh
        return torch.cat([q, qd], dim=-1)
    except Exception:
        zeros = torch.zeros((env.scene.num_envs, 1), device=env.device, dtype=torch.float32)
        return torch.cat([zeros, zeros], dim=-1)

def get_serve_progress(env: "TableTennisEnv"):
    """Serve progress (t_serve / t_serve,max)"""
    # 90 是 tt_env.py 中硬编码的发球超时 step 阈值
    return (env.ball_episode_length_buf.float() / 90.0).unsqueeze(-1)

def get_episode_progress(env: "TableTennisEnv"):
    """Episode progress (t_episode / t_episode,max)"""
    # 🛡️ 规避 Isaac Lab 初始化时的 Dry Run (试运行) 报错
    if not hasattr(env, "episode_length_buf") or not hasattr(env, "max_episode_length"):
        # 如果变量还没创建，返回一个全 0 的占位张量，形状为 [num_envs, 1]
        return torch.zeros((env.scene.num_envs, 1), device=env.device, dtype=torch.float32)
        
    return (env.episode_length_buf.float() / env.max_episode_length).unsqueeze(-1)

def get_touch_own_table(env: "TableTennisEnv"):
    """Ball touching own table (b_owntable)"""
    return env.has_touch_own_table.float().unsqueeze(-1)

def get_touch_paddle(env: "TableTennisEnv"):
    """Ball touching paddle (b_paddle)"""
    return env.has_touch_paddle.float().unsqueeze(-1)
