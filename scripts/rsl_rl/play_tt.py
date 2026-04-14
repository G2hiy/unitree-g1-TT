"""Script to play a checkpoint of an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import numpy as np

from isaaclab.app import AppLauncher

# local imports
import cli_args  # 确保路径匹配你的实际 cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="G1-TableTennis-v0", help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")

# 乒乓球专属参数
parser.add_argument("--predictor", action="store_true", help="Use predictor-augmented runner for loading predictor weights and inference")
parser.add_argument("--record_action", action="store_true", help="Record observations and actions during play.")

# append RSL-RL cli arguments (通常包含 --wandb_path, --load_run, --checkpoint 等)
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import pathlib
import torch

from whole_body_tracking.utils.on_policy_predictor_regression_runner import OnPolicyPredictorRegressionRunner 
from rsl_rl.runners import OnPolicyRunner


from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    
    # 播放模式覆盖配置
    env_cfg.scene.env_spacing = 5.0
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    # ==========================
    # WandB 或 本地检查点加载逻辑
    # ==========================
    if hasattr(args_cli, "wandb_path") and args_cli.wandb_path:
        import wandb
        run_path = args_cli.wandb_path
        api = wandb.Api()
        
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
            
        wandb_run = api.run(run_path)
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"
    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    log_dir = os.path.dirname(resume_path)

    # ==========================
    # 创建 Isaac 环境与 Wrappers
    # ==========================
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    base_env = env.unwrapped
    if hasattr(base_env, "set_debug_vis"):
        base_env.set_debug_vis(False)
        
    # ==========================
    # 加载 Runner 与 Policy
    # ==========================
    if args_cli.predictor:
        runner = OnPolicyPredictorRegressionRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=base_env.device)

    # ==========================
    # 模型导出 (Predictor, ONNX, JIT)
    # ==========================
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)
    
    if args_cli.predictor:
        try:
            orig_device = next(runner._predictor.parameters()).device
        except StopIteration:
            orig_device = torch.device("cpu")
        runner._predictor.to("cpu").eval()
        ts_predictor = torch.jit.script(runner._predictor)
        ts_predictor.save(os.path.join(export_model_dir, "predictor.pt"))
        runner._predictor.to(orig_device)
        
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx")

    # ==========================
    # 录制与统计初始化
    # ==========================
    obs, _ = env.get_observations()
    timestep = 0
    
    if args_cli.record_action:
        obs_records, action_records, obs_slice_records = [], [], []

    succ_total, hit_total, serve_total = 0, 0, 0
    try:
        serve_success_flag = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
        serve_hit_flag = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
    except Exception:
        serve_success_flag, serve_hit_flag = None, None
        
    step_count = 0

    # ==========================
    # 推理主循环
    # ==========================
    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # 策略推理
                actions = policy(obs)
                
                # 数据录制 (若开启)
                if args_cli.record_action:
                    try:
                        obs0 = obs[0].detach().cpu().numpy()
                        act0 = actions[0].detach().cpu().numpy()
                        obs_records.append(obs0)
                        action_records.append(act0)
                        if obs0.shape[0] >= 69:
                            obs_slice_records.append(obs0[48:69])
                        else:
                            start, end = 48, min(69, obs0.shape[0])
                            pad_len = 69 - end
                            safe_slice = obs0[start:end]
                            if pad_len > 0:
                                safe_slice = np.pad(safe_slice, (0, pad_len), mode="constant")
                            obs_slice_records.append(safe_slice)
                    except Exception:
                        pass

                # 环境步进
                obs, _, _, _ = env.step(actions)
                
                # 更新预测器落点
                if args_cli.predictor:
                    try:
                        if hasattr(base_env, "ball_pos"):
                            runner.env.ball_pos = base_env.ball_pos
                        if hasattr(base_env, "update_prediction"):
                            runner.env.update_prediction = base_env.update_prediction
                        
                        runner._record_ball_positions()
                        runner._maybe_predict_and_update_env()
                    except Exception:
                        print(f"⚠️ 预测器运行出错: {e}")
                        pass
                
                # 统计击球率与上台率
                try:
                    if serve_success_flag is not None:
                        event_mask = (base_env.has_touch_opponent_table_just_now & base_env.has_touch_paddle)
                        serve_success_flag |= event_mask
                        
                    if serve_hit_flag is not None:
                        hit_mask = (base_env.ball_contact_rew > 0.0)
                        serve_hit_flag |= hit_mask

                    if hasattr(base_env, "ball_reset_ids") and base_env.ball_reset_ids is not None:
                        ids = base_env.ball_reset_ids
                        if isinstance(ids, torch.Tensor) and ids.numel() > 0 and serve_success_flag is not None:
                            serve_total += int(ids.numel())
                            succ_total += int(serve_success_flag[ids].sum().item())
                            hit_total += int(serve_hit_flag[ids].sum().item())
                            serve_success_flag[ids] = False
                            serve_hit_flag[ids] = False
                except Exception:
                    pass
                
                step_count += 1
                if step_count % 50 == 0:
                    succ_rate = (succ_total / serve_total) if serve_total > 0 else 0.0
                    hit_rate = (hit_total / serve_total) if serve_total > 0 else 0.0
                    print(f"[Play] Success {succ_total}/{serve_total} ({succ_rate:.3f}) | Hits {hit_total}/{serve_total} ({hit_rate:.3f})")

            # 视频录制控制
            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    print(f"[INFO] Video recording complete. Length: {args_cli.video_length} steps.")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C detected. Finalizing...")
    finally:
        # 保存录制数据
        if args_cli.record_action:
            try:
                os.makedirs(log_dir, exist_ok=True)
                obs_arr = np.asarray(obs_records) if len(obs_records) > 0 else np.empty((0,))
                act_arr = np.asarray(action_records) if len(action_records) > 0 else np.empty((0,))
                slice_arr = np.asarray(obs_slice_records) if len(obs_slice_records) > 0 else np.empty((0,))
                npz_path = os.path.join(log_dir, "play_obs_actions.npz")
                np.savez(npz_path, obs=obs_arr, actions=act_arr, obs_slice=slice_arr)
                print(f"[INFO] Saved obs/action records to: {npz_path}")
            except Exception as e:
                print(f"[WARN] Failed to save recordings: {e}")
        
        env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()