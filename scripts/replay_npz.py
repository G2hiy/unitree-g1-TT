"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_motion.py --registry_name source/whole_body_tracking/whole_body_tracking/assets/g1/motions/lafan_walk_short.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")
parser.add_argument("--motion_hz", type=int, default=50, help="Frame rate of motion data (default: 50Hz)")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1_wo_racket import G1_CYLINDER_CFG
from whole_body_tracking.tasks.bydmimic.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()  # 仿真步长（默认0.02s=50Hz）
    motion_hz = args_cli.motion_hz
    motion_dt = 1.0 / motion_hz  # 动作数据步长

    # 计算动作帧的采样间隔（确保仿真和动作帧率匹配）
    frame_skip = max(1, int(round(sim_dt / motion_dt)))
    print(f"=== 帧率匹配信息 ===")
    print(f"仿真步长: {sim_dt}s | 动作步长: {motion_dt}s | 帧间隔: {frame_skip}")

    # 加载动作数据
    registry_name = args_cli.registry_name
    if ":" not in registry_name:
        registry_name += ":latest"
    import pathlib
    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    # 初始化MotionLoader
    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )

    # ========== 关键修复1：手动验证动作总帧数（不依赖motion.time_step_total） ==========
    # 直接从MotionLoader的实际数据中获取总帧数（最准确）
    actual_motion_frames = motion.body_pos_w.shape[0]
    # 兼容不同MotionLoader版本：如果body_pos_w是三维，取第一维为帧数
    if len(motion.body_pos_w.shape) >= 3:
        actual_motion_frames = motion.body_pos_w.shape[0]
    # 打印对比信息，定位问题
    print(f"\n=== 动作帧数验证 ===")
    print(f"motion.time_step_total 返回值: {motion.time_step_total}")
    print(f"从body_pos_w获取的实际帧数: {actual_motion_frames}")

    # ========== 关键修复2：修正重置判断逻辑 ==========
    # 最大有效索引 = 实际总帧数 - 1（索引从0开始）
    max_time_step = actual_motion_frames - 1
    # 初始化时间步（单环境）
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    # 记录是否已经播放完一轮（避免提前打印重置信息）
    has_played_complete = False

    # Simulation loop
    frame_counter = 0
    while simulation_app.is_running():
        # 按帧间隔更新动作（避免帧率不匹配导致跳帧）
        if frame_counter % frame_skip == 0:
            current_step = time_steps[0].item()
            
            # ========== 关键修复3：仅当播放完最后一帧后才重置 ==========
            if current_step > max_time_step:
                # 重置到0，并标记完成一轮
                time_steps[0] = 0
                has_played_complete = True
                print("✅ Motion loop reset! (完整播放一轮后重置)")
            else:
                # 未播放完，正常推进
                has_played_complete = False
                # 读取当前帧的动作数据
                root_states = robot.data.default_root_state.clone()
                # 读取当前帧的根节点数据（确保维度正确）
                root_states[:, :3] = motion.body_pos_w[current_step][0] + scene.env_origins[:, None, :]
                root_states[:, 3:7] = motion.body_quat_w[current_step][0]
                root_states[:, 7:10] = motion.body_lin_vel_w[current_step][0]
                root_states[:, 10:] = motion.body_ang_vel_w[current_step][0]

                # 写入机器人状态
                robot.write_root_state_to_sim(root_states)
                robot.write_joint_state_to_sim(
                    motion.joint_pos[current_step], 
                    motion.joint_vel[current_step]
                )
                scene.write_data_to_sim()

                # 仅在未到最后一帧时递增步长
                time_steps += 1

                # 打印进度（调试用）
                if current_step % 50 == 0:
                    print(f"播放进度: {current_step}/{max_time_step} 帧")

        # 渲染+更新仿真（必须执行step保证循环正常）
        sim.step(render=True)
        scene.update(sim_dt)

        # 更新相机视角
        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        frame_counter += 1


def main():
    # 配置仿真（确保步长和动作匹配）
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02  # 50Hz，和常见动作数据帧率一致
    sim = SimulationContext(sim_cfg)

    # 创建场景（单环境）
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # 运行仿真
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()