# HITTER 论文复现项目

## 项目目标
复现论文 "HITTER: A HumanOId Table TEnnis Robot via Hierarchical Planning 
and Learning"（arXiv:2508.21043）的仿真部分。不依赖真实硬件，只做
IsaacSim 中的仿真验证。

## 当前进度
- [x] 视频 → GVHMR → GMR 动作重定向完成
- [x] BeyondMimic 框架下挥拍动作模仿完成（正手/反手各一段，94帧，1.88s，
      击球在第43帧/0.86s）
- [ ] Model-based Planner（IV节）
- [ ] WBC RL 训练（V节）
- [ ] 二者集成

## 技术栈
- 仿真：IsaacSim
- RL 框架：PPO，参考 rsl_rl 实现
- 机器人：Unitree G1，27 DOF
- Python 3.8，PyTorch 2.0，CUDA 11.8

## 论文关键参数（不要修改这些值）
- 控制频率：50 Hz（策略输出频率）
- 物理仿真：200 Hz（substeps=4）
- 击球平面 x = -1.37 m（世界坐标系，机器人在 x<0 侧）
- 球拍半径：7.5 cm（临界位置误差阈值）
- 空气阻力系数 k = 0.006（待标定，这是初始值）
- 水平恢复系数 Ch = 0.95
- 垂直恢复系数 Cv = 0.80
- 击球恢复系数 Cr = 0.9
- episode 长度：10 秒（500 步 @ 50Hz）
- 每段参考动作：94帧，击球在第43帧

## 代码规范
- 所有张量操作必须支持批量 env（N 个并行环境）
- 奖励函数在 rewards.py 中独立实现，每个 reward term 单独一个函数
- 论文公式编号作为注释写在对应代码旁（如 # Eq.(6) in HITTER paper）

## 当前任务
实现在仿真环境中学会人类的挥拍动作且成功击球把球打到对方球桌

## 已知问题
- IsaacGym 对高速小球的接触检测不稳定，需要用解析式碰撞检测替代

