# X2 Full-Body Training Debug Rollback Checklist And Session Summary

## Debug 设置回放清单

目标：在“当前训练已恢复正常”的基础上，逐项恢复之前为了 debug 临时退掉的设置，避免一次性引入多个变量。

### Phase 0: 先冻结当前可训练版本

在恢复任何设置前，先确认并保留当前基线：

- `x2_fullbody_env.py` 中 DOF 相关 reward 已经只统计 lower-body 12 joints
- `x2_fullbody.urdf` 中可疑的 `elbow joint origin rpy = -0.5` 已修正，不要改回去
- 当前能稳定启动训练，`mean_episode_length` 不再快速塌缩

建议动作：

- 保存当前训练配置快照
- 训练出一个可用 checkpoint
- 用这个 checkpoint 先接 middleware/mock 验证

### Phase 1: 先做 middleware/mock 验证

目标：验证“训练恢复正常后，部署链路是否也更稳定”。

观察重点：

- 站立是否明显优于旧 checkpoint
- 上半身 PD 保持下，是否仍然快速后仰
- 是否还出现之前那种 fall logger 中持续后倒、动作范数暴涨的模式

如果这一步已经明显改善，就先不要继续恢复高风险 debug 设置。

### Phase 2: 按顺序回放 debug 设置

以下设置建议按顺序，一次只恢复一类。

#### 2.1 恢复 `ActorCriticRecurrent`

目的：验证 LSTM 是否仍有收益。

操作：

- 将 `policy_class_name` 从 `ActorCritic` 改回 `ActorCriticRecurrent`
- 恢复：
  - `rnn_type`
  - `rnn_hidden_size`
  - `rnn_num_layers`

通过标准：

- 训练不中途再次出现 `unpad_trajectories` 相关维度报错
- `mean_episode_length` 不出现持续性下滑

失败信号：

- 再次出现 `24 vs 23` 一类 recurrent rollout 错位
- episode length 在中期快速退化

结论处理：

- 如果失败，说明当前 full-body 环境仍不适合先上 RNN
- 先保留 `ActorCritic`，不要强行回 RNN

#### 2.2 恢复轻量 Domain Randomization

前提：只有在 `ActorCritic` 或已验证稳定的 recurrent 版本下再做。

建议恢复顺序：

1. 只开 friction randomization
2. 再开 base mass randomization
3. 最后再开 push_robots

注意：当前训练栈里 DR 默认是关的，只有传 `--domain_rand` 或命令行覆写 `domain_rand.*=...` 才会开启。

建议不要一步到位全开。

#### 2.3 恢复更接近部署的 upper-body PD 参数

目标：让训练时的 upper-body hold 更接近 middleware/mock 中的部署设定。

建议：

- 一次只调一组：waist / arm / head
- 优先保持 arm/head 稳定，不要先把 waist 调得很激进

通过标准：

- episode length 不明显下降
- tracking reward 不被大幅破坏

#### 2.4 最后再考虑 self-collision

这是高风险项，放到最后。

原因：

- full-body 资产一旦启用 self-collision，最容易重新引入短 episode、抖动、接触异常
- 即使 URDF 已经修过 elbow frame，也不代表 collision 分布已经完全可靠

建议：

- 只有在“无 self-collision”版本已经能稳定训练和通过 mock 验证后，再做 A/B 测试
- 开启前后只比较一个变量

## 不建议回退的内容

这些内容当前不要改回去：

- 不要把 `x2_fullbody_env.py` 中 lower-body reward 限定改回全 DOF 统计
- 不要把修正后的 elbow joint origin 改回 `rpy = -0.5`
- 不要在还没完成 middleware/mock 验证前，同时恢复 RNN + DR + self-collision

## 本次对话开发与调试摘要

### 1. Middleware / mock 侧开发内容

已完成：

- 上半身固定控制能力开发
- 按真实 HAL 路径拆分发布：
  - `leg`
  - `waist`
  - `arm`
  - `head`
- 新增 upper-body 配置与命令构建模块
- MuJoCo mock bridge 扩展为支持上半身多 topic 命令
- 新建 full-body mock XML，使上半身命令能够真正进入 MuJoCo
- 增加 fall logger，用 CSV 记录倒地前关键控制量

### 2. Middleware / mock 调试过程

现象：

- 上半身命令可以发布，但一旦在 mock 中开放上半身动力学，机器人很快后倒
- 调整上半身 `kp/kd` 后仍无法稳定站立
- 通过 fall logger 观察到：
  - `torso_pitch` 持续向后发散
  - `waist_pitch_target` 长时间饱和
  - 下半身 `action_norm` 暴涨

结论：

- 单靠部署侧补偿，无法让原本基于“固定上半身”训练出来的下半身策略适配 full-body 动力学
- middleware/mock 路径是必要的，它在不上机的前提下提前暴露了稳定性问题

### 3. 训练侧开发方向确认

确认采用的训练思路：

- lower body 继续由 RL 控制
- upper body 在训练环境中保持 active，但由 implicit high-gain PD 接管
- policy action space 仍保持 12D

为此新增：

- `x2_fullbody.urdf`
- `x2_fullbody_config.py`
- `x2_fullbody_env.py`
- 任务注册 `x2_fullbody`

### 4. 训练侧关键调试过程

#### 第一类问题：recurrent training 崩溃

现象：

- 训练到 300+ iteration 左右时，出现 `unpad_trajectories` 的 `24 vs 23` 维度错位报错

初步判断：

- 不是 RNN 本身先坏，而是环境已经先变得很不稳定
- 极短 episode / invalid rollout 导致 recurrent packing 先崩

处理：

- 先切换到普通 `ActorCritic`
- 暂时不优先使用 `ActorCriticRecurrent`

#### 第二类问题：训练虽然能跑，但 `mean_episode_length` 持续下降

最关键发现：

- `x2_fullbody_env.py` 已经把 observation 切回 lower-body 12 DOF
- 但大量 reward 仍然继承基类，对 full-body 全 DOF 统计
- 这导致训练目标和原本的 `x2_12dof` locomotion 任务不一致

具体影响项：

- `torques`
- `dof_vel`
- `dof_acc`
- `dof_pos_limits`
- `dof_vel_limits`
- `torque_limits`
- `stand_still`

解决方案：

- 在 `x2_fullbody_env.py` 中覆写这些 reward
- 只统计 lower-body 12 joints

结果：

- 训练恢复正常
- 说明之前训练学不到合适策略的主因，确实是 reward 没有从 `x2_12dof` 任务语义正确映射过来

### 5. 关于 DR 的最终结论

确认结果：

- 当前训练栈中，domain randomization 默认并不会开启
- 即使 config 中 `domain_rand` 写成 `True`
- 训练入口仍会通过 `update_cfg_from_args()` 用命令行参数覆写
- 只有显式传 `--domain_rand`，或使用 `domain_rand.*=...` 覆写时，DR 才真正生效

因此：

- 这次 full-body 训练崩溃和 episode length 退化，不能优先归因到 DR

### 6. 关于 `x2_fullbody.urdf` 的当前结论

已确认：

- `x2_fullbody.urdf` 与 `x2_ultra.urdf` 并非关键参数完全一致
- 最可疑的结构问题是两侧 elbow joint 的 `origin rpy = -0.5`
- 这项已经修正

当前建议：

- 先保留修正后的版本
- 暂时不要继续大改 URDF
- 也不要把 elbow origin 改回旧写法
- 先以“当前 reward 已修正、训练恢复正常”的版本继续推进

## 当前阶段的推荐下一步

1. 用当前 reward 修正后的 full-body 版本训练出一个稳定 checkpoint
2. 用该 checkpoint 接 middleware/mock 做可视化验证
3. 再按本清单逐项恢复：
   - `ActorCriticRecurrent`
   - 轻量 DR
   - 更接近部署的 upper-body PD
   - self-collision

## 建议保留的关键文件

- `legged_gym/envs/x2/x2_fullbody_env.py`
- `legged_gym/envs/x2/x2_fullbody_config.py`
- `resources/robots/x2/urdf/x2_fullbody.urdf`
- `wbc_middleware/ros2/control_node.py`
- `wbc_middleware/mock_bridge/mujoco_hal_bridge.py`
- `logs/csv/control_middleware_fall_log.csv`
- `doc/FULL_BODY_TRAINING_EVALUATION_AND_HISTORY.md`
- `doc/FULL_BODY_IMPLICIT_PD_TRAINING_MINIMAL_DESIGN.md`
