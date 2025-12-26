"""STIN MARL Environment - 完整实现总结

本文档总结了 STIN (Satellite-Terrestrial Integrated Network) 多智能体强化学习环境的完整实现。

**最新更新 (2025-12-15)**:
- ✅ 实现完整的论文时隙操作流程 (Section II-B, 5个步骤)
- ✅ 添加任务属性观测 (current_task_data_size, workload, max_delay, etc.)
- ✅ 实现信息收集和控制消息分发机制
- ✅ 添加服务卫星切换 (handover) 处理

==============================================================================
0. 论文时隙操作流程实现 (NEW!)
==============================================================================

根据论文 Section II-B，每个时隙 τ 内的网络控制和数据传输按以下步骤执行：

**Step 1: 信息收集** - 通过观测空间获取
  由观测空间完成网络信息收集，RL 智能体通过观测获取：
  - 本地资源状态：电池、存储、CPU 频率（SatProperties）
  - 当前任务属性：数据量、工作负载、时延约束、剩余时间（task_request）
  - 邻居卫星状态：ISL 距离、电池、队列、CPU（STINRelativeObservations）
  - 时间和光照状态（Time, Eclipse）
  
  实现位置: ComputationSatellite.observation_spec (行 429-468)
  总观测维度: ~40 维（卫星基础10维 + STIN状态3维 + 任务属性5维 + 邻居20维 + 时间光照2维）

**Step 2: 决策** - 由 RL 智能体完成
  基于观测生成动作：
  - 宏观切分: alpha_local (UD), alpha_cloud (云端)
  - 微观切分: x_self, x_neighbor1, ..., x_neighbor4
  - 资源分配: cpu_ratio, tx_power_ratio
  
  实现位置: 外部 RL 算法 → STINContinuousAction (10D)

**Step 3: 控制消息生成** - 通过动作解析和属性存储
  RL 动作被解析为控制参数，存储在卫星属性中：
  - 资源分配: set_resource_allocation() 方法处理动作[0-2]
    * cpu_ratio → 映射到实际 CPU 频率
    * tx_power_ratio → 映射到发射功率
    * platform_power_ratio → 映射到平台功耗
  - 任务切分: schedule_collaboration_action() 方法处理动作[3-9]
    * alpha_local, alpha_cloud → 一级切分
    * x_0-x_4 → 二级卫星间切分
  
  实现位置: 
    - set_resource_allocation(): 行 588-633
    - schedule_collaboration_action(): 行 802-965
  存储位置: satellite.current_control_message (逻辑记录，可选)

**Step 4: 任务执行** - 分两个阶段执行
  
  阶段 4a: 任务计算与初步处理 (execute_local_compute)
    - 处理 task_queue 中的计算任务（输入数据）
    - UD 本地处理（逻辑记录）
    - 云端处理（通过 SGL，逻辑记录）
    - 卫星边缘计算：
      * schedule_collaboration_action(): 新任务的切片分配
      * execute_local_compute(): 当前任务的计算执行
    - 计算完成时检查可见性：
      * 有可见性 → 直接完成（计算时延 + 下行传播时延）
      * 无可见性 → 转发到 result_queue（接力处理）
    
    实现位置: ComputationSatellite.execute_local_compute()
  
  阶段 4b: 结果接力与回传 (execute_result_relay)
    - 处理 result_queue 中的计算结果（输出数据）
    - 简化模型：仅处理回传路径和接力转发
    - 检查对 UD 的可见性：
      * 有可见性 → 直接下行回传（仅传播时延）
      * 无可见性 → 通过 ISL 接力转发（仅传播时延）
    - 与 task_queue 分离，避免队列混用
    
    实现位置: ComputationSatellite.execute_result_relay()

**Step 5: 服务卫星切换处理** - handle_handover()  这部分处理暂时不添加，因为我们可以训练agent在可见时间内完成，然后回传我们也使用了接力模式，因此未处理的切换暂时不考虑
  当前服务卫星失去可见性时，通过 ISL 切换到新服务卫星：
  - 同步 task_queue 中的所有待处理任务到新卫星
  - 传递 current_control_message 和 network_info
  - 累加 ISL 传播时延到每个任务
  - 更新任务的 current_holder（当前卫星）和 hop_count（转发跳数）
  
  实现位置: ComputationSatellite.handle_handover() (行 735-778)

**完整时隙执行流程**:
  每个环境 step 中，以下操作按顺序执行：
  1. **观测获取** (Step 1): observation_spec 自动收集网络信息
  2. **RL 决策** (Step 2): RL 智能体基于观测输出 10D 动作
  3. **动作执行** (Step 3): 
     - set_resource_allocation() 设置 CPU 频率和功耗
     - schedule_collaboration_action() 执行任务切分
  4. **任务执行** (Step 4a/4b):
     - execute_local_compute() 处理计算任务
     - execute_result_relay() 处理结果回传
  5. **切换处理** (Step 5): handle_handover() 处理服务卫星切换
  
  这 5 个步骤构成论文中描述的完整时隙操作
  
==============================================================================
1. 核心组件
==============================================================================

1.1 ComputationDynModel (src/bsk_rl/sim/dyn/computation_dynamics.py)
-------------------------------------------------------------------
- CPU 动力学建模：频率可调、功耗计算
- 继承 GroundStationDynModel → ImagingDynModel
- 参数（通过 @default_args）：
  * cpuMaxFrequency: 最大 CPU 频率 [Hz]
  * cpuMinFrequency: 最小 CPU 频率 [Hz]
  * cpuWorkload: 计算复杂度 [cycles/bit]
  * taskMinimumElevation: 任务可见性最小仰角 [rad]

1.2 ComputationSatellite (src/bsk_rl/sats/computation_satellite.py)
-------------------------------------------------------------------
- MARL 智能体实现
- 核心数据结构：
  * TaskSlice: 任务切片（包含完整时延建模）
  * TaskStatus: 任务状态枚举
  * task_queue: 待计算任务队列（输入数据）
  * result_queue: 待回传结果队列（输出数据）- 新增

- 核心方法：
  * set_resource_allocation(): 设置 CPU 频率和发射功率
  * receive_task_from_scenario(): 接收来自 Scenario 的原始任务
  * schedule_collaboration_action(): 执行任务切分和卸载
  * execute_local_compute(): 执行本地计算 + 初步可见性检查
  * execute_result_relay(): 处理结果接力和回传 - 新增
  * process_incoming_slice(): 接收其他卫星卸载的切片

- 队列分离的好处：
  * task_queue: 处理计算密集型任务（需要计算时延）
  * result_queue: 处理回传和接力（仅传播时延，简化模型）
  * 避免队列混用导致的数据丢失问题

1.3 STINContinuousAction (src/bsk_rl/act/stin_continuous_actions.py)
--------------------------------------------------------------------
- 10 维连续动作空间：
  [0]: cpu_ratio - CPU 频率比例 [0,1]
  [1]: tx_power_ratio - 发射功率比例 [0,1]
  [2]: platform_power_ratio - 平台功耗比例 [0,1]
  [3]: alpha_local - UD 本地保留比例 [0,1]（一级切分）
  [4]: alpha_cloud - 云端偏好因子 [0,1]（一级切分）
  [5-9]: x_0-x_4 - 卫星协作切分比例 [0,1]（二级切分）

- **层次化切分逻辑**（解决训练初期 alpha_sat=0 问题）：
  
  第一层：UD 本地和卸载分配
  ├─ alpha_local = action[3] (直接使用)
  └─ alpha_offload = 1 - alpha_local (卸载到云端和卫星)
  
  第二层：卸载部分中云端和卫星的相对比例
  ├─ preference_cloud = action[4] (云端偏好因子)
  ├─ alpha_cloud_final = alpha_offload × preference_cloud
  └─ alpha_sat_final = alpha_offload × (1 - preference_cloud)
  
  例如：action[3]=0.5, action[4]=0.5
  ├─ alpha_local = 0.5 (50% 本地)
  ├─ alpha_offload = 0.5
  ├─ alpha_cloud = 0.5 × 0.5 = 0.25 (25% 云端)
  └─ alpha_sat = 0.5 × 0.5 = 0.25 (25% 卫星)
  
  好处：即使两个动作都 0.5，卫星也能分到 25% 的任务
  
  第三层：卫星间协作切分
  └─ action[5-9] 的 x_0+x_1+...+x_4 会被归一化到 1.0
     表示 alpha_sat 部分在卫星间的分配

1.4 STINRelativeObservations (src/bsk_rl/obs/stin_relative_observations.py)
--------------------------------------------------------------------------
- 可配置的邻居观测（灵活设计）
- 默认配置（5 维 × 4 邻居 = 20 维）：
  * ISL 距离 [m]
  * 电池电量分数 [0, 1]
  * 任务队列长度
  * 排队工作量 [cycles]
  * CPU 频率 [Hz]

- 预定义属性函数：
  * get_isl_distance, get_battery_fraction, get_storage_fraction
  * get_task_queue_size, get_queue_workload, get_cpu_freq
  * r_DC_N (相对位置 3D), v_DC_N (相对速度 3D)

1.5 观测空间完整组成 (NEW!) 维度太多后续可能会减少
---------------------------
ComputationSatellite 的完整观测包含：

A. 卫星基础状态 (SatProperties):
   - battery_charge_fraction [0, 1]
   - storage_level_fraction [0, 1]
   - r_BN_P (位置向量 3D)
   - v_BN_P (速度向量 3D)

B. STIN 计算状态 (SatProperties, name="stin_state"):
   - current_cpu_freq [GHz]
   - task_queue_size
   - queue_workload [Gcycles]

C. **当前任务属性 (SatProperties, name="task_request") - 新增！**:
   对应论文 Step 1 "任务请求信息收集"
   来自 task_queue[0]（最新到达的任务）：
   - current_task_data_size [Mb] - 输入数据量
   - current_task_workload [kcycles/bit] - 计算复杂度
   - current_task_max_delay [s] - 时延约束
   - current_task_remaining_time [s] - 距离超时还剩时间
   - current_task_uplink_delay [ms] - 上行传播时延
   
   **重要性**: RL 智能体需要这些信息来决策：
   - 数据量 → 影响切分比例和资源分配
   - 计算复杂度 → 影响 CPU 频率分配（复杂度高需高频率）
   - 时延约束 → 影响优先级和能否通过云端/ISL 接力
   - 剩余时间 → 紧急程度指标

D. 邻居卫星状态 (STINRelativeObservations):
   默认 20 维 (5 属性 × 4 邻居)

E. 时间与光照:
   - Time()
   - Eclipse(norm=5700.0)

总观测维度 (默认配置): ~40 维
- 卫星基础: 10 维
- STIN 状态: 3 维
- 任务属性: 5 维 (新增)
- 邻居状态: 20 维
- 时间光照: 2 维

==============================================================================
2. 场景与任务
==============================================================================

2.1 ComputationTask (src/bsk_rl/scene/stin_scenario.py)
-------------------------------------------------------
- 表示用户终端（UD）发起的计算请求
- 属性：
  * r_LP_P / origin_position: 任务来源位置
  * data_size: 输入数据量 [bits]
  * workload: 计算复杂度 [cycles/bit]
  * max_delay: 最大容忍时延 [s]
  * priority: 优先级
  * uplink_rate: 上行传输速率 [bps]

2.2 STINTaskScenario
-------------------
- 生成计算任务分布
- 子类 CityTaskScenario: 基于城市分布

2.3 任务分配机制
---------------
- assign_tasks_to_satellites(): 检查可见性并分配任务
- 使用 AccessSatellite 的 opportunities 机制

==============================================================================
3. 时延模型（完整）
==============================================================================

**并行处理模型**: T_total = max(T_UD, T_SAT, T_CLOUD)

各路径时延计算：

T_UD = (α_local × data_size × workload) / f_ud
  - UD 本地处理，无上行传输（数据已在本地）
  - 计算时延由 UD CPU 频率决定

T_SAT = T_tx_up(α_sat) + T_prop_up + T_queue + T_compute(α_sat) + T_isl + T_prop_down
  其中：
  - T_tx_up(α_sat) = (α_sat × data_size) / uplink_rate
    * 按 alpha_sat 比例计算上行传输时延
    * 默认上行速率: 10 Mbps
  
  - T_prop_up = distance_up / c
    * 上行传播时延，c = 299792458 m/s
  
  - T_queue: 排队时延（任务在队列等待时间）
  
  - T_compute(α_sat) = (α_sat × data_size × workload) / cpu_freq
    * 卫星计算时延，按 alpha_sat 比例计算
  
  - T_isl: ISL 时延（根据是协作卸载还是结果接力而不同）
    见下面的 "ISL 时延的两种情况"
  
  - T_prop_down = distance_down / c
    * 下行传播时延，在任务完成时计算

T_CLOUD = T_tx_up(α_cloud) + T_tx_sgl(α_cloud) + 2×(T_prop_up + T_prop_sgl + T_fiber) + T_compute_cloud
  其中：
  - T_tx_up(α_cloud) = (α_cloud × data_size) / uplink_rate
    * 按 alpha_cloud 比例计算上行传输
  
  - T_tx_sgl(α_cloud) = (α_cloud × data_size) / sgl_rate
    * 卫星到网关的传输
  
  - 2×(T_prop_up + T_prop_sgl + T_fiber)
    * 往返传播（任务上传 + 结果返回）
  
  - T_compute_cloud = (α_cloud × data_size × workload) / f_cloud
    * 云端计算时延，默认 f_cloud = 10 GHz

**为什么是 max 而不是叠加？**
  - UD、卫星、云端并行处理各自分配的数据
  - 任务完成时间由最慢的路径决定
  - 符合论文的分层协作模型

**ISL 时延的两种情况**:

1. 协作卸载 (Collaboration Offloading - execute_local_compute 中的 schedule_collaboration_action):
   - 发生时机：在 schedule_collaboration_action() 中将任务切片分配给邻居卫星处理
   - 是否计算传输时延：**是的，需要计算**
   - 理由：需要传输大量输入数据到邻居卫星，传输时延显著
   - ISL 时延计算：
     * T_tx_isl = slice_data / ISL_rate （传输卸载的任务数据量）
     * T_prop_isl = distance_isl / c （传播时延）
     * 总 ISL = T_tx_isl + T_prop_isl
   - 累加位置：TaskSlice.t_tx_isl_total 和 t_prop_isl_total
   - 代码位置：schedule_collaboration_action() 第 4a 节点

2. 结果接力 (Result Relay - execute_result_relay 中的接力转发):
   - 发生时机：在 execute_result_relay() 中，当前卫星对 UD 不可见，需要转发给接力卫星
   - 是否计算传输时延：**否，仅计算传播时延**（简化工程模型）
   - 理由：结果数据量远小于原始任务数据（默认 10%），传输时延可忽略
   - ISL 时延计算（简化）：
     * T_prop_isl = distance_isl / c （仅传播时延）
     * T_tx_isl = 0 （忽略）
   - 累加位置：TaskSlice.t_prop_isl_total （不累加 t_tx_isl_total）
   - 代码位置：execute_result_relay() 的 relay_satellite 分支

**重要区别总结**：
- 协作卸载：数据量多（~100%），传输时延不可忽略，需计算 T_tx + T_prop
- 结果接力：数据量小（~10%），传输时延可忽略，仅计算 T_prop

**简化假设**：
- 结果数据量 = 10% × 输入数据量（可配置）
- 结果回传仅计算传播时延，忽略传输时延
- 下行回传仅计算传播时延（简化实际工程）
- 保持论文的时延建模一致性

==============================================================================
4. 奖励系统
==============================================================================

4.1 STINTaskData (src/bsk_rl/data/stin_task_data.py)
---------------------------------------------------
- 记录任务完成情况、时延、能耗
- 数据单元：
  * completed_tasks: 成功完成的任务列表
  * expired_tasks: 超时失败的任务列表
  * processed_data: 处理的数据量
  * offloaded_data: 卸载的数据量
  * energy_consumed: 消耗的能量

4.2 STINTaskReward
-----------------
奖励组成：

1. 任务完成奖励：
   R_complete = base_reward * (1 - delay_weight * normalized_delay)

2. 任务超时惩罚：
   R_timeout = -timeout_penalty

3. 能耗惩罚（可选）：
   R_energy = -energy_weight * normalized_energy

4. 负载均衡奖励（可选）：
   R_balance = -balance_weight * std(queue_lengths)

参数：
- base_reward: 基础完成奖励（默认 1.0）
- timeout_penalty: 超时惩罚系数（默认 2.0）
- delay_weight: 时延权重 [0, 1]（默认 0.5）
- energy_weight: 能耗权重（默认 0.0）
- balance_weight: 负载均衡权重（默认 0.0）

==============================================================================
5. 可选高级功能
==============================================================================

5.1 ISL 接力回传
---------------
- forward_completed_tasks_via_isl(): ISL 转发方法（占位符）
- get_isl_neighbors(): 获取 ISL 可见邻居
- 路由策略：
  * nearest_to_ud: 最接近 UD
  * best_visibility: 最佳可见性
  * shortest_path: 最短路径

5.2 通信协议
-----------
- 基于 BSK-RL 的 CommunicationMethod
- 可扩展实现数据共享和协作调度

==============================================================================
6. 使用示例
==============================================================================

参见: examples/stin_marl_example.py

基本流程：
```python
from bsk_rl import ConstellationTasking
from bsk_rl.data import STINTaskReward
from bsk_rl.scene import CityTaskScenario
from bsk_rl.sats import ComputationSatellite
from bsk_rl.utils.orbital import walker_delta_args

# 1. 创建环境
env = ConstellationTasking(
    satellites=[ComputationSatellite(...) for _ in range(6)],
    sat_arg_randomizer=walker_delta_args(...),
    scenario=CityTaskScenario(...),
    data=STINTaskReward(...),
)

# 2. 训练循环
obs, info = env.reset()
done = False
while not done:
    actions = policy(obs)  # 使用 RL 算法
    obs, rewards, terminated, truncated, info = env.step(actions)
    done = any(terminated.values()) or any(truncated.values())

env.close()
```

==============================================================================
7. 关键设计决策
==============================================================================

7.1 参数传递
-----------
- 使用 @default_args 装饰器在 dyn/fsw 层定义参数
- collect_default_args() 自动收集到 sat_args
- 避免硬编码，提高可配置性

7.2 任务生命周期与队列设计
-------------------------
- **task_queue**: 待计算任务（PENDING → COMPUTING → COMPUTED）
- **result_queue**: 待回传结果（COMPUTED → RELAYED → RETURNING → COMPLETED/EXPIRED）
- 队列分离避免数据丢失，明确的任务状态转移

7.3 时延模型简化原则
-------------------
**论文一致性与实际工程的平衡**：

1. 完整时延建模的部分：
   - 上行时延：T_tx_up + T_prop_up（任务上传）
   - 计算时延：T_compute（本地/卫星/云端）
   - 排队时延：T_queue（队列等待）
   - ISL 协作卸载时延：T_tx_isl + T_prop_isl（传输+传播）
   - ISL 结果接力时延：T_prop_isl（仅传播，简化）
   - 下行时延：T_prop_down（直接回传时）

2. 简化处理的部分：
   - 结果回传传输时延：**不计算**（result_data_size=10% × input_size）
   - 下行传输时延：**不计算**（假设下行速率很高）
   - UD 本地处理：逻辑记录，无上行传输（数据本地）
   - 云端处理：逻辑记录，无网络物理仿真

3. 简化的正当性：
   - 与论文时延公式一致（论文也未强调下行传输）
   - 反映实际系统下行通常比上行更容易的特点
   - 避免过度复杂化，便于 RL 智能体学习

7.4 可扩展性
-----------
- 模块化设计，易于扩展
- 支持自定义奖励函数
- 支持自定义任务分布
- 预留 ISL 接力回传接口

==============================================================================
8. 未来工作
==============================================================================

1. 实现完整的 ISL 接力回传路由算法
2. 添加更复杂的能耗建模（热管理、电池退化）
3. 实现联邦学习场景（隐私保护的协作学习）
4. 添加信道质量建模（多普勒效应、衰减）
5. 支持异构星座（不同性能的卫星）
6. 添加故障恢复机制
7. 实现更复杂的通信协议（MAC、路由）

==============================================================================
9. 文件清单
==============================================================================

核心实现：
- src/bsk_rl/sim/dyn/computation_dynamics.py
- src/bsk_rl/sats/computation_satellite.py
- src/bsk_rl/act/stin_continuous_actions.py
- src/bsk_rl/obs/stin_relative_observations.py
- src/bsk_rl/scene/stin_scenario.py
- src/bsk_rl/data/stin_task_data.py

示例：
- examples/stin_marl_example.py

文档：
- 本文件

==============================================================================
10. 参考文献与致谢
==============================================================================

本实现基于：
- BSK-RL 框架 (https://github.com/AVSLab/bsk_rl)
- Basilisk 航天器仿真器 (https://hanspeterschaub.info/basilisk/)
- STIN 任务卸载相关研究文献

"""
