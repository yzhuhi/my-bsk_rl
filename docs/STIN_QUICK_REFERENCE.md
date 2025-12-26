# STIN MARL 快速参考

## 🎯 论文完整性检查

### ✅ 已实现的论文核心内容

| 论文要素 | 实现位置 | 说明 |
|---------|----------|------|
| **时隙操作流程 (5步)** | `ComputationSatellite` | 完整实现 Step 1-5 |
| Step 1: 信息收集 | `observation_spec` | 观测空间自动收集网络信息 |
| Step 2: 决策 | 外部 RL + `STINContinuousAction` | 任务卸载和资源分配决策 |
| Step 3: 控制消息生成 | `set_resource_allocation()` + `schedule_collaboration_action()` | 动作解析为控制参数 |
| Step 4a: 任务计算 | `execute_local_compute()` | 计算执行 + 初步可见性检查 |
| Step 4b: 结果接力 | `execute_result_relay()` | 结果回传 + ISL 接力 |
| Step 5: 切换处理 | `handle_handover()` | ISL 状态同步 |
| **完整时延模型** | `TaskSlice` | T_tx + T_prop + T_queue + T_compute |
| **分层协作** | `schedule_collaboration_action()` | 一级切分(UD/云/卫星) + 二级切分(卫星间) |
| **连续资源分配** | `set_resource_allocation()` | CPU 频率、发射功率 |
| **任务属性观测** | `observation_spec` | 数据量、工作负载、时延约束等 |

### 📊 观测空间 (约 40 维)

```python
observation_spec = [
    # 1. 卫星基础状态 (10 维)
    obs.SatProperties(
        dict(prop="battery_charge_fraction", ...),
        dict(prop="storage_level_fraction", ...),
        dict(prop="r_BN_P", ...),  # 3D
        dict(prop="v_BN_P", ...),  # 3D
    ),
    
    # 2. STIN 计算状态 (3 维)
    obs.SatProperties(
        dict(prop="current_cpu_freq", ...),
        dict(prop="task_queue_size", ...),
        dict(prop="queue_workload", ...),
        name="stin_state"
    ),
    
    # 3. 当前任务属性 (5 维) - 对应论文 Step 1
    obs.SatProperties(
        dict(prop="current_task_data_size", ...),      # 数据量
        dict(prop="current_task_workload", ...),       # 计算复杂度
        dict(prop="current_task_max_delay", ...),      # 时延约束
        dict(prop="current_task_remaining_time", ...), # 剩余时间
        dict(prop="current_task_uplink_delay", ...),   # 上行时延
        name="task_request"
    ),
    
    # 4. 邻居状态 (20 维 = 5×4)
    obs.STINRelativeObservations(
        dict(prop="get_isl_distance", norm=1e7),
        dict(prop="get_battery_fraction"),
        dict(prop="get_task_queue_size", norm=20.0),
        dict(prop="get_queue_workload", norm=1e10),
        dict(prop="get_cpu_freq", norm=2e9),
        max_neighbors=4,
    ),
    
    # 5. 时间光照 (2 维)
    obs.Time(),
    obs.Eclipse(norm=5700.0),
]
```

### 🎮 动作空间 (10 维连续)

```python
action = [
    action[0],  # cpu_ratio: CPU 频率比例 [0,1]
    action[1],  # tx_power_ratio: 发射功率比例 [0,1]
    action[2],  # platform_power_ratio: 平台功耗比例 [0,1]
    action[3],  # alpha_local: UD 本地处理比例 [0,1]
    action[4],  # alpha_cloud: 云端处理偏好因子 [0,1]
    action[5],  # x_0: 自身处理比例 [0,1]
    action[6],  # x_1: 邻居1处理比例 [0,1]
    action[7],  # x_2: 邻居2处理比例 [0,1]
    action[8],  # x_3: 邻居3处理比例 [0,1]
    action[9],  # x_4: 邻居4处理比例 [0,1]
]
```

动作约束（层次化切分）：
- `alpha_local` 决定 UD 本地保留比例
- `alpha_cloud` 在剩余部分中作为偏好因子（决定云端和卫星的相对比例）
- `alpha_sat = (1 - alpha_local) × (1 - alpha_cloud)`
- 训练初期两个动作都是 0.5 时：Local=50%, Cloud=25%, Sat=25%
- `x_0 + x_1 + ... + x_4` 被归一化到 1（卫星间相对切分）

**层次化的好处**：确保即使两个动作都接近 0.5，卫星也能分到任务（不会出现 alpha_sat=0）

### ⏱️ 正确的并行时延模型

**重要修正**: 三条路径并行处理，总时延取最大值！各路径按比例计算传输时延！

```
T_total = max(T_UD, T_SAT, T_CLOUD)

其中：
├─ T_UD       = (α_local × data_size × workload) / f_ud  (UD 本地处理，无上行传输)
│
├─ T_SAT      = T_tx_up(α_sat) + T_prop_up + T_queue + T_compute + T_isl + T_prop_down
│   ├─ T_tx_up(α_sat) = (α_sat × data_size) / R_uplink  (按比例计算上行传输)
│   ├─ T_prop_up      = d_up / c
│   ├─ T_sat_process  = T_queue + (α_sat × data_size × workload) / f_cpu
│   ├─ T_isl          = T_tx_isl + T_prop_isl  (ISL 传输 + 传播)
│   └─ T_prop_down    = d_down / c
│
└─ T_CLOUD    = T_tx_up(α_cloud) + T_tx_sgl(α_cloud) + 2×(T_prop) + T_cloud_compute
    ├─ T_tx_up(α_cloud)  = (α_cloud × data_size) / R_uplink  (按比例计算)
    ├─ T_tx_sgl(α_cloud) = (α_cloud × data_size) / R_sgl
    ├─ 2×T_prop = 2×(T_prop_up + T_prop_sgl + T_fiber)  (往返传播)
    └─ T_cloud  = (α_cloud × data_size × workload) / f_cloud

光速 c = 299,792,458 m/s
ISL 速率默认 = 100 Mbps
云端 CPU 频率默认 = 10 GHz
UD 本地 CPU 频率默认 = 1 GHz
```

**为什么是 max 而不是叠加？**
- UD、卫星、云端**并行处理**各自分配的数据切片
- 任务完成时间由**最慢的路径**决定
- 这符合论文的分层协作模型

**ISL 时延的两种情况**:
```
1. 协作卸载 (Collaboration Offloading) - execute_local_compute 阶段:
   - 发生时机：决策阶段将任务切片分配给邻居卫星处理
   - ISL 时延 = 传输时延 + 传播时延
   - T_tx_isl = slice_data / ISL_rate (传输卸载的任务数据)
   - T_prop_isl = distance / c
   - 累加到 TaskSlice.t_tx_isl_total 和 t_prop_isl_total

2. 结果接力 (Result Relay) - execute_result_relay 阶段:
   - 发生时机：计算完成后，当前卫星对 UD 不可见
   - ISL 时延 = 仅传播时延（简化模型）
   - 理由：结果数据量远小于原始任务数据（默认 10%），传输时延可忽略
   - T_prop_isl = distance / c
   - 仅累加到 TaskSlice.t_prop_isl_total（不计算传输时延）
```

**可见性约束**:
```
如果当前卫星失去对 UD 的可见性：
├─ 必须在失去可见性前完成已分配的计算
└─ 通过 ISL 将结果转发给接力卫星
    ├─ 接力卫星选择标准：
    │   1. ISL 可达（< 5000 km）
    │   2. 对 UD 当前或即将可见
    │   3. 距离 UD 最近
    └─ 接力卫星负责与地面建立链路，完成交付
```

## 🔧 核心 API 使用

### 1. 创建环境

```python
from bsk_rl import ConstellationTasking
from bsk_rl.data import STINTaskReward
from bsk_rl.scene import CityTaskScenario
from bsk_rl.sats import ComputationSatellite
from bsk_rl.utils.orbital import walker_delta_args

env = ConstellationTasking(
    satellites=[
        ComputationSatellite(
            name=f"sat-{i}",
            sat_args={
                "cpuMaxFrequency": 2.0e9,    # 2 GHz
                "cpuMinFrequency": 0.5e9,    # 0.5 GHz
                "cpuWorkload": 500.0,        # 500 cycles/bit
                "taskMinimumElevation": 10.0 * np.pi / 180,
            }
        )
        for i in range(6)
    ],
    sat_arg_randomizer=walker_delta_args(
        n_planes=2, n_sats_per_plane=3, altitude=800, inc=60
    ),
    scenario=CityTaskScenario(
        n_tasks=20,
        data_size_range=(1e6, 10e6),      # 1-10 Mb
        workload_range=(100, 1000),        # cycles/bit
        max_delay_range=(5.0, 20.0),       # 5-20 s
    ),
    data=STINTaskReward(
        base_reward=1.0,
        timeout_penalty=2.0,
        delay_weight=0.5,
    ),
    max_step_duration=60.0,  # 60 s per step
)
```

### 2. 时隙操作执行

```python
# 方法 1: 使用标准 Gym API (推荐)
obs, info = env.reset()
action = env.action_space.sample()  # 或使用 RL 策略
obs, reward, terminated, truncated, info = env.step(action)

# 方法 2: 直接调用时隙方法 (高级用法)
satellite = env.satellites["sat-0"]
result = satellite.execute_slot_step(
    action=action_vector,
    step_duration=60.0,
    neighbor_satellites=neighbor_list,
)
```

### 3. 访问状态信息

```python
# 任务队列
satellite.task_queue  # List[TaskSlice]

# 当前资源分配
satellite.current_cpu_freq       # [Hz]
satellite.current_tx_power       # [W]

# 统计量
satellite.completed_tasks_count  # 成功完成数
satellite.expired_tasks_count    # 超时失败数
satellite.processed_data_total   # [bits]

# 时隙控制状态
satellite.network_info           # 收集的网络信息 (Step 1)
satellite.current_control_message  # 控制消息 (Step 3)
```

## 📦 文件清单

```
src/bsk_rl/
├── sim/dyn/
│   └── computation_dynamics.py      # CPU 动力学建模
├── sats/
│   └── computation_satellite.py     # MARL 智能体 (793+ 行)
├── act/
│   └── stin_continuous_actions.py   # 10D 连续动作空间
├── obs/
│   └── stin_relative_observations.py  # 可配置邻居观测
├── scene/
│   └── stin_scenario.py             # 任务场景生成
└── data/
    └── stin_task_data.py            # 奖励计算

examples/
└── stin_marl_example.py             # 完整使用示例

docs/
├── STIN_IMPLEMENTATION_SUMMARY.md   # 详细实现文档
└── STIN_QUICK_REFERENCE.md          # 本文件
```

## 🔍 关键设计亮点

1. **论文一致性**: 完整实现 5 步时隙操作流程
2. **信息完整性**: 任务属性纳入观测空间 (Step 1)
3. **控制消息**: 显式建模下发给 UD/云端/卫星的指令 (Step 3)
4. **状态追踪**: `network_info` 和 `current_control_message`
5. **模块化**: 每个步骤可独立调用或整体执行
6. **可扩展**: 预留 ISL 路由、切换处理等高级功能接口

## 🚀 快速测试

```bash
# 激活环境
conda activate Basilisk

# 运行示例
cd e:/py_project/bsk_rl
python examples/stin_marl_example.py
```

## 📚 参考

- 详细文档: `docs/STIN_IMPLEMENTATION_SUMMARY.md`
- 使用示例: `examples/stin_marl_example.py`
- BSK-RL 文档: https://github.com/AVSLab/bsk_rl
