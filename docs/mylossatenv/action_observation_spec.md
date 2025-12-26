# 动作与观测空间规范

## 动作空间 (10D 连续)

| 索引 | 名称 | 范围 | 说明 |
|------|------|------|------|
| 0 | `CPU_Ratio` | [0, 1] | CPU 频率分配比例 → [f_min, f_max] |
| 1 | `TX_Power_Ratio` | [0, 1] | 发射功率分配比例 |
| 2 | `Platform_Power_Ratio` | [0, 1] | 平台功耗管理比例 |
| 3 | `Alpha_Local` | [0, 1] | UD 本地保留比例（第一层切分）|
| 4 | `Alpha_Cloud` | [0, 1] | 云端偏好因子（第二层切分）|
| 5 | `x_0` | [0, 1] | 自身处理比例（归一化后）|
| 6-9 | `x_1` ~ `x_4` | [0, 1] | 邻居 1-4 协作比例 |

### 层次化切分模型

```
第一层: α_local 决定 UD 本地保留
        ↓
第二层: 卸载部分 (1 - α_local) 按 α_cloud 分配
        ├── 云端: (1 - α_local) × α_cloud
        └── 卫星: (1 - α_local) × (1 - α_cloud)
                    ↓
        第三层: 卫星部分按 x_0~x_4 在自身和邻居间切分
```

## 观测空间

### 卫星状态 (4D)
| 名称 | 归一化 | 说明 |
|------|--------|------|
| `battery_charge_fraction` | - | 电池电量分数 [0, 1] |
| `storage_level_fraction` | - | 存储空间分数 [0, 1] |
| `r_BN_P` | REQ_EARTH×1e3 | 位置向量 [m] |
| `v_BN_P` | 7616.5 | 速度向量 [m/s] |

### STIN 状态 (3D)
| 名称 | 归一化 | 说明 |
|------|--------|------|
| `current_cpu_freq` | 1e9 | 当前 CPU 频率 [GHz] |
| `task_queue_size` | - | 任务队列长度 |
| `queue_workload` | 1e9 | 队列工作量 [Gcycles] |

### 任务属性 (5D)
| 名称 | 单位 | 说明 |
|------|------|------|
| `current_task_data_size` | Mb | 当前任务数据量 |
| `current_task_workload` | kcycles/bit | 计算复杂度 |
| `current_task_max_delay` | s | 最大时延要求 |
| `current_task_remaining_time` | s | 剩余可用时间 |
| `current_task_uplink_delay` | ms | 上行链路时延 |

### 邻居状态 (每邻居 5D × max_neighbors)
| 名称 | 归一化 | 说明 |
|------|--------|------|
| `isl_distance` | 1e7 | ISL 距离 [m] |
| `battery_fraction` | - | 电池电量分数 |
| `task_queue_size` | 20.0 | 任务队列长度 |
| `queue_workload` | 1e10 | 工作量 [cycles] |
| `cpu_freq` | 2e9 | CPU 频率 [Hz] |

### 时间 (2D)
| 名称 | 说明 |
|------|------|
| `Time` | 仿真时间（归一化）|
| `Eclipse` | 日食状态 |

## 示例配置

```python
observation_spec = [
    obs.SatProperties(
        dict(prop="battery_charge_fraction", module="dynamics"),
        dict(prop="r_BN_P", module="dynamics", norm=REQ_EARTH * 1e3),
    ),
    obs.STINRelativeObservations(
        dict(prop="get_isl_distance", norm=1e7),
        dict(prop="get_battery_fraction"),
        max_neighbors=4,
    ),
    obs.Time(),
]
```
