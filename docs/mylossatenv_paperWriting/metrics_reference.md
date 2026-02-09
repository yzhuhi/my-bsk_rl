# STIN 物理量日志指标参考

> 本文档记录所有自定义的物理量指标，用于 WandB 训练监控。

---

## 1. 基础任务指标

| 指标               | 公式                             | 单位 | 说明                         |
| ------------------ | -------------------------------- | ---- | ---------------------------- |
| `ep_completed`     | count(completed_task_ids)        | 个   | Episode 内完成的任务数       |
| `ep_expired`       | count(expired_task_ids)          | 个   | Episode 内超时的任务数       |
| `ep_unassigned`    | len(arrived_tasks)               | 个   | 到达但未被任何卫星接收的任务 |
| `ep_total_arrived` | unassigned + completed + expired | 个   | 实际到达的总任务数           |

---

## 2. 核心 KPI 指标

| 指标                   | 公式                              | 范围   | 目标       |
| ---------------------- | --------------------------------- | ------ | ---------- |
| `ep_completion_rate`   | completed / (completed + expired) | [0, 1] | ↑ 越高越好 |
| `ep_avg_latency`       | Σ(task_latency) / completed       | 秒     | ↓ 越低越好 |
| `ep_energy_efficiency` | processed_bits / energy_consumed  | bits/J | ↑ 越高越好 |

---

## 3. 卸载行为指标 (Process Metrics)

| 指标                      | 公式                                    | 范围   | 理想值                    |
| ------------------------- | --------------------------------------- | ------ | ------------------------- |
| `ep_offload_ratio`        | offloaded_tasks / (completed + expired) | [0, 1] | 仅参考，已过时            |
| `ep_relay_ratio`          | relay_tasks / (completed + expired)     | [0, 1] | 20%~40%                   |
| `ep_active_offload_ratio` | first_hop_offloaded / raw_received      | [0, 1] | 20%~40%（**主动**卸载率） |
| `ep_avg_hops`             | Σ(hop_count) / completed                | ≥0     | 1.0~3.0（协作深度）       |

---

## 4. 协作效果指标 (Outcome Metrics)

### 4.1 成功任务来源分析
| 指标                     | 公式                                   | 范围   | 说明         |
| ------------------------ | -------------------------------------- | ------ | ------------ |
| `ep_collab_contribution` | remote_completed / completed           | [0, 1] | 协作完成占比 |
| `ep_solo_success_ratio`  | solo_completed / completed             | [0, 1] | 单独完成占比 |
| **恒等式**               | collab_contribution + solo_success = 1 |        |              |

### 4.2 失败任务来源分析
| 指标                      | 公式                              | 范围   | 说明         |
| ------------------------- | --------------------------------- | ------ | ------------ |
| `ep_collab_failure_ratio` | offloaded_expired / expired       | [0, 1] | 协作失败占比 |
| `ep_solo_failure_ratio`   | solo_expired / expired            | [0, 1] | 单独失败占比 |
| **恒等式**                | collab_failure + solo_failure = 1 |        |              |

### 4.3 卸载质量
| 指标                      | 公式                                  | 范围   | 目标       |
| ------------------------- | ------------------------------------- | ------ | ---------- |
| `ep_offload_success_rate` | offloaded_completed / offloaded_total | [0, 1] | ↑ 越高越好 |

---

## 5. 内部调试指标（不记录到 WandB）

| 指标            | 说明                        |
| --------------- | --------------------------- |
| `_step_energy`  | 本步能耗 (J)                |
| `_ep_energy`    | Episode 累积能耗 (J)        |
| `_ep_processed` | Episode 累积处理数据 (bits) |

---

## 6. 训练曲线解读

### 理想的 WandB 曲线
- `ep_active_offload_ratio`: 稳定在 20%~40%（按需分配）
- `ep_collab_contribution`: 与卸载率正相关，越高越好
- `ep_avg_hops`: 1.0~3.0 之间（利用多跳网络）
- `ep_completion_rate`: 逐渐上升
- `ep_avg_latency`: 逐渐下降
- `ep_energy_efficiency`: 逐渐上升

### 异常信号
| 现象                               | 可能原因                   |
| ---------------------------------- | -------------------------- |
| `collab_contribution` = 0          | 无协作，各自为战           |
| `collab_failure` >> `solo_failure` | 协作策略差，卸载给忙碌卫星 |
| `avg_hops` > 5                     | 任务在网络中无效跳转       |
| `offload_success_rate` < 0.3       | 卸载策略失败               |




指标	说明
任务层面	
ep_completed	成功任务数
ep_expired	失败任务数
ep_completion_rate	成功率
ep_total_arrived	到达总数
ep_unassigned	未分配数
性能层面	
ep_avg_latency	平均时延
ep_energy_efficiency	能效 (bits/J)
协作层面	
ep_sat_path_ratio	卫星路径任务比例（分配给卫星的任务/所有决策任务）
ep_active_offload_ratio	主动卸载率（卸载数据/接收数据）
ep_avg_hops	平均跳数（协作深度）
ep_collab_contribution	协作贡献率（协作完成/总完成）
ep_collab_failure_ratio	协作失败率（诊断用）