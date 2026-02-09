# STIN 分层 RL（HOP-PPO → H-MAPPO）实现计划

## 目标
在不破坏现有框架与对比实验的前提下，实现“分层 RL”方案：
- **第一层（层次化切分）**：只学习 `action[3:5]`（一级切分参数）。
- **第二层（多星协作）**：在第一层输出基础上，学习资源与路由（CPU/TX/平台功耗/优先级阈值/路由），采用 **HOP-PPO 条件化混合动作** 扩展到 **MARL (H‑MAPPO)**。

## 方案拆分

### 方案 A（现有动作空间 + 模式 B 改进） 已经实现
保持现有 16 维动作，但修复模式 B 的“平均优先级”逻辑，使其对队列逐切片判断高/低优先级并使用对应路由比例。

**预期变更**
- 在模式 B 中按切片优先级逐一选择 `high_split_ratios` / `low_split_ratios`。
- 支持每个切片独立路由（不再使用平均优先级）。

### 方案 B（新增分层 RL + H‑MAPPO）考虑
新增脚本与动作/观测，不覆盖现有框架。

#### 分层策略
1. **第一层（High‑Level Policy）**
   - 维度: 3 + K (K=4 时为 7D)
   - [alpha_ud, alpha_cloud, alpha_sat0(自己-接入卫星), alpha_sat1, ..., alpha_satK]
   - 对 raw 任务队列中的每个任务，按此比例切分到 UD本地/云端/接入卫星/各邻居卫星
   - 归一化后 sum = 1

2. **第二层（Low‑Level Policy, MARL + HOP‑PPO）**
   - 输入：环境嵌入 + 第一层输出 + 选择的协作目标特征。
   就相当于说切片不可再分了，对任务切片队列就是做路由；
但是由于我们的step是一个duration time，我们要考虑动作怎么设计才能对当前step内的切片做一个轮询处理；
- 连续部分:
  * 资源参数 (4D): [cpu, tx, platform, priority_threshold]
  * 高优先级切片路由 (K+1)D: [high_self, high_n0, high_n1, ..., high_nK-1]
  * 低优先级切片路由 (K+1)D: [low_self, low_n0, low_n1, ..., low_nK-1]
- 离散部分 (K+1): 主路由目标（fallback）
   - 采用条件化解码：先选路由，再基于该目标特征输出连续动作。

## 动作空间设计（方案 B）

### 第一层动作（High‑Level）
- `alpha_local ∈ [0, 1]`
- `alpha_cloud ∈ [0, 1]`
- `alpha_sat0(自己-接入卫星) ∈ [0, 1]`
- `alpha_sat1 ∈ [0, 1]`
- `alpha_sat2 ∈ [0, 1]`
- `alpha_sat3 ∈ [0, 1]`

### 第二层动作（Low‑Level, Hybrid）
- **离散**：`route_target ∈ {0..K}`，0 表示本地，1..K 表示邻居索引
- **连续**：
  - `cpu_ratio ∈ [0,1]`
  - `tx_power_ratio ∈ [0,1]`
  - `platform_power_ratio ∈ [0,1]`
  - `priority_threshold ∈ [0,1]`
  - 怎么设计还有疑问

### 联合概率（PPO 计算）
需要用联合概率来计算 ratio：

$$
\pi(a|s) = \pi_d(d|s) \cdot \pi_c(c|s, d)
$$

$$
\log \pi(a|s) = \log \pi_d(d|s) + \log \pi_c(c|s, d)
$$

## 观测空间（方案 B）

### 第一层观测
- 任务队列负载、平均延迟、资源消耗、电量、地面接入机会等（偏“任务/资源”视角）

### 第二层观测
- 第一层输出（`alpha_local`, `alpha_cloud`）
- 邻居特征列表（信道、距离、可见性、队列）
- 自身资源与状态

建议使用 **Transformer 编码器** 处理邻居列表，支持动态邻居数量。

## 新增脚本建议（不覆盖现有）

### Action
- 新增：`bsk_rl/act/stin_hier_hybrid_actions.py`
  - 定义 High‑Level 与 Low‑Level 动作解码
  - Low‑Level 采用条件化混合动作（HOP‑PPO）

### Observation
- 新增：`bsk_rl/obs/stin_hier_observations.py`
  - 输出 High‑Level 与 Low‑Level 各自的观测

### Algorithm
- 新增：`bsk_rl/algorithms/hmappo_hier.py`
  - 两层 PPO 训练或交替更新
  - 集中训练/分散执行（CTDE）

### Model
- 新增：`bsk_rl/models/hier_hop_actor_critic.py`
  - Transformer 观测编码器
  - Discrete + Conditional Continuous 头

## 训练流程建议
1. 先固定第二层，只训练第一层，确保一级切分收敛。
2. 固定第一层后训练第二层（协作与路由）。
3. 交替训练（或联合训练）以优化整体性能。

## 风险点与验证项
- **动作一致性**：第一层输出必须稳定传给第二层。
- **路由掩码**：不可达邻居必须在 `route_target` 中被 mask。
- **奖励分解**：建议设计两层 reward 或共享全局 reward。
- **收敛性**：建议先单星调通再扩展多星。

## 里程碑
1. 完成方案 A（修复模式 B 路由细粒度逻辑）。
2. 新增分层动作类与观测类（不覆盖原始代码）。
3. 实现 H‑MAPPO 训练器与 Actor‑Critic。
4. 完成对比实验（原 MAPPO vs 分层 H‑MAPPO）。
