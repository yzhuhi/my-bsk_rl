"""Data system for STIN task offloading and completion tracking.

This module defines the reward calculation for satellite-terrestrial integrated network (STIN) 
multi-agent reinforcement learning environments.
"""

import logging
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from bsk_rl.data.base import Data, DataStore, GlobalReward

if TYPE_CHECKING:
    from bsk_rl.sats import ComputationSatellite
    from bsk_rl.sats.computation_satellite import TaskSlice

logger = logging.getLogger(__name__)


class STINTaskData(Data):
    """Data for STIN task completion tracking.
    
    记录任务完成情况、时延、能耗等**增量数据**。
    
    注意：本类只存储增量/累计数据，不存储瞬时状态。 存增量
    瞬时状态（如电池电量、CPU利用率）应在需要时从卫星动态读取。

    任务列表：像是写日记。旧日记 + 新日记，但如果本子太厚了，就撕掉最早的前几页，只留最近 100 页。
    能耗/流量：像是存钱罐。旧的钱 + 新的钱 = 总钱数。
    平均时延：像是算绩点 (GPA)。根据学分（任务数量）进行加权平均，更新当前的平均表现。
    """

    def __init__(
        self,
        completed_tasks: Optional[list["TaskSlice"]] = None,
        expired_tasks: Optional[list["TaskSlice"]] = None,
        processed_data: float = 0.0,
        offloaded_data: float = 0.0,
        energy_consumed: float = 0.0,
        avg_latency: float = 0.0,
    ) -> None:
        """构建 STIN 任务数据单元。
        
        Args:
            completed_tasks: 本步成功完成的任务切片列表。
            expired_tasks: 本步超时失败的任务切片列表。
            processed_data: [bits] 本步处理的数据量（增量）。
            offloaded_data: [bits] 本步卸载的数据量（增量）。
            energy_consumed: [J] 本步消耗的能量（增量）。
            avg_latency: [s] 本步完成任务的平均时延。
        """
        self.completed_tasks = completed_tasks or []
        self.expired_tasks = expired_tasks or []
        self.processed_data = processed_data
        self.offloaded_data = offloaded_data
        self.energy_consumed = energy_consumed
        self.avg_latency = avg_latency

    def __add__(self, other: "STINTaskData") -> "STINTaskData":
        """合并两个数据单元（只合并增量数据）。
        
        Args:
            other: 另一个数据单元。
            
        Returns:
            合并后的数据单元。
        """
        # 合并任务列表，但限制大小以防止内存泄漏
        MAX_BUFFER_SIZE = 100  # 最多保留最近100个任务
        all_completed = (self.completed_tasks + other.completed_tasks)[-MAX_BUFFER_SIZE:]
        all_expired = (self.expired_tasks + other.expired_tasks)[-MAX_BUFFER_SIZE:]
        
        # 计算平均时延（加权平均）
        total_completed = len(self.completed_tasks) + len(other.completed_tasks)
        if total_completed > 0:
            n1 = len(self.completed_tasks)
            n2 = len(other.completed_tasks)
            avg_latency = (self.avg_latency * n1 + other.avg_latency * n2) / total_completed
        else:
            avg_latency = 0.0
        
        # 安全检查：防止累加产生 Infinity
        sum_energy = self.energy_consumed + other.energy_consumed
        if np.isnan(sum_energy) or np.isinf(sum_energy):
            sum_energy = max(self.energy_consumed, other.energy_consumed) if not (
                np.isnan(self.energy_consumed) or np.isinf(self.energy_consumed)
            ) else 0.0
        
        return STINTaskData(
            completed_tasks=all_completed,
            expired_tasks=all_expired,
            processed_data=self.processed_data + other.processed_data,
            offloaded_data=self.offloaded_data + other.offloaded_data,
            energy_consumed=sum_energy,
            avg_latency=avg_latency,
        )

class STINTaskStore(DataStore):
    """DataStore for STIN task completion tracking."""

    data_type = STINTaskData
    
    # 警告去重标志（类级别）
    _battery_read_warning_issued = False

    def __init__(self, *args, **kwargs) -> None:
        """初始化 STIN 任务数据存储。
        
        跟踪每个卫星的任务完成状态、资源使用情况。
        """
        super().__init__(*args, **kwargs)
        self.prev_completed_count = 0
        self.prev_expired_count = 0
        self.prev_processed_data = 0.0
        self.prev_offloaded_data = 0.0
        self.prev_battery_level = 0.0

    def get_log_state(self) -> dict:
        """记录当前步骤结束时的状态（用于计算增量）。
        
        Returns:
            包含任务统计和资源状态的字典。
        """
        sat: "ComputationSatellite" = self.satellite
        
        # 读取电池电量（用于计算能耗增量）
        battery_level = 0.0
        if hasattr(sat, 'dynamics') and hasattr(sat.dynamics, 'powerMonitor'):
            try:
                battery_msg = sat.dynamics.powerMonitor.batPowerOutMsg.read()
                battery_level = battery_msg.storageLevel  # [W·s]
            except Exception as e:
                # 首次警告，后续调试级别（避免日志洪泛）
                if not STINTaskStore._battery_read_warning_issued:
                    logger.warning(f"Cannot read battery level: {e}. Future errors will be debug level.")
                    STINTaskStore._battery_read_warning_issued = True
                else:
                    logger.debug(f"Cannot read battery level: {e}")
        
        # 记录仿真时间
        sim_time = sat.simulator.sim_time
        
        # 记录各硬件当前的瞬时功耗 (W)
        # 注意：BSK 中功耗通常记录为负值，我们取绝对值
        cpu_power = abs(sat.dynamics.cpuPowerSink.nodePowerOut) if hasattr(sat.dynamics, 'cpuPowerSink') and sat.dynamics.cpuPowerSink else 0.0
        tx_power = abs(sat.dynamics.transmitterPowerSink.nodePowerOut) if hasattr(sat.dynamics, 'transmitterPowerSink') and sat.dynamics.transmitterPowerSink else 0.0
        base_power = abs(sat.dynamics.basePowerSink.nodePowerOut) if hasattr(sat.dynamics, 'basePowerSink') and sat.dynamics.basePowerSink else 0.0
        
        return {
            "completed_count": sat.completed_tasks_count,
            "expired_count": sat.expired_tasks_count,
            "processed_data": sat.processed_data_total,
            "offloaded_data": sat.offloaded_data_total,
            "battery_level": battery_level,  # 用于奖励计算中的电池安全检查
            "sim_time": sim_time,            # 用于计算精细能耗
            "total_power": cpu_power + tx_power + base_power,
            "task_queue": list(sat.task_queue),
            "result_queue": list(sat.result_queue) if hasattr(sat, 'result_queue') else [],
        }

    def compare_log_states(
        self, old_state: dict, new_state: dict
    ) -> STINTaskData:
        """对比两个状态，生成增量数据。
        
        Args:
            old_state: 上一步的状态。
            new_state: 当前步的状态。
            
        Returns:
            本步骤产生的增量数据（不含瞬时状态）。
        """
        # 检测新完成的任务
        completed_delta = new_state["completed_count"] - old_state["completed_count"]
        expired_delta = new_state["expired_count"] - old_state["expired_count"]
        
        # 数据增量
        processed_delta = new_state["processed_data"] - old_state["processed_data"]
        offloaded_delta = new_state["offloaded_data"] - old_state["offloaded_data"]
        
        # 能耗计算：改为基于物理做功 (功耗 * 时间)，不再受太阳能充电干扰
        # 这种方式精度更高，且能反映任务执行的真实代价 
        dt = new_state["sim_time"] - old_state["sim_time"]
        if dt > 0:
            # 使用功率均值 (假设本步内功率恒定，BSK-RL 中 agent 动作为步级恒定)
            total_power = old_state.get("total_power", 0.0)
            # 安全检查：防止 NaN 或 Infinity 传播
            if np.isnan(total_power) or np.isinf(total_power):
                logger.warning(f"Invalid total_power: {total_power}, using 0.0")
                total_power = 0.0
            energy_consumed = total_power * dt
            # 防止能耗值异常
            if np.isnan(energy_consumed) or np.isinf(energy_consumed):
                logger.warning(f"Invalid energy: power={total_power}, dt={dt}")
                energy_consumed = 0.0
        else:
            energy_consumed = 0.0
        
        # 从卫星的历史缓冲区提取完成和超时的任务
        sat: "ComputationSatellite" = self.satellite
        completed_tasks = []
        expired_tasks = []
        
        if hasattr(sat, 'completed_tasks_buffer'):
            # 提取本 step 完成的任务
            completed_tasks = sat.completed_tasks_buffer.copy()
            # 清空缓冲区（避免重复计数）
            sat.completed_tasks_buffer.clear()
        
        if hasattr(sat, 'expired_tasks_buffer'):
            # 提取本 step 超时的任务
            expired_tasks = sat.expired_tasks_buffer.copy()
            # 清空缓冲区
            sat.expired_tasks_buffer.clear()
        
        # 计算平均时延（从实际完成的任务中提取）
        latencies = []
        for task in completed_tasks:
            # 使用 TaskSlice 中实现的 get_total_delay_parallel
            # 这会返回 max(T_UD, T_SAT, T_CLOUD)
            if hasattr(sat, 'simulator'):
                current_time = sat.simulator.sim_time
                latency = task.get_total_delay_parallel(current_time)
                latencies.append(latency)
                
        avg_latency = np.mean(latencies) if latencies else 0.0
        
        return STINTaskData(
            completed_tasks=completed_tasks,
            expired_tasks=expired_tasks,
            processed_data=processed_delta,
            offloaded_data=offloaded_delta,
            energy_consumed=energy_consumed,
            avg_latency=avg_latency,
        )


class STINTaskReward(GlobalReward):
    """GlobalReward for STIN task offloading environments.
    
    完整奖励设计：
        R_total = R_completion + R_latency + R_energy + R_balance
    
    其中：
    1. 任务完成奖励 (R_completion)：
       - R_complete = priority × (1 - delay_weight × normalized_delay) × base_reward
       - 优先级越高、完成越快，奖励越大 (✅ 已在第394行实现 priority_factor)
       
    2. 时延敏感奖励 (R_latency)：
       - R_latency = -latency_penalty × (actual_delay / max_delay)²
       - 二次惩罚函数，越接近deadline惩罚越重
       
    3. 能耗与电池安全约束 (R_energy)：
       - 线性能耗惩罚：R_energy_linear = -energy_weight × normalized_energy
       - 电池安全软约束：当 battery_soc < safety_threshold 时：
         R_battery_safety = -battery_penalty × exp((safety_threshold - battery_soc) / decay_rate)
       - 指数惩罚防止卫星"猝死"
       
    4. 负载均衡奖励 (R_balance)：
       - R_balance = -balance_weight × std(queue_lengths)
       - 鼓励任务在卫星间均匀分配
    
    参数说明：
        - base_reward: 基础完成奖励（默认 1.0）
        - timeout_penalty: 超时惩罚系数（默认 2.0）
        - delay_weight: 时延权重 [0, 1]（默认 0.5）
        - latency_penalty: 时延二次惩罚系数（默认 0.1）
        - energy_weight: 能耗权重（默认 0.01）
        - battery_penalty: 电池安全惩罚系数（默认 5.0）
        - battery_safety_threshold: 电池安全阈值 SOC（默认 0.2，即 20%）
        - balance_weight: 负载均衡权重（默认 0.0）
        - collaboration_weight: 协作奖励权重（默认 0.5）    
    """

    data_store_type = STINTaskStore

    def __init__(
        self,
        base_reward: float = 1.0,
        timeout_penalty: float = 2.0,
        delay_weight: float = 0.5,
        latency_penalty: float = 0.1,
        energy_weight: float = 0.01,
        battery_penalty: float = 5.0,
        battery_safety_threshold: float = 0.2,
        balance_weight: float = 0.0,
        collaboration_weight: float = 0.5,  # 协作奖励权重：卸载方获得完成奖励的比例
        reward_fn: Optional[Callable] = None,
    ) -> None:
        """初始化 STIN 任务奖励计算器。
        
        Args:
            base_reward: 基础完成奖励。
            timeout_penalty: 超时惩罚系数。
            delay_weight: 完成任务时延权重 [0, 1]，越高越重视低时延。
            latency_penalty: 时延二次惩罚系数（越接近 deadline 惩罚越重）。
            energy_weight: 能耗线性惩罚权重。
            battery_penalty: 电池安全指数惩罚系数。
            battery_safety_threshold: 电池安全阈值（SOC，0-1）。
            balance_weight: 负载均衡权重。
            collaboration_weight: 协作奖励权重。    
            reward_fn: 自定义奖励函数 fn(task_slice, satellite) -> reward。
        """
        super().__init__()
        self.base_reward = base_reward
        self.timeout_penalty = timeout_penalty
        self.delay_weight = delay_weight
        self.latency_penalty = latency_penalty
        self.energy_weight = energy_weight
        self.battery_penalty = battery_penalty
        self.battery_safety_threshold = battery_safety_threshold
        self.balance_weight = balance_weight
        self.collaboration_weight = collaboration_weight
        self.reward_fn = reward_fn
        self._satellite_cache: dict = {}  # 缓存卫星名称到对象的映射
        self._collaboration_rewards: dict = {}  # 协作奖励缓冲区

    def _get_satellite_by_name(self, sat_name: str):
        """根据名称获取卫星对象（带缓存）。
        
        Args:
            sat_name: 卫星名称。
            
        Returns:
            卫星对象，如果找不到返回 None。
        """
        # 使用缓存避免重复查找
        if sat_name in self._satellite_cache:
            return self._satellite_cache[sat_name]
        
        # 遍历卫星列表查找
        for sat in self.scenario.satellites:
            if sat.name == sat_name:
                self._satellite_cache[sat_name] = sat
                return sat
        
        return None

    def reset_overwrite_previous(self) -> None:
        """重置时清空卫星缓存和协作奖励缓冲区。"""
        super().reset_overwrite_previous()
        self._satellite_cache = {}
        self._collaboration_rewards = {}  # 防止跨 Episode 残留协作奖励

    def calculate_reward(
        self, new_data_dict: dict[str, STINTaskData]
    ) -> dict[str, float]:
        """计算每个卫星的步骤奖励（完整的时延敏感 + 能量安全奖励）。
        
        Args:
            new_data_dict: 每个卫星的新数据字典。
            
        Returns:
            每个卫星的奖励字典。
        """
        reward = {}
        
        for sat_id, new_data in new_data_dict.items():
            sat_reward = 0.0
            
            # ============================================================
            # 0. 奖励塑形 (Reward Shaping) - 提供中间信号
            # ============================================================
            # 0.1 每步生存奖励 - 鼓励卫星保持运行
            # sat_reward += 0.0001
            
            # # 0.2 任务接收奖励 - 鼓励接收任务
            # # 使用 new_data.queue_size 代替直接访问卫星对象，避免副作用
            # if new_data.queue_size > 0:
            #     sat_reward += 0.005 * min(new_data.queue_size, 5)  # 最多 0.25
                    
            # # 0.3 计算进度奖励 - 鼓励开始处理任务
            # # 使用 new_data.computing_count 代替直接访问卫星对象
            # if new_data.computing_count > 0:
            #     sat_reward += 0.001 * min(new_data.computing_count, 3)  # 最多 0.3
            

            # ============================================================
            # 1. 任务完成奖励 (R_completion)
            # ============================================================
            for task in new_data.completed_tasks:
                task_reward = 0.0
                if self.reward_fn:
                    sat = self._get_satellite_by_name(sat_id)
                    if sat:
                        task_reward = self.reward_fn(task, sat)
                else:
                    # 默认奖励：考虑（优先级）和时延    
                    normalized_delay = task.get_elapsed_time(
                        task.compute_end_time
                    ) / task.max_delay
                    # delay_factor = 1.0 - self.delay_weight * normalized_delay
                    priority_factor = getattr(task, 'priority', 0.1)    
                    delay_factor = max(0.1, 1.0 - self.delay_weight * normalized_delay)
                    task_reward = self.base_reward * delay_factor * priority_factor
                
                # 本卫星获得完整奖励
                sat_reward += task_reward
                
                # 协作奖励：如果任务是从其他卫星卸载来的，给原始卫星也发奖励
                origin_sat = getattr(task, 'origin_satellite', '')
                if origin_sat and origin_sat != sat_id and self.collaboration_weight > 0:
                    collab_reward = task_reward * self.collaboration_weight
                    if origin_sat not in self._collaboration_rewards:
                        self._collaboration_rewards[origin_sat] = 0.0
                    self._collaboration_rewards[origin_sat] += collab_reward
            
            # ============================================================
            # 2. 任务超时惩罚 (R_timeout) - 改进版
            # ============================================================
            for task in new_data.expired_tasks:
                # --- 改进点 1: 基于任务优先级的动态惩罚 ---
                # 假设 task.priority 范围是 1.0 ~ 5.0
                # 越重要的任务，失败了扣分越狠
                priority_factor = getattr(task, 'priority', 0.1)
                
                # 计算基础惩罚 (确保它是负值)
                current_penalty = self.timeout_penalty * priority_factor
                
                sat_reward -= current_penalty
                
                # --- 改进点 2: 连带责任 (Chain of Accountability) ---
                # 如果这个任务是别的卫星托付给我的，我搞砸了，我也要害那个卫星被扣分
                origin_sat = getattr(task, 'origin_satellite', '')
                
                if origin_sat and origin_sat != sat_id:
                    # 这是一个协作任务，我搞砸了
                    # 1. (可选) 对我(当前卫星)施加额外惩罚，因为我破坏了协作契约
                    sat_reward -= current_penalty * 0.5 
                    
                    # 2. 对原始卫星(委托方)施加连带惩罚
                    # 告诉它："你选错人了，下次别发给我"
                    if self.collaboration_weight > 0:
                        if origin_sat not in self._collaboration_rewards:
                            self._collaboration_rewards[origin_sat] = 0.0
                        
                        # 原始卫星也要承担一部分责任 (比如 50%)
                        self._collaboration_rewards[origin_sat] -= current_penalty * self.collaboration_weight
            
            # ============================================================
            # 3a. 时延敏感惩罚 (R_latency) - 二次惩罚 （ 上面失败任务已经惩罚了，这个惩罚可加可不加）
            # ============================================================
            # 从 scenario 动态读取系统最大时延，避免硬编码
            if new_data.avg_latency > 0 and len(new_data.completed_tasks) > 0:
                # 从 scenario 读取 max_delay_range 的上界作为归一化分母
                system_max_delay = 300.0  # 默认值
                if hasattr(self, 'scenario') and hasattr(self.scenario, 'max_delay_range'):
                    system_max_delay = self.scenario.max_delay_range[1]
                
                normalized_latency = min(1.0, new_data.avg_latency / system_max_delay)
                # 二次惩罚：越接近 deadline 惩罚越重
                latency_penalty_value = self.latency_penalty * (normalized_latency ** 2)
                sat_reward -= latency_penalty_value
            
            # ============================================================
            # 3b. 能耗惩罚 (R_energy) - 基于电池消耗比例
            # ============================================================
            # 使用相对能耗（消耗占电池容量的比例）而非绝对焦耳数，更具通用性
            # 参考: Wang et al., "Deep Reinforcement Learning for Task Offloading in Satellite IoT"
            if self.energy_weight > 0 and new_data.energy_consumed > 0:
                sat = self._get_satellite_by_name(sat_id)
                battery_capacity = 2700000.0  # 默认电池容量 [J]
                if sat and hasattr(sat, 'dynamics') and hasattr(sat.dynamics, 'powerMonitor'):
                    battery_capacity = getattr(sat.dynamics.powerMonitor, 'storageCapacity', battery_capacity)
                
                # 归一化：消耗的能量 / 电池容量 = 消耗占比 ∈ [0, 1]
                energy_ratio = new_data.energy_consumed / battery_capacity
                sat_reward -= self.energy_weight * energy_ratio
            
            # ============================================================
            # 3c. 电池安全软约束 (R_battery_safety) - 指数惩罚
            # ============================================================
            if self.battery_penalty > 0:
                sat = self._get_satellite_by_name(sat_id)
                # [修复 B] 获取电池 SOC，默认值改为 0.7 避免读取失败时误惩罚
                battery_soc = 0.7  # 默认假设电池正常（70%）
                if sat and hasattr(sat, 'dynamics') and hasattr(sat.dynamics, 'powerMonitor'):
                    try:
                        battery_msg = sat.dynamics.powerMonitor.batPowerOutMsg.read()
                        battery_capacity = sat.dynamics.powerMonitor.storageCapacity
                        if battery_capacity > 0:
                            battery_soc = battery_msg.storageLevel / battery_capacity
                    except Exception as e:
                        logger.debug(f"Cannot read battery SOC for {sat_id}: {e}. Using default 0.7")
                
                # 当电池低于安全阈值时施加指数惩罚
                if battery_soc < self.battery_safety_threshold:
                    # 指数衰减：SOC 越低，惩罚指数增长
                    decay_rate = 0.1  # 衰减率
                    safety_violation = self.battery_safety_threshold - battery_soc
                    raw_penalty = self.battery_penalty * np.exp(safety_violation / decay_rate)
                    
                    # [修复 C] 截断惩罚上限，防止梯度爆炸
                    MAX_BATTERY_PENALTY = 15.0  # 最大惩罚不超过 15
                    battery_safety_penalty = min(raw_penalty, MAX_BATTERY_PENALTY)
                    sat_reward -= battery_safety_penalty
                    
                    # 记录警告
                    if battery_soc < 0.1:  # 10% 严重警告
                        logger.warning(
                            f"{sat_id} battery critically low: SOC={battery_soc:.2%}, "
                            f"penalty={battery_safety_penalty:.2f} (raw={raw_penalty:.2f})"
                        )
            
            reward[sat_id] = sat_reward
        
        # ============================================================
        # 4. 负载均衡奖励 (R_balance) - 全局奖励 
        # ============================================================
        if self.balance_weight > 0:
            queue_lengths = []
            for sat_id in new_data_dict.keys():
                sat = self._get_satellite_by_name(sat_id)
                # 增加防御性判断，防止 task_queue 属性不存在
                if sat and hasattr(sat, 'task_queue'):
                    queue_lengths.append(len(sat.task_queue))
            
            if len(queue_lengths) > 1:
                queue_std = np.std(queue_lengths)
                
                # 使用绝对标准差作为负载不均衡度量
                # 注：也可考虑使用变异系数 queue_std / (np.mean(queue_lengths) + 1e-6)
                balance_reward = -self.balance_weight * queue_std
                
                # 归一化到单星，防止卫星数量变化影响总奖励量级
                per_agent_penalty = balance_reward / len(reward)
                
                for sat_id in reward:
                    reward[sat_id] += per_agent_penalty

        # ============================================================
        # 5. 发放协作奖励 (R_collaboration)
        # ============================================================
        # 将累积的协作奖励发放给发起卸载的卫星
        for origin_sat_id, collab_reward in self._collaboration_rewards.items():
            # 确保 ID 类型匹配 (有些环境 ID 是 int, 有些是 str)
            # 如果 reward key 是 int, 而 origin_sat_id 是 "EO-1", 这里需要转换逻辑
            # 假设你已经处理好了 ID 一致性：
            if origin_sat_id in reward:
                reward[origin_sat_id] += collab_reward
                # 只有在非零奖励时才打日志，减少刷屏
                if abs(collab_reward) > 0.001:
                    logger.debug(f"[Collab] {origin_sat_id} received +{collab_reward:.4f} offloading bonus")
        
        # 必须清空！否则奖励会重复累加
        self._collaboration_rewards.clear()
        
        return reward

    def is_truncated(self, satellite) -> bool:
        """检查是否应该截断 episode（资源耗尽或其他软约束违反）。
        
        Args:
            satellite: 卫星对象。
            
        Returns:
            True 如果应该截断 episode。
        """
        # 检查电池是否耗尽（硬约束）
        if hasattr(satellite, 'dynamics') and hasattr(satellite.dynamics, 'powerMonitor'):
            try:
                battery_msg = satellite.dynamics.powerMonitor.batPowerOutMsg.read()
                battery_capacity = satellite.dynamics.powerMonitor.storageCapacity
                if battery_capacity > 0:
                    battery_soc = battery_msg.storageLevel / battery_capacity
                    # 电池 SOC < 5% 时截断（防止"猝死"） 卫星大概率有充电机制，这个我感觉不需要，并且强制截断回合
                    # 其他正在运行的卫星也会出问题，不太符合协作理念
                    if battery_soc < 0.00:
                        logger.warning(
                            f"{satellite.name} battery critically depleted: "
                            f"SOC={battery_soc:.2%}, truncating episode"
                        )
                        return True
            except Exception as e:
                logger.debug(f"Cannot check battery level: {e}")
        
        return False

    def is_terminated(self, satellite) -> bool:
        """检查是否应该终止 episode（任务完成）。"""
        # 简化实现：当所有任务都完成或超时时终止
        if hasattr(self.scenario, 'tasks'):
            total_tasks = len(self.scenario.tasks)
            # 注意：self.scenario.satellites 是列表，不是字典
            completed = sum(
                sat.completed_tasks_count 
                for sat in self.scenario.satellites
                if hasattr(sat, 'completed_tasks_count')
            )
            expired = sum(
                sat.expired_tasks_count 
                for sat in self.scenario.satellites
                if hasattr(sat, 'expired_tasks_count')
            )
            
            if completed + expired >= total_tasks:
                logger.info(
                    f"All tasks processed: {completed} completed, {expired} expired"
                )
                return True
        
        return False


__doc_title__ = "STIN Task Data"
__all__ = ["STINTaskReward", "STINTaskStore", "STINTaskData"]

