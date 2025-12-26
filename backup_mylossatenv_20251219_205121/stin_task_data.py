"""Data system for STIN task offloading and completion tracking.

This module defines the reward calculation for satellite-terrestrial integrated
network (STIN) multi-agent reinforcement learning environments.
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
    
    注意：本类只存储增量/累计数据，不存储瞬时状态。
    瞬时状态（如电池电量、CPU利用率）应在需要时从卫星动态读取。
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
        # 合并任务列表
        all_completed = self.completed_tasks + other.completed_tasks
        all_expired = self.expired_tasks + other.expired_tasks
        
        # 计算平均时延（加权平均）
        total_completed = len(all_completed)
        if total_completed > 0:
            n1 = len(self.completed_tasks)
            n2 = len(other.completed_tasks)
            avg_latency = (self.avg_latency * n1 + other.avg_latency * n2) / total_completed
        else:
            avg_latency = 0.0
        
        return STINTaskData(
            completed_tasks=all_completed,
            expired_tasks=all_expired,
            processed_data=self.processed_data + other.processed_data,
            offloaded_data=self.offloaded_data + other.offloaded_data,
            energy_consumed=self.energy_consumed + other.energy_consumed,
            avg_latency=avg_latency,
        )


class STINTaskStore(DataStore):
    """DataStore for STIN task completion tracking."""

    data_type = STINTaskData

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
                logger.debug(f"Cannot read battery level: {e}")
        
        return {
            "completed_count": sat.completed_tasks_count,
            "expired_count": sat.expired_tasks_count,
            "processed_data": sat.processed_data_total,
            "offloaded_data": sat.offloaded_data_total,
            "battery_level": battery_level,  # 只用于计算增量
            "task_queue": list(sat.task_queue),  # 浅拷贝当前队列
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
        
        # 能耗增量（电池电量下降）
        battery_delta = old_state["battery_level"] - new_state["battery_level"]
        energy_consumed = max(0.0, battery_delta)  # 确保非负
        
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
       - 优先级越高、完成越快，奖励越大
       
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
        self.reward_fn = reward_fn
        self._satellite_cache: dict = {}  # 缓存卫星名称到对象的映射

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
        """重置时清空卫星缓存。"""
        super().reset_overwrite_previous()
        self._satellite_cache = {}

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
            # 1. 任务完成奖励 (R_completion)
            # ============================================================
            for task in new_data.completed_tasks:
                if self.reward_fn:
                    sat = self._get_satellite_by_name(sat_id)
                    if sat:
                        sat_reward += self.reward_fn(task, sat)
                else:
                    # 默认奖励：考虑优先级和时延
                    normalized_delay = task.get_elapsed_time(
                        task.compute_end_time
                    ) / task.max_delay
                    delay_factor = 1.0 - self.delay_weight * normalized_delay
                    sat_reward += self.base_reward * delay_factor
            
            # ============================================================
            # 2. 任务超时惩罚 (R_timeout)
            # ============================================================
            for task in new_data.expired_tasks:
                sat_reward -= self.timeout_penalty
            
            # ============================================================
            # 3a. 时延敏感惩罚 (R_latency) - 二次惩罚
            # ============================================================
            if new_data.avg_latency > 0 and len(new_data.completed_tasks) > 0:
                # 计算归一化时延（假设 max_delay 的平均值）
                avg_max_delay = sum(t.max_delay for t in new_data.completed_tasks) / len(new_data.completed_tasks)
                if avg_max_delay > 0:
                    normalized_latency = new_data.avg_latency / avg_max_delay
                    # 二次惩罚：越接近 deadline 惩罚越重
                    latency_penalty_value = self.latency_penalty * (normalized_latency ** 2)
                    sat_reward -= latency_penalty_value
            
            # ============================================================
            # 3b. 能耗线性惩罚 (R_energy_linear)
            # ============================================================
            if self.energy_weight > 0 and new_data.energy_consumed > 0:
                # 归一化能耗到 MJ（百万焦耳）
                normalized_energy = new_data.energy_consumed / 1e6
                sat_reward -= self.energy_weight * normalized_energy
            
            # ============================================================
            # 3c. 电池安全软约束 (R_battery_safety) - 指数惩罚
            # ============================================================
            if self.battery_penalty > 0:
                sat = self._get_satellite_by_name(sat_id)
                # 获取电池 SOC（State of Charge）
                battery_soc = 0.0
                if hasattr(sat, 'dynamics') and hasattr(sat.dynamics, 'powerMonitor'):
                    try:
                        battery_msg = sat.dynamics.powerMonitor.batPowerOutMsg.read()
                        battery_capacity = sat.dynamics.powerMonitor.storageCapacity
                        if battery_capacity > 0:
                            battery_soc = battery_msg.storageLevel / battery_capacity
                    except Exception as e:
                        logger.debug(f"Cannot read battery SOC: {e}")
                
                # 当电池低于安全阈值时施加指数惩罚
                if battery_soc < self.battery_safety_threshold:
                    # 指数衰减：SOC 越低，惩罚指数增长
                    decay_rate = 0.1  # 衰减率
                    safety_violation = self.battery_safety_threshold - battery_soc
                    battery_safety_penalty = self.battery_penalty * np.exp(safety_violation / decay_rate)
                    sat_reward -= battery_safety_penalty
                    
                    # 记录警告
                    if battery_soc < 0.1:  # 10% 严重警告
                        logger.warning(
                            f"{sat_id} battery critically low: SOC={battery_soc:.2%}, "
                            f"penalty={battery_safety_penalty:.2f}"
                        )
            
            reward[sat_id] = sat_reward
        
        # ============================================================
        # 4. 负载均衡奖励 (R_balance) - 全局奖励
        # ============================================================
        if self.balance_weight > 0:
            queue_lengths = []
            for sat_id in new_data_dict.keys():
                sat = self._get_satellite_by_name(sat_id)
                if sat and hasattr(sat, 'task_queue'):
                    queue_lengths.append(len(sat.task_queue))
            
            if len(queue_lengths) > 1:  # 至少需要两个卫星才能计算标准差
                queue_std = np.std(queue_lengths)
                balance_reward = -self.balance_weight * queue_std
                # 平分给所有卫星（全局协作激励）
                for sat_id in reward:
                    reward[sat_id] += balance_reward / len(reward)
        
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
                    # 电池 SOC < 5% 时截断（防止"猝死"）
                    if battery_soc < 0.05:
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

