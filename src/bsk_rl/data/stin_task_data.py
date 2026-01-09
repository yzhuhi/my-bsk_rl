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
        battery_soc: float = 1.0,  # 电池 SOC，用于软约束
    ) -> None:
        """构建 STIN 任务数据单元。
        
        Args:
            completed_tasks: 本步成功完成的任务切片列表。
            expired_tasks: 本步超时失败的任务切片列表。
            processed_data: [Gbits] 本步处理的数据量（增量）。
            offloaded_data: [Gbits] 本步卸载的数据量（增量）。
            energy_consumed: [J] 本步消耗的能量（增量）。
            avg_latency: [s] 本步完成任务的平均时延。
            battery_soc: 电池 SOC [0, 1]，用于软约束惩罚。
        """
        self.completed_tasks = completed_tasks or []
        self.expired_tasks = expired_tasks or []
        self.processed_data = processed_data
        self.offloaded_data = offloaded_data
        self.energy_consumed = energy_consumed
        self.avg_latency = avg_latency
        self.battery_soc = battery_soc

    def __add__(self, other: "STINTaskData") -> "STINTaskData":
        """合并两个数据单元（增量累加）。
        
        Args:
            other: 另一个数据单元。
            
        Returns:
            合并后的数据单元。
        """
        # 合并任务列表，但限制大小以防止内存泄漏
        MAX_BUFFER_SIZE = 100
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
        
        # 增量累加（符合库设计）
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
        cpu_power = 0.0
        tx_power = 0.0
        base_power = 0.0
        
        if hasattr(sat.dynamics, 'cpuPowerSink') and sat.dynamics.cpuPowerSink:
            raw_cpu = sat.dynamics.cpuPowerSink.nodePowerOut
            cpu_power = abs(raw_cpu) if not (np.isnan(raw_cpu) or np.isinf(raw_cpu)) else 0.0
        
        if hasattr(sat.dynamics, 'transmitterPowerSink') and sat.dynamics.transmitterPowerSink:
            raw_tx = sat.dynamics.transmitterPowerSink.nodePowerOut
            tx_power = abs(raw_tx) if not (np.isnan(raw_tx) or np.isinf(raw_tx)) else 0.0
        
        if hasattr(sat.dynamics, 'basePowerSink') and sat.dynamics.basePowerSink:
            raw_base = sat.dynamics.basePowerSink.nodePowerOut
            base_power = abs(raw_base) if not (np.isnan(raw_base) or np.isinf(raw_base)) else 0.0
        
        total_power = cpu_power + tx_power + base_power
        
        # 调试日志：每100步打印一次功耗详情
        # if int(sim_time) % 100 == 0:
        #     logger.info(f"[POWER DEBUG] {sat.name} @ t={sim_time:.1f}s: CPU={cpu_power:.2f}W, TX={tx_power:.2f}W, Base={base_power:.2f}W, Total={total_power:.2f}W")
        
        # ⚠️ 安全检查：防止 inf/nan 传播
        processed_data = sat.processed_data_total
        offloaded_data = sat.offloaded_data_total
        if np.isinf(processed_data) or np.isnan(processed_data):
            logger.error(f"[OVERFLOW SOURCE] {sat.name} processed_data_total={processed_data}, resetting to 0")
            sat.processed_data_total = 0.0
            processed_data = 0.0
        if np.isinf(offloaded_data) or np.isnan(offloaded_data):
            logger.error(f"[OVERFLOW SOURCE] {sat.name} offloaded_data_total={offloaded_data}, resetting to 0")
            sat.offloaded_data_total = 0.0
            offloaded_data = 0.0
        
        return {
            "completed_count": sat.completed_tasks_count,
            "expired_count": sat.expired_tasks_count,
            "processed_data": processed_data,
            "offloaded_data": offloaded_data,
            "raw_data_received": getattr(sat, 'raw_data_received', 0.0),  # 从 UD 接收的原始数据
            "battery_level": battery_level,  # 用于奖励计算中的电池安全检查
            "sim_time": sim_time,            # 用于计算精细能耗
            "total_power": total_power,
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
        
        # 数据增量（带保护）
        processed_delta = new_state["processed_data"] - old_state["processed_data"]
        offloaded_delta = new_state["offloaded_data"] - old_state["offloaded_data"]
        
        # ⚠️ 安全检查：防止 inf/nan 传播
        if np.isinf(processed_delta) or np.isnan(processed_delta) or processed_delta < 0:
            processed_delta = 0.0
        if np.isinf(offloaded_delta) or np.isnan(offloaded_delta) or offloaded_delta < 0:
            offloaded_delta = 0.0
        
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
            # 调试日志：每100步打印一次能耗详情
            # if int(new_state["sim_time"]) % 100 == 0:
            #     logger.info(f"[ENERGY DEBUG] dt={dt:.1f}s, power={total_power:.2f}W, energy={energy_consumed:.2f}J")
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
            processed_data=processed_delta / 1e9,  # 转为 Gbits
            offloaded_data=offloaded_delta / 1e9,  # 转为 Gbits
            energy_consumed=energy_consumed,
            avg_latency=avg_latency,
            battery_soc=new_state.get("battery_level", 1.0),  # 当前电池 SOC
        )


class STINTaskReward(GlobalReward):
    """GlobalReward for STIN task offloading environments.
    
    **结果导向的简化奖励设计**（推荐）：
        R_total = R_completion + R_timeout + R_energy
    
    核心理念：
        - 只看结果：任务完成得奖，失败扣分，能耗是成本
        - 硬约束（电量）：通过 Action Masking/Replacement 处理，不放入奖励
        - 软约束（负载均衡）：通过观测空间（邻居队列长度）让 Agent 自己学习
    
    奖励分量：
    1. 任务完成奖励 (R_completion) - 稀疏奖励
       - R_complete = w_task × delay_factor × priority
       - delay_factor = max(0, 1 - elapsed_time / max_delay)，线性衰减
       - 只在任务真正完成时才给分
       
    2. 任务失败惩罚 (R_timeout) - 稀疏惩罚
       - R_timeout = -w_task × 0.5 × priority
       - 失败惩罚 < 成功奖励，鼓励 Agent 尝试接收任务
       
    3. 能耗惩罚 (R_energy) - 纯成本项
       - R_energy = -energy_consumed / 1000 × w_energy
       - 归一化后的能耗成本
    
    4. 协作激励
       - 任务发起者获得完成奖励的 20%
       - 任务接收者搞砸时，发起者也受连带惩罚
    
    参数（极简，只需调 3 个）：
        - w_task: 完成任务的基础奖励（默认 10.0）
        - w_energy: 能耗惩罚权重（默认 0.1）
        - w_delay: 时延敏感度（默认 1.0）
    
    注意：
        - 不再使用 baseline_penalty, progress_reward_weight, idle_penalty
        - 负载均衡通过观测空间实现，不在奖励函数中
        - 电池安全通过 Action Masking 实现，不在奖励函数中
    """

    data_store_type = STINTaskStore

    def __init__(
        self,
        # === 新版校准参数（里程碑奖励）===
        w_progress: float = 5.0,      # 面包屑：每处理 1 Gbits 给 5 分
        w_complete: float = 10.0,     # 主奖励：任务完成基础奖励
        w_expired: float = 3.0,       # 超时惩罚系数
        w_energy: float = 0.0001,     # 能耗成本（每焦耳扣分）
        w_delay: float = 1.0,         # 延迟惩罚系数
        w_offload_instant: float = 0.1,  # 即时承诺：卸载发生时立即奖励（路费报销）
        w_offload_delayed: float = 0.5,  # 延迟分红：任务完成后反向分配（业绩奖金）
        gamma_hop: float = 0.9,       # 每跳奖励衰减因子
        liability_ratio: float = 0.3, # 连带惩罚比例：发起者承担失败惩罚的比例
        w_efficiency: float = 0.0,    # 能效奖励：每 bits/J 的奖励系数
        # === 向后兼容的旧参数（deprecated）===
        # w_collab: float = 0.3,        # deprecated, 协作分成比例 被信度分配替代
        # w_task: float = None,         # deprecated, 使用 w_complete
        # w_delay: float = 1.0,         # 时延敏感度 (保留)
        # collaboration_bonus: float = None,  # deprecated, 使用 w_collab
        # base_reward: float = None,    # deprecated
        # timeout_penalty: float = None,  # deprecated
        # delay_weight: float = None,   # deprecated
        # latency_penalty: float = 0.0,  # deprecated
        # energy_weight: float = None,  # deprecated
        # battery_penalty: float = 0.0,  # deprecated
        # battery_safety_threshold: float = 0.2,  # deprecated
        # balance_weight: float = 0.0,  # deprecated
        # collaboration_weight: float = None,  # deprecated
        # idle_penalty: float = 0.0,    # deprecated
        # idle_queue_threshold: int = 10,  # deprecated
        # progress_reward_weight: float = None,  # deprecated
        # baseline_penalty: float = 0.0,  # deprecated
        reward_fn: Optional[Callable] = None,
    ) -> None:
        """初始化 STIN 任务奖励计算器（里程碑奖励版本）。
        
        Args:
            w_progress: 面包屑奖励，每处理 1 Gbits 数据的奖励。
            w_complete: 任务完成基础奖励（会乘以时延折扣和优先级）。
            w_expired: 超时惩罚系数（惩罚 = w_expired × priority）。
            w_energy: 能耗成本（每焦耳扣分）。
            w_collab: 协作分成比例，发起者获得完成奖励的比例。
            w_delay: 时延敏感度 [0, 1]。
        """
        super().__init__()
        
        # 新版参数
        self.w_progress = w_progress
        self.w_complete = w_complete
        self.w_expired = w_expired
        self.w_energy = w_energy
        # self.w_collab = w_collab
        self.w_delay = w_delay
        self.w_offload_instant = w_offload_instant  # 即时承诺（路费报销）
        self.w_offload_delayed = w_offload_delayed  # 延迟分红（业绩奖金）
        self.gamma_hop = gamma_hop                  # 每跳衰减因子
        self.liability_ratio = liability_ratio      # 连带惩罚比例
        self.w_efficiency = w_efficiency            # 能效奖励系数
        
        # 向后兼容处理（仅打印警告，不再实际使用这些参数）
        # if base_reward is not None:
        #     logger.warning("base_reward is deprecated, use w_complete instead")
        # if timeout_penalty is not None:
        #     logger.warning("timeout_penalty is deprecated, use w_expired instead")
        # if delay_weight is not None:
        #     logger.warning("delay_weight is deprecated, use w_delay instead")
        # if energy_weight is not None:
        #     logger.warning("energy_weight is deprecated, use w_energy instead")
        # if collaboration_weight is not None:
        #     logger.warning("collaboration_weight is deprecated, use w_collab instead")
        # if battery_penalty is not None and battery_penalty > 0:
        #     logger.warning("battery_penalty is deprecated, use Action Shield instead")
        # if balance_weight is not None and balance_weight > 0:
        #     logger.warning("balance_weight is deprecated, neighbor queue info is in observation space")
        # if idle_penalty is not None and idle_penalty > 0:
        #     logger.warning("idle_penalty is deprecated, sparse rewards are more stable")
        # if baseline_penalty is not None and baseline_penalty > 0:
        #     logger.warning("baseline_penalty is deprecated, sparse rewards are more stable")
        # if progress_reward_weight is not None and progress_reward_weight > 0:
        #     logger.warning("progress_reward_weight is deprecated, use w_progress instead")
        
        # 不再保留旧属性，避免混淆（已全部废弃）
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

    # ========================================================================
    # 奖励计算辅助方法
    # ========================================================================
    
    def _calc_progress_reward(self, processed_data: float) -> float:
        """计算面包屑奖励（处理数据量）。"""
        if processed_data > 0:
            return processed_data * self.w_progress
        return 0.0
    
    def _calc_completion_reward(self, task) -> tuple[float, str]:
        """计算任务完成奖励。
        
        Returns:
            (reward, origin_satellite) 元组
        """
        time_ratio = task.get_elapsed_time(task.compute_end_time) / task.max_delay
        delay_factor = max(0.0, 1.0 - self.w_delay * time_ratio)
        priority = getattr(task, 'priority', 1.0)
        reward = self.w_complete * delay_factor * priority
        origin_sat = getattr(task, 'origin_satellite', '')
        return reward, origin_sat
    
    def _calc_expired_penalty(self, task) -> float:
        """计算超时惩罚。"""
        priority = getattr(task, 'priority', 1.0)
        return self.w_expired * priority
    
    def _calc_energy_cost(self, energy_consumed: float) -> float:
        """计算能耗成本。"""
        if energy_consumed > 0:
            return energy_consumed * self.w_energy
        return 0.0

    # ========================================================================
    # 主奖励计算方法
    # ========================================================================

    def calculate_reward(self, new_data_dict: dict[str, STINTaskData]) -> dict[str, float]:
        """里程碑奖励函数（模块化版）。
        
        组件：
            1. Progress: 处理数据量 × w_progress
            2. Completion: 任务完成 × delay_factor × priority
            3. Expired: 超时惩罚 × priority
            4. Energy: 能耗成本
            5. Collaboration: 发起者分成
            6. Liability: 连带责任（失败时发起者承担部分惩罚）
        """
        # ✅ 修复 Bug 1: 初始化所有卫星奖励为 0，防止覆盖问题
        reward = {sat_id: 0.0 for sat_id in new_data_dict.keys()}

        for sat_id, new_data in new_data_dict.items():
            # ✅ 使用 += 累加，不使用临时变量赋值
            
            # 1. 面包屑奖励（处理进度）
            reward[sat_id] += self._calc_progress_reward(new_data.processed_data)
            
            # 1.5 即时承诺奖励 (Promise)：卸载发生时立即奖励
            # 作用：打破僵局，抵消通信能耗，诱导 Agent 迈出第一步（路费报销）
            if new_data.offloaded_data > 0:
                reward[sat_id] += new_data.offloaded_data * self.w_offload_instant

            # 2. 主奖励 + 延迟分红 (Dividend)
            for task in new_data.completed_tasks:
                task_reward, _ = self._calc_completion_reward(task)
                reward[sat_id] += task_reward  # 完成者拿全额
                
                # 延迟分红：反向分配给链路上所有卫星 设置成w_offload_delayed=1 就是GAE
                offload_chain = getattr(task, 'offload_chain', [])
                if len(offload_chain) > 0:
                    for hop, chain_sat in enumerate(offload_chain):
                        if chain_sat != sat_id and chain_sat in reward:
                            delayed_bonus = task_reward * (self.gamma_hop ** hop) * self.w_offload_delayed
                            reward[chain_sat] += delayed_bonus

            # 3. 失败惩罚 + 连带责任 (Liability)
            for task in new_data.expired_tasks:
                penalty = self._calc_expired_penalty(task)
                reward[sat_id] -= penalty  # 当前卫星承担主要责任
                
                # ✅ 修复 Bug 2: 连带责任 - 发起者承担部分惩罚
                # 防止 A 把任务甩给忙碌的 B，一旦失败 A 也要负责
                origin_sat = getattr(task, 'origin_satellite', '')
                if origin_sat and origin_sat != sat_id and origin_sat in reward:
                    reward[origin_sat] -= penalty * self.liability_ratio  # 发起者承担连带惩罚

            # # 4. 能耗成本
            # reward[sat_id] -= self._calc_energy_cost(new_data.energy_consumed)
            
            # 5. 能效奖励：处理数据量 / 能耗 越高越好
            if self.w_efficiency > 0 and new_data.energy_consumed > 0:
                current_efficiency = new_data.processed_data / new_data.energy_consumed  # bits/J
                # 奖励 = 能效 × 系数，归一化到合理范围（假设基线能效 ~1e6 bits/J = 1 Mbits/J）
                efficiency_reward = (current_efficiency / 1e5) * self.w_efficiency
                reward[sat_id] += efficiency_reward

        # 安全检查
        for k, v in reward.items():
            if np.isnan(v) or np.isinf(v):
                logger.error(f"[Reward Error] {k} reward is {v}, resetting to -10.0")
                reward[k] = -10.0

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
        """检查是否应该终止 episode（任务完成）。
        
        ✅ 修正逻辑(2026-01-05):
        只有当任务池空了 **且** 所有已到达任务都处理完时才终止。
        
        ⚠️ 注意：使用任务ID集合统计，避免因slice分割导致重复计数。
        """
        if not hasattr(self.scenario, 'task_pool'):
            return False
        
        # 获取当前仿真时间
        sim_time = satellite.simulator.sim_time if hasattr(satellite, 'simulator') else 0
        
        # 检查任务池剩余
        pool_remaining = len(self.scenario.task_pool)
        
        # 获取已到达的任务数
        total_arrived = len(self.scenario.arrived_tasks) if hasattr(self.scenario, 'arrived_tasks') else 0
        
        # ✅ 使用任务ID集合统计唯一任务数（避免slice重复计数）
        all_completed_ids = set()
        all_expired_ids = set()
        for sat in self.scenario.satellites:
            if hasattr(sat, 'completed_task_ids'):
                all_completed_ids.update(sat.completed_task_ids)
            if hasattr(sat, 'expired_task_ids'):
                all_expired_ids.update(sat.expired_task_ids)
        
        # ⚠️ 关键修复：一个任务可能有些slice完成，有些过期
        # 如果一个任务有任何slice完成，就算completed（不算expired）
        # 只有完全没有完成的才算expired
        pure_expired_ids = all_expired_ids - all_completed_ids
        
        completed = len(all_completed_ids)
        expired = len(pure_expired_ids)  # 使用纯过期数（排除已完成的）
        all_processed_ids = all_completed_ids | all_expired_ids  # 所有已处理的任务
        processed = len(all_processed_ids)
        pending = total_arrived - processed
        
        # 🔍 每500秒输出一次状态（仅对第一个卫星输出，避免重复）
        first_sat = self.scenario.satellites[0] if self.scenario.satellites else None
        is_first_sat = (satellite == first_sat)
        if is_first_sat and sim_time > 0 and int(sim_time) % 500 == 0:
            logger.warning(
                f"📊 [TASK STATUS @ {sim_time:.0f}s] "
                f"Pool: {pool_remaining} | Arrived: {total_arrived} | "
                f"Processed: {processed} (C:{completed} E:{expired}) | "
                f"Pending: {pending}"
            )
        
        # 任务池还有剩余，不能终止
        if pool_remaining > 0:
            return False
        
        # 任务池已空，检查已到达的任务是否都处理完
        # 使用 processed（唯一任务ID的并集）而不是 completed + expired
        if processed >= total_arrived and total_arrived > 0:
            if is_first_sat:  # 只打印一次
                logger.info(
                    f"✅ [TERMINATED @ {sim_time:.0f}s] All tasks processed: "
                    f"{completed} completed, {expired} expired out of {total_arrived} arrived."
                )
            return True
        else:
            # 任务池空了但还有未处理的任务（在卫星队列中）
            pass  # 静默等待
        
        return False


__doc_title__ = "STIN Task Data"
__all__ = ["STINTaskReward", "STINTaskStore", "STINTaskData"]

