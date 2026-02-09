"""Data system for STIN task offloading and completion tracking.

This module defines the reward calculation for satellite-terrestrial integrated network (STIN) 
multi-agent reinforcement learning environments.
"""

import logging
from typing import TYPE_CHECKING, Optional

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
        raw_queue_len: int = 0,  # 原始任务队列长度（用于排队惩罚）
        queue_len: int = 0,      # 切片任务队列长度（用于排队惩罚）
        slice_queue_len: int = 0,
        queue_data_gbits: float = 0.0,
        queue_data_delta_gbits: float = 0.0,
        slice_queue_data_gbits: float = 0.0,
        slice_queue_data_delta_gbits: float = 0.0,
        energy_tx: float = 0.0,  # 🆕 [J] 发送能耗 E_tr = P_tr × t_tx
        energy_rx: float = 0.0,  # 🆕 [J] 接收能耗 E_re = P_re × t_rx
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
            energy_tx: [J] 发送能耗（论文 E_tr）。
            energy_rx: [J] 接收能耗（论文 E_re）。
        """
        self.completed_tasks = completed_tasks or []
        self.expired_tasks = expired_tasks or []
        self.processed_data = processed_data
        self.offloaded_data = offloaded_data
        self.energy_consumed = energy_consumed
        self.avg_latency = avg_latency
        self.battery_soc = battery_soc
        self.raw_queue_len = raw_queue_len
        self.queue_len = queue_len
        self.slice_queue_len = slice_queue_len
        self.queue_data_gbits = queue_data_gbits
        self.queue_data_delta_gbits = queue_data_delta_gbits
        self.slice_queue_data_gbits = slice_queue_data_gbits
        self.slice_queue_data_delta_gbits = slice_queue_data_delta_gbits
        self.energy_tx = energy_tx  # 🆕
        self.energy_rx = energy_rx  # 🆕

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
            raw_queue_len=other.raw_queue_len,
            queue_len=other.queue_len,
            slice_queue_len=other.slice_queue_len,
            queue_data_gbits=other.queue_data_gbits,
            queue_data_delta_gbits=other.queue_data_delta_gbits,
            slice_queue_data_gbits=other.slice_queue_data_gbits,
            slice_queue_data_delta_gbits=other.slice_queue_data_delta_gbits,
            energy_tx=self.energy_tx + other.energy_tx,  # 🆕
            energy_rx=self.energy_rx + other.energy_rx,  # 🆕
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
        total_power = 0.0
        cpu_power = 0.0
        tx_power = 0.0
        base_power = 0.0
        
        # 如果 powerMonitor 读取失败，回退到手动加总
        if hasattr(sat.dynamics, 'cpuPowerSink') and sat.dynamics.cpuPowerSink:
            raw_cpu = sat.dynamics.cpuPowerSink.nodePowerOut
            cpu_power = abs(raw_cpu) if not (np.isnan(raw_cpu) or np.isinf(raw_cpu)) else 0.0
        
        if hasattr(sat.dynamics, 'transmitterPowerSink') and sat.dynamics.transmitterPowerSink:
            raw_tx = sat.dynamics.transmitterPowerSink.nodePowerOut
            tx_power = abs(raw_tx) if not (np.isnan(raw_tx) or np.isinf(raw_tx)) else 0.0
        
        if hasattr(sat.dynamics, 'basePowerSink') and sat.dynamics.basePowerSink:
            raw_base = sat.dynamics.basePowerSink.nodePowerOut
            base_power = abs(raw_base) if not (np.isnan(raw_base) or np.isinf(raw_base)) else 0.0
        
        # 🆕 接收机功耗（论文能耗模型 E_re）
        rx_power = 0.0
        if hasattr(sat.dynamics, 'rxPowerSink') and sat.dynamics.rxPowerSink:
            raw_rx = sat.dynamics.rxPowerSink.nodePowerOut
            rx_power = abs(raw_rx) if not (np.isnan(raw_rx) or np.isinf(raw_rx)) else 0.0
        
        # total_power = cpu_power + tx_power + base_power + rx_power
        total_power = cpu_power + base_power

    
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
            # 🆕 收发统计（用于论文能耗模型）
            "sent_data_this_step": getattr(sat, 'sent_data_this_step', 0.0),
            "received_data_this_step": getattr(sat, 'received_data_this_step', 0.0),
            "tx_time_this_step": getattr(sat, 'tx_time_this_step', 0.0),
            "rx_time_this_step": getattr(sat, 'rx_time_this_step', 0.0),
            "raw_task_queue": list(getattr(sat, 'raw_task_queue', [])),
            "slice_queue": list(getattr(sat, 'slice_queue', [])),
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
            
            # 🆕 论文能耗模型分项计算
            # E_tr = P_tr × t_tx (发送能耗)
            # E_re = P_re × t_rx (接收能耗)
            tx_time = new_state.get("tx_time_this_step", 0.0)
            rx_time = new_state.get("rx_time_this_step", 0.0)
            
            # 获取功率参数（从 dynamics 或使用默认值）
            sat = self.satellite
            p_tx = abs(getattr(sat.dynamics, 'max_tx_power_draw', 15.0)) if hasattr(sat, 'dynamics') else 15.0
            p_rx = abs(getattr(sat.dynamics, 'rx_power_draw', 8.0)) if hasattr(sat, 'dynamics') else 8.0
            
            energy_tx = p_tx * tx_time  # [J] 发送能耗
            energy_rx = p_rx * rx_time  # [J] 接收能耗
            energy_consumed += energy_tx + energy_rx
        else:
            energy_consumed = 0.0
            energy_tx = 0.0
            energy_rx = 0.0
        
        # 从卫星的历史缓冲区提取完成和超时的任务
        sat: "ComputationSatellite" = self.satellite
        completed_tasks = []
        expired_tasks = []
        
        if hasattr(sat, 'completed_tasks_buffer'):
            # 提取本 step 完成的任务 (复制引用，不清空)
            # ⚠️ 不在这里清空 buffer！由 gym.py 的 _get_info() 统一管理
            completed_tasks = list(sat.completed_tasks_buffer)
        
        if hasattr(sat, 'expired_tasks_buffer'):
            # 提取本 step 超时的任务 (复制引用，不清空)
            # ⚠️ 不在这里清空 buffer！由 gym.py 的 _get_info() 统一管理
            expired_tasks = list(sat.expired_tasks_buffer)
        
        # 计算平均时延（从实际完成的任务中提取）
        # ✅ 直接从任务对象计算时延: T_total = max(T_UD, T_SAT, T_CLOUD)
        latencies = []
        for task in completed_tasks:
            t_ud = getattr(task, 't_ud_path', 0.0)
            t_sat = getattr(task, 't_sat_path', 0.0)
            t_cloud = getattr(task, 't_cloud_path', 0.0)
            latency = max(t_ud, t_sat, t_cloud)
            if latency > 0:
                latencies.append(latency)
                
        avg_latency = np.mean(latencies) if latencies else 0.0

        def _queue_data_gbits(queue):
            return sum(getattr(task, "data_size", 0.0) for task in queue) / 1e9 if queue else 0.0

        queue_data_gbits = (
            _queue_data_gbits(new_state.get("raw_task_queue", []))
            + _queue_data_gbits(new_state.get("task_queue", []))
            + _queue_data_gbits(new_state.get("slice_queue", []))
        )
        prev_queue_data_gbits = (
            _queue_data_gbits(old_state.get("raw_task_queue", []))
            + _queue_data_gbits(old_state.get("task_queue", []))
            + _queue_data_gbits(old_state.get("slice_queue", []))
        )
        queue_data_delta_gbits = queue_data_gbits - prev_queue_data_gbits
        slice_queue_data_gbits = _queue_data_gbits(new_state.get("slice_queue", []))
        prev_slice_queue_data_gbits = _queue_data_gbits(old_state.get("slice_queue", []))
        slice_queue_data_delta_gbits = (
            slice_queue_data_gbits - prev_slice_queue_data_gbits
        )
        
        return STINTaskData(
            completed_tasks=completed_tasks,
            expired_tasks=expired_tasks,
            processed_data=processed_delta / 1e9,  # 转为 Gbits
            offloaded_data=offloaded_delta / 1e9,  # 转为 Gbits
            energy_consumed=energy_consumed,
            avg_latency=avg_latency,
            battery_soc=new_state.get("battery_level", 1.0),  # 当前电池 SOC
            raw_queue_len=len(new_state.get("raw_task_queue", [])),
            queue_len=len(new_state.get("task_queue", [])),
            slice_queue_len=len(new_state.get("slice_queue", [])),
            queue_data_gbits=queue_data_gbits,
            queue_data_delta_gbits=queue_data_delta_gbits,
            slice_queue_data_gbits=slice_queue_data_gbits,
            slice_queue_data_delta_gbits=slice_queue_data_delta_gbits,
            energy_tx=energy_tx,  # 🆕 发送能耗
            energy_rx=energy_rx,  # 🆕 接收能耗
        )


class STINTaskReward(GlobalReward):
    """GlobalReward for STIN task offloading environments.

    目标：最大化完成率，最小化时延与能耗。
    可选 shaping：进度、队列、利用率、PBRS（由配置开关控制）。
    """

    data_store_type = STINTaskStore

    def __init__(
        self,
        w_progress: float = 0.0,      # 面包屑：默认关闭，避免刷分
        w_complete: float = 60.0,     # 主奖励：任务完成基础奖励
        w_expired: float = 20.0,      # 超时惩罚系数
        w_energy: float = 0.000003,   # 能耗成本（每焦耳扣分）
        w_delay: float = 0.6,         # 延迟惩罚系数
        w_queue_penalty: float = 0.5, # 排队惩罚：每个排队任务的罚分（鼓励清队）
        w_slice_drain: float = 0.0,   # 切片队列净下降奖励（轻量 shaping）
        max_queue_size: int = 40,     # 队列长度归一化上限
        progress_norm_gbits: float = 1.0,
        global_reward_ratio: float = 0.3,  # 全局结果奖励占比 [0, 1]
        w_utilization: float = 0.5,       # 资源利用率奖励：高负载时高资源利用=好
        w_pbrs_queue: float = 0.0,        # PBRS队列势函数系数（状态改善奖励）
        pbrs_gamma: float = 0.99,         # PBRS折扣因子
    ) -> None:
        """初始化 STIN 任务奖励计算器。

        Args:
            w_progress: 面包屑奖励，每处理 1 Gbits 数据的奖励。
            w_complete: 任务完成基础奖励（会乘以时延折扣和优先级）。
            w_expired: 超时惩罚系数（惩罚 = w_expired × priority）。
            w_energy: 能耗成本（每焦耳扣分）。
            w_delay: 时延敏感度 [0, 1]。
            w_queue_penalty: 队列长度惩罚系数。
            max_queue_size: 队列长度归一化上限。
            progress_norm_gbits: 面包屑奖励归一化尺度。
            global_reward_ratio: 全局结果奖励混合比例 [0, 1]。
            w_utilization: 资源利用率奖励系数。
            w_pbrs_queue: PBRS 队列势函数系数。
            pbrs_gamma: PBRS 折扣因子。
        """
        super().__init__()

        self.w_progress = w_progress
        self.w_complete = w_complete
        self.w_expired = w_expired
        self.w_energy = w_energy
        self.w_delay = w_delay
        self.w_queue_penalty = w_queue_penalty
        self.w_slice_drain = w_slice_drain
        self.max_queue_size = max_queue_size
        self.progress_norm_gbits = progress_norm_gbits
        self.global_reward_ratio = global_reward_ratio
        self.w_utilization = w_utilization
        self.w_pbrs_queue = w_pbrs_queue
        self.pbrs_gamma = pbrs_gamma

        # PBRS 状态追踪（每个卫星的上一步势函数值）
        self._prev_potential: dict[str, float] = {}

        logger.info(
            "[Reward Config] w_progress=%s w_complete=%s w_expired=%s w_energy=%s "
            "w_delay=%s w_queue_penalty=%s w_slice_drain=%s max_queue_size=%s progress_norm_gbits=%s "
            "global_reward_ratio=%s w_utilization=%s w_pbrs_queue=%s pbrs_gamma=%s",
            self.w_progress,
            self.w_complete,
            self.w_expired,
            self.w_energy,
            self.w_delay,
            self.w_queue_penalty,
            self.w_slice_drain,
            self.max_queue_size,
            self.progress_norm_gbits,
            self.global_reward_ratio,
            self.w_utilization,
            self.w_pbrs_queue,
            self.pbrs_gamma,
        )

        self._satellite_cache: dict = {}  # 缓存卫星名称到对象的映射
        self.last_reward_components: dict[str, dict[str, float]] = {}
        self._rewarded_task_ids: set[str] = set()
        self._penalized_task_ids: set[str] = set()

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
        """重置时清空卫星缓存和任务ID记录（防止跨 Episode 状态泄漏）。"""
        super().reset_overwrite_previous()
        self._satellite_cache = {}
        # 清空任务 ID 记录，防止跨 Episode 状态泄漏
        self._rewarded_task_ids = set()
        self._penalized_task_ids = set()
        # 重置 PBRS 状态
        self._prev_potential = {}

    # ========================================================================
    # 奖励计算辅助方法
    # ========================================================================
    
    def _calc_potential(self, queue_pressure: float) -> float:
        """计算PBRS势函数 Φ(s) = -α × queue_pressure。
        
        队列压力越大，势函数值越负，状态越"差"。
        当状态改善（队列压力降低）时，F = γΦ(s') - Φ(s) > 0，获得正奖励。
        
        Args:
            queue_pressure: 归一化队列压力 [0, 1]
            
        Returns:
            势函数值
        """
        return -self.w_pbrs_queue * queue_pressure
    
    def _calc_pbrs_shaping(self, sat_id: str, current_queue_pressure: float) -> float:
        """计算PBRS奖励整形 F = γΦ(s') - Φ(s)。
        
        理论保证：PBRS 不改变最优策略，只加速学习。
        
        Args:
            sat_id: 卫星ID
            current_queue_pressure: 当前队列压力
            
        Returns:
            PBRS整形奖励
        """
        if self.w_pbrs_queue <= 0:
            return 0.0
        
        current_potential = self._calc_potential(current_queue_pressure)
        prev_potential = self._prev_potential.get(sat_id, current_potential)
        
        # F = γΦ(s') - Φ(s)
        pbrs_reward = self.pbrs_gamma * current_potential - prev_potential
        
        # 更新状态
        self._prev_potential[sat_id] = current_potential
        
        return pbrs_reward
    
    def _calc_progress_reward(self, processed_data: float) -> float:
        """计算面包屑奖励（处理数据量，单位已是 Gbits）。"""
        if processed_data <= 0 or self.w_progress <= 0:
            return 0.0
        if self.progress_norm_gbits and self.progress_norm_gbits > 0:
            scaled = np.tanh(processed_data / self.progress_norm_gbits)
            return self.w_progress * scaled
        return processed_data * self.w_progress
    
    def _calc_completion_reward(self, task, total_delay: Optional[float] = None) -> tuple[float, str]:
        """计算任务完成奖励（乘法耦合版）。
        
        奖励公式 = w_complete × delay_factor × priority
        
        能效因子设计：
        - 使用任务对象中记录的实际时延信息估算能耗
        - 计算能耗基于实际处理时间
        - 通信能耗基于 ISL 时延记录
        - 归一化到 [0.5, 1.5] 区间
        
        Returns:
            (reward, origin_satellite) 元组
        """
        # ✅ 防止除零：max_delay 至少为 1.0 秒
        max_delay = max(task.max_delay, 1.0)
        if total_delay is None:
            total_delay = task.get_elapsed_time(getattr(task, "compute_end_time", 0.0))
        time_ratio = total_delay / max_delay
        delay_factor = max(0.0, 1.0 - self.w_delay * time_ratio)
        priority = getattr(task, 'priority', 1.0)
        # 乘法耦合：所有因子相乘
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

    def _get_task_completion_state(self, task) -> tuple[bool, float, str]:
        """判断任务级完成（含回传）与总时延。"""
        task_id = getattr(task, "task_id", None)
        if not task_id:
            return False, 0.0, ""

        access_sat_name = getattr(task, "access_satellite", "") or getattr(task, "origin_satellite", "")
        access_sat = self._get_satellite_by_name(access_sat_name) if access_sat_name else None
        status = None
        if access_sat is not None and hasattr(access_sat, "slice_registry"):
            status = access_sat.slice_registry.get(task_id)

        if access_sat is not None and hasattr(access_sat, "get_task_total_delay"):
            total_delay = access_sat.get_task_total_delay(task_id)
        else:
            total_delay = max(
                getattr(task, "t_ud_path", 0.0),
                getattr(task, "t_sat_path", 0.0),
                getattr(task, "t_cloud_path", 0.0),
            )

        expired = False
        if access_sat is not None and hasattr(access_sat, "expired_task_ids"):
            expired = task_id in access_sat.expired_task_ids
        else:
            for sat in getattr(self.scenario, "satellites", []):
                if hasattr(sat, "expired_task_ids") and task_id in sat.expired_task_ids:
                    expired = True
                    break

        is_complete = False
        task_max_delay = getattr(task, "max_delay", 0.0)
        if status is not None:
            total = status.get("total", 0)
            completed = status.get("completed", 0)
            if total <= 0:
                is_complete = (not expired) and total_delay > 0 and total_delay <= task_max_delay
            else:
                is_complete = (not expired) and completed >= total and total_delay <= task_max_delay
        else:
            is_complete = (not expired) and total_delay > 0 and total_delay <= task_max_delay

        return is_complete, total_delay, access_sat_name

    # ========================================================================
    # 主奖励计算方法
    # ========================================================================

    def calculate_reward(self, new_data_dict: dict[str, STINTaskData]) -> dict[str, float]:
        """计算每个卫星的奖励。

        奖励由三类构成：
        1) 结果项：完成奖励 / 超时惩罚
        2) 成本项：能耗成本
        3) 轻量 shaping：队列惩罚、进度、利用率、PBRS（可配置开关）

        结果项可按 global_reward_ratio 进行全局混合，促进系统级协同。
        """
        reward = {sat_id: 0.0 for sat_id in new_data_dict.keys()}
        outcome_reward = {sat_id: 0.0 for sat_id in new_data_dict.keys()}
        reward_components = {
            sat_id: {
                "completion_reward": 0.0,
                "expired_penalty": 0.0,
                "energy_cost": 0.0,
                "queue_penalty": 0.0,
                "slice_drain_reward": 0.0,
                "progress_reward": 0.0,
                "utilization_reward": 0.0,
                "pbrs_reward": 0.0,
                "total_reward": 0.0,
            }
            for sat_id in new_data_dict.keys()
        }

        if not hasattr(self, "_rewarded_task_ids"):
            self._rewarded_task_ids = set()
        if not hasattr(self, "_penalized_task_ids"):
            self._penalized_task_ids = set()

        for sat_id, new_data in new_data_dict.items():
            max_queue_size = getattr(
                self, "max_queue_size", getattr(self.scenario, "max_task_queue_size", 40)
            )
            total_queue_len = getattr(new_data, "raw_queue_len", 0) + getattr(
                new_data, "queue_len", 0
            ) + getattr(new_data, "slice_queue_len", 0)

            # 1) 轻量 shaping（队列/进度）
            if self.w_queue_penalty > 0 and max_queue_size > 0:
                ratio = min(total_queue_len / (max_queue_size * 3), 1.0)
                queue_penalty = self.w_queue_penalty * ratio
                reward[sat_id] -= queue_penalty
                reward_components[sat_id]["queue_penalty"] = queue_penalty

            if self.w_slice_drain > 0:
                slice_drain = -float(
                    getattr(new_data, "slice_queue_data_delta_gbits", 0.0)
                )
                if slice_drain > 0:
                    slice_drain_reward = self.w_slice_drain * slice_drain
                    reward[sat_id] += slice_drain_reward
                    reward_components[sat_id]["slice_drain_reward"] = slice_drain_reward

            progress_reward = self._calc_progress_reward(new_data.processed_data)
            reward[sat_id] += progress_reward
            reward_components[sat_id]["progress_reward"] = progress_reward

            # 2) 结果项：完成与超时
            for task in new_data.completed_tasks:
                task_id = getattr(task, "task_id", None)
                if not task_id:
                    continue
                if task_id in self._rewarded_task_ids or task_id in self._penalized_task_ids:
                    continue

                is_complete, total_delay, _ = self._get_task_completion_state(task)
                if not is_complete:
                    continue

                task_reward, _ = self._calc_completion_reward(task, total_delay=total_delay)
                reward[sat_id] += task_reward
                outcome_reward[sat_id] += task_reward
                reward_components[sat_id]["completion_reward"] += task_reward
                self._rewarded_task_ids.add(task_id)

            for task in new_data.expired_tasks:
                task_id = getattr(task, "task_id", None)
                if not task_id:
                    continue
                if task_id in self._penalized_task_ids or task_id in self._rewarded_task_ids:
                    continue
                penalty = self._calc_expired_penalty(task)
                reward[sat_id] -= penalty
                outcome_reward[sat_id] -= penalty
                reward_components[sat_id]["expired_penalty"] += penalty
                self._penalized_task_ids.add(task_id)

            # 3) 成本项（能耗）
            energy_cost = self._calc_energy_cost(new_data.energy_consumed)
            reward[sat_id] -= energy_cost
            reward_components[sat_id]["energy_cost"] = energy_cost

            # 4) 资源利用率 shaping
            if self.w_utilization > 0:
                max_queue = max(self.max_queue_size, 1)
                queue_pressure = min(total_queue_len / max_queue, 1.0)
                if new_data.processed_data > 0 and queue_pressure > 0.3:
                    processed_factor = np.tanh(new_data.processed_data)
                    util_reward = self.w_utilization * queue_pressure * processed_factor
                    reward[sat_id] += util_reward
                    reward_components[sat_id]["utilization_reward"] = util_reward

            # 5) PBRS shaping
            if self.w_pbrs_queue > 0 and max_queue_size > 0:
                queue_pressure = min(total_queue_len / (max_queue_size * 3), 1.0)
                pbrs_reward = self._calc_pbrs_shaping(sat_id, queue_pressure)
                reward[sat_id] += pbrs_reward
                reward_components[sat_id]["pbrs_reward"] = pbrs_reward

        # 6) 安全检查
        for k, v in reward.items():
            if np.isnan(v) or np.isinf(v):
                logger.error(f"[Reward Error] {k} reward is {v}, resetting to -10.0")
                reward[k] = -10.0

        # 7) 全局混合（仅结果项）
        if self.global_reward_ratio > 0 and len(reward) > 1:
            global_outcome = sum(outcome_reward.values()) / len(outcome_reward)
            for sat_id in reward:
                reward[sat_id] = (
                    reward[sat_id]
                    + self.global_reward_ratio * (global_outcome - outcome_reward[sat_id])
                )

        for sat_id in reward:
            reward_components[sat_id]["total_reward"] = reward[sat_id]

        self.last_reward_components = reward_components
        return reward


    def is_truncated(self, satellite) -> bool:
        """检查是否应该截断 episode（资源耗尽或其他软约束违反）。
        
        ✅ 多星协作逻辑：
        - 单颗卫星违反 DoD 约束：只标记警告，不截断（其他卫星继续运行）
        - 所有卫星都违反 DoD 约束：截断 episode
        
        Args:
            satellite: 卫星对象。
            
        Returns:
            True 如果应该截断 episode。
        """
        # 🆕 使用 battery_dod_valid() 软约束
        # 注意：这里只检查单颗卫星，环境层应该检查所有卫星
        if hasattr(satellite, 'dynamics') and hasattr(satellite.dynamics, 'battery_dod_valid'):
            if not satellite.dynamics.battery_dod_valid():
                # DoD 约束违反：记录警告，但不直接返回 True
                # 截断决策应由环境层检查所有卫星后决定
                logger.warning(
                    f"{satellite.name} battery below DoD threshold, "
                    f"SOC={satellite.dynamics.battery_charge_fraction:.2%}"
                )
                # 返回 True 表示这颗卫星建议截断
                # 环境层会聚合所有卫星的结果来做最终决策
                return True
        
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
        
        # 任务失败判定：任意切片过期即任务失败
        expired_ids = all_expired_ids
        completed_ids = all_completed_ids - expired_ids
        
        completed = len(completed_ids)
        expired = len(expired_ids)
        all_processed_ids = all_completed_ids | all_expired_ids  # 所有已处理的任务
        processed = len(all_processed_ids)
        pending = total_arrived - processed
        
        # 仅在回合结束时打印一次状态（避免频繁刷屏）
        first_sat = self.scenario.satellites[0] if self.scenario.satellites else None
        is_first_sat = (satellite == first_sat)
        
        if is_first_sat and sim_time > 0 and int(sim_time) % 7000 == 0:
            reason_summary = ""
            reason_counts = {}
            total_reason = 0
            for sat in self.scenario.satellites:
                counts = getattr(sat, "expired_reason_counts", {})
                for reason, count in counts.items():
                    if count <= 0:
                        continue
                    reason_counts[reason] = reason_counts.get(reason, 0) + count
                    total_reason += count
            if total_reason > 0:
                sorted_reasons = sorted(
                    reason_counts.items(), key=lambda item: item[1], reverse=True
                )
                top_parts = []
                for reason, count in sorted_reasons[:4]:
                    pct = 100.0 * count / total_reason
                    top_parts.append(f"{reason}={pct:.0f}%")
                if top_parts:
                    reason_summary = " | Expired reasons: " + ", ".join(top_parts)
            logger.warning(
                f"[TASK STATUS @ {sim_time:.0f}s] "
                f"Pool: {pool_remaining} | Arrived: {total_arrived} | "
                f"Processed: {processed} (C:{completed} E:{expired}) | "
                f"Pending: {pending}{reason_summary}"
            )
        
        # 任务池还有剩余，不能终止
        if pool_remaining > 0:
            return False
        
        # 任务池已空，检查已到达的任务是否都处理完
        # 使用 processed（唯一任务ID的并集）而不是 completed + expired
        if processed >= total_arrived and total_arrived > 0:
            if is_first_sat:  # 只打印一次
                logger.info(
                    f"[TERMINATED @ {sim_time:.0f}s] All tasks processed: "
                    f"{completed} completed, {expired} expired out of {total_arrived} arrived."
                )
            return True
        else:
            # 任务池空了但还有未处理的任务（在卫星队列中）
            pass  # 静默等待
        
        return False


__doc_title__ = "STIN Task Data"
__all__ = ["STINTaskReward", "STINTaskStore", "STINTaskData"]

