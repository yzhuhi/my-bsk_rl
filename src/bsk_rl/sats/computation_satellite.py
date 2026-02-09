"""STIN Computation Satellite Agent.

This module defines the ComputationSatellite class for satellite-terrestrial 
integrated network (STIN) multi-agent reinforcement learning environments.
"""

import logging
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from Basilisk.utilities.orbitalMotion import REQ_EARTH

from bsk_rl import sats, act, obs
from bsk_rl.sim import dyn, fsw
from bsk_rl.obs.relative_observations import r_DC_N  # 复用相对位置计算
from bsk_rl.utils.constants import (
    SPEED_OF_LIGHT,
    KAPPA,
    DEFAULT_UPLINK_RATE,
    DEFAULT_ISL_RATE,
    DEFAULT_FIBER_DISTANCE,
    DEFAULT_CLOUD_CPU_FREQ,
    DEFAULT_UD_CPU_FREQ,
    DEFAULT_RESULT_RATIO,
    # 动态速率模型 (论文公式 5, 6, 10)
    calculate_isl_rate,
    calculate_sgl_rate,
)
from bsk_rl.utils import vizard
import time

try:
    from Basilisk.architecture import messaging
    from Basilisk.ExternalModules import taskVizController
    TASK_VIZ_AVAILABLE = True
except ImportError:
    TASK_VIZ_AVAILABLE = False

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.act.actions import Action

logger = logging.getLogger(__name__)

# --- 防循环常量 ---
# MAX_HOPS = 5  # 最大跳数限制，超过后强制本地处理


# --- 任务状态枚举 ---
class TaskStatus:
    """任务状态常量。"""
    PENDING = "pending"           # 等待计算
    COMPUTING = "computing"       # 正在计算
    COMPUTED = "computed"         # 计算完成，等待回传
    RELAYED = "relayed"           # 已通过 ISL 转发给接力卫星
    RETURNING = "returning"       # 正在回传（ISL 接力中）
    COMPLETED = "completed"       # 任务完成（结果已送达）
    EXPIRED = "expired"           # 超时失败


# --- 辅助类: 任务切片结构 ---
class TaskSlice:
    """定义任务切片的结构，包含任务属性和状态。
    
    **正确的并行时延模型**：
        三条路径并行处理，取最大值：
        
        T_total = max(T_UD, T_SAT, T_CLOUD)
        datasize_total = [α_local, α_sat, α_cloud] × data_size

        其中：
        - T_UD: UD 本地处理时延（无上行传输，数据已在本地）
          T_UD = (α_local × data_size × workload) / f_ud  没问题！
        
        - T_SAT: 卫星边缘处理路径总时延（任务级别，取所有切片中最晚完成的）
          T_SAT = last_completion_time - creation_time
          
          单个切片的时延组成：
          t_slice = T_tx_up(α_sat) + T_prop_up + T_queue + T_compute + T_isl + T_prop_down
          
          其中 T_tx_up(α_sat) = (α_sat × data_size) / R_uplink(fixed)
          
          T_isl = T_tx_isl + T_prop_isl:
            - 协作切分：T_tx_isl (传输数据到邻居) + T_prop_isl (传播)
            - 接力中继：仅 T_prop_isl (结果数据量小，传输时延可忽略)
          
          注意：使用绝对完成时间而非累加时延，可正确处理：
            - 多切片异步处理（不同切片到达不同卫星）
            - 排队等待（队列阻塞导致的额外延迟）
            - 接力中继（多跳传递的累积时延）
        
        - T_CLOUD: 云端处理路径总时延
          T_CLOUD = T_tx_up(α_cloud) + T_tx_sgl(α_cloud) + 2×(T_prop_up + T_prop_sgl + T_fiber) + T_compute_cloud 没问题！
          
          其中：
            - T_tx_up(α_cloud) = (α_cloud × data_size) / R_uplink
            - T_tx_sgl(α_cloud) = (α_cloud × data_size) / R_sgl
            - 2× 表示往返传播（任务上传 + 结果返回）
    
    **层次化一级切分**：
        解决训练初期 α_local + α_cloud ≈ 1 导致 α_sat = 0 的问题：
        
        第一层: α_local 决定 UD 本地保留比例
        第二层: 在卸载部分 (1 - α_local) 中，α_cloud 作为偏好因子
                决定云端和卫星的相对比例
        
        示例 (两个动作都是 0.5):
          α_local = 0.5 → 50% 本地
          α_offload = 0.5 → 剩余 50% 待分配
          preference_cloud = 0.5 → 云端 25%, 卫星 25% (1-α_local)*(1-preference_cloud) >= 0
    
    **可见性约束**：
        - 当服务卫星失去对 UD 的可见性时，必须：
          1. 完成已分配的计算任务（需合理决策数据量）
          2. 或通过 ISL 接力将结果转发给可见卫星完成交付
            
    任务生命周期：PENDING → COMPUTING → COMPLETED/EXPIRED/RELAYED

    代码审查：时延建模没有问题，切分比例存在的问题也能够解决；
    """
    
    # 常量从 bsk_rl.utils.constants 模块导入
    # 保留类级别引用以保持向后兼容
    SPEED_OF_LIGHT = SPEED_OF_LIGHT
    DEFAULT_UPLINK_RATE = DEFAULT_UPLINK_RATE # 符合论文
    # DEFAULT_ISL_RATE = DEFAULT_ISL_RATE # 用不上了，动态计算
    DEFAULT_FIBER_DISTANCE = DEFAULT_FIBER_DISTANCE # 符合论文
    DEFAULT_CLOUD_CPU_FREQ = DEFAULT_CLOUD_CPU_FREQ # 符合论文
    DEFAULT_UD_CPU_FREQ = DEFAULT_UD_CPU_FREQ
    DEFAULT_RESULT_RATIO = DEFAULT_RESULT_RATIO

    def __init__(
        self, 
        task_id: int, 
        data_size: float, 
        workload: float,  # 计算复杂度
        max_delay: float, # 最大时延
        origin_position: np.ndarray = None,  # UD 位置（用于可见性检查）
        uplink_distance: float = 0.0,        # 上行链路距离 [m]
        uplink_rate: float = None,           # 上行传输速率 [bps]
        priority: float = 0.5,               # 任务优先级 [0, 1]
    ):
        """初始化任务切片。
        
        task_id: 任务 ID。
        data_size: [bits] 任务切片携带的数据量 (x_v * d_t)。
        workload: [cycles/bit] 任务计算复杂度。
        max_delay: [s] 任务容忍的最大时延 (含所有时延)。
        origin_position: [m] 任务来源位置 (Planet-fixed)，用于可见性检查。
        uplink_distance: [m] UD 到接入卫星的上行链路距离。
        uplink_rate: [bps] 上行传输速率，None 则使用默认值。
        priority: 任务优先级 [0, 1]，用于观测空间和奖励计算。
        """
        self.task_id = task_id # 任务 ID （有几类属性，只要切片属于同一个任务，这些属性都一样）
        self.data_size = data_size           # 剩余待处理数据量
        self.original_data_size = data_size  # 原始输入数据量
        self.result_data_size = data_size * self.DEFAULT_RESULT_RATIO  # 结果数据量（输出）只做设置，实际可能不用作时延计算的部分（相比传输速率太小）
        self.workload = workload # 计算复杂度
        self.max_delay = max_delay # 最大时延
        self.priority = priority  # 任务优先级 [0, 1]
        self.origin_position = origin_position  # UD 位置
        
        # 上行传输时延建模 √
        uplink_rate = uplink_rate or self.DEFAULT_UPLINK_RATE       # joint 那篇论文不考虑，本项目固定
        self.uplink_rate = uplink_rate                              # [bps] 上行速率
        self.t_tx_up = data_size / uplink_rate                      # [s] 上行传输时延
        
        # 传播时延建模 
        self.uplink_distance = uplink_distance                      # [m] 上行链路距离
        self.t_prop_up = uplink_distance / self.SPEED_OF_LIGHT      # [s] 上行传播时延
        self.t_prop_down = 0.0                                      # [s] 下行传播时延（完成时计算）跟功率有关
        
        # ISL 时延建模（区分协作切分 vs 接力） 两者不一定一一对应，有的数据只需要接力不需要处理；
        self.t_tx_isl_total = 0.0                                   # [s] ISL 传输时延累计（协作切分时）跟功率有关
        self.t_prop_isl_total = 0.0                                 # [s] ISL 传播时延累计
        
        # 云端路径参数 √
        self.fiber_distance = self.DEFAULT_FIBER_DISTANCE           # [m] 网关到云的光纤距离
        self.cloud_cpu_freq = self.DEFAULT_CLOUD_CPU_FREQ           # [Hz] 云端 CPU 频率
        self.ud_cpu_freq = self.DEFAULT_UD_CPU_FREQ                 # [Hz] UD 本地 CPU 频率
        
        # 时间戳跟踪
        self.creation_time = 0.0      # [s] 任务在 UD 创建的时间
        self.arrival_time = 0.0       # [s] 任务到达当前卫星的时间 (含上行时延)
        self.compute_start_time = 0.0 # [s] 开始计算的时间
        self.compute_end_time = 0.0   # [s] 计算完成的时间
        
        # 状态跟踪
        self.status = TaskStatus.PENDING
        self.current_holder: str = ""  # 当前持有该任务的卫星名称
        self.hop_count = 0             # ISL 转发跳数
        self.current_hop = 0           # 路由跳数（仅计算卸载链路）
        self.is_slice = False          # ✅ 是否为切片（True=原子切片，False=原始任务）
        
        # 路径时延跟踪（并行处理模型）
        self.alpha_local = 0.0         # UD 本地处理比例
        self.alpha_cloud = 0.0         # 云端处理比例
        self.alpha_sat = 0.0           # 卫星处理比例
        self.t_ud_path = 0.0           # [s] UD 路径总时延
        self.t_sat_path = 0.0          # [s] 卫星路径总时延
        self.t_cloud_path = 0.0        # [s] 云端路径总时延
        
        # 协作奖励追踪
        self.origin_satellite: str = ""  # 发起卸载决策的卫星名称（用于协作奖励）
        self.access_satellite: str = ""  # 最初接入的卫星（从 UD 接收任务的卫星，保持不变）
        self.offload_chain: list = []    # 卸载链路：记录所有参与卸载的卫星（用于延迟奖励分配）
        self.visited_sats: set = set()   # 路由去重：记录已访问卫星（用于回环检测）
    
    @property
    def t_uplink_total(self) -> float:
        """上行总时延 [s] = 传输 + 传播。"""
        return self.t_tx_up + self.t_prop_up
        
    @property
    def total_cycles(self) -> float:
        """计算任务所需的总 CPU 周期数。"""
        return self.original_data_size * self.workload
    
    @property
    def remaining_workload(self) -> float:
        """剩余计算工作量 [cycles]。"""
        return self.data_size * self.workload
    
    def get_elapsed_time(self, current_time: float) -> float:
        """计算已用时间 [s]（从任务创建到当前时间）。
        
        Args:
            current_time: 当前仿真时间 [s]。
            
        Returns:
            已用时间 [s]。
        """
        return current_time - self.creation_time
    
    def get_remaining_time(self, current_time: float) -> float:
        """计算剩余可用时间 [s]。
        
        Args:
            current_time: 当前仿真时间 [s]。
            
        Returns:
            剩余时间 [s]，如果已超时则返回 0。
        """
        return max(0.0, self.max_delay - self.get_elapsed_time(current_time))
    
    def is_expired(self, current_time: float) -> bool:
        """检查任务是否已超时。"""
        return (current_time - self.creation_time) > self.max_delay
    
    @property
    def t_isl_total(self) -> float:
        """卫星处理部分：ISL 总时延 [s] = 传输 + 传播。
        
        注意：
        - 协作切分时：t_tx_isl + t_prop_isl（需要传输数据）
        - 接力时：仅 t_prop_isl（结果数据量很小，传输时延忽略）
        ·
        - 我们的策略：希望尽可能避免，任务在某颗卫星上处理一部分后由于可见性或者能量原因被迫去做协作切分；
        - 协作切分：我们期望的是 分到的那一时间，立即判断是否需要协作切分，如果需要则立即进行协作切分，否则继续处理；
        """
        return self.t_tx_isl_total + self.t_prop_isl_total
    
    def calculate_ud_path_delay(self) -> float:
        """计算 UD 本地处理路径时延。
        
        T_UD = (alpha_local × data_size × workload) / f_ud
        
        注意：UD 本地处理不需要上行传输，数据已在 UD 本地。
        
        Returns:
            UD 路径时延 [s]。 √
        """
        if self.alpha_local < 1e-6:
            return 0.0
        
        # UD 本地处理的数据量
        data_local = self.original_data_size * self.alpha_local
        cycles_local = data_local * self.workload
        t_compute_ud = cycles_local / self.ud_cpu_freq
        
        # UD 本地处理不需要上行时延，只有计算时延
        return t_compute_ud
    
    def calculate_cloud_path_delay(self, sgl_distance: float = 600e3, tx_power: float = 15.0) -> float:
        """计算云端处理路径时延。
        
        T_CLOUD = T_tx_up(α_cloud) + T_tx_sgl(α_cloud) + 2*(T_prop_up + T_prop_sgl + T_fiber) + T_cloud_compute
        
        其中：
        - T_tx_up(α_cloud): 按 α_cloud 比例计算的上行传输时延（UD→卫星）
        - T_tx_sgl(α_cloud): 按 α_cloud 比例计算的 SGL 传输时延（卫星→网关）
        - 传播时延 ×2: 上行+下行（任务数据上传 + 结果返回）
        - T_cloud_compute: 云端计算时延
        - 带宽通常是 Tbps 级别。发送 1GB 数据可能只需要毫秒级。在这种情况下，传输时延相对于卫星链路可以忽略。但是，物理距离带来的传播时延（光跑几千公里需要的时间）是物理定律决定的，无法被带宽消除，所以传播时延必须保留。
        Args:
            sgl_distance: [m] 卫星到地面网关的距离，默认 600 km (LEO)。
            tx_power: [W] 发射功率 (Agent 可控)，用于动态 SGL 速率计算。
        
        Returns:
            云端路径时延 [s]。 √
        """
        if self.alpha_cloud < 1e-6:
            return 0.0
        
        # 按 alpha_cloud 比例计算的数据量
        data_cloud = self.original_data_size * self.alpha_cloud
        
        # 上行传输时延（UD → 卫星，按 α_cloud 比例）
        t_tx_up_cloud = data_cloud / self.uplink_rate
        
        # SGL 传输时延（卫星 → 网关，按 α_cloud 比例）
        # 动态 SGL 速率：基于发射功率 (论文公式 10)
        sgl_rate = calculate_sgl_rate(tx_power)
        t_tx_sgl = data_cloud / sgl_rate
        
        # 传播时延（×2 表示往返：任务上传 + 结果返回）
        t_prop_up = self.t_prop_up
        t_prop_sgl = sgl_distance / self.SPEED_OF_LIGHT
        t_fiber = self.fiber_distance / self.SPEED_OF_LIGHT
        t_prop_round_trip = 2 * (t_prop_up + t_prop_sgl + t_fiber)
        
        # 云端计算时延 没问题 符合公式
        cycles_cloud = data_cloud * self.workload
        t_compute_cloud = cycles_cloud / self.cloud_cpu_freq
        
        t_cloud = t_tx_up_cloud + t_tx_sgl + t_prop_round_trip + t_compute_cloud
        
        return t_cloud

    def __repr__(self) -> str:
        return (f"TaskSlice(id={self.task_id}, status={self.status}, "
                f"data={self.data_size/1e6:.2f}Mb)")


# --- 辅助函数 ---
# KAPPA 常量已从 bsk_rl.utils.constants 模块导入

def compute_energy_consumption(cpu_freq: float, cycles: float) -> float:
    """计算计算任务的能耗: E = κ * f² * cycles。
    
    Args:
        cpu_freq: [Hz] CPU 频率。
        cycles: [cycles] 总计算周期数。
        
    Returns:
        [J] 能耗。
    """
    if cpu_freq <= 0:
        return 0.0
    return KAPPA * (cpu_freq ** 2) * cycles


class ComputationSatellite(sats.AccessSatellite):
    """STIN 计算节点卫星智能体（论文模型实现）。
    
    扩展自 AccessSatellite，支持连续资源分配和任务切片的 MARL 智能体。
    
    **核心功能**:
        1. 连续资源分配 (CPU 频率、发射功率)
        2. 分层协作调度 (本地/云端/卫星间)
        3. 任务切片处理与队列管理
        4. 绝对时间模型的时延计算（处理多切片异步、排队、接力）
    
    **时隙操作流程**（论文 Section II-B）:
        每个时隙 τ (对应max_step_duration) 内，网络控制和数据传输按以下步骤执行：
        
        **Step 1: 信息收集** (gather_network_information)
            - 服务卫星通过 GSL 收集网络信息和任务请求
            - 包括：信道状态信息(CSI)、UD位置、任务属性、邻居状态等 
            - 信息收集完成后，服务卫星开始执行任务卸载和资源分配决策
            - 模型不考虑这部分时延（认为信息下发时延很小）
        
        **Step 2: 决策** (make_offloading_decision)
            - 基于收集的信息，服务卫星做出任务卸载和资源分配决策
            - 由 RL 智能体生成 10D 连续动作：
              * CPU 频率比例 + 发射功率比例（资源分配）
              * α_local、α_cloud（一级切分）
              * split_ratios[4]（二级卫星间切分）
        
        **Step 3: 控制消息分发** (distribute_control_messages)
            - 控制消息下发给 UD 和云计算中心
            - 包括：本地处理比例、云端处理比例、资源分配等
            - 这部分时延不考虑
        
        **Step 4: 任务执行** (execute_scheduled_tasks)
            - 任务按控制消息在 UD、卫星、云端并行执行
            - 开始执行任务后，任务的时延开始计算
            - 三条路径并行：T_total = max(T_UD, T_SAT, T_CLOUD)
        
        **Step 5: 切换处理** (handle_handover)
            - 服务卫星切换时，状态通过 ISL 同步到下一个服务卫星
            - 只处理未完成任务（接力中继）
    
    **时延模型**（绝对完成时间）:
        - T_UD: UD 本地处理时延（无传输，数据已在本地）
        - T_SAT: 卫星路径 = max(所有切片的完成时间) - 任务创建时间
                 ✓ 自动处理多切片异步、排队、接力的累积延迟
        - T_CLOUD: 云端处理路径（含上下行传输、传播、计算）
        
        使用绝对时间而非累加延迟的优势：
          ✓ 多切片异步处理（不同切片到达不同卫星）
          ✓ 排队等待（队列阻塞导致的额外延迟自动计入）
          ✓ 接力中继（多跳传递的累积延迟反映在完成时间差中）
    
    **任务生命周期**:
        PENDING → COMPUTING → COMPLETED/EXPIRED/RELAYED
        
        其中 RELAYED 表示任务计算完成但未回传，需通过 ISL 转发结果。
    
    **参数配置**:
        所有 Satellite Mobility Edge Computing (SMEC) 参数 (CPU频率、工作负载等) 
        通过 ComputationDynModel (仿真器动力学) 的 @default_args 装饰器定义，
        无需在此类重写 default_sat_args。
        
    **Example**:
        >>> sat = ComputationSatellite(
        ...     name="sat_1",
        ...     sat_args={"cpuMaxFrequency": 2.0e9, "cpuWorkload": 500.0}
        ... )
    """
    
    # --- 类属性: 模型类型定义 ---
    dyn_type = dyn.LoSComputationDynaModel
    # fsw_type = fsw.ImagingFSWModel
    fsw_type = fsw.SteeringImagerFSWModel
    
    # --- 类属性: 动作空间定义 --- 
    
    action_spec: List["Action"] = [act.STINContinuousAction(max_neighbors=4)]  # 连续动作空间（必须是实例）10D
    # --- 类属性: 观测空间定义 --- 40D 参数都做了归一化，不然NN计算会导致巨大的数值；加速收敛；
    observation_spec = [
        # 卫星状态 
        obs.SatProperties(
            dict(prop="battery_charge_fraction", module="dynamics"), # 电池电量
            dict(prop="storage_level_fraction", module="dynamics"), # 存储使用率
            dict(prop="r_BN_P", module="dynamics", norm=REQ_EARTH * 1e3), # 卫星位置
            dict(prop="v_BN_P", module="dynamics", norm=7616.5), # 卫星速度
        ),
        # STIN 特有状态 (简化模型: 不追踪结果队列)
        obs.SatProperties(
            dict(prop="current_cpu_freq", fn=lambda sat: sat.current_cpu_freq / 1e9), # CPU频率 [0, 2]
            dict(prop="task_queue_size", fn=lambda sat: min(len(sat.task_queue), 50) / 50.0), # 任务队列大小 [0, 1] ✅ Fix1: 归一化
            dict(prop="queue_workload", fn=lambda sat: min(sum(t.remaining_workload for t in sat.task_queue) / 1e9, 100.0) / 100.0), # 任务队列负载 [0, 1] ✅ Fix1: 归一化+截断
            name="stin_state"
        ),
        # 当前任务属性 (论文 Step 1: 任务请求信息)
        # -----------------------------
        # 单一任务级别-非统计
        # obs.SatProperties(
        #     dict(prop="current_task_data_size", fn=lambda sat: sat.task_queue[0].data_size / 1e6 if sat.task_queue else 0.0),  # [Mb]
        #     dict(prop="current_task_workload", fn=lambda sat: sat.task_queue[0].workload / 1e3 if sat.task_queue else 0.0),     # [kcycles/bit]
        #     dict(prop="current_task_max_delay", fn=lambda sat: sat.task_queue[0].max_delay if sat.task_queue else 0.0),        # [s]
        #     dict(prop="current_task_remaining_time", fn=lambda sat: sat.task_queue[0].get_remaining_time(sat.simulator.sim_time) if sat.task_queue else 0.0),  # [s]
        #     # dict(prop="current_task_uplink_delay", fn=lambda sat: sat.task_queue[0].t_uplink_total * 1000 if sat.task_queue else 0.0),  # [ms]
        #     dict(prop="current_task_priority", fn=lambda sat: getattr(sat.task_queue[0], 'priority', 0.5) if sat.task_queue else 0.0),  # [0, 1] 任务优先级
        #     name="task_request"
        # ),
        # -----------------------------
        # ✅ 原始任务队列统计信息 (raw_task_queue - 可切分任务)
        obs.SatProperties(
            dict(prop="raw_queue_size", fn=lambda sat: min(len(sat.raw_task_queue), 50) / 50.0),  # 原始任务数量 [0, 1] ✅ Fix1: 归一化
            dict(prop="raw_queue_data", fn=lambda sat: min(sum(t.data_size for t in sat.raw_task_queue) / 1e9, 50.0) / 50.0 if sat.raw_task_queue else 0.0),  # [Gb] 原始任务总数据量 [0, 1] ✅ Fix1: 归一化
            dict(prop="raw_avg_priority", fn=lambda sat: float(np.nanmean([getattr(t, 'priority', 0.5) for t in sat.raw_task_queue])) if sat.raw_task_queue else 0.5),  # [0, 1] 平均优先级 ✅ Fix1: nanmean + 默认0.5
            name="raw_task_stats"
        ),
        # -----------------------------
        # 切片队列统计信息 (slice_queue/task_queue - 原子切片，只能执行或转发)
        obs.SatProperties(
            dict(prop="avg_task_data_size", fn=lambda sat: min(float(np.nanmean([t.data_size for t in sat.task_queue])) / 1e6, 500.0) / 500.0 if sat.task_queue else 0.0),  # [Mb] 平均数据量 [0, 1] ✅ Fix1: nanmean + 归一化
            dict(prop="avg_task_workload", fn=lambda sat: min(float(np.nanmean([t.workload for t in sat.task_queue])) / 1e3, 10.0) / 10.0 if sat.task_queue else 0.0),     # [kcycles/bit] 平均复杂度 [0, 1] ✅ Fix1: nanmean + 归一化
            dict(prop="min_remaining_time", fn=lambda sat: min(min((t.get_remaining_time(sat.simulator.sim_time) for t in sat.task_queue), default=0.0), 300.0) / 300.0),  # [s] 最紧急任务剩余时间 [0, 1] ✅ Fix1: 归一化
            dict(prop="avg_priority", fn=lambda sat: float(np.nanmean([getattr(t, 'priority', 0.5) for t in sat.task_queue])) if sat.task_queue else 0.5),  # [0, 1] 平均优先级 ✅ Fix1: nanmean + 默认0.5
            dict(prop="total_queue_data", fn=lambda sat: min(sum(t.data_size for t in sat.task_queue) / 1e9, 50.0) / 50.0 if sat.task_queue else 0.0),  # [Gb] 队列总数据量 [0, 1] ✅ Fix1: 归一化
            name="slice_queue_stats"
        ),
        # -----------------------------
        # 邻居卫星状态（灵活配置）
        obs.STINRelativeObservations(
            dict(prop="get_isl_distance", norm=1e7),           # ISL 距离 [m]，归一化到 10,000 km
            dict(prop="get_battery_fraction"),                 # 电池电量分数 [0, 1]
            dict(prop="get_task_queue_size", norm=40.0),       # 任务队列长度
            dict(prop="get_queue_workload", norm=1e10),        # 队列工作量 [cycles]
            dict(prop="get_cpu_freq", norm=2e9),               # CPU 频率 [Hz]
            dict(prop="get_isl_channel_quality"),              # 🆕 ISL 信道质量 [0, 1]
            max_neighbors=4,  # 最多观测 4 个邻居
        ),
        # 时间
        obs.Time(),
        obs.Eclipse(norm=5700.0),
    ]
    
    # 全局去重：同一任务只打印一次 EXPIRED
    _expired_task_ids_logged: set[str] = set()

    def __init__(self, *args, **kwargs) -> None:
        """初始化计算卫星。
        
        sat_args 中的参数会自动通过 collect_default_args 从 dyn_type 和 
        fsw_type 中收集。用户可通过 sat_args 覆盖默认值。
        
        Args (via kwargs):
            training_mode: 训练模式（默认 False）。启用后禁用日志和可视化以提升性能。
            shield_config: Action Shield 配置字典（可选）。
            ud_config: UD 处理能力配置字典（可选）。
        """
        # 提取 training_mode 参数（在调用 super 前，避免传递给父类）
        self.training_mode = kwargs.pop('training_mode', True)
        
        # 提取 shield_config 参数（用于 Action 层读取）
        self.shield_config = kwargs.pop('shield_config', {})
        
        # 提取 ud_config 参数（用于协作调度）
        self.ud_config = kwargs.pop('ud_config', {})
        
        # 提取 sim_step 参数（用于协作调度）
        self.sim_step = kwargs.pop('sim_step', 15)
        # 提取 ISL 最大距离参数（不进入 sat_args）
        self.isl_max_distance_km = kwargs.pop("isl_max_distance_km", None)
        
        super().__init__(*args, **kwargs)
        
        # MARL 核心追踪变量 - 任务处理
        self.task_queue: List[TaskSlice] = []      # 待计算任务队列（输入数据）
        self.result_queue: List[TaskSlice] = []    # 待回传结果队列（输出数据）
        
        # 任务历史缓冲区（用于 STINTaskStore 提取物理指标）
        self.completed_tasks_buffer: List[TaskSlice] = []   # 本 step 完成的任务列表
        self.expired_tasks_buffer: List[TaskSlice] = []     # 本 step 超时的任务列表
        
        # 资源状态
        self.current_cpu_freq = 0.0                # [Hz] 当前分配的 CPU 频率
        self.current_tx_power = 0.0                # [W] 当前分配的发射功率
        self.current_base_power_draw = 0.0         # [W] 当前平台基础功耗
        
        # 统计量
        self.processed_data_total = 0.0            # [bits] 累计处理数据量（卫星本地执行）
        self.offloaded_data_total = 0.0            # [bits] 累计卸载数据量（卫星发给别人，包含所有跳）
        self.first_hop_offloaded = 0.0             # [bits] 首次卸载（从原始接收卫星卸载出去）
        self.relay_offloaded = 0.0                 # [bits] 中继卸载（已接收的切片再次卸载）
        self.raw_data_received = 0.0               # [bits] 从 UD 接收的原始任务数据总量
        self.completed_tasks_count = 0             # 成功完成的任务数（slice级别，可能重复）
        self.expired_tasks_count = 0               # 超时失败的任务数（slice级别，可能重复）
        self.expired_reason_counts: dict = {}      # 超时原因统计（任务级）
        # ✅ 任务级别统计（使用集合追踪唯一任务ID，避免slice重复计数）
        self.completed_task_ids: set = set()      # 已完成的原始任务ID集合
        self.expired_task_ids: set = set()        # 已过期的原始任务ID集合
        self.offloaded_task_ids: set = set()      # 已卸载的任务ID集合（首跳卸载）
        self.relay_task_ids: set = set()          # 被中继的任务ID集合（二跳及以上）
        self.result_relay_task_ids: set = set()   # 结果回传经 ISL 接力的任务ID集合
        
        # 时隙控制状态 (论文 Step 3)
        self.current_control_message: dict = {}   # 当前时隙的控制消息
        self.network_info: dict = {}               # 收集的网络信息 (Step 1)
        
        # Vizard 可视化事件发送器（训练模式下禁用）
        if TASK_VIZ_AVAILABLE and not self.training_mode:
            self.task_event_msg = messaging.TaskEventMsgPayload()
            self.task_event_writer = messaging.TaskEventMsg_C()
            self.task_event_enabled = True
        else:
            self.task_event_enabled = False

    
    def __getstate__(self):
        """Exclude unpicklable SWIG objects from serialization."""
        state = self.__dict__.copy()
        # Remove unpicklable entries
        if 'task_event_msg' in state:
            del state['task_event_msg']
        if 'task_event_writer' in state:
            del state['task_event_writer']
        return state
    
    def __setstate__(self, state):
        """Restore state and reinitialize SWIG objects."""
        self.__dict__.update(state)
        # Reinitialize SWIG objects if enabled
        if TASK_VIZ_AVAILABLE and getattr(self, 'task_event_enabled', False):
            self.task_event_msg = messaging.TaskEventMsgPayload()
            self.task_event_writer = messaging.TaskEventMsg_C()

    def reset_overwrite_previous(self) -> None:
        super().reset_overwrite_previous()
        ComputationSatellite._expired_task_ids_logged = set()
        # ✅ 三队列架构
        self.raw_task_queue = []   # 原始任务队列（可切分）
        self.slice_queue = []      # 原子切片队列（只能执行或转发）
        self.result_queue = []     # 结果队列（等待回传）
        # 兼容性别名：旧代码可能引用 task_queue
        self.task_queue = self.slice_queue
        # 重置任务历史缓冲区
        self.completed_tasks_buffer = []
        self.expired_tasks_buffer = []
        # 重置资源状态
        self.current_cpu_freq = 0.0
        self.current_tx_power = 0.0
        self.current_base_power_draw = 0.0
        # 重置统计量
        self.processed_data_total = 0.0
        self.offloaded_data_total = 0.0
        self.first_hop_offloaded = 0.0
        self.relay_offloaded = 0.0
        self.raw_data_received = 0.0
        self.completed_tasks_count = 0
        self.expired_tasks_count = 0
        # ✅ 重置任务ID集合
        self.completed_task_ids = set()
        self.expired_task_ids = set()
        self.offloaded_task_ids = set()
        self.relay_task_ids = set()
        self.result_relay_task_ids = set()
        
        # ✅ 累积协作指标统计（用于 gym.py 直接读取，避免 buffer 时序问题）
        self.total_hops_accumulated = 0       # 累计跳数（所有完成任务的 hop_count 之和）
        self.total_latency_accumulated = 0.0  # 累计时延（所有完成任务的 T_total 之和）
        self.remote_completed_count = 0       # 异地完成的任务数
        
        # 🆕 收发数据量统计（用于论文能耗模型 E_tr 和 E_re）
        self.sent_data_this_step = 0.0        # [bits] 本步发送的数据量
        self.received_data_this_step = 0.0    # [bits] 本步接收的数据量
        self.tx_time_this_step = 0.0          # [s] 本步发送时间
        self.rx_time_this_step = 0.0          # [s] 本步接收时间
        self.offload_attempts_this_step = 0   # 🆕 本步卸载尝试次数（用于论文指标）
        self.level2_decisions_this_step = 0   # 🆕 Level-2 路由决策次数
        self.level2_self_selected_this_step = 0  # 🆕 Level-2 选择本地处理次数
        
        # ✅ 切片聚合追踪：task_id -> {"total": n, "completed": n, "expired": n}
        self.slice_registry: dict = {}
        # 重置时隙控制状态
        self.current_control_message = {}
        self.network_info = {}

    def _should_log_task_expired(self, task_id: str) -> bool:
        if task_id in ComputationSatellite._expired_task_ids_logged:
            return False
        ComputationSatellite._expired_task_ids_logged.add(task_id)
        return True

    def _log_task_expired_once(
        self,
        task_id: str,
        reason: str = "",
        total_delay: float | None = None,
    ) -> None:
        if not self._should_log_task_expired(task_id):
            return
        if reason:
            self.expired_reason_counts[reason] = self.expired_reason_counts.get(reason, 0) + 1
        reason_part = f" reason={reason}" if reason else ""
        delay_part = f" T_total={total_delay:.4f}s" if total_delay is not None else ""
        self.logger.info(
            f"[TASK EXPIRED] Task {task_id}{delay_part}{reason_part}"
        )

    # --- 切片聚合追踪方法 ---
    
    def register_slice(self, task_id: str, slice_count: int = 1, creation_time: float = 0.0) -> None:
        """注册任务的切片数量（在切片创建时调用）。
        
        Args:
            task_id: 任务ID
            slice_count: 切片数量
            creation_time: 任务创建时间（用于计算真实时延）
        """
        if task_id not in self.slice_registry:
            self.slice_registry[task_id] = {
                "total": 0, 
                "completed": 0, 
                "expired": 0,
                "creation_time": creation_time,       # ✅ 任务创建时间
                "last_completion_time": 0.0,          # ✅ 最后一个切片完成的绝对时间
                "t_ud_path": 0.0,                     # UD路径时延（任务级别共享）
                "t_cloud_path": 0.0,                  # 云端路径时延（任务级别共享）
            }
        self.slice_registry[task_id]["total"] += slice_count
        # 更新创建时间（如果尚未设置）
        if self.slice_registry[task_id]["creation_time"] == 0.0 and creation_time > 0.0:
            self.slice_registry[task_id]["creation_time"] = creation_time
    
    def mark_slice_completed(self, task_id: str, completion_time: float = 0.0) -> bool:
        """标记一个切片完成，返回该任务是否全部完成。
        
        Args:
            task_id: 任务ID
            completion_time: 该切片完成的绝对时间点
            
        Returns:
            True 如果所有切片都已完成
        """
        if task_id not in self.slice_registry:
            # 未注册的切片（可能是 UD 处理的），直接返回 True
            return True
        self.slice_registry[task_id]["completed"] += 1
        # ✅ 更新最后完成时间（取最大值，即最后一个完成的切片）
        if completion_time > self.slice_registry[task_id]["last_completion_time"]:
            self.slice_registry[task_id]["last_completion_time"] = completion_time
        status = self.slice_registry[task_id]
        return status["completed"] == status["total"]
    
    def get_task_total_delay(self, task_id: str) -> float:
        """获取任务的总时延 = max(T_UD, T_SAT, T_CLOUD)。
        
        T_SAT = 最后一个切片完成时间 - 任务创建时间
        """
        if task_id not in self.slice_registry:
            return 0.0
        status = self.slice_registry[task_id]
        # ✅ T_SAT 使用真实的完成时间差
        t_sat = status["last_completion_time"] - status["creation_time"]
        return max(
            status.get("t_ud_path", 0.0),
            t_sat,
            status.get("t_cloud_path", 0.0)
        )
    
    def mark_slice_expired(self, task_id: str) -> None:
        """标记一个切片超时。"""
        if task_id not in self.slice_registry:
            self.slice_registry[task_id] = {"total": 1, "completed": 0, "expired": 0}
        self.slice_registry[task_id]["expired"] += 1
    
    def _find_satellite_by_name(self, name: str):
        """通过名称查找卫星对象。"""
        if not hasattr(self, 'simulator') or self.simulator is None:
            return None
        for sat in self.simulator.satellites:
            if sat.name == name:
                return sat
        return None

    # --- 存储容量属性（用于观测空间）---
    
    @property
    def storage_level(self) -> float:
        """当前队列数据量 [bits]。"""
        return sum(t.data_size for t in self.task_queue)
    
    @property
    def storage_capacity(self) -> float:
        """存储容量 [bits]，从动力学模型获取。"""
        return getattr(self.dynamics, 'data_storage_capacity', float('inf'))
    
    @property
    def storage_fraction(self) -> float:
        """存储使用率 [0, 1]，用于观测归一化。"""
        capacity = self.storage_capacity
        if capacity <= 0 or capacity == float('inf'):
            return 0.0
        return min(self.storage_level / capacity, 1.0)

    # --- 修复父类的 numpy 数组比较问题 ---
    
    def _add_window(self, object, new_window, type, r_LP_P=None, merge_time=None):
        """覆盖父类方法以修复 numpy 数组比较问题。
        
        原父类 AccessSatellite._add_window 在比较 opportunity["object"] == object
        时，当 object 是 numpy 数组时会抛出 ValueError。
        
        Args:
            object: 目标对象（可能是标量或 numpy 数组）
            new_window: 新窗口时间范围 (start, end)
            type: 窗口类型
            r_LP_P: 相对位置向量
            merge_time: 合并时间点
        """
        import bisect
        
        if new_window[0] == merge_time or merge_time is None:
            for opportunity in self.opportunities:
                # 安全的对象比较（支持 numpy 数组）
                try:
                    objects_equal = opportunity["object"] == object
                    # 如果是数组比较，numpy 会返回数组，需要用 all()
                    if hasattr(objects_equal, '__iter__'):
                        objects_equal = all(objects_equal)
                except (ValueError, TypeError):
                    # 回退到身份比较
                    objects_equal = opportunity["object"] is object
                
                if (
                    opportunity["type"] == type
                    and objects_equal
                    and opportunity["window"][1] == new_window[0]
                ):
                    opportunity["window"] = (opportunity["window"][0], new_window[1])
                    return
        
        bisect.insort(
            self.opportunities,
            {"object": object, "window": new_window, "type": type, "r_LP_P": r_LP_P},
            key=lambda x: x["window"][1],
        )

    # --- 资源分配执行接口 (对应动作维度: CPU频率、发射功率) ---
    
    def set_resource_allocation(
        self, 
        cpu_ratio: float, 
        tx_power_ratio: float,
        platform_power_ratio: float
    ) -> None:
        """
        卫星通常分为两大部分：载荷（Payload） 和 平台（Platform/Bus）。
        载荷功耗（前两项）：是你为了完成任务（比如计算任务、通信任务）额外消耗的电。
        平台功耗（第三项 base_power）：是不管你做不做任务，只要卫星还“活着”就需要消耗的电。它包括
        根据比例设置 CPU 频率和发射功率。
        
        平台功耗是一个相对固定的值（即 max_base_power）。代码引入了 platform_power_ratio，
        这意味你在仿真中引入了**“休眠模式”或“省电模式”**的机制。

        Args:
            cpu_ratio: CPU 频率比例 ∈ [0, 1]，映射到 [f_min, f_max]。
            tx_power_ratio: 发射功率比例 ∈ [0, 1]，映射到 [0, max_tx_power]。
            platform_power_ratio: 平台功耗比例 ∈ [0, 1]，映射到 [0, max_base_power]。  # ← 添加此行
        """
        # 注意：电池安全硬约束已通过 STINActionShield 在动作空间层实现
        # 无需在此处再次裁剪，避免双重保护导致的策略学习困难
        
        # === 正常资源分配流程 ===
        # 从 dynamics 层获取参数 (由 @default_args 定义)
        min_f = self.dynamics.cpu_min_frequency
        max_f = self.dynamics.cpu_max_frequency
        
        # 1. 连续 CPU 频率控制 (f_u)
        self.current_cpu_freq = min_f + (max_f - min_f) * np.clip(cpu_ratio, 0.0, 1.0)
        
        # 2. 连续发射功率控制 (p_u)
        max_tx_power = abs(self.sat_args.get("transmitterPowerDraw", -15.0))
        self.current_tx_power = max_tx_power * np.clip(tx_power_ratio, 0.0, 1.0)
        
        # 3. 平台功耗管理 (e_platform) 引入比例意味着
        #    你可以在仿真中引入**“休眠模式”或“省电模式”**的机制。 
        max_base_power = abs(self.sat_args.get("basePowerDraw", 0.0))
        self.current_base_power_draw = -max_base_power * np.clip(platform_power_ratio, 0.0, 1.0)
        
        # 4. 更新 BSK 物理层功耗
        # CPU 功耗: P = P_max * (f/f_max)²
        freq_ratio = self.current_cpu_freq / max_f if max_f > 0 else 0.0
        cpu_power = self.dynamics.cpu_power_draw * (freq_ratio ** 2)
        self.dynamics.set_cpu_power(cpu_power)
        
        # 平台基础功耗
        self.dynamics.set_base_power(self.current_base_power_draw)
        
        self.logger.debug(
            f"Resource: F={self.current_cpu_freq/1e9:.2f} GHz, "
            f"P_tx={self.current_tx_power:.2f} W, P_cpu={abs(cpu_power):.2f} W, "
            f"P_base={abs(self.current_base_power_draw):.2f} W"
        )

    def handle_handover(
        self,
        next_serving_satellite: "ComputationSatellite",
    ) -> bool:
        """Step 5: 处理服务卫星切换。
        
        当服务卫星切换时，通过 ISL 将当前任务执行状态和连接状态
        同步到下一个服务卫星，确保任务完成的连续性。
        agent要尽可能保证卸载的任务能够被卫星计算完成，而不会频繁切换，频繁切换就说明策略失败；
        Args:
            next_serving_satellite: 接手服务的下一个卫星。
        
        Returns:
            True 如果切换成功。

               # --- Step 5: 服务卫星切换 (handle_handover) ---
                # 设计决策：不实现未完成任务的切换转发
                # 理由：
                #   1. Agent 应学习在可见窗口内完成任务分配
                #   2. 结果回传使用接力模式 (execute_result_relay)
                #   3. 频繁切换意味着策略失败，应通过奖励惩罚学习避免
                # 如需实现，参考 docs/STIN_IMPLEMENTATION_SUMMARY.md 中的设计
        """
        if not self.task_queue:
            self.logger.info(f"[Step 5] Handover to {next_serving_satellite.name}: No active tasks.")
            return True
        
        # 计算 ISL 传播时延
        isl_distance = self._compute_isl_distance(next_serving_satellite)
        isl_delay = isl_distance / TaskSlice.SPEED_OF_LIGHT
        
        # 同步所有未完成任务
        tasks_transferred = 0
        while self.task_queue:
            task_slice = self.task_queue.pop(0)
            
            # 累加 ISL 传播时延
            task_slice.t_prop_isl_total += isl_delay
            task_slice.hop_count += 1
            task_slice.current_holder = next_serving_satellite.name
            
            # 将任务转移到下一个卫星
            next_serving_satellite.task_queue.append(task_slice)
            tasks_transferred += 1
        
        # 同步控制状态
        next_serving_satellite.current_control_message = self.current_control_message.copy()
        next_serving_satellite.network_info = self.network_info.copy()
        
        self.logger.info(
            f"[Step 5] Handover to {next_serving_satellite.name}: "
            f"{tasks_transferred} tasks transferred, ISL delay={isl_delay*1000:.2f}ms"
        )
        
        return True
    
    # --- 协作调度执行接口 (分层协作: UD/云端/卫星间) ---

    def schedule_collaboration_action(
        self, 
        alpha_local: float, 
        alpha_cloud: float,
        priority_threshold: float,
        high_split_ratios: np.ndarray,
        low_split_ratios: np.ndarray,
        neighbor_satellites: List["ComputationSatellite"]
    ) -> None:
        """基于优先级的差异化切分，循环处理队列中所有需要切分的任务。
        
        实现三层分层协作模型:
            1. 一级切分: α_local (UD本地) + α_cloud (云端) + α_sat (卫星) - 统一应用
            2. 二级切分: 根据任务优先级选择不同的卫星间协作比例
               - 高优先级 (priority >= threshold): 使用 high_split_ratios
               - 低优先级 (priority < threshold): 使用 low_split_ratios

        Args:
            alpha_local: UD 本地处理比例（统一应用）。
            alpha_cloud: 云端处理比例（统一应用）。
            priority_threshold: 高/低优先级分界阈值 [0, 1]。
            high_split_ratios: 高优先级任务的卫星间切分比例 [x_self, x_n1, ...]。
            low_split_ratios: 低优先级任务的卫星间切分比例 [x_self, x_n1, ...]。
            neighbor_satellites: 邻居卫星列表。
        """
        # ✅ 三队列架构：只处理原始任务队列（可切分任务）
        # 切片队列（slice_queue）由 execute_local_compute 处理
        max_tasks_per_step = 30  # 防止无限循环
        tasks_processed = 0
        
        while self.raw_task_queue and tasks_processed < max_tasks_per_step:
            current_raw_task = self.raw_task_queue[0]
            tasks_processed += 1
            
            # 根据任务优先级选择切分比例
            task_priority = getattr(current_raw_task, 'priority', 0.5)
            if task_priority >= priority_threshold:
                split_ratios = high_split_ratios
                priority_label = "HIGH"
            else:
                split_ratios = low_split_ratios
                priority_label = "LOW"
            
            # 调用内部切分处理方法
            self._process_single_task_split(
                current_raw_task, 
                alpha_local, 
                alpha_cloud, 
                split_ratios, 
                neighbor_satellites,
                priority_label
            )
            
            # 任务已被切分，从原始任务队列移除
            self.raw_task_queue.pop(0)
        
        if tasks_processed > 0:
            self.requires_retasking = True
    
    def _process_single_task_split(
        self,
        current_raw_task,
        alpha_local: float,
        alpha_cloud: float,
        split_ratios: np.ndarray,
        neighbor_satellites: List["ComputationSatellite"],
        priority_label: str = ""
    ) -> None:
        """处理单个任务的切分逻辑（从原 schedule_collaboration_action 提取）。
        
        Args:
            current_raw_task: 当前要切分的任务
            alpha_local: UD 本地保留比例
            alpha_cloud: 云端偏好因子
            split_ratios: 卫星间切分比例
            neighbor_satellites: 邻居卫星列表
            priority_label: 优先级标签（用于日志）
        """
        
        # --------------------------------------------------------
        # Step 1: 层次化一级切分 - 确保卫星总能分到任务
        # --------------------------------------------------------
        # 
        # 问题：如果使用 alpha_local + alpha_cloud = 1 - alpha_sat 的剩余法，
        # 训练初期 Actor 输出的两个 alpha 都在 0.5 左右时，alpha_sat = 0。
        #
        # 解决方案：层次化切分
        #   第一层: alpha_local 决定 UD 本地保留多少
        #   第二层: 在卸载部分 (1 - alpha_local) 中，alpha_cloud 作为偏好因子
        #          决定云端和卫星的相对比例
        #
        # 这样即使两个动作都是 0.5：
        #   alpha_local = 0.5 → 50% 本地
        #   alpha_offload = 0.5 → 剩余 50% 待分配
        #   preference_cloud = 0.5 → 云端 25%, 卫星 25%
        # --------------------------------------------------------
        
        # 1. 第一层：UD 本地保留比例
        alpha_local_final = np.clip(alpha_local, 0.0, 1.0)
        
        # ✅ 关键约束：非接入卫星不能使用 UD 和云端路径
        # 协作任务（α_sat 部分）一旦决定在卫星间处理，就只能在卫星间传递
        is_access_satellite = (getattr(current_raw_task, 'access_satellite', '') == '' or 
                               getattr(current_raw_task, 'access_satellite', '') == self.name)
        if not is_access_satellite:
            # 非接入卫星：强制 α_local=0, α_cloud=0，全部由卫星处理
            alpha_local_final = 0.0
            alpha_cloud = 0.0  # 覆盖输入
            self.logger.debug(
                f"[COLLAB CONSTRAINT] Non-access sat {self.name} processing task from {current_raw_task.access_satellite}. "
                f"Forcing α_local=0, α_cloud=0"
            )
        
        # 2. 计算卸载部分
        alpha_offload_total = 1.0 - alpha_local_final
        
        # 3. 第二层：在卸载部分中分配云端和卫星
        # alpha_cloud 输入作为偏好因子 [0, 1]
        preference_cloud = np.clip(alpha_cloud, 0.0, 1.0)
        
        if alpha_offload_total > 1e-6:
            alpha_cloud_final = alpha_offload_total * preference_cloud
            alpha_sat_final = alpha_offload_total * (1.0 - preference_cloud)
        else:
            alpha_cloud_final = 0.0
            alpha_sat_final = 0.0

        # 只有在无卫星切分时，UD 完成才可视为任务级完成
        allow_ud_completion = alpha_sat_final <= 1e-6

        self.logger.warning (
            f"[COLLAB] α_local={alpha_local_final:.2%}, α_cloud={alpha_cloud_final:.2%}, "
            f"α_sat={alpha_sat_final:.2%} | split_ratios={[f'{r:.2f}' for r in split_ratios]}"
        )
        
        # --------------------------------------------------------
        # Step 2: 规范化二级切分 (x_v) - 使用 Softmax 确保和为 1
        # --------------------------------------------------------
        
        # ✅ 模式 A: Softmax 归一化（用于原始任务切分）
        # 这确保切分比例总和为 1，且所有值为正
        exp_logits = np.exp(split_ratios - np.max(split_ratios))  # 数值稳定性
        normalized_xv_ratios = exp_logits / np.sum(exp_logits)
             
        # --------------------------------------------------------
        # Step 3: 执行切分和分配 (Task Slice Creation & Routing)
        # --------------------------------------------------------
        
        raw_data_size = current_raw_task.data_size 
        task_workload = current_raw_task.workload
        
        # 记录路径分配比例（用于并行时延计算）
        current_raw_task.alpha_local = alpha_local_final
        current_raw_task.alpha_cloud = alpha_cloud_final
        current_raw_task.alpha_sat = alpha_sat_final

        # 3.1 远端/本地切分 (宏观协作)
        
        # UD 本地处理 (Alpha_Local)
        # 简化模型：UD 计算能力有限，可通过 ud_cpu_ratio 配置
        # 如果分配给 UD 的任务在 max_delay 内完不成，视为超时
        if alpha_local_final > 1e-6:
            data_local = raw_data_size * alpha_local_final
            
            # UD 处理能力估算（从配置读取 ud_cpu_ratio）
            ud_cpu_ratio = self.ud_config.get('ud_cpu_ratio', 0.5)
            ud_cpu_freq = self.dynamics.cpu_min_frequency * ud_cpu_ratio  # 默认 ~1 GHz
            ud_process_time = (data_local * task_workload) / ud_cpu_freq
            remaining_time = current_raw_task.get_remaining_time(self.simulator.sim_time)
            
            # ✅ 记录 UD 路径时延（用于任务级别 max 计算）
            # T_UD = 计算时延（无上行，数据已在本地）
            t_ud_path = ud_process_time
            
            if ud_process_time <= remaining_time:
                # UD 能完成 → 算作成功（但奖励归 UD，卫星不拿）
                self.logger.debug(
                    f"Task {current_raw_task.task_id}: UD Local can complete "
                    f"{data_local/1e6:.2f} Mb in {ud_process_time:.1f}s (remaining: {remaining_time:.1f}s)"
                )
                # ✅ 在 slice_registry 中记录 UD 路径时延
                task_id = current_raw_task.task_id
                if task_id not in self.slice_registry:
                    self.slice_registry[task_id] = {
                        "total": 0, "completed": 0, "expired": 0,
                        "creation_time": current_raw_task.creation_time,
                        "last_completion_time": 0.0,
                        "t_ud_path": 0.0, "t_cloud_path": 0.0
                    }
                self.slice_registry[task_id]["t_ud_path"] = t_ud_path
                
                # ✅ 仅当无卫星切分时，UD 完成才计为任务级完成
                if allow_ud_completion:
                    if current_raw_task.task_id not in self.completed_task_ids:
                        self.completed_task_ids.add(current_raw_task.task_id)
                        # ✅ 创建 UD 完成的虚拟切片并添加到 buffer
                        ud_completed_slice = TaskSlice(
                            task_id=current_raw_task.task_id,
                            data_size=data_local,
                            workload=task_workload,
                            max_delay=current_raw_task.max_delay,
                            origin_position=current_raw_task.origin_position,
                            uplink_distance=current_raw_task.uplink_distance,
                            uplink_rate=current_raw_task.uplink_rate,
                        )
                        ud_completed_slice.status = TaskStatus.COMPLETED
                        ud_completed_slice.priority = getattr(current_raw_task, 'priority', 1.0)
                        ud_completed_slice.t_ud_path = t_ud_path
                        ud_completed_slice.t_sat_path = 0.0  # ✅ UD本地完成，无卫星路径
                        ud_completed_slice.t_cloud_path = 0.0  # ✅ UD本地完成，无云端路径
                        ud_completed_slice.creation_time = current_raw_task.creation_time
                        ud_completed_slice.compute_end_time = self.simulator.sim_time
                        ud_completed_slice.access_satellite = self.name  # UD 完成，接入卫星即自己
                        ud_completed_slice.hop_count = 0  # ✅ UD本地完成，没有转发
                        self.completed_tasks_buffer.append(ud_completed_slice)
            else:
                # UD 完不成 → 超时，惩罚落在接入卫星头上
                self.logger.debug(
                    f"Task {current_raw_task.task_id}: UD Local TIMEOUT! "
                    f"Need {ud_process_time:.1f}s but only {remaining_time:.1f}s remaining"
                )
                # 创建一个虚拟的超时切片，让奖励函数能检测到
                expired_ud_slice = TaskSlice(
                    task_id=current_raw_task.task_id,
                    data_size=data_local,
                    workload=task_workload,
                    max_delay=current_raw_task.max_delay,
                    origin_position=current_raw_task.origin_position,
                    uplink_distance=current_raw_task.uplink_distance,
                    uplink_rate=current_raw_task.uplink_rate,
                )
                expired_ud_slice.status = TaskStatus.EXPIRED
                expired_ud_slice.priority = getattr(current_raw_task, 'priority', 1.0)
                expired_ud_slice.origin_satellite = self.name  # 接入卫星负责
                # ✅ 修复重复计数
                if expired_ud_slice.task_id not in self.expired_task_ids:
                    self.expired_tasks_count += 1
                    self.expired_task_ids.add(expired_ud_slice.task_id)
                self.expired_tasks_buffer.append(expired_ud_slice)
                self._log_task_expired_once(
                    expired_ud_slice.task_id, reason="ud_timeout"
                )
        
        # 云端处理 (Alpha_Cloud)
        if alpha_cloud_final > 1e-6:
            data_cloud = raw_data_size * alpha_cloud_final
            
            # ✅ 计算云端路径时延（用于任务级别 max 计算）
            # T_CLOUD = T_tx_up + T_tx_sgl + 2×(T_prop_up + T_prop_sgl + T_fiber) + T_compute_cloud
            # 使用 TaskSlice 的方法计算
            t_cloud_path = current_raw_task.calculate_cloud_path_delay(
                sgl_distance=self._get_downlink_distance(current_raw_task, "GS"),
                tx_power=self.current_tx_power
            )
            
            # ✅ 在 slice_registry 中记录云端路径时延
            task_id = current_raw_task.task_id
            if task_id not in self.slice_registry:
                self.slice_registry[task_id] = {
                    "total": 0, "completed": 0, "expired": 0,
                    "creation_time": current_raw_task.creation_time,
                    "last_completion_time": 0.0,
                    "t_ud_path": 0.0, "t_cloud_path": 0.0
                }
            self.slice_registry[task_id]["t_cloud_path"] = t_cloud_path
            
            self.logger.info(
                f"Task {current_raw_task.task_id}: Routed {data_cloud/1e6:.2f} Mb to Cloud via SGL. "
                f"T_cloud={t_cloud_path:.4f}s"
            )
        
        # 3.2 卫星协作切分 (Alpha_Sat)
        if alpha_sat_final > 1e-6:
            
            all_sat_nodes = [self] + neighbor_satellites  # 自身 + 邻居卫星
            num_actual_nodes = len(all_sat_nodes)  # 实际可用节点数 (1 ~ 5)
            
            # 处理邻居数量不足的情况：
            # 如果实际节点 < 5，将多余的比例重新分配给自身 (索引 0)
            if num_actual_nodes < len(normalized_xv_ratios):
                # 计算多余的比例（分配给不存在的邻居的比例）
                unused_ratio = np.sum(normalized_xv_ratios[num_actual_nodes:])
                # 重新分配给自身
                adjusted_ratios = normalized_xv_ratios[:num_actual_nodes].copy()
                adjusted_ratios[0] += unused_ratio
                # 重新归一化
                adjusted_ratios = adjusted_ratios / np.sum(adjusted_ratios) if np.sum(adjusted_ratios) > 0 else adjusted_ratios
            else:
                adjusted_ratios = normalized_xv_ratios[:num_actual_nodes]
            
            # ✅ 预计算：非接入卫星二次切分时，计算新增切片数
            is_access_sat = (getattr(current_raw_task, 'access_satellite', '') == '' or 
                            getattr(current_raw_task, 'access_satellite', '') == self.name)
            if not is_access_sat:
                # 计算会创建多少个有效切片（ratio > 1e-6 的数量）
                new_slice_count = sum(1 for r in adjusted_ratios if r > 1e-6)
                if new_slice_count > 1:
                    # 原来1个切片变成 N 个，需要额外注册 N-1 个
                    access_sat_name = getattr(current_raw_task, 'access_satellite', '')
                    access_sat = self._find_satellite_by_name(access_sat_name)
                    if access_sat:
                        access_sat.register_slice(current_raw_task.task_id, new_slice_count - 1)
                        self.logger.debug(
                            f"[SLICE REGISTRY] Secondary split: 1 slice → {new_slice_count} slices, "
                            f"registered {new_slice_count - 1} extra to access sat {access_sat_name}"
                        )
            
            for i, target_node in enumerate(all_sat_nodes):
                relative_ratio = adjusted_ratios[i]
                
                if relative_ratio > 1e-6:
                    # 计算实际数据量: Alpha_sat_final * 相对比例
                    slice_data = raw_data_size * alpha_sat_final * relative_ratio 
                    
                    # ✅ 防循环检查: 禁止直接回传给上一跳发送者
                    task_origin = getattr(current_raw_task, 'origin_satellite', '')
                    if target_node != self and target_node.name == task_origin:
                        self.logger.debug(
                            f"[LOOP PREVENTION] Skip sending to {target_node.name} - it's the origin of this task"
                        )
                        # 将这部分比例重新分配给自己
                        adjusted_ratios[0] += relative_ratio
                        continue
                    
                    # 计算 ISL 时延 (如果卸载到其他卫星)
                    # 协作切分时：需要传输数据，有传输时延 + 传播时延
                    isl_tx_delay = 0.0
                    isl_prop_delay = 0.0
                    if target_node != self:
                        isl_distance = self._compute_isl_distance(target_node)
                        # 动态 ISL 速率：基于发射功率和距离 (论文公式 6)
                        isl_rate = calculate_isl_rate(
                            self.current_tx_power,
                            isl_distance,
                            max_distance=self._get_isl_max_distance_m(),
                        )
                        # ISL 传输时延 = 数据量 / 动态速率
                        isl_tx_delay = slice_data / isl_rate
                        # ISL 传播时延 = 距离 / 光速
                        isl_prop_delay = isl_distance / TaskSlice.SPEED_OF_LIGHT
                    
                    # 创建新的切片，继承原始任务的属性
                    new_slice = TaskSlice(
                        task_id=current_raw_task.task_id,
                        data_size=slice_data,
                        workload=task_workload,
                        max_delay=current_raw_task.max_delay,  # 保持时延约束
                        origin_position=current_raw_task.origin_position,  # 继承 UD 位置
                        uplink_distance=current_raw_task.uplink_distance,  # 继承上行距离
                        uplink_rate=current_raw_task.uplink_rate,  # 继承上行速率
                        priority=current_raw_task.priority,  # 继承任务优先级
                    )
                    
                    # 累加 ISL 时延（协作切分：传输 + 传播）
                    new_slice.t_tx_isl_total = current_raw_task.t_tx_isl_total + isl_tx_delay
                    new_slice.t_prop_isl_total = current_raw_task.t_prop_isl_total + isl_prop_delay
                    new_slice.hop_count = current_raw_task.hop_count + (1 if target_node != self else 0)
                    new_slice.creation_time = current_raw_task.creation_time  # 继承原始创建时间
                    
                    # 继承云端和 UD 参数
                    new_slice.fiber_distance = current_raw_task.fiber_distance
                    new_slice.cloud_cpu_freq = current_raw_task.cloud_cpu_freq
                    new_slice.ud_cpu_freq = current_raw_task.ud_cpu_freq
                    new_slice.alpha_local = alpha_local_final
                    new_slice.alpha_cloud = alpha_cloud_final
                    new_slice.alpha_sat = alpha_sat_final
                    new_slice.is_slice = True  # ✅ 标记为原子切片（不可再切分）
                    new_slice.isl_tx_power = self.current_tx_power  # 🆕 记录 ISL 传输时的发射功率
                    
                    # 记录发起卸载的卫星（用于协作奖励）
                    new_slice.origin_satellite = self.name
                    
                    # 记录接入卫星（第一个从 UD 收到任务的卫星，保持不变）
                    # 如果原始任务已有 access_satellite，继承它；否则设为当前卫星
                    original_access = getattr(current_raw_task, 'access_satellite', '')
                    new_slice.access_satellite = original_access if original_access else self.name
                    
                    # 记录卸载链路（用于延迟奖励分配）
                    if not new_slice.offload_chain:
                        new_slice.offload_chain = [self.name]  # 首次卸载，发起者是链路第一个
                    # 目标节点会在接收时追加自己
                    
                    # ✅ 修复：处理返回值，如果目标拒绝则不记录卸载统计
                    accepted = target_node.process_incoming_slice(new_slice)
                    if not accepted:
                        self.logger.warning(
                            f"[OFFLOAD REJECTED] Target {target_node.name} rejected slice {new_slice.task_id}, "
                            f"keeping locally"
                        )
                        # 目标拒绝，切片回到本地切片队列（原子切片）
                        self.slice_queue.append(new_slice)
                        continue
                    
                    # ✅ 注册切片（仅接入卫星首次切分时注册）
                    # 非接入卫星二次切分的注册已在循环前统一处理
                    is_access_sat_for_slice = (new_slice.access_satellite == self.name)
                    if is_access_sat_for_slice:
                        # 接入卫星首次切分：每个新切片注册一次，传入任务创建时间
                        self.register_slice(
                            new_slice.task_id, 
                            1, 
                            creation_time=new_slice.creation_time
                        )
                    
                    if target_node != self:
                        self.offloaded_data_total += slice_data
                        
                        # 区分首次卸载和中继卸载
                        # 如果当前任务的 origin_satellite 为空或等于自己，说明是首次卸载
                        # 否则是中继卸载（转发别人发来的切片）
                        task_origin = getattr(current_raw_task, 'origin_satellite', '')
                        is_first_hop = (task_origin == '' or task_origin == self.name)
                        if is_first_hop:
                            self.first_hop_offloaded += slice_data
                            # 追踪卸载的任务ID（按任务数统计）
                            self.offloaded_task_ids.add(current_raw_task.task_id)
                        else:
                            self.relay_offloaded += slice_data
                            # 追踪中继的任务ID（二跳及以上）
                            self.relay_task_ids.add(current_raw_task.task_id)
                        
                        self.logger.info(
                            f"[Collab] {'First-hop' if is_first_hop else 'Relay'} offloaded {slice_data/1e6:.2f} Mb to {target_node.name}. "
                            f"ISL delay: tx={isl_tx_delay*1000:.2f}ms + prop={isl_prop_delay*1000:.2f}ms"
                        )
                        # 🆕 可视化星间卸载连线（黄色）
                        self.draw_isl_offload_line(target_node.name)

                    else:
                        self.logger.info(f"Self-allocated {slice_data/1e6:.2f} Mb for local execution.")
                        
        # ✅ 注意：任务从队列移除的操作已移到 schedule_collaboration_action 循环中
    
    def receive_task_from_scenario(self, computation_task) -> None:
        """接收来自 Scenario 的原始计算任务并转换为 TaskSlice。
        
        当卫星成为任务的接入点时调用此方法。
        
        Args:
            computation_task: ComputationTask 对象（来自 STINTaskScenario）。
        """
        # 计算上行链路距离
        r_sat = np.array(self.dynamics.r_BN_N)
        r_ud = computation_task.origin_position
        uplink_distance = np.linalg.norm(r_sat - r_ud)
        
        # 创建初始任务切片（完整任务，未切分）
        initial_slice = TaskSlice(
            task_id=computation_task.task_id,
            data_size=computation_task.data_size,
            workload=computation_task.workload,
            max_delay=computation_task.max_delay,
            origin_position=computation_task.origin_position,
            uplink_distance=uplink_distance,
            uplink_rate=computation_task.uplink_rate,
            priority=getattr(computation_task, 'priority', 0.5),  # 继承任务优先级
        )
        
        # ✅ 存储对原始任务的引用，用于后续删除BSK事件
        initial_slice.origin_task = computation_task
        
        # 设置云端和 UD 参数（从 ComputationTask 继承）
        if hasattr(computation_task, 'fiber_distance'):
            initial_slice.fiber_distance = computation_task.fiber_distance
        if hasattr(computation_task, 'cloud_cpu_freq'):
            initial_slice.cloud_cpu_freq = computation_task.cloud_cpu_freq
        if hasattr(computation_task, 'ud_cpu_freq'):
            initial_slice.ud_cpu_freq = computation_task.ud_cpu_freq
        
        # 设置任务创建时间
        initial_slice.creation_time = self.simulator.sim_time
        initial_slice.arrival_time = self.simulator.sim_time + initial_slice.t_uplink_total
        initial_slice.current_holder = self.name
        initial_slice.access_satellite = self.name  # 设置接入卫星（用于协作检测）
        
        # 🆕 存储容量检查：防止队列数据量超过卫星存储容量
        current_total_data = (
            sum(t.data_size for t in self.raw_task_queue) +
            sum(t.data_size for t in getattr(self, 'slice_queue', []))
        )
        storage_capacity = getattr(self.dynamics, 'data_storage_capacity', float('inf'))
        
        if current_total_data + initial_slice.data_size > storage_capacity:
            # 存储已满，拒绝任务并标记为过期
            self._log_task_expired_once(
                initial_slice.task_id, reason="storage_full"
            )
            self.logger.debug(
                f"[STORAGE FULL] Dropping task {initial_slice.task_id}: "
                f"current {current_total_data/1e9:.2f} Gb + new {initial_slice.data_size/1e6:.2f} Mb > "
                f"capacity {storage_capacity/1e9:.2f} Gb"
            )
            initial_slice.status = TaskStatus.EXPIRED
            if initial_slice.task_id not in self.expired_task_ids:
                self.expired_tasks_count += 1
                self.expired_task_ids.add(initial_slice.task_id)
            self.expired_tasks_buffer.append(initial_slice)
            return  # 不加入队列
        
        # 加入原始任务队列（可切分）
        self.raw_task_queue.append(initial_slice)
        
        # 记录从 UD 接收的原始任务数据量
        self.raw_data_received += initial_slice.data_size
        
        self.logger.info(
            f"Received NEW task {initial_slice.task_id}: "
            f"{initial_slice.data_size/1e6:.2f} Mb, "
            f"max_delay={initial_slice.max_delay:.2f}s, "
            f"uplink_delay={initial_slice.t_uplink_total*1000:.2f}ms "
            f"(tx={initial_slice.t_tx_up*1000:.2f}ms + prop={initial_slice.t_prop_up*1000:.2f}ms). "
            f"raw_queue={len(self.raw_task_queue)}, "
            f"slice_queue={len(getattr(self, 'slice_queue', []))}, "
            f"task_queue={len(self.task_queue)}"
        )
        
        # Emit uplink visualization event
        self._emit_viz_event(4, "UD", initial_slice.origin_position)
        
        self.requires_retasking = True
    
    def process_incoming_slice(self, task_slice: TaskSlice) -> bool:
        """接收并处理来自其他卫星的任务切片。
        
        Args:
            task_slice: 接收到的任务切片对象。
            
        Returns:
            True 如果成功接收，False 如果因存储容量不足被拒绝。
        """
        # ✅ 存储容量检查：基于数据量而非任务数量
        current_queue_data = sum(t.data_size for t in self.slice_queue)
        storage_capacity = getattr(self.dynamics, 'data_storage_capacity', float('inf'))
        
        if current_queue_data + task_slice.data_size > storage_capacity:
            self.logger.warning(
                f"[STORAGE FULL] Rejecting slice {task_slice.task_id}: "
                f"queue {current_queue_data/1e9:.2f} Gb + new {task_slice.data_size/1e6:.2f} Mb > "
                f"capacity {storage_capacity/1e9:.2f} Gb"
            )
            return False  # 拒绝接收
        
        # 记录切片到达时间
        task_slice.arrival_time = self.simulator.sim_time
        task_slice.current_holder = self.name
        # 记录访问轨迹（用于回环检测）
        if not hasattr(task_slice, "visited_sats") or task_slice.visited_sats is None:
            task_slice.visited_sats = set()
        elif not isinstance(task_slice.visited_sats, set):
            task_slice.visited_sats = set(task_slice.visited_sats)
        task_slice.visited_sats.add(self.name)
        
        # 追加到卸载链路（用于延迟奖励分配）
        if hasattr(task_slice, 'offload_chain'):
            task_slice.offload_chain.append(self.name)
        
        # 加入切片队列（原子切片，不可再切分）
        self.slice_queue.append(task_slice)
        
        # 🆕 统计接收数据量（用于论文能耗模型 E_re）
        self.received_data_this_step += task_slice.data_size
        # 接收时间 = 发送方传输时间（从切片中获取，如果有记录）
        isl_tx_delay = getattr(task_slice, 'last_isl_tx_delay', 0.0)
        self.rx_time_this_step += isl_tx_delay
        
        self.logger.info(
            f"Received slice {task_slice.task_id}: {task_slice.data_size/1e6:.2f} Mb, "
            f"workload={task_slice.workload}, max_delay={task_slice.max_delay}s. "
            f"Queue: {len(self.slice_queue)} slices, {(current_queue_data + task_slice.data_size)/1e9:.2f} Gb"
        )
        
        # 标记需要重新调度
        self.requires_retasking = True
        return True
    
    def _mask_and_normalize_routing_probs(
        self,
        base_ratios: np.ndarray,
        neighbor_satellites: List["ComputationSatellite"],
    ) -> np.ndarray:
        """Mask out invalid neighbors and normalize probabilities.
        
        Args:
            base_ratios: Array of shape (1 + num_neighbors,). Index 0 is self, rest are neighbors.
            neighbor_satellites: List of neighbor satellite objects. None means invalid/disconnected.
            
        Returns:
            Normalized probability distribution.
        """
        # Clip negative values to 0 (for linear normalization)
        valid_logits = np.maximum(base_ratios.copy(), 0.0)
        
        # Mask invalid neighbors (None or disconnected)
        for i in range(1, len(valid_logits)):
            neighbor_idx = i - 1
            if neighbor_idx >= len(neighbor_satellites) or neighbor_satellites[neighbor_idx] is None:
                valid_logits[i] = 0.0
                
        # Normalize
        total = np.sum(valid_logits)
        if total > 1e-6:
            return valid_logits / total
        else:
            # Fallback to self only if all invalid
            probs = np.zeros_like(valid_logits)
            probs[0] = 1.0
            return probs
    
    def process_slice_queue(
        self,
        neighbor_satellites: List["ComputationSatellite"],
        max_slices: int = None,  # ✅ 限制处理的切片数量（观测-动作一致性）
        split_ratios: Optional[np.ndarray] = None,
        high_split_ratios: Optional[np.ndarray] = None,
        low_split_ratios: Optional[np.ndarray] = None,
        priority_threshold: Optional[float] = None,
        use_argmax: bool = False,
    ) -> None:
        """处理切片队列的路由决策（分布采样/argmax）。
        
        对于 slice_queue 中的切片，按分布采样决定：
        - winner_index == 0: 本地执行，保留在队列等待 execute_local_compute
        - winner_index > 0: 转发给邻居，从队列移除
        
        Args:
            neighbor_satellites: 邻居卫星列表
            max_slices: 最多处理的切片数量（用于观测-动作一致性，只处理 step 开始时存在的切片）
            split_ratios: 统一路由分布 [x_self, x_n1, x_n2, ...]
            high_split_ratios: 高优先级路由分布
            low_split_ratios: 低优先级路由分布
            priority_threshold: 高/低优先级分界阈值
            use_argmax: 是否使用 argmax 作为确定性策略
        """
        max_forwards_per_step = 80  # 防止无限循环
        forwards_count = 0
        processed_count = 0  # ✅ 跟踪已处理的切片数量
        
        # ✅ 添加 max_slices 检查：只处理 step 开始时就存在的切片
        while self.slice_queue and forwards_count < max_forwards_per_step:
            # 检查是否已处理足够多的切片（观测-动作一致性）
            if max_slices is not None and processed_count >= max_slices:
                break
            
            current_slice = self.slice_queue[0]
            
            # 检查是否超时
            if current_slice.is_expired(self.simulator.sim_time):
                expired_slice = self.slice_queue.pop(0)
                expired_slice.status = TaskStatus.EXPIRED
                if expired_slice.task_id not in self.expired_task_ids:
                    self.expired_tasks_count += 1
                    self.expired_task_ids.add(expired_slice.task_id)
                self.expired_tasks_buffer.append(expired_slice)
                self._log_task_expired_once(
                    expired_slice.task_id, reason="slice_queue_timeout"
                )
                processed_count += 1
                continue
            
            # 逐切片选择高/低路由分布（若提供 split_ratios 则保持统一分布）
            if split_ratios is not None:
                routing_ratios = split_ratios
            else:
                slice_priority = getattr(current_slice, 'priority', 0.5)
                threshold = 0.5 if priority_threshold is None else priority_threshold
                if high_split_ratios is None or low_split_ratios is None:
                    raise ValueError("high_split_ratios and low_split_ratios are required when split_ratios is None")
                routing_ratios = high_split_ratios if slice_priority >= threshold else low_split_ratios

            # mask 不可达邻居并归一化
            probs = self._mask_and_normalize_routing_probs(
                base_ratios=routing_ratios,
                neighbor_satellites=neighbor_satellites,
            )
            
            # 采样或 argmax 路由目标
            if use_argmax:
                winner_index = int(np.argmax(probs))
            else:
                winner_index = int(np.random.choice(len(probs), p=probs))
            
            if winner_index == 0:
                # 自己处理 → 保留在队列，让 execute_local_compute 处理
                processed_count += 1  # ✅ 计入已处理
                break
            else:
                # 转发给邻居
                neighbor_idx = winner_index - 1
                if neighbor_idx < len(neighbor_satellites) and neighbor_satellites[neighbor_idx] is not None:
                    target_neighbor = neighbor_satellites[neighbor_idx]

                    # 防 ping-pong：禁止立即回传给上一跳
                    prev_hop = None
                    chain = getattr(current_slice, "offload_chain", [])
                    if len(chain) >= 2:
                        prev_hop = chain[-2]
                    if prev_hop and target_neighbor.name == prev_hop:
                        self.logger.debug(
                            f"[PINGPONG] Skip backtrack for slice {current_slice.task_id} "
                            f"from {self.name} to {target_neighbor.name}"
                        )
                        processed_count += 1  # 计入已处理，避免重复选择
                        break
                    
                    # 🆕 统计卸载尝试（用于论文指标）
                    self.offload_attempts_this_step += 1
                    
                    # 更新切片信息
                    current_slice.hop_count += 1
                    current_slice.current_holder = target_neighbor.name
                    if hasattr(current_slice, "offload_chain"):
                        if not current_slice.offload_chain or current_slice.offload_chain[-1] != self.name:
                            current_slice.offload_chain.append(self.name)
                        current_slice.offload_chain.append(target_neighbor.name)
                    
                    # 计算 ISL 传播时延
                    isl_distance = self._compute_isl_distance(target_neighbor)
                    isl_prop_delay = isl_distance / TaskSlice.SPEED_OF_LIGHT
                    current_slice.t_prop_isl_total += isl_prop_delay
                    
                    # 🆕 计算 ISL 传输时延（使用物理模型动态速率）
                    try:
                        tx_power = getattr(self, 'current_tx_power', 0.0)
                        if tx_power <= 0:
                            tx_power = abs(getattr(self.dynamics, 'transmitter_power_draw', 15.0))
                        isl_rate = calculate_isl_rate(
                            tx_power,
                            isl_distance,
                            max_distance=self._get_isl_max_distance_m(),
                        )
                    except Exception:
                        isl_rate = abs(getattr(self.dynamics, 'transmitter_baud_rate', 50e6))
                    
                    # --- Scheme A: 链路容量截断与残留 (Backlog) ---
                    max_transferable = isl_rate * self.sim_step
                    
                    is_congested = False
                    if current_slice.data_size > max_transferable:
                        is_congested = True
                        remaining_data = current_slice.data_size - max_transferable
                        
                        # 创建残留切片 (Backlog)
                        import copy
                        backlog_slice = copy.deepcopy(current_slice)
                        backlog_slice.data_size = remaining_data
                        backlog_slice.priority = getattr(current_slice, "priority", 0.5)
                        backlog_slice.is_slice = True
                        
                        # 修改当前切片为截断后的大小
                        current_slice.data_size = max_transferable
                    
                    isl_tx_delay = current_slice.data_size / isl_rate  # [s]
                    current_slice.t_tx_isl_total = getattr(current_slice, 't_tx_isl_total', 0.0) + isl_tx_delay
                    current_slice.last_isl_tx_delay = isl_tx_delay  # 供接收方使用

                    # Offload metrics (count actual transmitted bits)
                    if hasattr(self, "offloaded_data_total"):
                        self.offloaded_data_total += current_slice.data_size
                        task_origin = getattr(current_slice, "origin_satellite", "")
                        is_first_hop = (task_origin == "" or task_origin == self.name)
                        if is_first_hop:
                            if hasattr(self, "first_hop_offloaded"):
                                self.first_hop_offloaded += current_slice.data_size
                            if hasattr(self, "offloaded_task_ids"):
                                self.offloaded_task_ids.add(current_slice.task_id)
                        else:
                            if hasattr(self, "relay_offloaded"):
                                self.relay_offloaded += current_slice.data_size
                            if hasattr(self, "relay_task_ids"):
                                self.relay_task_ids.add(current_slice.task_id)

                    # 🆕 统计发送数据量和时间（用于论文能耗模型 E_tr）
                    self.sent_data_this_step += current_slice.data_size
                    self.tx_time_this_step += isl_tx_delay
                    
                    # 转发
                    target_neighbor.slice_queue.append(current_slice)
                    self.slice_queue.pop(0)
                    
                    if is_congested:
                        # 将残留切片插入队列头部，留待下一时隙处理
                        self.slice_queue.insert(0, backlog_slice)
                        # 停止当前 step 的队列处理（队首阻塞）
                        self.logger.info(
                            f"[CONGESTION] Link to {target_neighbor.name} saturated. "
                            f"Sent {max_transferable/1e6:.2f}Mb, backlog {remaining_data/1e6:.2f}Mb."
                        )
                        break
                    
                    forwards_count += 1
                    processed_count += 1  # ✅ 计入已处理
                    
                    self.logger.info(
                        f"[RELAY] Forwarding slice {current_slice.task_id} to {target_neighbor.name}. "
                        f"hop_count={current_slice.hop_count}, tx_delay={isl_tx_delay*1000:.2f}ms, prop_delay={isl_prop_delay*1000:.2f}ms"
                    )
                    # 🆕 可视化中继转发连线（黄色）
                    self.draw_isl_offload_line(target_neighbor.name)
                else:
                    # 邻居不可达，强制本地执行
                    self.logger.warning(
                        f"[RELAY] No valid neighbor for slice {current_slice.task_id}, forcing local execution"
                    )
                    break
                
    # 固定比例路由版本；                    
    # def process_slice_queue(
    #     self,
    #     neighbor_satellites: List["ComputationSatellite"],
    #     max_slices: int = None,  # ✅ 限制处理的切片数量（观测-动作一致性）
    #     split_ratios: Optional[np.ndarray] = None,
    #     high_split_ratios: Optional[np.ndarray] = None,
    #     low_split_ratios: Optional[np.ndarray] = None,
    #     priority_threshold: Optional[float] = None,
    # ) -> None:
    #     """模式 B: 处理切片队列的路由决策（Argmax 选择）。
        
    #     对于 slice_queue 中的切片，使用 Argmax 决定：
    #     - winner_index == 0: 本地执行，保留在队列等待 execute_local_compute
    #     - winner_index > 0: 转发给邻居，从队列移除
        
    #     Args:
    #         neighbor_satellites: 邻居卫星列表
    #         max_slices: 最多处理的切片数量（用于观测-动作一致性，只处理 step 开始时存在的切片）
    #         split_ratios: 兼容旧接口的路由权重 [x_self, x_n1, x_n2, ...]
    #         high_split_ratios: 高优先级路由权重
    #         low_split_ratios: 低优先级路由权重
    #         priority_threshold: 高/低优先级分界阈值
    #     """
    #     max_forwards_per_step = 20  # 防止无限循环
    #     forwards_count = 0
    #     processed_count = 0  # ✅ 跟踪已处理的切片数量
        
    #     # ✅ 添加 max_slices 检查：只处理 step 开始时就存在的切片
    #     while self.slice_queue and forwards_count < max_forwards_per_step:
    #         # 检查是否已处理足够多的切片（观测-动作一致性）
    #         if max_slices is not None and processed_count >= max_slices:
    #             break
            
    #         current_slice = self.slice_queue[0]
            
    #         # 检查是否超时
    #         if current_slice.is_expired(self.simulator.sim_time):
    #             expired_slice = self.slice_queue.pop(0)
    #             expired_slice.status = TaskStatus.EXPIRED
    #             if expired_slice.task_id not in self.expired_task_ids:
    #                 self.expired_tasks_count += 1
    #                 self.expired_task_ids.add(expired_slice.task_id)
    #             self.expired_tasks_buffer.append(expired_slice)
    #             continue
            
    #         # ✅ 模式 B: Argmax 路由选择
    #         # 逐切片选择高/低路由比例（若提供 split_ratios 则保持旧行为）
    #         if split_ratios is not None:
    #             routing_ratios = split_ratios
    #         else:
    #             slice_priority = getattr(current_slice, 'priority', 0.5)
    #             threshold = 0.5 if priority_threshold is None else priority_threshold
    #             if high_split_ratios is None or low_split_ratios is None:
    #                 raise ValueError("high_split_ratios and low_split_ratios are required when split_ratios is None")
    #             routing_ratios = high_split_ratios if slice_priority >= threshold else low_split_ratios

    #         # 构建有效邻居 mask（不可达邻居设为 -inf）
    #         valid_logits = routing_ratios.copy()
    #         for i in range(1, len(valid_logits)):
    #             neighbor_idx = i - 1
    #             if neighbor_idx >= len(neighbor_satellites) or neighbor_satellites[neighbor_idx] is None:
    #                 valid_logits[i] = -np.inf  # 不可达
            
    #         winner_index = np.argmax(valid_logits) # 使用 Argmax 选择路由目标 单纯
            
    #         if winner_index == 0:
    #             # 自己处理 → 保留在队列，让 execute_local_compute 处理
    #             processed_count += 1  # ✅ 计入已处理
    #             break
    #         else:
    #             # 转发给邻居
    #             neighbor_idx = winner_index - 1
    #             if neighbor_idx < len(neighbor_satellites) and neighbor_satellites[neighbor_idx] is not None:
    #                 target_neighbor = neighbor_satellites[neighbor_idx]
                    
    #                 # 🆕 统计卸载尝试（用于论文指标）
    #                 self.offload_attempts_this_step += 1
                    
    #                 # 更新切片信息
    #                 current_slice.hop_count += 1
    #                 current_slice.current_holder = target_neighbor.name
                    
    #                 # 计算 ISL 传播时延
    #                 isl_distance = self._compute_isl_distance(target_neighbor)
    #                 isl_prop_delay = isl_distance / TaskSlice.SPEED_OF_LIGHT
    #                 current_slice.t_prop_isl_total += isl_prop_delay
                    
    #                 # 🆕 计算 ISL 传输时延（论文公式：t_tx = x / C）
    #                 isl_rate = getattr(self.dynamics, 'transmitter_baud_rate', 50e6)  # bps
    #                 isl_rate = abs(isl_rate) if isl_rate else 50e6
    #                 isl_tx_delay = current_slice.data_size / isl_rate  # [s]
    #                 current_slice.t_tx_isl_total = getattr(current_slice, 't_tx_isl_total', 0.0) + isl_tx_delay
    #                 current_slice.last_isl_tx_delay = isl_tx_delay  # 供接收方使用
                    
    #                 # 🆕 统计发送数据量和时间（用于论文能耗模型 E_tr）
    #                 self.sent_data_this_step += current_slice.data_size
    #                 self.tx_time_this_step += isl_tx_delay
                    
    #                 # 转发
    #                 target_neighbor.slice_queue.append(current_slice)
    #                 self.slice_queue.pop(0)
    #                 forwards_count += 1
    #                 processed_count += 1  # ✅ 计入已处理
                    
    #                 self.logger.info(
    #                     f"[RELAY] Forwarding slice {current_slice.task_id} to {target_neighbor.name}. "
    #                     f"hop_count={current_slice.hop_count}, tx_delay={isl_tx_delay*1000:.2f}ms, prop_delay={isl_prop_delay*1000:.2f}ms"
    #                 )
    #             else:
    #                 # 邻居不可达，强制本地执行
    #                 self.logger.warning(
    #                     f"[RELAY] No valid neighbor for slice {current_slice.task_id}, forcing local execution"
    #                 )
    #                 break

    def execute_local_compute(self, duration: float) -> None:
        """执行本地计算任务切片（在每个 step 中调用）。
        
        ✅ 改进：使用时间预算循环，在 step 时间内连续处理多个任务。
        
        Args:
            duration: [s] 本次计算的持续时间（通常为 step_duration）。
        """
        # [DIAG] 诊断入口日志（训练模式下跳过）
        if not self.training_mode:
            self.logger.debug(
                f"[DIAG] execute_local_compute called: queue_size={len(self.task_queue)}, "
                f"cpu_freq={self.current_cpu_freq:.2e}, duration={duration:.2f}s"
            )
        
        # Emit viz event based on state
        if len(self.task_queue) > 0:
            self._emit_viz_event(1)  # COMPUTING
        else:
            self._emit_viz_event(0)  # IDLE
        
        if not self.task_queue:
            if not self.training_mode:
                self.logger.debug("[DIAG] RETURN: task_queue is empty!")
            return
        
        # 检查 CPU 是否有效分配 (使用 dynamics 层参数)
        if self.current_cpu_freq < self.dynamics.cpu_min_frequency:
            if not self.training_mode:
                self.logger.warning(
                    f"[DIAG] RETURN: CPU freq too low! current={self.current_cpu_freq:.2e}, "
                    f"min={self.dynamics.cpu_min_frequency:.2e}"
                )
            return
        
        # ✅ 时间预算循环：在剩余时间内连续处理任务
        remaining_time = duration
        max_tasks_per_step = 40  # 防止无限循环
        tasks_processed = 0
        
        while self.task_queue and remaining_time > 1e-6 and tasks_processed < max_tasks_per_step:
            current_slice = self.task_queue[0]
            tasks_processed += 1
            
            if current_slice.is_expired(self.simulator.sim_time):
                expired_slice = self.task_queue.pop(0)
                expired_slice.status = TaskStatus.EXPIRED
                # ✅ 修复重复计数
                is_new_expire = expired_slice.task_id not in self.expired_task_ids
                if is_new_expire:
                    self.expired_tasks_count += 1
                    self.expired_task_ids.add(expired_slice.task_id)
                self.expired_tasks_buffer.append(expired_slice)
                
                # 删除超时任务的BSK事件
                if hasattr(expired_slice, 'origin_task'):
                    try:
                        self.remove_location_for_access_checking(expired_slice.origin_task)
                    except Exception as e:
                        self.logger.debug(f"Failed to remove access event: {e}")
                self._log_task_expired_once(
                    expired_slice.task_id, reason="compute_timeout"
                )
                continue  # 继续处理下一个任务
            
            # 计算处理速率 (bits/s) = f / c_t
            R_proc = self.current_cpu_freq / current_slice.workload

            # 实际可处理数据量 (受限于时间窗口和剩余数据)
            actual_processed_data = min(current_slice.data_size, R_proc * remaining_time)

            if actual_processed_data > 1e-6:
                # 计算实际用时
                time_taken = actual_processed_data / R_proc
                remaining_time -= time_taken  # ✅ 扣除已用时间
                
                # 计算能耗
                cycles_used = actual_processed_data * current_slice.workload
                energy_consumed = compute_energy_consumption(self.current_cpu_freq, cycles_used)
                
                # 更新切片状态
                current_slice.data_size -= actual_processed_data
                self.processed_data_total += actual_processed_data
                
                # 检查切片是否完成计算
                if current_slice.data_size < 1e-6:
                    completed_slice = self.task_queue.pop(0)
                    completed_slice.compute_end_time = self.simulator.sim_time
                    completed_slice.processing_cpu_freq = self.current_cpu_freq  # 🆕 记录处理时的 CPU 频率
                    completed_slice.status = TaskStatus.COMPUTED
                    
                    # 计算卫星路径总时延
                    t_sat_compute = completed_slice.compute_end_time - completed_slice.arrival_time
                    completed_slice.t_sat_path = (
                        completed_slice.t_uplink_total +
                        t_sat_compute +
                        completed_slice.t_isl_total
                    )
                    
                    # 放入 result_queue
                    self.result_queue.append(completed_slice)
                    self.logger.debug(
                        f"Task {completed_slice.task_id} computed, moved to result_queue"
                    )
                    # 继续处理下一个任务（如果还有时间）
                else:
                    current_slice.status = TaskStatus.COMPUTING
                    if current_slice.compute_start_time == 0.0:
                        current_slice.compute_start_time = self.simulator.sim_time
                    
                    self.logger.debug(
                        f"Task slice {current_slice.task_id} in progress: "
                        f"{current_slice.data_size/1e6:.2f} Mb remaining."
                    )
                    break  # 当前任务未完成，退出循环等待下一 step
            else:
                break  # 处理量不足，退出
            
        self.requires_retasking = True

    
    def execute_result_relay(self, duration: float) -> None:
        """处理结果队列（result_queue）中的任务传输和接力。
        
        职责分离设计：
        - execute_local_compute：只负责计算，完成后放入 result_queue
        - execute_result_relay：负责所有结果的回传/接力
        
        流程：
            result_queue 中的任务
                ↓
            检查 UD 可见性
                ├── 可见 → 直接下传完成 (COMPLETED)
                └── 不可见 → 找接力卫星
                        ├── 有接力卫星 → 放入 relay_sat.result_queue
                        └── 无接力卫星 → 留在队列等待
        
        Args:
            duration: [s] 本次步长的持续时间。
        """
        # ✅ 循环处理队列中所有结果（最多处理 10 个，防止卡死）
        max_process_per_step = 40
        processed_count = 0
        
        while self.result_queue and processed_count < max_process_per_step:
            result_slice = self.result_queue[0]
            processed_count += 1
            
            if result_slice.is_expired(self.simulator.sim_time):
                expired_result = self.result_queue.pop(0)
                expired_result.status = TaskStatus.EXPIRED
                # ✅ 修复重复计数
                is_new_expire = expired_result.task_id not in self.expired_task_ids
                if is_new_expire:
                    self.expired_tasks_count += 1
                    self.expired_task_ids.add(expired_result.task_id)
                # ✅ Bug修复：添加到 expired_tasks_buffer 用于协作失败率统计
                self.expired_tasks_buffer.append(expired_result)
                self._log_task_expired_once(
                    expired_result.task_id, reason="relay_timeout"
                )
                continue  # ✅ 继续处理下一个
            
            # 检查对 UD 的可见性
            has_visibility = self._check_ud_visibility(result_slice.origin_position)
        
            if has_visibility:
                # 可见 - 直接下行
                completed_result = self.result_queue.pop(0)
                downlink_distance = self._get_downlink_distance(completed_result, "UD")
                completed_result.t_prop_down = downlink_distance / TaskSlice.SPEED_OF_LIGHT
                
                # 计算总时延
                completed_result.t_sat_path += completed_result.t_prop_down
                completed_result.t_ud_path = completed_result.calculate_ud_path_delay()
                completed_result.t_cloud_path = completed_result.calculate_cloud_path_delay(
                    sgl_distance=self._get_downlink_distance(completed_result, "GS"),
                    tx_power=self.current_tx_power
                )
                
                total_delay = max(
                    completed_result.t_ud_path,
                    completed_result.t_sat_path,
                    completed_result.t_cloud_path
                )
                
                if total_delay <= completed_result.max_delay:
                    completed_result.status = TaskStatus.COMPLETED
                    
                    # ✅ 切片聚合：传入完成的绝对时间点
                    access_sat_name = getattr(completed_result, 'access_satellite', '')
                    access_sat = None
                    is_task_complete = False
                    if access_sat_name:
                        access_sat = self._find_satellite_by_name(access_sat_name)
                        if access_sat:
                            is_task_complete = access_sat.mark_slice_completed(
                                completed_result.task_id, 
                                completion_time=self.simulator.sim_time  # ✅ 传入完成的绝对时间
                            )
                    else:
                        # 无 access_satellite 的任务视为单切片任务
                        is_task_complete = True

                    # ✅ 任务级完成：三端全部完成时只打印一次
                    if (
                        is_task_complete
                        and access_sat
                        and completed_result.task_id not in access_sat.expired_task_ids
                        and completed_result.task_id in access_sat.slice_registry
                    ):
                        task_status = access_sat.slice_registry[completed_result.task_id]
                        if task_status.get("total", 0) > 0:
                            task_total_delay = access_sat.get_task_total_delay(completed_result.task_id)
                            if task_total_delay <= completed_result.max_delay:
                                self.logger.info(
                                    f"[TASK COMPLETED] Task {completed_result.task_id} all parts done. "
                                    f"T_total={task_total_delay:.4f}s"
                                )
                    
                    if is_task_complete:
                        # ✅ 修复重复计数
                        if completed_result.task_id not in self.completed_task_ids:
                            self.completed_tasks_count += 1
                            self.completed_task_ids.add(completed_result.task_id)

                        # ✅ 任务级完成才计入 buffer
                        self.completed_tasks_buffer.append(completed_result)
                        
                        # ✅ 更新累积协作指标（gym.py 直接读取，避免 buffer 时序问题）
                        self.total_hops_accumulated += completed_result.hop_count
                        self.total_latency_accumulated += total_delay
                        # 检查是否是异地完成（access_satellite != self）
                        access_sat = getattr(completed_result, 'access_satellite', '')
                        if access_sat and access_sat != self.name:
                            self.remote_completed_count += 1
                        elif completed_result.hop_count > 0:
                            self.remote_completed_count += 1
                        
                        self.logger.debug(
                            f"Result {completed_result.task_id} COMPLETED (relayed via {completed_result.hop_count} hops). "
                            f"T_total={total_delay:.4f}s"
                        )
                else:
                    completed_result.status = TaskStatus.EXPIRED
                    
                    # ✅ 切片聚合
                    access_sat_name = getattr(completed_result, 'access_satellite', '')
                    if access_sat_name:
                        access_sat = self._find_satellite_by_name(access_sat_name)
                        if access_sat:
                            access_sat.mark_slice_expired(completed_result.task_id)
                    
                    # ✅ 修复重复计数
                    is_new_expire = completed_result.task_id not in self.expired_task_ids
                    if is_new_expire:
                        self.expired_tasks_count += 1
                        self.expired_task_ids.add(completed_result.task_id)
                    # ✅ Bug修复：添加到 expired_tasks_buffer 用于协作失败率统计
                    self.expired_tasks_buffer.append(completed_result)
                    self._log_task_expired_once(
                        completed_result.task_id,
                        reason="deadline_miss",
                        total_delay=total_delay,
                    )
            else:
                # 不可见 - 继续接力
                relay_satellite = self._find_best_relay_satellite(result_slice)
                
                if relay_satellite:
                    relayed_result = self.result_queue.pop(0)
                    isl_distance = self._compute_isl_distance(relay_satellite)
                    
                    # 简化模型：结果回传只计算传播时延，忽略传输时延
                    isl_prop_delay = isl_distance / TaskSlice.SPEED_OF_LIGHT
                    
                    # 仅累加传播时延
                    relayed_result.t_prop_isl_total += isl_prop_delay
                    relayed_result.hop_count += 1
                    relayed_result.current_holder = relay_satellite.name
                    self.result_relay_task_ids.add(relayed_result.task_id)
                    
                    relay_satellite.result_queue.append(relayed_result)
                    
                    self.logger.info(
                        f"[Relay] Result {relayed_result.task_id} forwarded to {relay_satellite.name}, "
                        f"ISL prop delay={isl_prop_delay*1000:.2f}ms (tx ignored), "
                        f"hops={relayed_result.hop_count}"
                    )
                else:
                    # 无接力卫星 - 移到队尾，让后续任务有机会处理
                    # 超时由循环开头的 is_expired() 自动判断
                    self.logger.debug(
                        f"Result {result_slice.task_id} waiting for relay (no visible neighbor)"
                    )
                    self.result_queue.append(self.result_queue.pop(0))
    
    def _get_downlink_distance(self, task_slice: TaskSlice, target: str) -> float:
        """计算下行链路距离（用于传播时延计算）。

        Args:
            task_slice: 任务切片对象。
            target: 目标类型，"UD"（用户终端）或 "GS"（地面站）。
            
        Returns:
            下行链路距离 [m]。如果无法计算则返回默认 LEO 距离。
        """
        if target == "UD":
            # UD 位置计算直接下行距离
            if task_slice.origin_position is not None:
                r_sat = np.array(self.dynamics.r_BN_N)
                r_ud_N = task_slice.origin_position
                return np.linalg.norm(r_sat - r_ud_N)
            # UD 位置未知，使用默认斜距
            return 800e3
            
        elif target == "GS":
            # 尝试获取最近可见地面站的真实距离
            try:
                if hasattr(self.simulator, 'world') and hasattr(self.simulator.world, 'groundStations'):
                    r_sat = np.array(self.dynamics.r_BN_N)
                    min_distance = float('inf')
                    
                    for gs in self.simulator.world.groundStations:
                        # 获取地面站在惯性系中的位置
                        # groundLocation 输出 r_LP_P (Planet-fixed frame)
                        # 需要通过 PN 矩阵转换到惯性系
                        gs_state = gs.currentGroundStateOutMsg.read()
                        r_gs_N = np.array(gs_state.r_LN_N)  # 惯性系位置
                        
                        distance = np.linalg.norm(r_sat - r_gs_N)
                        if distance < min_distance:
                            min_distance = distance
                    
                    if min_distance < float('inf'):
                        return min_distance
            except Exception:
                pass
            
            # 无法获取真实距离，使用默认 LEO 斜距
            return 600e3
        
        # 未知目标类型，返回默认值
        return 800e3
    
    def _has_return_path(self, task_slice: TaskSlice) -> bool:
        """检查是否存在结果回传路径。
        
        简化模型：检查 UD 直接可见 或 地面站可见。
        
        Args:
            task_slice: 任务切片对象。
            
        Returns:
            True 如果存在回传路径。
        """
        # 策略 1: 地面站可见 (通过地面网络路由回 UD)
        if self._has_ground_station_access():
            return True
        
        # 策略 2: UD 直接可见
        if task_slice.origin_position is not None:
            return self._check_ud_visibility(task_slice.origin_position)
        
        # 默认：假设有回传路径（乐观模型）
        return True

    # --- 辅助方法: 可见性检查与距离计算 ---
    
    def _compute_isl_distance(self, target_satellite: "ComputationSatellite") -> float:
        """计算到目标卫星的 ISL 距离。
        
        复用 bsk_rl.obs.relative_observations.r_DC_N 计算相对位置，
        与 stin_relative_observations.get_isl_distance 保持一致。
        
        Args:
            target_satellite: 目标卫星对象。
            
        Returns:
            ISL 距离 [m]。
        """
        r_rel = r_DC_N(deputy=target_satellite, chief=self)
        return np.linalg.norm(r_rel)

    def _get_isl_max_distance_m(self) -> float:
        """获取 ISL 最大通信距离 [m]（优先使用 sat_args 配置）。"""
        from bsk_rl.utils.constants import ISL_MAX_DISTANCE
        try:
            if self.isl_max_distance_km is not None and float(self.isl_max_distance_km) > 0:
                return float(self.isl_max_distance_km) * 1e3
        except Exception:
            pass
        try:
            max_km = (getattr(self, "sat_args", {}) or {}).get(
                "isl_max_distance_km", None
            )
            if max_km is not None and float(max_km) > 0:
                return float(max_km) * 1e3
        except Exception:
            pass
        return ISL_MAX_DISTANCE
    
    def _has_ground_station_access(self) -> bool:
        """检查是否有地面站当前可访问。"""
        # 使用 AccessSatellite 的 opportunities 机制
        gs_opportunities = self.find_next_opportunities(
            n=1, types="ground_station", pad=False
        )
        if gs_opportunities:
            window = gs_opportunities[0].get("window", (float("inf"), float("inf")))
            # 检查当前时间是否在窗口内
            return window[0] <= self.simulator.sim_time <= window[1]
        return False
    
    def _check_ud_visibility(self, ud_position: np.ndarray) -> bool:
        """检查 UD 位置是否对卫星可见。
        
        Args:
            ud_position: UD 的 Planet-fixed 位置 [m]。
            
        Returns:
            True 如果 UD 可见。
        """
        # 简化实现：检查仰角是否满足最小要求
        # 更精确的实现需要使用 BSK 的 groundLocation 模块
        
        r_sat = np.array(self.dynamics.r_BN_N)
        
        # 将 UD 位置从 Planet-fixed 转换到惯性系 (简化：假设地球不转动)
        # 实际应该使用 world.PN 旋转矩阵
        r_ud_N = ud_position  # 简化
        
        # 计算卫星到 UD 的向量
        r_sat_to_ud = r_ud_N - r_sat
        
        # 计算仰角 (从 UD 看卫星)
        r_ud_norm = np.linalg.norm(r_ud_N)
        if r_ud_norm < 1e-6:
            return False
        
        cos_elev = np.dot(r_ud_N, r_sat_to_ud) / (r_ud_norm * np.linalg.norm(r_sat_to_ud))
        elev = np.arcsin(np.clip(cos_elev, -1, 1))
        
        min_elev = self.dynamics.task_minimum_elevation
        return elev >= min_elev
    
    def _find_best_relay_satellite(
        self, 
        task_slice: TaskSlice
    ) -> "ComputationSatellite":
        """查找最佳接力卫星（对 UD 可见且 ISL 可达）。
        
        选择策略（仅从 ISL 邻居中选择）：
        1. ISL 可达（已在邻居列表中）
        2. 对 UD 当前可见
        3. 优先选择距离 UD 最近的卫星
        
        Args:
            task_slice: 需要转发的任务切片。
        
        Returns:
            最佳接力卫星，如果没有则返回 None。
        """
        # 从动作实例获取邻居列表
        neighbors = []
        if hasattr(self, 'action_builder') and hasattr(self.action_builder, '_action'):
            action_instance = self.action_builder._action
            neighbors = getattr(action_instance, 'neighbor_satellites', [])
        
        if not neighbors:
            self.logger.debug("No ISL neighbors available for relay.")
            return None
        
        best_relay = None
        best_score = float('inf')
        
        for neighbor in neighbors:
            # 检查是否为 ComputationSatellite
            if not isinstance(neighbor, ComputationSatellite): 
                continue
            
            # 检查对 UD 的可见性
            if not neighbor._check_ud_visibility(task_slice.origin_position):
                continue
            
            # 计算 ISL 距离（邻居已在范围内，但用于评分）
            isl_distance = self._compute_isl_distance(neighbor)
            
            # 计算评分：距离 UD 的距离（越近越好）
            r_neighbor = np.array(neighbor.dynamics.r_BN_N)
            r_ud = task_slice.origin_position
            distance_to_ud = np.linalg.norm(r_neighbor - r_ud)
            
            # 综合评分：ISL 距离 + UD 距离
            score = 0.3 * isl_distance + 0.7 * distance_to_ud
            
            if score < best_score:
                best_score = score
                best_relay = neighbor
        
        return best_relay

    # --- ISL 接力回传（可选高级功能）---
    
    def get_isl_neighbors(self) -> list["ComputationSatellite"]:
        """获取当前可通过 ISL 连接的邻居卫星。
        
        使用 LOSCommDynModel 检查星间链路可见性。
        
        Returns:
            可连接的邻居卫星列表。
        """
        neighbors = []
        
        # 检查是否有 LOSCommDynModel
        if not hasattr(self.dynamics, 'transmitters'):
            return neighbors
        
        # 遍历所有其他卫星，检查 ISL 可见性
        for other_sat in self.simulator.satellites:
            if other_sat == self:
                continue
            # 距离阈值过滤（硬件最大距离）
            max_distance_m = self._get_isl_max_distance_m()
            isl_distance = self._compute_isl_distance(other_sat)
            if isl_distance > max_distance_m:
                continue
            
            # 检查是否有 transmitter 连接
            for transmitter in self.dynamics.transmitters:
                if transmitter.access_checker:
                    # 简化：假设有 access_checker 就表示可见
                    # 实际应该查询 transmitter.hasAccess
                    neighbors.append(other_sat)
                    break
        
        return neighbors

    @vizard.visualize
    def update_sprite_color(self, color_name: str, vizSupport=None, vizInstance=None):
        """Update satellite sprite color in Vizard to indicate status.
        
        Args:
            color_name: Color to set ('blue', 'red', 'yellow', 'green', etc.)
            vizSupport: Vizard support module (injected by decorator)
            vizInstance: Vizard instance (injected by decorator)
        """
        if vizInstance is None or vizSupport is None:
            return
        
        # Create new sprite with the desired color
        new_sprite = vizSupport.setSprite("SQUARE", color=color_name)
        
        # Update the spacecraft's sprite
        # Note: This requires finding the spacecraft in the vizInterface's spacecraft list
        # For simplicity, we set an attribute that could be read on next reset
        # In practice, dynamic sprite updates are limited in Vizard
        
        # Store for potential later use
        self.vizard_current_color = color_name
        self.vizard_color_change_time = time.time()
    
    @vizard.visualize
    def draw_isl_offload_line(
        self, target_sat_name: str, vizSupport=None, vizInstance=None
    ) -> None:
        """绘制星间卸载连线（红色）。
        
        当卫星将任务切片卸载到邻居卫星时调用此方法，在 Vizard 中绘制连线以可视化星间路由。
        使用缓存策略避免重复创建相同的线。
        
        Args:
            target_sat_name: 目标卫星名称
            vizSupport: Vizard support module (由装饰器注入)
            vizInstance: Vizard instance (由装饰器注入)
        """
        if vizInstance is None or vizSupport is None:
            return
        
        # 初始化线缓存
        if not hasattr(self, '_isl_lines'):
            self._isl_lines = {}
            
        link_key = (self.name, target_sat_name)
        
        try:
            if link_key not in self._isl_lines:
                # 首次创建：使用红色 RGBA [255, 50, 50, 255]
                vizSupport.createTargetLine(
                    vizInstance,
                    fromBodyName=self.name,
                    toBodyName=target_sat_name,
                    lineColor=[255, 50, 50, 255],
                )
                self._isl_lines[link_key] = vizSupport.targetLineList[-1]
                self.logger.info(f"[VIZ] Created ISL line: {self.name} -> {target_sat_name}")
            
            # 更新目标位置
            self._isl_lines[link_key].toBodyName = target_sat_name
            vizSupport.updateTargetLineList(vizInstance)
            
        except Exception as e:
            self.logger.debug(f"[VIZ] Failed to draw ISL line to {target_sat_name}: {e}")
    
    
    def _emit_viz_event(self, event_type: int, target_name: str = "", target_pos=None):
        """
        目的：这是核心可视化通信函数。它负责把 Python 端的决策（如"我要卸载给 sat-2"）打包成 C++ 消息发给 
        TaskVizController
        流程：
        检查 task_event_enabled（如果 training_mode=True 这里会直接返回）。
        打印调试日志（前5次）。
        填充 TaskEventMsgPayload C++ 结构体：
        eventType: 0=空闲, 1=计算, 2=ISL, 3=下传
        targetName: 目标卫星名
        调用 self.task_event_writer.write() 将消息发送到底层 C++ 模块。
        Emit task event for Vizard visualization.
        
        Args:
            event_type: 0=IDLE, 1=COMPUTING, 2=ISL_OFFLOAD, 3=DOWNLINK
            target_name: Target satellite/task name
            target_pos: Target position [m] (numpy array)
        """
        if not self.task_event_enabled:
            return
        
        # Debug: Print first few events
        if not hasattr(self, '_viz_event_count'):
            self._viz_event_count = 0
        self._viz_event_count += 1
        if self._viz_event_count <= 5:  # Only print first 5 events per satellite
            event_names = ["IDLE", "COMPUTING", "ISL_OFFLOAD", "DOWNLINK", "UPLINK"]
            event_name = event_names[event_type] if event_type < len(event_names) else f"UNKNOWN({event_type})"
            print(f"[VIZ_EVENT] {self.name}: {event_name} -> {target_name if target_name else 'N/A'}")
        
        
        # NOTE: ISL visualization is now handled by C++ TaskVizController
        # via vizLiveSettings.targetLineList. The C++ module adds/removes
        # PointLines dynamically based on event type (yellow=ISL, green=Downlink)

        
        # Fill message
        self.task_event_msg.satelliteName = self.name
        self.task_event_msg.eventType = event_type
        self.task_event_msg.eventTime = self.simulator.sim_time
        
        if target_name:
            self.task_event_msg.targetName = target_name
        else:
            self.task_event_msg.targetName = ""
        
        if target_pos is not None:
            self.task_event_msg.targetPosition_N = [float(target_pos[0]), float(target_pos[1]), float(target_pos[2])]
        else:
            self.task_event_msg.targetPosition_N = [0.0, 0.0, 0.0]
        
        # Write message
        self.task_event_writer.write(self.task_event_msg, self.simulator.sim_time_ns)

