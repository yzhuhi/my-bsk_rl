"""STIN Computation Satellite Agent.

This module defines the ComputationSatellite class for satellite-terrestrial 
integrated network (STIN) multi-agent reinforcement learning environments.
"""

import logging
from typing import TYPE_CHECKING, Any, List

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
        
        - T_SAT: 卫星边缘处理路径总时延
          T_SAT = T_tx_up(α_sat) + T_prop_up + T_queue + T_compute + T_isl + T_prop_down  没问题！
          
          其中 T_tx_up(α_sat) = (α_sat × data_size) / R_uplink(fixed)
          
          T_isl = T_tx_isl + T_prop_isl:
            - 协作切分：T_tx_isl (传输数据到邻居) + T_prop_isl (传播)
            - 接力中继：仅 T_prop_isl (已处理完的部分数据:量小，传输时延可忽略) 
        
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
    DEFAULT_ISL_RATE = DEFAULT_ISL_RATE # 用不上了，动态计算
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
        uplink_rate = uplink_rate or self.DEFAULT_UPLINK_RATE       # 固定（符合论文）
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
        
        # 路径时延跟踪（并行处理模型）
        self.alpha_local = 0.0         # UD 本地处理比例
        self.alpha_cloud = 0.0         # 云端处理比例
        self.alpha_sat = 0.0           # 卫星处理比例
        self.t_ud_path = 0.0           # [s] UD 路径总时延
        self.t_sat_path = 0.0          # [s] 卫星路径总时延
        self.t_cloud_path = 0.0        # [s] 云端路径总时延
        
        # 协作奖励追踪
        self.origin_satellite: str = ""  # 发起卸载决策的卫星名称（用于协作奖励）
        self.offload_chain: list = []    # 卸载链路：记录所有参与卸载的卫星（用于延迟奖励分配）
    
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
    
    # 这个函数有问题（不太准确，其实也难以估算）
    def estimate_sat_delay(self, current_time: float, downlink_distance: float = 0.0) -> float:
        """估算星间部分的任务完成时的总时延 [s]。
        
        Args:
            current_time: 当前仿真时间 [s]。怎么能表示后面就卫星就不需要处理了呢？
            downlink_distance: 下行链路距离 [m]（用于估算下行传播时延）。
            
        Returns:
            估算的总时延 [s] = T_elapsed + T_prop_down_est。
        """
        t_prop_down_est = downlink_distance / self.SPEED_OF_LIGHT
        return self.get_elapsed_time(current_time) + t_prop_down_est
    
    # [已删除] elapsed_time 和 remaining_time properties
    # @property
    # def elapsed_time(self) -> float:
    #     """已用时间 [s]（从创建到现在）。"""
    #     # 需要外部传入当前时间来计算
    #     return 0.0  # Placeholder, 由外部计算
    
    # @property
    # def remaining_time(self) -> float:
    #     """剩余时间 [s]（到达 max_delay 前的可用时间）。"""
    #     return max(0.0, self.max_delay - self.elapsed_time)
    # 原因：TaskSlice 对象不持有仿真时间引用，必须使用 get_elapsed_time(current_time) 方法
    
    def is_expired(self, current_time: float) -> bool:
        """检查任务是否已超时。"""
        return (current_time - self.creation_time) > self.max_delay
    
    @property
    def t_isl_total(self) -> float:
        """ISL 总时延 [s] = 传输 + 传播。
        
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
    
    def calculate_sat_path_delay(self, t_queue: float = 0.0, t_compute: float = 0.0) -> float:
        """计算卫星边缘处理路径时延。
        
        T_SAT = T_tx_up(α_sat) + T_prop_up + T_queue + T_compute + T_isl + T_prop_down
        
        其中：
        - T_tx_up(α_sat): 按 α_sat 比例计算的上行传输时延
        - T_isl = T_tx_isl + T_prop_isl:
          - 协作切分：T_tx_isl (传输) + T_prop_isl (传播)
          - 接力：仅 T_prop_isl (传播)，因为结果数据量很小
        
        Args:
            t_queue: 排队时延 [s]。
            t_compute: 计算时延 [s]。
        
        Returns:
            卫星路径时延 [s]。
        """
        if self.alpha_sat < 1e-6:
            return 0.0
        
        # 按 alpha_sat 比例计算上行传输时延
        data_sat = self.original_data_size * self.alpha_sat
        t_tx_up_sat = data_sat / self.uplink_rate
        
        t_sat = (
            t_tx_up_sat +              # 上行传输（按 α_sat 比例）
            self.t_prop_up +           # 上行传播
            t_queue +                  # 排队
            t_compute +                # 计算
            self.t_isl_total +         # ISL：传输(协作切分) + 传播(协作切分/接力)
            self.t_prop_down           # 下行传播
        )
        
        return t_sat
    
    def calculate_cloud_path_delay(self, sgl_distance: float = 600e3, tx_power: float = 15.0) -> float:
        """计算云端处理路径时延。
        
        T_CLOUD = T_tx_up(α_cloud) + T_tx_sgl(α_cloud) + 2*(T_prop_up + T_prop_sgl + T_fiber) + T_cloud_compute
        
        其中：
        - T_tx_up(α_cloud): 按 α_cloud 比例计算的上行传输时延（UD→卫星）
        - T_tx_sgl(α_cloud): 按 α_cloud 比例计算的 SGL 传输时延（卫星→网关）
        - 传播时延 ×2: 上行+下行（任务数据上传 + 结果返回）
        - T_cloud_compute: 云端计算时延
        
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
    
    def get_total_delay_parallel(self, current_time: float = None) -> float:
        """计算并行路径模型的总时延。
        
        T_total = max(T_UD, T_SAT, T_CLOUD)
        
        Args:
            current_time: 当前仿真时间 [s]，用于估算未完成路径。
        
        Returns:
            并行处理的总时延 [s]。 √
        """
        # 如果有记录的路径时延，使用记录值
        if self.t_ud_path > 0 or self.t_sat_path > 0 or self.t_cloud_path > 0:
            return max(self.t_ud_path, self.t_sat_path, self.t_cloud_path)
        
        # 否则估算当前时延（用于未完成任务）
        elapsed = current_time - self.creation_time if current_time else 0.0
        return elapsed
    
    def convert_to_result(self, ratio: float = 0.05) -> None:
        """将计算任务转换为结果数据包。
        
        Args:
            ratio: 结果数据量与原始输入数据量的比例 (例如 5%)
        """
        # 重置数据量为结果大小
        self.data_size = self.original_data_size * self.alpha_sat * ratio
        # 清除计算负载 (结果不需要再计算，只需要传输)
        self.workload = 0.0 
        self.status = TaskStatus.COMPUTED

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
    """STIN 计算节点卫星智能体。
    
    扩展自 AccessSatellite，支持连续资源分配和任务切片的 MARL 智能体。
    
    核心功能:
        1. 连续资源分配 (CPU 频率、发射功率)
        2. 分层协作调度 (本地/云端/卫星间)
        3. 任务切片处理与队列管理
    
    时隙操作流程（论文 Section II-B）:
        每个时隙 τ (对应max_step_duration) 内，网络控制和数据传输按以下步骤执行：
        
        Step 1: 信息收集 (gather_network_information)
            - 服务卫星通过 GSL 收集网络信息和任务请求
            - 包括：信道状态信息(CSI)、UD位置、任务属性、邻居状态等 
            - 信息收集完成后，服务卫星开始执行任务卸载和资源分配决策
            - 我们的模型不考虑这部分时延，我们认为这部分信息以及后续的指令下发的数据 计算得到的传输和传播时延都比较小
        
        Step 2: 决策 (make_offloading_decision)
            - 基于收集的信息，服务卫星做出任务卸载和资源分配决策
            - 由 RL 智能体生成动作 → 解析为切分比例和资源分配
        
        Step 3: 控制消息分发 (distribute_control_messages)
            - 控制消息下发给 UE 和云计算中心
            - 包括：本地处理比例、云端处理比例、资源分配等
            - 这部分时延不考虑
        
        Step 4: 任务执行 (execute_scheduled_tasks)
            - 任务按控制消息在 UD、卫星、云端并行执行
            - 开始执行任务后，任务的时延开始计算
        
        Step 5: 切换处理 (handle_handover)
            - 服务卫星切换时，状态通过 ISL 同步到下一个服务卫星
            - 接力模式；
    
    参数配置:
        所有 Satelite Mobility Edge Computing (SMEC) 参数 (CPU频率、工作负载等) 通过 ComputationDynModel (仿真器动力学)
        的 @default_args 装饰器定义，无需在此类重写 default_sat_args。
        
    Example:
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
    action_spec: List["Action"] = [act.STINContinuousAction()]  # 连续动作空间（必须是实例）10D
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
            dict(prop="current_cpu_freq", fn=lambda sat: sat.current_cpu_freq / 1e9), # CPU频率
            dict(prop="task_queue_size", fn=lambda sat: len(sat.task_queue)), # 任务队列大小
            dict(prop="queue_workload", fn=lambda sat: sum(t.remaining_workload for t in sat.task_queue) / 1e9), # 任务队列负载
            name="stin_state"
        ),
        # 当前任务属性 (论文 Step 1: 任务请求信息)
        obs.SatProperties(
            dict(prop="current_task_data_size", fn=lambda sat: sat.task_queue[0].data_size / 1e6 if sat.task_queue else 0.0),  # [Mb]
            dict(prop="current_task_workload", fn=lambda sat: sat.task_queue[0].workload / 1e3 if sat.task_queue else 0.0),     # [kcycles/bit]
            dict(prop="current_task_max_delay", fn=lambda sat: sat.task_queue[0].max_delay if sat.task_queue else 0.0),        # [s]
            dict(prop="current_task_remaining_time", fn=lambda sat: sat.task_queue[0].get_remaining_time(sat.simulator.sim_time) if sat.task_queue else 0.0),  # [s]
            # dict(prop="current_task_uplink_delay", fn=lambda sat: sat.task_queue[0].t_uplink_total * 1000 if sat.task_queue else 0.0),  # [ms]
            dict(prop="current_task_priority", fn=lambda sat: getattr(sat.task_queue[0], 'priority', 0.5) if sat.task_queue else 0.0),  # [0, 1] 任务优先级
            name="task_request"
        ),
        # 邻居卫星状态（灵活配置）
        obs.STINRelativeObservations(
            dict(prop="get_isl_distance", norm=1e7),           # ISL 距离 [m]，归一化到 10,000 km
            dict(prop="get_battery_fraction"),                 # 电池电量分数 [0, 1]
            dict(prop="get_task_queue_size", norm=20.0),       # 任务队列长度
            dict(prop="get_queue_workload", norm=1e10),        # 队列工作量 [cycles]
            dict(prop="get_cpu_freq", norm=2e9),               # CPU 频率 [Hz]
            max_neighbors=4,  # 最多观测 4 个邻居
        ),
        # 时间
        obs.Time(),
        obs.Eclipse(norm=5700.0),
    ]
    
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
        # ✅ 任务级别统计（使用集合追踪唯一任务ID，避免slice重复计数）
        self.completed_task_ids: set = set()      # 已完成的原始任务ID集合
        self.expired_task_ids: set = set()        # 已过期的原始任务ID集合
        self.offloaded_task_ids: set = set()      # 已卸载的任务ID集合（首跳卸载）
        self.relay_task_ids: set = set()          # 被中继的任务ID集合（二跳及以上）
        
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
        # 重置队列
        self.task_queue = []
        self.result_queue = []
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
        # 重置时隙控制状态
        self.current_control_message = {}
        self.network_info = {}

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
            cpu_ratio: 分配的 CPU 频率比例 ∈ [0, 1]，映射到 [f_min, f_max]。
            tx_power_ratio: 分配的发射功率比例 ∈ [0, 1]。
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
        
        Args:
            next_serving_satellite: 接手服务的下一个卫星。
        
        Returns:
            True 如果切换成功。
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
    
    # def _get_battery_soc(self) -> float:
    #     """获取电池 SOC（State of Charge）。
        
    #     Returns:
    #         电池 SOC ∈ [0, 1]，如果无法获取返回 1.0（满电）。
    #     """
    #     try:
    #         if hasattr(self.dynamics, 'powerMonitor'):
    #             battery_msg = self.dynamics.powerMonitor.batPowerOutMsg.read()
    #             battery_capacity = self.dynamics.powerMonitor.storageCapacity
    #             if battery_capacity > 0:
    #                 return battery_msg.storageLevel / battery_capacity
    #     except Exception as e:
    #         self.logger.debug(f"Cannot read battery SOC: {e}")
        
    #     return 1.0  # 默认满电
    
    # def _apply_battery_hard_constraint(
    #     self, 
    #     cpu_ratio: float, 
    #     tx_power_ratio: float
    # ) -> tuple:
    #     """[DEPRECATED] 硬约束：线性插值电池安全裁剪。
        
    #     .. deprecated::
    #         硬约束已通过 STINActionShield 在动作空间层实现。
    #         此方法保留仅为向后兼容，不再被 set_resource_allocation 调用。
        
    #     设计理念：
    #     - 10% 以下强制关机（limit=0）
    #     - 10%-30% 线性插值降速
    #     - 30% 以上全速运行（limit=1）
        
    #     比阶跃函数更平滑，避免动作空间突变。
        
    #     Args:
    #         cpu_ratio: 原始 CPU 分配比例
    #         tx_power_ratio: 原始传输功率比例
            
    #     Returns:
    #         裁剪后的 (cpu_ratio, tx_power_ratio)
    #     """
    #     battery_soc = self._get_battery_soc()
        
    #     # 定义缓降区间
    #     safe_min = 0.10  # 10% 电量以下强制关机
    #     safe_max = 0.30  # 30% 电量以上全速运行
        
    #     # 线性插值计算物理上限
    #     clip_limit = np.clip(
    #         (battery_soc - safe_min) / (safe_max - safe_min), 
    #         0.0, 1.0
    #     )
        
    #     # 记录原始动作
    #     original_cpu = cpu_ratio
        
    #     # 应用裁剪
    #     final_cpu = min(cpu_ratio, clip_limit)
    #     final_tx = min(tx_power_ratio, clip_limit)
        
    #     # 仅在大幅度限制时打印日志，避免刷屏
    #     if original_cpu - final_cpu > 0.1:
    #         self.logger.debug(
    #             f"🔋 Low Battery ({battery_soc:.1%}): "
    #             f"Throttling {original_cpu:.2f} → {final_cpu:.2f}"
    #         )
            
    #     return final_cpu, final_tx

    # --- 协作调度执行接口 (分层协作: UD/云端/卫星间) ---

    def schedule_collaboration_action(
        self, 
        alpha_local: float, 
        alpha_cloud: float,
        split_ratios: np.ndarray, 
        neighbor_satellites: List["ComputationSatellite"]
    ) -> None:
        """解析切分比例，将任务切片分配给各节点。
        
        实现三层分层协作模型:
            1. 一级切分: α_local (UD本地) + α_cloud (云端) + α_sat (卫星)
            2. 二级切分: 卫星间协作比例 x_v (自身 + 邻居)

            不需要在 schedule 函数里强制检查“能不能完成”。让 Agent 去试错。如果它分给了一个无法回传的卫星，导致任务超时，它下次就不敢这么分了。这是 RL 的核心。
            您的思路非常清晰：接力的“计算任务转发”是不存在的（那是 Action 决定的 Offloading），只有协作的时候切分才是我们的目的。如果，
            任务没有计算完就被迫接力给下一颗卫星这说明当时的这个决策本身就并不是最好的。 我们尽量减少这种。
            也就是说，接力只针对“结果数据转发”（这是 FSW 决定的 Relaying）。也就是当下这颗星完成了任务但是没有来的及传回去的这种接力模式，
            将会减少传输时延，而只有传播时延，这对我们算法去实现低时延的目标更贴近。
        
        Args:
            alpha_local: UD 本地处理比例。
            alpha_cloud: 云端处理比例。
            split_ratios: 卫星间切分比例数组 [x_self, x_neighbor1, ...]。
            neighbors: 邻居卫星列表。

        """
        # 假设接入卫星只处理队列中的第一个任务（最新到达的任务）
        if not self.task_queue:
            self.logger.debug("No active task to schedule. Idle.")
            return
            
        current_raw_task = self.task_queue[0] # 这是一个包含原始 d_t, c_t 的任务结构
        
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

        self.logger.warning(
            f"[COLLAB] α_local={alpha_local_final:.2%}, α_cloud={alpha_cloud_final:.2%}, "
            f"α_sat={alpha_sat_final:.2%} | split_ratios={[f'{r:.2f}' for r in split_ratios]}"
        )
        
        # --------------------------------------------------------
        # Step 2: 规范化二级切分 (x_v) - 确保总和为 alpha_sat
        # --------------------------------------------------------
        
        # 强制将 split_ratios (x_v) 的和归一化到 1.0 (内部相对切分)
        total_xv_ratio = np.sum(split_ratios)
        if total_xv_ratio > 1e-6:
             normalized_xv_ratios = split_ratios / total_xv_ratio
        else:
             normalized_xv_ratios = np.zeros_like(split_ratios)
             normalized_xv_ratios[0] = 1.0 # 默认自己处理
             
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
            
            if ud_process_time <= remaining_time:
                # UD 能完成 → 算作成功（但奖励归 UD，卫星不拿）
                # 简化处理：直接标记这部分任务完成
                self.logger.debug(
                    f"Task {current_raw_task.task_id}: UD Local can complete "
                    f"{data_local/1e6:.2f} Mb in {ud_process_time:.1f}s (remaining: {remaining_time:.1f}s)"
                )
                # 注意：UD 完成的任务不算卫星的 completed_tasks，也不给卫星奖励
                # 这鼓励卫星协作处理而不是推给 UD
            else:
                # UD 完不成 → 超时，惩罚落在接入卫星头上
                self.logger.warning(
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
                self.expired_tasks_count += 1
                self.expired_task_ids.add(expired_ud_slice.task_id)  # ✅ 追踪唯一任务ID
                self.expired_tasks_buffer.append(expired_ud_slice)
        
        # 云端处理 (Alpha_Cloud)
        if alpha_cloud_final > 1e-6:
            data_cloud = raw_data_size * alpha_cloud_final
            # TODO: 实际的 SGL 传输指令，这里只是逻辑记录
            self.logger.info(f"Task {current_raw_task.task_id}: Routed {data_cloud/1e6:.2f} Mb to Cloud via SGL.")
        
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
            
            for i, target_node in enumerate(all_sat_nodes):
                relative_ratio = adjusted_ratios[i]
                
                if relative_ratio > 1e-6:
                    # 计算实际数据量: Alpha_sat_final * 相对比例
                    slice_data = raw_data_size * alpha_sat_final * relative_ratio 
                    
                    # 计算 ISL 时延 (如果卸载到其他卫星)
                    # 协作切分时：需要传输数据，有传输时延 + 传播时延
                    isl_tx_delay = 0.0
                    isl_prop_delay = 0.0
                    if target_node != self:
                        isl_distance = self._compute_isl_distance(target_node)
                        # 动态 ISL 速率：基于发射功率和距离 (论文公式 6)
                        isl_rate = calculate_isl_rate(self.current_tx_power, isl_distance)
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
                    
                    # 记录发起卸载的卫星（用于协作奖励）
                    new_slice.origin_satellite = self.name
                    
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
                        # 目标拒绝，切片回到本地队列
                        self.task_queue.append(new_slice)
                        continue
                    
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
                        # TODO: Proper ISL visualization needs Vizard transceiver beams

                    else:
                        self.logger.info(f"Self-allocated {slice_data/1e6:.2f} Mb for local execution.")
                        
        # 4. 任务完成调度
        self.task_queue.pop(0) # 原始任务已被完全切分，从接入卫星的任务队列中移除
        self.requires_retasking = True
    
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
        
        # 加入任务队列
        self.task_queue.append(initial_slice)
        
        # 记录从 UD 接收的原始任务数据量
        self.raw_data_received += initial_slice.data_size
        
        self.logger.info(
            f"Received NEW task {initial_slice.task_id}: "
            f"{initial_slice.data_size/1e6:.2f} Mb, "
            f"max_delay={initial_slice.max_delay:.2f}s, "
            f"uplink_delay={initial_slice.t_uplink_total*1000:.2f}ms "
            f"(tx={initial_slice.t_tx_up*1000:.2f}ms + prop={initial_slice.t_prop_up*1000:.2f}ms). "
            f"Queue size: {len(self.task_queue)}"
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
        current_queue_data = sum(t.data_size for t in self.task_queue)
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
        
        # 追加到卸载链路（用于延迟奖励分配）
        if hasattr(task_slice, 'offload_chain'):
            task_slice.offload_chain.append(self.name)
        
        # 加入本地任务队列
        self.task_queue.append(task_slice)
        
        self.logger.info(
            f"Received slice {task_slice.task_id}: {task_slice.data_size/1e6:.2f} Mb, "
            f"workload={task_slice.workload}, max_delay={task_slice.max_delay}s. "
            f"Queue: {len(self.task_queue)} tasks, {(current_queue_data + task_slice.data_size)/1e9:.2f} Gb"
        )
        
        # 标记需要重新调度
        self.requires_retasking = True
        return True
    
    def execute_local_compute(self, duration: float) -> None:
        """执行本地计算任务切片（在每个 step 中调用）。
        
        简化模型：计算完成 + 回传路径存在 = 任务完成
        传播时延在任务完成时计算并记录到 TaskSlice 中。
        
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
            
        current_slice = self.task_queue[0]
        
        # --- [新增] 检查任务类型：是需要计算，还是仅仅需要当前这颗卫星作为中继节点转发？ ---
        if current_slice.status in [TaskStatus.RELAYED, TaskStatus.COMPUTED]:
            # 如果是接力（默认接力的是已经处理好的任务，这也迫使卫星能够做出更加智能的决策）过来的任务，或者已计算完等待回传的任务
            # 应该调用下行/转发逻辑，而不是计算逻辑
            self._handle_result_transmission(current_slice, duration)
            return
        # --------------------------------------------------------
        
        # 检查任务是否已超时
        if current_slice.is_expired(self.simulator.sim_time):
            expired_slice = self.task_queue.pop(0)
            expired_slice.status = TaskStatus.EXPIRED
            self.expired_tasks_count += 1
            self.expired_task_ids.add(expired_slice.task_id)  # ✅ 追踪唯一任务ID
            self.expired_tasks_buffer.append(expired_slice)  # 添加到超时缓冲区
            
            # ✅ 关键修复:删除超时任务的BSK事件,防止event map膨胀
            if hasattr(expired_slice, 'origin_task'):
                try:
                    self.remove_location_for_access_checking(expired_slice.origin_task)
                except Exception as e:
                    self.logger.debug(f"Failed to remove access event for expired task {expired_slice.task_id}: {e}")
            if not self.training_mode:
                self.logger.warning(
                    f"Task slice {expired_slice.task_id} EXPIRED during compute. "
                    f"Elapsed: {expired_slice.get_elapsed_time(self.simulator.sim_time):.2f}s > "
                    f"max_delay: {expired_slice.max_delay:.2f}s"
                )
            self.requires_retasking = True
            return
        
        # 检查 CPU 是否有效分配 (使用 dynamics 层参数)
        if self.current_cpu_freq < self.dynamics.cpu_min_frequency:
            if not self.training_mode:
                self.logger.warning(
                    f"[DIAG] RETURN: CPU freq too low! current={self.current_cpu_freq:.2e}, "
                    f"min={self.dynamics.cpu_min_frequency:.2e}"
                )
            return
        
        # 计算处理速率 (bits/s) = f / c_t
        R_proc = self.current_cpu_freq / current_slice.workload

        # 实际可处理数据量 (受限于时间窗口和剩余数据)
        actual_processed_data = min(current_slice.data_size, R_proc * duration)

        if actual_processed_data > 1e-6:
            # 计算实际用时
            time_taken = actual_processed_data / R_proc
            
            # 计算能耗
            cycles_used = actual_processed_data * current_slice.workload
            energy_consumed = compute_energy_consumption(self.current_cpu_freq, cycles_used)
            
            # TODO: 实际扣除数据和能量（需要与 dyn.ComputationDynModel 对接）
            # if hasattr(self.dynamics, 'storageUnit'):
            #     self.dynamics.storageUnit.storageLevel -= actual_processed_data
            # if hasattr(self.dynamics, 'battery'):
            #     self.dynamics.battery.storageLevel -= energy_consumed
            
            # 更新切片状态
            current_slice.data_size -= actual_processed_data
            self.processed_data_total += actual_processed_data
            
            # 检查切片是否完成计算
            if current_slice.data_size < 1e-6:
                completed_slice = self.task_queue.pop(0)
                completed_slice.compute_end_time = self.simulator.sim_time
                completed_slice.status = TaskStatus.COMPUTED
                
                # 计算卫星路径总时延（使用正确的 ISL 时延）
                t_sat_compute = completed_slice.compute_end_time - completed_slice.arrival_time
                completed_slice.t_sat_path = (
                    completed_slice.t_uplink_total +      # 上行时延: 传输 + 传播
                    t_sat_compute +                       # 计算时延
                    completed_slice.t_isl_total           # ISL 时延: 传输(协作) + 传播(协作/接力) 做出决策的时候就已经计算好了
                )
                
                # ========================================================
                # 可见性检查与 ISL 接力逻辑
                # ========================================================
                
                # 检查当前卫星对 UD 的可见性
                has_direct_visibility = self._check_ud_visibility(completed_slice.origin_position)
                
                if has_direct_visibility:
                    # 情况 1: 直接可见 - 直接回传
                    downlink_distance = self._get_downlink_distance(completed_slice, "UD")
                    completed_slice.t_prop_down = downlink_distance / TaskSlice.SPEED_OF_LIGHT
                    completed_slice.t_sat_path += completed_slice.t_prop_down
                    
                    # 计算并行路径时延 T_total = max(T_UD, T_SAT, T_CLOUD)
                    # 使用正确的路径时延计算方法
                    completed_slice.t_ud_path = completed_slice.calculate_ud_path_delay()
                    # 注意：SGL 距离（到地面网关）与下行距离（到 UD）不同
                    completed_slice.t_cloud_path = completed_slice.calculate_cloud_path_delay(
                        sgl_distance=self._get_downlink_distance(completed_slice, "GS"),
                        tx_power=self.current_tx_power
                    )
                    
                    total_delay = max(
                        completed_slice.t_ud_path,
                        completed_slice.t_sat_path,
                        completed_slice.t_cloud_path
                    )
                    
                    # 检查是否满足时延约束
                    if total_delay <= completed_slice.max_delay:
                        completed_slice.status = TaskStatus.COMPLETED
                        completed_slice.compute_end_time = self.simulator.sim_time  # 记录完成时间
                        self.completed_tasks_count += 1
                        self.completed_task_ids.add(completed_slice.task_id)  # ✅ 追踪唯一任务ID
                        self.completed_tasks_buffer.append(completed_slice)  # 添加到历史缓冲区
                        
                        self.logger.info(
                            f"[OK] Task {completed_slice.task_id} COMPLETED and results downlinked directly to UD. "
                            f"Total delay: {total_delay:.2f}s"
                        )
                        
                        # Emit downlink visualization event
                        self._emit_viz_event(3, "UD", completed_slice.origin_position)
                        self.logger.info(
                            f"Task slice {completed_slice.task_id} COMPLETED (direct). "
                            f"T_total={total_delay:.4f}s = max("
                            f"T_UD={completed_slice.t_ud_path:.4f}, "
                            f"T_SAT={completed_slice.t_sat_path:.4f}, "
                            f"T_CLOUD={completed_slice.t_cloud_path:.4f}). "
                            f"Energy: {energy_consumed/1000:.2f}kJ"
                        )
                        # TODO: Proper downlink visualization needs Vizard transceiver beams

                    else:
                        # 超时失败
                        completed_slice.status = TaskStatus.EXPIRED
                        self.expired_tasks_count += 1
                        self.expired_task_ids.add(completed_slice.task_id)  # ✅ 追踪唯一任务ID
                        self.expired_tasks_buffer.append(completed_slice)  # 添加到超时缓冲区
                        self.logger.warning(
                            f"Task slice {completed_slice.task_id} EXPIRED: "
                            f"T_total={total_delay:.4f}s > max_delay={completed_slice.max_delay:.4f}s"
                        )
                
                else:
                    # 情况 2: 不可见 - 需要 ISL 接力 （应该从邻居卫星中一步一步找）
                    relay_satellite = self._find_best_relay_satellite(completed_slice)
                    
                    if relay_satellite:
                        # 转发结果到接力卫星的 result_queue
                        isl_distance = self._compute_isl_distance(relay_satellite)
                        
                        # 动态模型：结果回传同样使用动态 ISL 速率 (论文公式 6)
                        # 结果数据量 = 原始数据量 × 结果比例
                        result_data_size = completed_slice.original_data_size * DEFAULT_RESULT_RATIO * completed_slice.alpha_sat
                        isl_rate = calculate_isl_rate(self.current_tx_power, isl_distance)
                        isl_tx_delay = result_data_size / isl_rate
                        isl_prop_delay = isl_distance / TaskSlice.SPEED_OF_LIGHT
                        
                        # 累加传输 + 传播时延
                        completed_slice.t_tx_isl_total += isl_tx_delay
                        completed_slice.t_prop_isl_total += isl_prop_delay
                        
                        completed_slice.hop_count += 1
                        completed_slice.status = TaskStatus.RELAYED
                        completed_slice.current_holder = relay_satellite.name
                        
                        # 放入 result_queue（不是 task_queue）
                        relay_satellite.result_queue.append(completed_slice)
                        
                        self.logger.info(
                            f"[Relay] Result {completed_slice.task_id} relayed to {relay_satellite.name}. "
                            f"ISL tx={isl_tx_delay*1000:.2f}ms + prop={isl_prop_delay*1000:.2f}ms, "
                            f"rate={isl_rate/1e6:.1f}Mbps, hops={completed_slice.hop_count}"
                        )
                    else:
                        # 无法找到接力卫星 - 任务失败
                        completed_slice.status = TaskStatus.EXPIRED
                        self.expired_tasks_count += 1
                        self.expired_task_ids.add(completed_slice.task_id)  # ✅ 追踪唯一任务ID
                        self.expired_tasks_buffer.append(completed_slice)  # 添加到超时缓冲区
                        self.logger.warning(
                            f"Task slice {completed_slice.task_id} FAILED: "
                            f"No visibility and no relay satellite available."
                        )
            else:
                current_slice.status = TaskStatus.COMPUTING
                if current_slice.compute_start_time == 0.0:
                    current_slice.compute_start_time = self.simulator.sim_time
                    
                self.logger.debug(
                    f"Task slice {current_slice.task_id} in progress: "
                    f"{actual_processed_data/1e6:.2f} Mb processed, "
                    f"{current_slice.data_size/1e6:.2f} Mb remaining."
                )
            
            self.requires_retasking = True
    
    def _handle_result_transmission(self, task_slice: TaskSlice, duration: float) -> None:
        """处理已完成计算或接力过来的任务的下行/转发。
        
        当 task_queue 中的任务状态为 RELAYED 或 COMPUTED 时调用此方法，
        将任务从 task_queue 移到 result_queue 并处理下行传输。
        下面的 result_queue 逻辑中包含了两种,可见不可见，不可见就继续接力；
        所以这个逻辑链条还是比较清晰：任务处理完但是无法下传->task_queue 判断是否可见，可见直接处理->
        不可见，传给result_queue再去判断是否可见，可见直接处理->不可见，传给接力卫星；

        Args:
            task_slice: 需要处理的任务切片。
            duration: [s] 本次步长的持续时间。
        """
        # 从 task_queue 移除并放入 result_queue
        if task_slice in self.task_queue:
            self.task_queue.remove(task_slice)
        
        # 检查对 UD 的可见性
        has_visibility = self._check_ud_visibility(task_slice.origin_position)
        
        if has_visibility:
            # 可见 - 直接完成下行回传
            downlink_distance = self._get_downlink_distance(task_slice, "UD")
            task_slice.t_prop_down = downlink_distance / TaskSlice.SPEED_OF_LIGHT
            task_slice.t_sat_path += task_slice.t_prop_down
            
            # 计算并行路径时延
            task_slice.t_ud_path = task_slice.calculate_ud_path_delay()
            # 注意：SGL 距离（到地面网关）与下行距离（到 UD）不同
            task_slice.t_cloud_path = task_slice.calculate_cloud_path_delay(
                sgl_distance=self._get_downlink_distance(task_slice, "GS"),
                tx_power=self.current_tx_power
            )
            
            total_delay = max(
                task_slice.t_ud_path,
                task_slice.t_sat_path,
                task_slice.t_cloud_path
            )
            
            if total_delay <= task_slice.max_delay:
                task_slice.status = TaskStatus.COMPLETED
                self.completed_tasks_count += 1
                self.completed_task_ids.add(task_slice.task_id)  # ✅ 追踪唯一任务ID
                self.completed_tasks_buffer.append(task_slice)
                
                # ✅ 删除已完成任务的BSK事件
                if hasattr(task_slice, 'origin_task'):
                    try:
                        self.remove_location_for_access_checking(task_slice.origin_task)
                    except Exception as e:
                        self.logger.debug(f"Failed to remove access event for completed task {task_slice.task_id}: {e}")
                
                self.logger.info(
                    f"Task {task_slice.task_id} COMPLETED (direct transmission). "
                    f"T_total={total_delay:.4f}s"
                )
            else:
                task_slice.status = TaskStatus.EXPIRED
                self.expired_tasks_count += 1
                self.expired_task_ids.add(task_slice.task_id)  # ✅ 追踪唯一任务ID
                self.expired_tasks_buffer.append(task_slice)
                
                # ✅ 删除已超时任务的BSK事件
                if hasattr(task_slice, 'origin_task'):
                    try:
                        self.remove_location_for_access_checking(task_slice.origin_task)
                    except Exception as e:
                        self.logger.debug(f"Failed to remove access event for expired task {task_slice.task_id}: {e}")
                
                self.logger.warning(
                    f"Task {task_slice.task_id} EXPIRED: T_total={total_delay:.4f}s"
                )
        else:
            # 不可见 - 放入 result_queue 等待接力
            self.result_queue.append(task_slice)
            self.logger.info(
                f"Task {task_slice.task_id} moved to result_queue for relay. "
                f"Status={task_slice.status}"
            )
        
        self.requires_retasking = True
    
    def execute_result_relay(self, duration: float) -> None:
        """处理结果接力队列（result_queue）。
        
        与 execute_local_compute 不同，这里只处理结果的转发/下行传输，
        不进行计算。
        
        Args:
            duration: [s] 本次步长的持续时间。
        """
        if not self.result_queue:
            return
        
        # 处理队列中的第一个结果
        result_slice = self.result_queue[0]
        
        # 检查是否超时
        if result_slice.is_expired(self.simulator.sim_time):
            expired_result = self.result_queue.pop(0)
            expired_result.status = TaskStatus.EXPIRED
            self.expired_tasks_count += 1
            self.expired_task_ids.add(expired_result.task_id)  # ✅ 追踪唯一任务ID
            self.logger.warning(
                f"Result slice {expired_result.task_id} EXPIRED during relay. "
                f"Elapsed: {expired_result.get_elapsed_time(self.simulator.sim_time):.2f}s"
            )
            return
        
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
            # 注意：SGL 距离（到地面网关）与下行距离（到 UD）不同
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
                self.completed_tasks_count += 1
                self.completed_task_ids.add(completed_result.task_id)  # ✅ 追踪唯一任务ID
                self.logger.info(
                    f"Result {completed_result.task_id} COMPLETED (relayed via {completed_result.hop_count} hops). "
                    f"T_total={total_delay:.4f}s"
                )
            else:
                completed_result.status = TaskStatus.EXPIRED
                self.expired_tasks_count += 1
                self.expired_task_ids.add(completed_result.task_id)  # ✅ 追踪唯一任务ID
                self.logger.warning(
                    f"Result {completed_result.task_id} EXPIRED: T_total={total_delay:.4f}s"
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
                
                relay_satellite.result_queue.append(relayed_result)
                
                self.logger.info(
                    f"[Relay] Result {relayed_result.task_id} forwarded to {relay_satellite.name}, "
                    f"ISL prop delay={isl_prop_delay*1000:.2f}ms (tx ignored), "
                    f"hops={relayed_result.hop_count}"
                )
            else:
                # 无接力卫星
                failed_result = self.result_queue.pop(0)
                failed_result.status = TaskStatus.EXPIRED
                self.expired_tasks_count += 1
                self.expired_task_ids.add(failed_result.task_id)  # ✅ 追踪唯一任务ID
                self.logger.warning(
                    f"Result {failed_result.task_id} FAILED: No relay satellite available."
                )
    
    def _get_downlink_distance(self, task_slice: TaskSlice, target: str) -> float:
        """计算下行链路距离（用于传播时延计算）。
        既然我们有 
        GroundStationWorldModel
        提供的真实地面站位置，我们完全可以计算动态的卫星到地面站距离，而不是使用固定的 600km。
        
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

