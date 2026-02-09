"""
STINRelativeObservations: 针对 STIN 场景定制的相对观测。

获取当前卫星相对于其 N 个最近邻居的状态，以支持 MARL 决策。
支持灵活配置观测属性和归一化参数。

设计说明:
    - 继承自 Observation 基类（而非 RelativeProperties）
    - RelativeProperties 设计用于 1 对 1 (deputy vs chief) 场景
    - STINRelativeObservations 设计用于 1 对 N (self vs N neighbors) 场景
    - 复用 relative_observations 模块中的辅助函数 (如 r_DC_N, v_DC_N) 计算相对状态

使用示例::

    obs.STINRelativeObservations(
        dict(prop="r_DC_N", norm=1e7),           # 相对位置 [m]
        dict(prop="v_DC_N", norm=1e4),           # 相对速度 [m/s]
        dict(prop="battery_fraction"),            # 电池电量分数
        dict(prop="task_queue_size", norm=20.0),  # 任务队列负载
        max_neighbors=4,
    )
"""

import logging
from typing import TYPE_CHECKING, List, Dict, Any, Callable, Optional

import numpy as np
from gymnasium import spaces

from bsk_rl.obs import Observation
from bsk_rl.obs.relative_observations import r_DC_N, v_DC_N  # 复用框架提供的相对状态计算

if TYPE_CHECKING:
    from bsk_rl.sats.computation_satellite import ComputationSatellite
    from bsk_rl.act.stin_continuous_actions import STINContinuousAction

logger = logging.getLogger(__name__)


# 预定义的邻居属性获取函数
def get_battery_fraction(neighbor, self_sat) -> float:
    """获取邻居的电池电量分数（兼容 simpleBattery 和 powerMonitor）。"""
    try:
        if hasattr(neighbor.dynamics, 'simpleBattery'):
            battery = neighbor.dynamics.simpleBattery
            return battery.storageLevel / battery.storageCapacity
        elif hasattr(neighbor.dynamics, 'powerMonitor'):
            # LoSComputationDynaModel 使用 powerMonitor
            battery_msg = neighbor.dynamics.powerMonitor.batPowerOutMsg.read()
            capacity = getattr(neighbor.dynamics, 'battery_capacity', 2700000.0)
            return battery_msg.storageLevel / capacity
    except Exception:
        pass
    return 0.5  # 默认中等电量


def get_storage_fraction(neighbor, self_sat) -> float:
    """获取邻居的存储空间分数（兼容不同动力学模型）。"""
    try:
        if hasattr(neighbor.dynamics, 'storageUnit'):
            storage = neighbor.dynamics.storageUnit
            return storage.storageLevel / storage.storageCapacity
    except Exception:
        pass
    return 0.5  # 默认中等存储


def get_task_queue_size(neighbor, self_sat) -> float:
    """获取邻居的任务队列长度。"""
    return float(len(getattr(neighbor, "task_queue", [])))


def get_queue_workload(neighbor, self_sat) -> float:
    """获取邻居的队列总工作量 [cycles]。"""
    task_queue = getattr(neighbor, "task_queue", [])
    # TaskSlice 使用 data_size * workload 计算总工作量
    return sum(
        getattr(t, "data_size", 0.0) * getattr(t, "workload", 0.0) 
        for t in task_queue
    )


def get_cpu_freq(neighbor, self_sat) -> float:
    """获取邻居的当前 CPU 频率 [Hz]。"""
    return getattr(neighbor, "current_cpu_freq", 0.0)


def get_isl_distance(neighbor, self_sat) -> float:
    """获取到邻居的 ISL 距离 [m]。"""
    r_rel = r_DC_N(deputy=neighbor, chief=self_sat)
    return np.linalg.norm(r_rel)


def get_current_task_priority(neighbor, self_sat) -> float:
    """获取邻居当前任务的优先级 [0, 1]。
    
    ✅ 已启用: TaskSlice 现已包含 priority 属性。
    可用于邻居观测空间：
        dict(prop="get_current_task_priority", norm=1.0)
    
    🔮 Future Work 扩展方向：
        1. 将邻居优先级加入 STINRelativeObservations（增加 4D）
        2. 添加动作空间维度："优先处理高优先级任务"开关 (11D 动作)
        3. Agent 可根据自身和邻居的任务优先级分布，决定卸载策略
           （高优先级任务不卸载，低优先级任务卸载给有空闲的邻居）
    
    Returns:
        当前队列首任务的优先级，无任务时返回 0.0。
    """
    task_queue = getattr(neighbor, "task_queue", [])
    if task_queue:
        # 尝试从 TaskSlice 获取优先级（如果继承自 ComputationTask）
        return getattr(task_queue[0], "priority", 0.5)
    return 0.0


# ============================================================
# 🆕 观测空间增强：任务队列统计 + 邻居信道质量
# ============================================================

def get_min_remaining_deadline(sat, self_sat) -> float:
    """获取任务队列中最紧急任务的剩余时延 [s]。
    
    用于观测空间，让 Agent 知道"接下来有没有急件"。
    
    Args:
        sat: 目标卫星（可以是自身或邻居）
        self_sat: 当前卫星（用于获取仿真时间）
        
    Returns:
        最小剩余时延 [s]，无任务时返回大值 (1000.0)。
    """
    task_queue = getattr(sat, "task_queue", [])
    if not task_queue:
        return 1000.0  # 无任务，返回大值表示"不紧急"
    
    # 获取当前仿真时间
    sim_time = 0.0
    if hasattr(sat, "simulator") and sat.simulator is not None:
        sim_time = sat.simulator.sim_time
    
    min_remaining = 1000.0
    for task in task_queue:
        # TaskSlice 有 deadline 属性（绝对时间）
        deadline = getattr(task, "deadline", None)
        if deadline is not None:
            remaining = max(0.0, deadline - sim_time)
            min_remaining = min(min_remaining, remaining)
    
    return min_remaining


def get_avg_task_priority(sat, self_sat) -> float:
    """获取任务队列的平均优先级 [0, 1]。
    
    用于观测空间，让 Agent 知道"手头任务整体优先级"。
    
    Args:
        sat: 目标卫星（可以是自身或邻居）
        self_sat: 当前卫星（未使用，保持接口一致）
        
    Returns:
        平均优先级，无任务时返回 0.0。
    """
    task_queue = getattr(sat, "task_queue", [])
    if not task_queue:
        return 0.0
    
    priorities = [getattr(t, "priority", 0.5) for t in task_queue]
    return sum(priorities) / len(priorities)


def get_isl_channel_quality(neighbor, self_sat) -> float:
    """获取到邻居的 ISL 信道质量评分 [0, 1]。
    
    基于 constants.calculate_isl_rate() 的 ISL 信道质量评分：
        R_I = B_I × log₂(1 + SNR)
        SNR = (p_I × g_I^tr × g_I^rc) / (L_I × N₀ × B_I)
    
    质量越高，传输速度越快、丢包率越低。
    
    Args:
        neighbor: 邻居卫星
        self_sat: 当前卫星
        
    Returns:
        信道质量 [0, 1]，1.0 表示最佳，0.0 表示边缘可见。
    """
    from bsk_rl.utils.constants import (
        calculate_isl_rate,
        ISL_MAX_DISTANCE,
        ISL_BANDWIDTH,  # 用于计算理论最大速率
    )
    
    distance = get_isl_distance(neighbor, self_sat)
    
    # 支持从卫星属性/ sat_args 覆盖最大 ISL 距离（km -> m）
    max_distance_m = ISL_MAX_DISTANCE
    try:
        max_km = getattr(self_sat, "isl_max_distance_km", None)
        if max_km is not None and max_km > 0:
            max_distance_m = float(max_km) * 1e3
    except Exception:
        pass
    if max_distance_m == ISL_MAX_DISTANCE:
        try:
            sat_args = getattr(self_sat, "sat_args", {}) or {}
            max_km = sat_args.get("isl_max_distance_km", None)
            if max_km is not None and max_km > 0:
                max_distance_m = float(max_km) * 1e3
        except Exception:
            pass
    # 超出最大距离时返回 0
    if distance > max_distance_m:
        return 0.0
    
    # 获取当前发射功率（优先从卫星属性获取，否则从 sat_args 获取默认值）
    tx_power = getattr(self_sat, 'current_tx_power', 0.0)
    if tx_power <= 0:
        # 回退到 sat_args 中的默认值（与 computation_satellite.py L781 一致）
        default_power = abs(getattr(self_sat, 'sat_args', {}).get('transmitterPowerDraw', -15.0))
        tx_power = default_power
    
    # 使用物理模型计算 ISL 速率 [bps]
    isl_rate = calculate_isl_rate(tx_power, distance, max_distance=max_distance_m)
    
    # 归一化到 [0, 1]
    # 理论最大速率 ≈ bandwidth × log2(1 + 高SNR) ≈ 1 Gbps
    # 使用 500 Mbps 作为"优秀"基准（实际工程上的高速链路）
    max_expected_rate = 5 * ISL_BANDWIDTH  # 500 Mbps = 5 × 100e6
    quality = min(isl_rate / max_expected_rate, 1.0)
    
    return quality



class STINRelativeObservations(Observation):
    """
    STIN 相对观测类：为每颗卫星提供其 N 个最近邻居的可配置状态信息。

    该类直接继承自 Observation 基类，用于获取多个邻居的状态。
    支持灵活配置观测属性、归一化参数和邻居数量。
    """

    def __init__(
        self, 
        *neighbor_properties: dict[str, Any],
        max_neighbors: int = 4,
        name: str = "neighbor_obs"
    ) -> None:
        """初始化相对观测。

        Args:
            neighbor_properties: 邻居属性规范。每个属性是一个字典，包含：
                * ``prop``: 属性名称（框架函数名或自定义函数）
                * ``norm`` `optional`: 归一化因子。默认 1.0。
                * ``name`` `optional`: 观测元素名称。默认为 ``prop`` 的值。
                * ``fn`` `optional`: 自定义函数 fn(neighbor, self_sat) -> value。
            max_neighbors: 最大邻居数量。
            name: 观测名称。

        示例::

            STINRelativeObservations(
                dict(prop="r_DC_N", norm=1e7),           # 相对位置
                dict(prop="v_DC_N", norm=1e4),           # 相对速度
                dict(prop="battery_fraction"),            # 电池
                dict(prop="task_queue_size", norm=20.0),  # 队列
                max_neighbors=4,
            )
        """
        super().__init__(name=name)
        
        self.max_neighbors = max_neighbors
        self._action_instance: Optional["STINContinuousAction"] = None
        
        # 处理属性规范（与 RelativeProperties 风格一致）
        self.neighbor_properties: List[Dict[str, Any]] = []
        for prop_spec in neighbor_properties:
            processed_spec = self._process_property_spec(prop_spec)
            self.neighbor_properties.append(processed_spec)
        
        # 计算观测维度
        self.num_features = sum(
            np.prod(prop["shape"]) for prop in self.neighbor_properties
        )
        self.num_total_obs = self.num_features * self.max_neighbors
        self.dtype = np.float64

    def _process_property_spec(self, prop_spec: dict[str, Any]) -> dict[str, Any]:
        """处理属性规范，与 RelativeProperties 风格一致。
        
        Args:
            prop_spec: 原始属性规范字典。
            
        Returns:
            处理后的属性规范，包含 fn, norm, name, shape。
        """
        # 验证键
        for key in prop_spec:
            if key not in ["prop", "norm", "name", "fn", "shape"]:
                raise ValueError(f"Invalid property key: {key}")
        
        processed = {}
        
        # 1. 处理函数
        if "fn" in prop_spec:
            processed["fn"] = prop_spec["fn"]
        else:
            # 尝试从全局命名空间或预定义函数中获取
            prop_name = prop_spec.get("prop")
            if not prop_name:
                raise ValueError("Must provide either 'fn' or 'prop'")
            
            # 首先尝试从 relative_observations 模块获取（如 r_DC_N, v_DC_N）
            import bsk_rl.obs.relative_observations as rel_obs_module
            if hasattr(rel_obs_module, prop_name):
                processed["fn"] = getattr(rel_obs_module, prop_name)
            # 然后尝试本模块的预定义函数（如 get_battery_fraction）
            elif prop_name in globals():
                processed["fn"] = globals()[prop_name]
            else:
                raise ValueError(
                    f"Property prop='{prop_name}' not found. "
                    "Provide a custom 'fn' or use predefined properties."
                )
        
        # 2. 处理归一化
        processed["norm"] = prop_spec.get("norm", 1.0)
        
        # 3. 处理名称
        if "name" in prop_spec:
            processed["name"] = prop_spec["name"]
        else:
            processed["name"] = prop_spec.get("prop", "custom")
            if processed["norm"] != 1.0:
                processed["name"] += "_normd"
        
        # 4. 推断形状（调用一次函数来获取输出形状）
        if "shape" in prop_spec:
            processed["shape"] = prop_spec["shape"]
        else:
            # 默认假设标量输出
            processed["shape"] = (1,)
        
        return processed

    @property
    def observation_space(self) -> spaces.Box:
        """定义观测空间：N_neighbors * N_features 的一维向量。"""
        dtype = self.dtype
        low = np.asarray(-np.inf, dtype=dtype) if dtype is not None else -np.inf
        high = np.asarray(np.inf, dtype=dtype) if dtype is not None else np.inf
        return spaces.Box(
            low=low,
            high=high,
            shape=(self.num_total_obs,),
            dtype=dtype,
        )

    def get_obs(self) -> np.ndarray:
        """
        构造卫星的相对观测向量。

        Returns:
            np.ndarray: 形状为 (num_total_obs,) 的观测向量
        """
        # 1. 获取邻居列表 (从动作实例中获取)
        if self._action_instance is None:
            # 第一次调用时绑定动作实例
            self._action_instance = self.satellite.action_builder._action

        neighbors = getattr(self._action_instance, "neighbor_satellites", [])

        obs_list: List[float] = []

        # 2. 遍历邻居，构造相对状态
        for i in range(self.max_neighbors):
            if i < len(neighbors):
                neighbor = neighbors[i]
                
                # 获取该邻居的所有属性
                for prop_spec in self.neighbor_properties:
                    fn = prop_spec["fn"]
                    norm = prop_spec["norm"]
                    
                    # 调用函数获取值
                    value = fn(neighbor, self.satellite)
                    
                    # 转换为 numpy 数组并归一化
                    if isinstance(value, (list, tuple)):
                        value = np.array(value)
                    elif not isinstance(value, np.ndarray):
                        value = np.array([value])
                    
                    value_normalized = value / norm
                    
                    # ✅ NaN/Inf 保护：防止异常值传播到神经网络
                    value_normalized = np.nan_to_num(
                        value_normalized, 
                        nan=0.0, 
                        posinf=1.0, 
                        neginf=-1.0
                    )
                    obs_list.extend(value_normalized.flatten())
                    
            else:
                # 填充零：如果邻居数量不足
                obs_list.extend([0.0] * self.num_features)

        return np.array(obs_list, dtype=self.dtype)


    @property
    def observation_description(self) -> Dict[str, Any]:
        """人类可读的观测空间描述。"""
        desc = {}
        for i in range(self.max_neighbors):
            for prop_spec in self.neighbor_properties:
                key = f"Neighbor{i+1}_{prop_spec['name']}"
                desc[key] = f"{prop_spec['name']} (norm={prop_spec['norm']})"
        return desc
    
__doc_title__ = "STIN Relative Observations"
__all__ = [
    "STINRelativeObservations",
    # 辅助函数（可在外部使用）
    "get_battery_fraction",
    "get_storage_fraction", 
    "get_task_queue_size",
    "get_queue_workload",
    "get_cpu_freq",
    "get_isl_distance",
    "get_current_task_priority",
    # 🆕 观测空间增强
    "get_min_remaining_deadline",   # 最紧急任务剩余时延
    "get_avg_task_priority",        # 平均任务优先级
    "get_isl_channel_quality",      # ISL 信道质量评分
]
