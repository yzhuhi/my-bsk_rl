"""
STINHierarchicalObservations: 针对分层动作空间的统一扁平观测。

设计理念：
    - L1 和 L2 使用**同一套观测**，观测本身不区分层级
    - 分头的区别在于**网络端**：L2 Selector 额外接收 L1 输出作为条件
    - 观测结构与 STINRelativeObservations 保持一致的扁平设计
    - 支持可配置属性规范，与旧版 STINRelativeObservations 风格对齐
    
观测内容：
    - 自身状态：队列、电池、CPU、TX 功率、任务统计（可配置）
    - 邻居特征：队列、电池、ISL 距离/质量、CPU 频率（可配置）
    - 邻居可达性掩码

使用示例::

    # 使用默认配置（向后兼容）
    obs.STINHierarchicalObservations()
    
    # 使用可配置风格
    obs.STINHierarchicalObservations(
        self_properties=[
            dict(prop="task_queue_size", norm=50.0),
            dict(prop="battery_fraction"),
        ],
        neighbor_properties=[
            dict(prop="task_queue_size", norm=50.0),
            dict(prop="isl_distance", norm=1e7),
        ],
        max_neighbors=4,
    )

注意：本文件为 Hier 方案的独立实现，不覆盖现有观测类。
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np
from gymnasium import spaces

from bsk_rl.obs.relative_observations import r_DC_N  # 用于精确计算 ISL 距离
from Basilisk.utilities.orbitalMotion import REQ_EARTH

from bsk_rl.obs import Observation

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats.hier_computation_satellite import HierComputationSatellite
    from bsk_rl.act.stin_hier_hybrid_actions import STINHierarchicalHybridAction

logger = logging.getLogger(__name__)


# ============== 从旧版模块复用辅助函数 ==============
# 统一签名为 get_xxx(sat, self_sat) 双参数形式
from bsk_rl.obs.stin_relative_observations import (
    get_battery_fraction,
    get_task_queue_size,
    get_queue_workload,
    get_isl_distance,
    get_isl_channel_quality,
    get_current_task_priority,
)

# ============== 新版独有的辅助函数 ==============
# 以下函数为分层 RL 专用，在旧版中不存在或实现不同
def get_raw_queue_size(sat: "HierComputationSatellite", self_sat: "HierComputationSatellite" = None) -> float:
    """获取原始任务队列长度。
    
    Args:
        sat: 目标卫星
        self_sat: 当前卫星（未使用，保持接口一致）
        
    Returns:
        原始队列长度（原始值），归一化由 norm 参数处理
    """
    return float(len(getattr(sat, 'raw_task_queue', [])))


def get_avg_task_priority(sat: "HierComputationSatellite", self_sat: "HierComputationSatellite" = None) -> float:
    """获取任务队列平均优先级。
    
    Args:
        sat: 目标卫星
        self_sat: 当前卫星（未使用，保持接口一致）
        
    Returns:
        平均优先级 [0, 1]，无任务时返回 0.5
    """
    if hasattr(sat, 'task_queue') and sat.task_queue:
        priorities = [getattr(t, 'priority', 0.5) for t in sat.task_queue]
        return float(np.nanmean(priorities))
    return 0.5


def get_min_remaining_deadline(sat: "HierComputationSatellite", self_sat: "HierComputationSatellite" = None) -> float:
    """获取最紧急任务的剩余时间 [s]。
    
    Args:
        sat: 目标卫星
        self_sat: 当前卫星（未使用，保持接口一致）
        
    Returns:
        最小剩余时延 [s]，无任务时返回大值 (1000.0)，归一化由 norm 参数处理
    """
    task_queue = getattr(sat, "task_queue", [])
    if not task_queue:
        return 1000.0  # 无任务时返回大值表示"不紧急"
    
    # 获取当前仿真时间
    sim_time = 0.0
    if hasattr(sat, "simulator") and sat.simulator is not None:
        sim_time = sat.simulator.sim_time
    
    min_remaining = 1000.0
    for task in task_queue:
        remaining = task.get_remaining_time(sim_time) if hasattr(task, 'get_remaining_time') else 1000.0
        min_remaining = min(min_remaining, remaining)
    
    return min_remaining


def get_cpu_freq_ratio(sat: "HierComputationSatellite", self_sat: "HierComputationSatellite" = None) -> float:
    """获取当前 CPU 频率比例（归一化到 [0, 1]）。
    
    Args:
        sat: 目标卫星
        self_sat: 当前卫星（未使用，保持接口一致）
        
    Returns:
        CPU 频率比例 [0, 1]
    """
    if hasattr(sat, 'current_cpu_freq') and hasattr(sat, 'dynamics'):
        max_freq = getattr(sat.dynamics, 'cpu_max_frequency', 2e9)
        return sat.current_cpu_freq / max_freq if max_freq > 0 else 0.0
    return 0.0


def get_tx_power_ratio(sat: "HierComputationSatellite", self_sat: "HierComputationSatellite" = None) -> float:
    """获取当前发射功率比例。
    
    Args:
        sat: 目标卫星
        self_sat: 当前卫星（未使用，保持接口一致）
        
    Returns:
        发射功率比例 [0, 1]
    """
    if hasattr(sat, 'current_tx_power') and hasattr(sat, 'dynamics'):
        max_power = getattr(sat.dynamics, 'transmitter_power_draw', 30.0)
        return sat.current_tx_power / max_power if max_power > 0 else 0.0
    return 0.0


def get_total_queue_data(sat: "HierComputationSatellite", self_sat: "HierComputationSatellite" = None) -> float:
    """获取队列总数据量 [bits]。
    
    Args:
        sat: 目标卫星
        self_sat: 当前卫星（未使用，保持接口一致）
        
    Returns:
        队列总数据量 [bits]，归一化由 norm 参数处理
    """
    task_queue = getattr(sat, 'task_queue', [])
    if task_queue:
        return sum(t.data_size for t in task_queue)
    return 0.0


def get_downstream_queue(sat: "HierComputationSatellite", self_sat: "HierComputationSatellite" = None) -> float:
    """获取下游拥塞信号：邻居的邻居的平均队列长度。
    
    用于 Backpressure Routing: 不仅考虑直接邻居的队列，还考虑其下游的拥塞情况。
    如果路由到某邻居，该邻居的邻居队列都很满，说明下游通道拥堵。
    
    计算方式：
        downstream_queue = mean(neighbor_j.queue_size for j in sat.neighbors if j != self_sat)
    
    Args:
        sat: 目标邻居卫星（我们要评估将任务路由到此卫星后的下游情况）
        self_sat: 当前卫星（排除自己，避免循环计算）
        
    Returns:
        下游平均队列长度 [任务数]，归一化由 norm 参数处理
        若无有效下游邻居，返回 0.0
    """
    # 获取 sat 的邻居（即 self_sat 的二跳邻居）
    downstream_neighbors = []
    
    # 尝试从缓存的快照获取
    cached = getattr(sat, "_neighbor_snapshot", None)
    if cached:
        downstream_neighbors = list(cached)
    elif hasattr(sat, "get_isl_neighbors"):
        downstream_neighbors = sat.get_isl_neighbors()
    
    if not downstream_neighbors:
        return 0.0
    
    queue_sizes = []
    for neighbor in downstream_neighbors:
        if neighbor is None:
            continue
        # 排除 self_sat（避免将自己计入下游）
        if self_sat is not None and neighbor is self_sat:
            continue
        # 获取该下游邻居的队列长度
        q_size = float(len(getattr(neighbor, 'task_queue', [])))
        queue_sizes.append(q_size)
    
    if not queue_sizes:
        return 0.0
    
    return float(np.mean(queue_sizes))


# ============== 统一扁平观测类 ==============

class STINHierarchicalObservations(Observation):
    """
    统一扁平观测：L1 和 L2 共享相同的观测输入。
    
    设计说明：
        - 观测本身不区分层级，网络端负责分头处理
        - L2 Selector 会额外接收 L1 输出作为条件输入（在模型端实现）
        - 结构与 STINRelativeObservations 保持一致
        - 支持可配置属性规范，与旧版风格对齐
    
    观测维度（默认配置）：
        - 邻居特征 (N x F): 每个邻居的状态（当前 F=8）
        - 邻居掩码 (N): 可达性标记
    """

    def __init__(
        self, 
        *neighbor_properties: dict[str, Any],
        max_neighbors: int = 4,
        name: str = "hier_neighbor_obs"
    ) -> None:
        """初始化分层观测。

        Args:
            neighbor_properties: 邻居属性规范列表。
            max_neighbors: 最大邻居数量。
            name: 观测名称。

        示例::

            # 使用默认配置
            STINHierarchicalObservations()
            
            # 使用可配置风格 (*args 传入属性)
            STINHierarchicalObservations(
                dict(prop="get_task_queue_size", norm=50.0),
                dict(prop="get_isl_distance", norm=1e7),
                max_neighbors=4,
            )
        """
        super().__init__(name=name)
        self.max_neighbors = max_neighbors
        self.dtype = np.float64
        self._action_instance: Optional["STINHierarchicalHybridAction"] = None
        
        # 处理邻居属性规范
        self.neighbor_properties: List[Dict[str, Any]] = []
        for prop_spec in neighbor_properties:
            processed = self._process_property_spec(prop_spec)
            self.neighbor_properties.append(processed)
        
        # 特征名称（用于描述输出）
        self.neighbor_feature_names = [p["name"] for p in self.neighbor_properties]
        
        # 计算维度（统一结构，支持向量特征）
        self.neighbor_features = int(sum(
            np.prod(p.get("shape", (1,))) for p in self.neighbor_properties
        ))  # 每个邻居的特征数
        self.neighbor_dim = max_neighbors * self.neighbor_features  # 邻居特征
        self.mask_dim = max_neighbors  # 邻居可达性掩码
        self.total_dim = self.neighbor_dim + self.mask_dim

    def _process_property_spec(self, prop_spec: Dict[str, Any]) -> Dict[str, Any]:
        """处理属性规范，与 STINRelativeObservations 风格一致。
        
        Args:
            prop_spec: 原始属性规范字典。
            
        Returns:
            处理后的属性规范，包含 fn, norm, name。
        """
        # 验证键
        for key in prop_spec:
            if key not in ["prop", "norm", "name", "fn", "shape"]:
                raise ValueError(f"Invalid property key: {key}")
        
        processed: Dict[str, Any] = {}
        
        # 1. 处理函数
        if "fn" in prop_spec:
            processed["fn"] = prop_spec["fn"]
        else:
            # 尝试从本模块的预定义函数中获取
            prop_name = prop_spec.get("prop")
            if not prop_name:
                raise ValueError("Must provide either 'fn' or 'prop'")
            
            # 首先尝试从 relative_observations 模块获取（如 r_DC_N, v_DC_N）
            import bsk_rl.obs.relative_observations as rel_obs_module
            if hasattr(rel_obs_module, prop_name):
                processed["fn"] = getattr(rel_obs_module, prop_name)
            # 然后从模块全局命名空间获取函数
            elif prop_name in globals():
                processed["fn"] = globals()[prop_name]
            else:
                raise ValueError(
                    f"Property prop='{prop_name}' not found. "
                    "Provide a custom 'fn' or use predefined properties like: "
                    "get_task_queue_size, get_battery_fraction, get_isl_distance, etc."
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
    def observation_space(self) -> spaces.Space:
        """观测空间定义（统一扁平向量，使用 inf 作为范围）。"""
        dtype = self.dtype
        low = np.asarray(-np.inf, dtype=dtype) if dtype is not None else -np.inf
        high = np.asarray(np.inf, dtype=dtype) if dtype is not None else np.inf
        return spaces.Box(
            low=low,
            high=high,
            shape=(self.total_dim,),
            dtype=dtype,
        )

    def get_obs(self) -> np.ndarray:
        """获取观测向量（仅包含邻居状态和 mask）。"""
        sat = self.satellite
        
        # 获取邻居列表（优先使用环境刷新后的快照）
        neighbors: List[Any] = []
        cached_neighbors = getattr(sat, "_neighbor_snapshot", None)
        if cached_neighbors:
            neighbors = list(cached_neighbors)
        else:
            if hasattr(sat, "action_builder") and hasattr(sat.action_builder, "_action"):
                action_instance = sat.action_builder._action
                if action_instance is not self._action_instance:
                    self._action_instance = action_instance
            if self._action_instance is not None:
                neighbors = getattr(self._action_instance, "neighbor_satellites", [])
        if (not neighbors) or all(n is None for n in neighbors):
            if hasattr(sat, "get_isl_neighbors"):
                try:
                    neighbors = sat.get_isl_neighbors() or []
                except Exception:
                    neighbors = []
        
        # ============ 邻居特征 (N x neighbor_features) ============
        all_neighbor_features = []
        neighbor_mask = np.zeros(self.max_neighbors, dtype=self.dtype)
        
        for i in range(self.max_neighbors):
            if i < len(neighbors) and neighbors[i] is not None:
                n = neighbors[i]
                current_neighbor_feats = []
                
                for prop_spec in self.neighbor_properties:
                    fn = prop_spec["fn"]
                    norm = prop_spec["norm"]
                    # 调用函数
                    value = fn(n, sat)
                    
                    # 处理标量或向量
                    if isinstance(value, (list, tuple)):
                        value = np.array(value)
                    elif not isinstance(value, np.ndarray):
                        value = np.array([value])
                        
                    # 归一化并展平
                    feat = (value / norm).flatten()
                    current_neighbor_feats.extend(feat)
                
                all_neighbor_features.extend(current_neighbor_feats)
                neighbor_mask[i] = 1.0  # 可达
            else:
                # 填充零：如果邻居不可达或不存在
                all_neighbor_features.extend([0.0] * self.neighbor_features)
                neighbor_mask[i] = 0.0  # 不可达

        mask_snapshot = getattr(sat, "_neighbor_mask_snapshot", None)
        if mask_snapshot is not None and len(mask_snapshot) == self.max_neighbors:
            neighbor_mask = np.asarray(mask_snapshot, dtype=self.dtype)

        if not hasattr(self, "_mask_diag_remaining"):
            self._mask_diag_remaining = 300
        action_neighbors = None
        if self._action_instance is not None:
            action_neighbors = getattr(self._action_instance, "neighbor_satellites", None)
        action_count = len(action_neighbors) if action_neighbors is not None else -1
        snapshot_neighbors = getattr(sat, "_neighbor_snapshot", None)
        snapshot_count = len(snapshot_neighbors) if snapshot_neighbors is not None else -1
        neighbor_count = int(sum(1 for n in neighbors if n is not None))
        mask_sum = float(np.sum(neighbor_mask))
        snapshot_mismatch = snapshot_count >= 0 and snapshot_count != neighbor_count
        action_mismatch = action_count >= 0 and action_count != neighbor_count
        is_zero = neighbor_count == 0 or mask_sum <= 0.0
        should_log = (snapshot_mismatch or action_mismatch or is_zero)
        if should_log and self._mask_diag_remaining > 0:
            self._mask_diag_remaining -= 1
            logger.warning(
                "[MASK DIAG] sat=%s neighbors=%d action_neighbors=%d snapshot_neighbors=%d mask_sum=%.0f",
                getattr(sat, "name", "unknown"),
                neighbor_count,
                action_count,
                snapshot_count,
                mask_sum,
            )

        # 扁平化输出：[neighbors_flat, mask]
        obs = np.concatenate([
            np.array(all_neighbor_features, dtype=self.dtype),
            neighbor_mask,
        ])

        # ✅ NaN/Inf 保护
        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    @property
    def observation_description(self) -> Dict[str, Any]:
        """人类可读的观测空间描述。"""
        desc: Dict[str, Any] = {}
        # 移除了 Self 描述
        for n_idx in range(self.max_neighbors):
            for f_idx, name in enumerate(self.neighbor_feature_names, start=1):
                desc[f"Neighbor{n_idx+1}_{f_idx}_{name}"] = name
            desc[f"Neighbor{n_idx+1}_mask"] = "reachable_mask"

        return desc

__doc_title__ = "STIN Hierarchical Observations"
__all__ = [
    "STINHierarchicalObservations",
    # 从 stin_relative_observations 复用的函数（再导出）
    "get_battery_fraction",
    "get_task_queue_size",
    "get_queue_workload",
    "get_isl_distance",
    "get_isl_channel_quality",
    "get_current_task_priority",
    # 本模块独有的辅助函数
    "get_raw_queue_size",
    "get_avg_task_priority",
    "get_min_remaining_deadline",
    "get_cpu_freq_ratio",
    "get_tx_power_ratio",
    "get_total_queue_data",
    "get_downstream_queue",  # 🆕 下游拥塞信号 (Backpressure Routing)
]
# =============================================================================
# 设计说明 (Hier-MAPPO 架构)
# =============================================================================
#
# 按照 Hier-MAPPO 框架设计，L1 和 L2 使用**同一套统一观测**：
#   - Head1 (Discrete/Task Split): 使用观测决定任务切分 [α_local, α_cloud, α_sat]
#   - Head2 (Continuous/Routing): 使用观测 + Head1 输出（条件化）决定路由
#
# 分头逻辑在**模型端**实现，而非观测端：
#   - L2 Selector 网络接收 [观测, L1输出] 作为输入
#   - 这使得 L2 的路由决策可以基于 L1 的切分意图
#
# 观测结构（扁平化，可配置）：
#   - self_features (默认 8D): 可配置的自身状态特征
#   - neighbor_features (N x 默认 6): 可配置的邻居特征
#   - neighbor_mask (N): 邻居可达性掩码
#
# 观测结构（扁平化，可配置）：
#   - neighbor_features (N x 默认 6): 可配置的邻居特征
#   - neighbor_mask (N): 邻居可达性掩码
#   - 自身特征由 obs.SatProperties 额外组合
#
# 默认总维度: (N * 6) + N = 7N
# 当 N=4 时: 28D
#
# =============================================================================


