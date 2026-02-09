"""
STINHierarchicalHybridAction: 分层连续动作空间（High‑Level + Low‑Level）。

定义星地融合网络的全连续动作空间。
继承自 Action 以遵循 BSK-RL 框架。

动作空间维度:
    REQUIRED_ACTION_DIMS_FORMULA = "K + 7"  (K = max_neighbors)
    K=4 时总维度: 11D
    
    Level-1 (High-Level) - 任务切分 (4 维):
        [alpha_ud, alpha_cloud, alpha_sat_local, alpha_sat_offload]
        
    Level-2 (Low-Level) - 资源与路由 (4 + 2*K 维):
        连续部分: [cpu, tx, priority_threshold, high_split..., low_split...]

**安全屏蔽 (Action Shielding)**：
    硬约束通过 STINActionShield 实现，在动作执行前自动修正违规动作：
    - 电池 < 20%：限制 CPU/TX 功率
    - 队列满：提高本地处理比例


设计目标（分层连续路由 - 双层分离）：

第一层 (Level-1) - 任务切分：
- 维度: 4
- [alpha_ud, alpha_cloud, alpha_sat_local, alpha_sat_offload]
- 对 raw 任务队列中的每个任务，按此比例切分到 UD本地/云端/接入卫星本地/接入卫星切片队列
- 归一化后 sum = 1

第二层 (Level-2) - 切片路由 + 资源（简化为纯连续分布）：
就相当于说切片不可再分了，对任务切片队列就是做路由；
但是由于我们的step是一个duration time，我们要考虑动作怎么设计才能对当前step内的切片做一个轮询处理；
形成两个分布，执行的时候根据分布采样；
- 连续部分:
    * 资源参数 (3D): [cpu, tx, priority_threshold]
* 高优先级切片路由 (K)D: [high_n0, high_n1, ..., high_nK-1]
* 低优先级切片路由 (K)D: [low_n0, low_n1, ..., low_nK-1]

简化动作空间设计：排除优先级阈值，将切片路由简化成[high_n1, ..., high_nK] 不给本地，同时只需要一套；


处理流程：
1. Level-1: 对任务队列中每个任务执行一级切分，生成切片 (只切分一次) （Zhang 等 - 2024- 只跳一次）
2. Level-2: 对切片队列轮询处理
     - priority >= threshold: 使用 high_split_ratios
     - priority < threshold: 使用 low_split_ratios
     - 对每个切片按对应分布采样路由目标

总维度: 4 + (4 + 2*(K+1)) = 2K + 10
K=4 时: 4 + 14 = 18D

执行语义（可切换）：
    - delayed: Level-1 在当前 step 新生成的切片不会被当前 step 的 Level-2 处理
    - immediate: Level-1 新生成的切片可在当前 step 立即参与 Level-2 路由
    - 默认 delayed，以保持与旧实验行为兼容

    首选 ratio_normalize：

在强化学习任务卸载场景中，我们通常希望 Agent 能够完全切断某条低效链路（例如直接丢弃或者完全不给云端）。线性归一化允许产生 0.0，这对于策略的稀疏性和可解释性非常重要。
代码默认值也是这个，保持不动即可。
慎用 softmax_all：
除非你的 Reward 函数非常平滑，否则 Softmax 容易导致训练初期探索不足（很难输出 0 或 1）。
grouped_softmax 是进阶选项：
如果你发现 Agent 总是学不会在“本地计算”和“邻居卸载”之间做细微权衡（因为地面链路的数值太大掩盖了它们），可以尝试切换到这个模式。
注意：本文件为新方案的独立实现，不覆盖现有 STINContinuousAction。
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import numpy as np
from gymnasium import spaces

from bsk_rl.act.continuous_actions import ContinuousAction 
from bsk_rl.act.actions import Action, ActionBuilder

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats.hier_computation_satellite import HierComputationSatellite
    from bsk_rl.sats.satellite import Satellite
    from bsk_rl.utils.shields import STINActionShield


logger = logging.getLogger(__name__)

class HierarchicalHybridActionBuilder(ActionBuilder):
    """ActionBuilder for hierarchical continuous action.
    
    Note:
        action_space 返回 flat_space（Box 类型）以兼容 BenchMARL/TorchRL。
        分层结构 Dict 空间可通过 hierarchical_space 属性访问。
    """

    def __init__(self, satellite: "HierComputationSatellite") -> None:
        self.action_spec: list[STINHierarchicalHybridAction]
        super().__init__(satellite)
        assert len(self.action_spec) == 1, "Only one hierarchical action is supported."

    @property
    def _action(self) -> "STINHierarchicalHybridAction":
        return self.action_spec[0]

    @property
    def action_space(self) -> spaces.Space:
        """返回扁平化动作空间（Box 类型），兼容 BenchMARL/TorchRL。
        
        gym.py 的 action_spaces 属性期望 Box 或 Discrete 类型。
        分层 Dict 空间可通过 hierarchical_space 属性访问。
        """
        return self._action.flat_space
    
    @property
    def hierarchical_space(self) -> spaces.Space:
        """返回分层动作空间（Dict 类型），用于需要层级结构的场景。"""
        return self._action.space

    @property
    def action_description(self) -> Any:
        return self._action.action_description

    def set_action(self, action: Any) -> None:
        self._action.set_action(action)


class STINHierarchicalHybridAction(Action):
    """分层连续动作（统一路由，K+7 维）。

    动作维度：
        REQUIRED_ACTION_DIMS_FORMULA = "K + 7" (K = max_neighbors)

    - Level-1 (4 维): [alpha_ud, alpha_cloud, alpha_sat_local, alpha_sat_offload]
    - Level-2 (2 + K+1 维): [cpu, tx, route_n0...route_nK-1, route_self]

    执行语义：
        - delayed: 本步切分、下步路由
        - immediate: 本步切分、立即路由
    """

    # 动作维度计算公式: K + 7 (K = max_neighbors)
    
    builder_type = HierarchicalHybridActionBuilder

    def __init__(
        self,
        name: str = "stin_hier_continuous_act",
        enable_shield: bool = True,
        battery_threshold: float = 0.15,
        max_neighbors: int = 4,
        level1_norm_mode: str = "ratio_normalize",
        routing_execution_mode: str = "delayed",
    ) -> None:
        """Initialize the action.

        Args:
            name: Action name.
            enable_shield: Whether to enable safety shielding.
            battery_threshold: Battery SOC threshold.
            max_neighbors: Maximum neighbors for routing.
            level1_norm_mode: Level-1 归一化模式，可选：
                - "grouped_softmax": 分组 softmax + 门控
                - "softmax_all": 全量 softmax（单头切分）
                - "ratio_normalize": 非负线性归一化（不做 softmax/门控）
        """
        super().__init__(name=name)
        self.neighbor_satellites: List['Satellite'] = []
        self.max_neighbors = max_neighbors  # 供 gym.py 读取
        # 统一路由版动作维度：4 + 2 + (K+1) = K + 7
        self.REQUIRED_ACTION_DIMS_FORMULA = "K + 7"

        # 安全屏蔽配置（默认值，会在 link_satellite 时从 sat_args 覆盖）
        self.enable_shield = enable_shield
        self.battery_threshold = battery_threshold
        self._shield = None  # 延迟初始化，避免循环导入
        self._shield_config = {}  # Shield 详细配置

        # Level-1 归一化模式
        self.level1_norm_mode = level1_norm_mode
        # delayed: 本步切分、下步路由; immediate: 本步切分、立即路由
        self.routing_execution_mode = str(routing_execution_mode).lower()
        if self.routing_execution_mode not in {"delayed", "immediate"}:
            logger.warning(
                "Unknown routing_execution_mode=%s, fallback to delayed",
                self.routing_execution_mode,
            )
            self.routing_execution_mode = "delayed"

        # 屏蔽统计
        self.shield_interventions = 0
        # Level-1 归一化统计
        self.level1_normalize_eps = 1e-3
        self.level1_normalize_count = 0
        # 动作后处理差异日志（用于诊断）
        self.action_diff_log_remaining = 0
        self.action_diff_log_threshold = 1e-1

    def link_satellite(self, satellite: "Satellite") -> None:
        """
        Link the action to a satellite and load Shield configuration.
        
        Shield 配置从 satellite.shield_config 读取，支持以下参数：
            - enable_shield: 是否启用安全屏蔽（默认 True）
            - battery_threshold: 触发电池保护的 SOC 阈值（默认 0.15）
            - low_battery_cpu_cap: 低电量时 CPU 功率上限（默认 0.3）
            - low_battery_tx_cap: 低电量时发射功率上限（默认 0.2）
            - max_queue_size: 触发队列保护的阈值（默认 50）
        
        Args:
            satellite: Satellite to link to
        """
        self.satellite = satellite
        
        # 从 satellite.shield_config 读取 Shield 配置
        shield_config = getattr(satellite, 'shield_config', {})
        self.enable_shield = shield_config.get('enable_shield', self.enable_shield)
        self.battery_threshold = shield_config.get('battery_threshold', self.battery_threshold)
        
        # 如果需要创建 Shield，也传递额外参数
        self._shield_config = {
            'battery_threshold': shield_config.get('battery_threshold', self.battery_threshold),
            'low_battery_cpu_cap': shield_config.get('low_battery_cpu_cap', 0.3),
            'low_battery_tx_cap': shield_config.get('low_battery_tx_cap', 0.2),
            'max_queue_size': shield_config.get('max_queue_size', 50),
        }
        self.action_diff_log_remaining = shield_config.get(
            'action_diff_log_remaining', self.action_diff_log_remaining
        )
        self.action_diff_log_threshold = shield_config.get(
            'action_diff_log_threshold', self.action_diff_log_threshold
        )
        self.routing_execution_mode = str(
            getattr(satellite, "routing_execution_mode", self.routing_execution_mode)
        ).lower()
        if self.routing_execution_mode not in {"delayed", "immediate"}:
            self.routing_execution_mode = "delayed"
        
        logger.debug(f"[{satellite.name}] Shield config: enable={self.enable_shield}, threshold={self.battery_threshold}")
        

    @property
    def space(self) -> spaces.Space:
        """
        分层连续动作空间（统一路由版）：
        - level1: Box(4,) -> [alpha_ud, alpha_cloud, alpha_sat_local, alpha_sat_offload]
        - level2: {"continuous": [cpu, tx, routing_probs...(含 self)]}

        其中 routing_probs 维度为 (K + 1)，K=max_neighbors。
        """
        K = self.max_neighbors
        continuous_dim = 2 + (K + 1)
        return spaces.Dict(
            {
                "level1": spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
                "level2": spaces.Dict(
                    {
                        "continuous": spaces.Box(
                            low=0.0,
                            high=1.0,
                            shape=(continuous_dim,),
                            dtype=np.float32,
                        ),
                    }
                ),  # 保留 Dict，便于后续扩展
            }
        )
    
    @property
    def flat_space(self) -> spaces.Space:
        """
        扁平化动作空间（用于 BenchMARL 兼容）：
        - [0:4]: level1 (alpha_ud, alpha_cloud, alpha_sat_local, alpha_sat_offload)
        - [4:6]: level2 资源 (cpu, tx)
        - [6:6+K+1]: level2 unified routing (K+1 维，含 self)

        总维度: 4 + 2 + (K+1) = K + 7
        K=4 时: 11
        """
        K = self.max_neighbors
        total_dim = 4 + 2 + (K + 1)
        return spaces.Box(low=0.0, high=1.0, shape=(total_dim,), dtype=np.float32)

    @property
    def action_description(self) -> List[str]:
        """动作描述（扁平化列表形式，与 flat_space 对齐）。

        Returns:
            List[str]: 按动作维度顺序排列的动作名称列表。
                长度为 K + 7 (K = max_neighbors)。
        """
        K = self.max_neighbors

        level1_names = [
            "alpha_ud (grouped split input)",
            "alpha_cloud (grouped split input)",
            "alpha_sat_local (grouped split input, self)",
            "alpha_sat_offload (grouped split input, routing)",
        ]

        resource_names = [
            "cpu_ratio (resource)",
            "tx_power_ratio (resource)",
        ]

        routing_names = [f"route_n{i} (slice routing)" for i in range(K)] + [
            "route_self (slice routing)"
        ]

        return level1_names + resource_names + routing_names

    @property
    def action_description_dict(self) -> Dict[str, Any]:
        """动作描述（分层字典形式）。"""
        K = self.max_neighbors
        level1_names = [
            "alpha_ud (grouped)",
            "alpha_cloud (grouped)",
            "alpha_sat_local (grouped)",
            "alpha_sat_offload (grouped)",
        ]
        routing_names = [f"route_n{i} (slice)" for i in range(K)] + ["route_self (slice)"]
        return {
            "level1": level1_names,
            "level2": {
                "continuous": [
                    "cpu_ratio (slice)",
                    "tx_power_ratio (slice)",
                ] + routing_names,
            },
        }

    def set_action(self, action: Any) -> None:
        """解析并执行分层连续动作（统一路由版）。

        支持以下输入格式：
        1. Dict: {"level1": array(4,), "level2": {"continuous": array(2+K+1,)}}
        2. Flat array: [level1..., cpu, tx, routing_split...]

        处理流程：
        1. 资源分配 (cpu, tx)
        2. Level-1 任务切分: 对 raw 任务队列按 alpha 比例切分，生成切片
        3. Level-2 切片路由: 对切片队列按统一路由分布进行处理

        安全屏蔽流程：
            1. 如果启用 shield，先对原始动作进行安全检查和修正
            2. 解析修正后的动作分量
            3. 执行资源分配和协作调度
        """
        # 维度检查仅对 Flat array 格式生效（Dict 格式在 _parse_action 处理）
        K = self.max_neighbors
        routing_targets = K + 1
        expected_dim = 4 + 2 + routing_targets
        if not isinstance(action, dict):
            action_size = np.asarray(action, dtype=np.float64).size
            if action_size != expected_dim:
                raise ValueError(
                    f"Action vector dimension mismatch. Expected {expected_dim}, got {action_size}"
                )

        # ====== 安全屏蔽（硬约束）======
        raw_parsed = self._parse_action(action)
        safe_action = action
        if self.enable_shield:
            safe_action = self._apply_shield(action)
        if not isinstance(safe_action, dict):
            safe_size = np.asarray(safe_action, dtype=np.float64).size
            if safe_size != expected_dim:
                raise ValueError(
                    f"Shielded action dimension mismatch. Expected {expected_dim}, got {safe_size}"
                )

        parsed = self._parse_action(safe_action)

        # Level-1: 任务切分参数 [alpha_ud, alpha_cloud, alpha_sat_local, alpha_sat_offload]
        level1_ratios = self._normalize_level1(parsed["level1"])
        alpha_ud = level1_ratios[0]
        alpha_cloud = level1_ratios[1]
        alpha_sat_local = level1_ratios[2]
        alpha_sat_offload = level1_ratios[3]

        # 记录 Level-1 分配比例，便于日志/诊断
        self.satellite.last_level1_ratios = np.asarray(level1_ratios, dtype=np.float64)
        self.satellite._alpha_updated = True

        # Level-2: 资源与统一路由
        continuous = parsed["level2"]["continuous"]
        cpu_ratio = float(continuous[0])
        tx_power_ratio = float(continuous[1])
        platform_power_ratio = 1.0  # 固定平台功耗，不作为动作维度

        routing_raw = continuous[2:2 + routing_targets]
        routing_split_ratios = np.asarray(routing_raw, dtype=np.float64)

        # 诊断：动作后处理前后差异（shield + level1 归一化）
        if raw_parsed is not None:
            raw_level1 = np.asarray(raw_parsed["level1"], dtype=np.float64)
            raw_cont = np.asarray(raw_parsed["level2"]["continuous"], dtype=np.float64)
            safe_level1 = np.asarray(parsed["level1"], dtype=np.float64)
            safe_cont = np.asarray(parsed["level2"]["continuous"], dtype=np.float64)
            raw_flat = np.concatenate([raw_level1, raw_cont])
            safe_flat = np.concatenate([safe_level1, safe_cont])
            post_flat = np.concatenate([np.asarray(level1_ratios, dtype=np.float64), safe_cont])
            shield_delta = raw_flat - safe_flat
            post_delta = safe_flat - post_flat
            shield_l2 = float(np.linalg.norm(shield_delta))
            post_l2 = float(np.linalg.norm(post_delta))
            shield_max = float(np.max(np.abs(shield_delta))) if shield_delta.size else 0.0
            post_max = float(np.max(np.abs(post_delta))) if post_delta.size else 0.0
            level1_max = float(np.max(np.abs(raw_level1 - safe_level1))) if raw_level1.size else 0.0
            l2_max = float(np.max(np.abs(raw_cont - safe_cont))) if raw_cont.size else 0.0
            should_log = (
                self.action_diff_log_remaining > 0
                or max(shield_max, post_max) > self.action_diff_log_threshold
            )
            if should_log:
                logger.warning(
                    "[ACTION DIFF] sat=%s shield_l2=%.3e post_l2=%.3e "
                    "shield_max=%.3e post_max=%.3e l1_max=%.3e l2_max=%.3e",
                    getattr(self.satellite, "name", "unknown"),
                    shield_l2,
                    post_l2,
                    shield_max,
                    post_max,
                    level1_max,
                    l2_max,
                )
                if self.action_diff_log_remaining > 0:
                    self.action_diff_log_remaining -= 1

        # 记录切片队列长度（Level-1 执行前）
        initial_slice_count = (
            len(self.satellite.slice_queue) if hasattr(self.satellite, "slice_queue") else 0
        )

        # 1) 资源分配（本地控制）
        self.satellite.set_resource_allocation(cpu_ratio, tx_power_ratio, platform_power_ratio)

        # 2) Level-1: 任务切分（对 raw 任务队列）
        task_split_ratios = np.array(
            [alpha_ud, alpha_cloud, alpha_sat_local, alpha_sat_offload], dtype=np.float64
        )

        self.satellite.schedule_collaboration_action_hier(
            task_split_ratios=task_split_ratios,
            neighbor_satellites=self.neighbor_satellites,
        )

        # 3) Level-2: 切片路由（对切片队列轮询）
        # delayed: 仅处理动作前已有切片；immediate: 包含本步新增切片。
        if self.routing_execution_mode == "immediate":
            max_slices = None
        else:
            max_slices = initial_slice_count

        should_route = (
            (max_slices is None and len(getattr(self.satellite, "slice_queue", [])) > 0)
            or (max_slices is not None and max_slices > 0)
        )
        processed_slices = 0
        if should_route:
            processed_slices = int(
                self.satellite.process_slice_queue_hier(
                high_split_ratios=routing_split_ratios,
                low_split_ratios=routing_split_ratios,
                priority_threshold=0.5,
                neighbor_satellites=self.neighbor_satellites,
                max_slices=max_slices,
                )
            )

        if hasattr(self.satellite, "level2_candidate_slices_this_step"):
            if max_slices is None:
                self.satellite.level2_candidate_slices_this_step = len(
                    getattr(self.satellite, "slice_queue", [])
                ) + max(processed_slices, 0)
            else:
                self.satellite.level2_candidate_slices_this_step = max(max_slices, 0)
        if hasattr(self.satellite, "level2_processed_count_this_step"):
            self.satellite.level2_processed_count_this_step = max(processed_slices, 0)

    def _normalize_ratios(self, ratios: np.ndarray) -> np.ndarray:
        """归一化比例数组，确保和为 1。
        
        Args:
            ratios: 原始比例数组
            
        Returns:
            归一化后的比例数组，sum = 1
        """
        ratios = np.asarray(ratios, dtype=np.float64)
        total = np.sum(ratios)
        if (
            np.all(ratios >= -self.level1_normalize_eps)
            and abs(total - 1.0) <= self.level1_normalize_eps
        ):
            return np.clip(ratios, 0.0, 1.0)

        self.level1_normalize_count += 1
        # 确保非负后再 clip，增强数值稳定性
        ratios = np.maximum(ratios, 0.0)
        ratios = np.clip(ratios, 1e-6, 1.0)
        total = np.sum(ratios)
        if total < 1e-6 or np.isnan(total) or np.isinf(total):
            # 均匀分配
            return np.full_like(ratios, 1.0 / len(ratios))
        return ratios / total

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        if logits.size == 0:
            return logits
        logits = np.nan_to_num(logits, nan=0.0, posinf=1.0, neginf=-1.0)
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        denom = np.sum(exp_logits)
        if denom <= 0 or np.isnan(denom) or np.isinf(denom):
            return np.full_like(logits, 1.0 / len(logits))
        return exp_logits / denom

    def _normalize_level1(self, ratios: np.ndarray) -> np.ndarray:
        ratios = np.asarray(ratios, dtype=np.float64)
        mode = getattr(self, "level1_norm_mode", "ratio_normalize")
        if ratios.size == 0:
            return ratios

        if mode == "grouped_softmax":
            return self._normalize_level1_grouped(ratios)
        if mode == "softmax_all":
            return self._softmax(ratios)
        if mode == "ratio_normalize":
            cleaned = np.nan_to_num(ratios, nan=0.0, posinf=0.0, neginf=0.0)
            clipped = np.maximum(cleaned, 0.0)
            total = np.sum(clipped)
            if total <= 0 or np.isnan(total) or np.isinf(total):
                return np.full_like(clipped, 1.0 / len(clipped))
            return clipped / total
        if mode == "passthrough":
            # Only sanitize non-finite values; keep policy output geometry unchanged.
            return np.nan_to_num(ratios, nan=0.0, posinf=0.0, neginf=0.0)

        logger.warning(
            f"Unknown level1_norm_mode={mode}, fallback to ratio_normalize"
        )
        clipped = np.maximum(np.nan_to_num(ratios, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        total = np.sum(clipped)
        if total <= 0 or np.isnan(total) or np.isinf(total):
            return np.full_like(clipped, 1.0 / len(clipped))
        return clipped / total

    def _normalize_level1_grouped(self, ratios: np.ndarray) -> np.ndarray:
        ratios = np.asarray(ratios, dtype=np.float64)
        if ratios.size < 3:
            return self._normalize_ratios(ratios)

        ud_cloud_raw = ratios[:2]
        sat_raw = ratios[2:]

        # Grouped softmax: split within group, then gate between groups
        ud_cloud_probs = self._softmax(ud_cloud_raw)
        sat_probs = self._softmax(sat_raw)
        group_logits = np.array(
            [np.mean(ud_cloud_raw), np.mean(sat_raw)], dtype=np.float64
        )
        group_probs = self._softmax(group_logits)

        grouped = np.concatenate(
            [group_probs[0] * ud_cloud_probs, group_probs[1] * sat_probs]
        )
        total = np.sum(grouped)
        if total <= 0 or np.isnan(total) or np.isinf(total):
            return np.full_like(ratios, 1.0 / len(ratios))
        return grouped / total

    def _parse_action(self, action: Any) -> Dict[str, Any]:
        """解析动作输入为标准格式（统一路由版）。"""
        K = self.max_neighbors
        routing_targets = K + 1  # neighbors + self
        level1_dim = 4
        continuous_dim = 2 + routing_targets  # cpu, tx + unified routing

        if isinstance(action, dict):
            default_level1 = np.full(level1_dim, 1.0 / level1_dim, dtype=np.float64)
            level1 = np.asarray(action.get("level1", default_level1), dtype=np.float64)
            if level1.size != level1_dim:
                raise ValueError(
                    f"Level1 dimension mismatch. Expected {level1_dim}, got {level1.size}"
                )

            level2 = action.get("level2", {})

            # 默认连续动作：资源 0.5，routing 均匀分配
            default_continuous = np.full(continuous_dim, 0.5, dtype=np.float64)
            if routing_targets > 0:
                default_continuous[2:] = 1.0 / routing_targets
            continuous = np.asarray(
                level2.get("continuous", default_continuous), dtype=np.float64
            )
            if continuous.size != continuous_dim:
                raise ValueError(
                    f"Level2 continuous dimension mismatch. Expected {continuous_dim}, got {continuous.size}"
                )
            return {"level1": level1, "level2": {"continuous": continuous}}

        action_arr = np.asarray(action, dtype=np.float64).flatten()
        expected_total = level1_dim + continuous_dim

        if action_arr.size != expected_total:
            raise ValueError(
                "Action vector dimension mismatch after flatten. "
                f"Expected {expected_total}, got {action_arr.size}"
            )

        level1 = action_arr[:level1_dim]
        continuous = action_arr[level1_dim : level1_dim + continuous_dim]

        return {"level1": level1, "level2": {"continuous": continuous}}

    def _apply_shield(self, action: Any) -> Any:
        """应用安全屏蔽，返回修正后的动作。

        硬约束检查：
            1. 电池 < threshold：限制 CPU/TX/Cloud 动作
            2. 队列满：提高云端比例

        Args:
            action: 原始动作（支持 Dict 或 flat array 格式）

        Returns:
            修正后的安全动作（保持输入格式）
        """
        # 延迟初始化 shield（避免循环导入）
        if self._shield is None:
            try:
                from bsk_rl.utils.shields import STINActionShield

                self._shield = STINActionShield(
                    battery_threshold=self._shield_config.get("battery_threshold", self.battery_threshold),
                    low_battery_cpu_cap=self._shield_config.get("low_battery_cpu_cap", 0.3),
                    low_battery_tx_cap=self._shield_config.get("low_battery_tx_cap", 0.2),
                    low_battery_cloud_cap=self._shield_config.get("low_battery_cloud_cap", 0.6),
                    max_queue_size=self._shield_config.get("max_queue_size", 50),
                    queue_full_local_boost=self._shield_config.get(
                        "queue_full_cloud_boost",
                        self._shield_config.get("queue_full_local_boost", 0.6),
                    ),
                    log_interventions=False,
                )
            except ImportError:
                logger.warning("Cannot import STINActionShield, disabling shield")
                self.enable_shield = False
                return action

        K = self.max_neighbors
        routing_targets = K + 1
        level1_dim = 4
        continuous_dim = 2 + routing_targets

        is_dict_input = isinstance(action, dict)
        if is_dict_input:
            level1 = np.asarray(action.get("level1", np.zeros(level1_dim)), dtype=np.float64)
            level2 = action.get("level2", {})
            level2_continuous = np.asarray(
                level2.get("continuous", np.zeros(continuous_dim)), dtype=np.float64
            )
        else:
            action_arr = np.asarray(action, dtype=np.float64).flatten()
            level1 = action_arr[:level1_dim]
            level2_continuous = action_arr[level1_dim : level1_dim + continuous_dim]

        safe_l1, safe_l2 = self._shield.apply_hierarchical(
            self.satellite, level1, level2_continuous
        )

        if not (np.allclose(level1, safe_l1) and np.allclose(level2_continuous, safe_l2)):
            self.shield_interventions += 1

        if is_dict_input:
            return {
                "level1": safe_l1.astype(np.float32),
                "level2": {"continuous": safe_l2.astype(np.float32)},
            }
        safe_action = np.concatenate([safe_l1, safe_l2])
        return safe_action.astype(np.float32)

    def reset_overwrite_previous(self) -> None:
        super().reset_overwrite_previous()
        self.neighbor_satellites.clear()
        self.shield_interventions = 0
        self.level1_normalize_count = 0

    def add_neighbor(self, neighbor: "HierComputationSatellite") -> None:
        """添加邻居卫星（通信模块调用）。"""
        if len(self.neighbor_satellites) < self.max_neighbors:
            self.neighbor_satellites.append(neighbor)


__doc_title__ = "STIN Hierarchical Hybrid Actions"
__all__ = [
    "STINHierarchicalHybridAction",
    "HierarchicalHybridActionBuilder",
]

