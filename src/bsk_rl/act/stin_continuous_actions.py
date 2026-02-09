"""
STINContinuousAction: 定义星地融合网络的全连续动作空间。
继承自 ContinuousAction 以遵循 BSK-RL 框架。

**安全屏蔽 (Action Shielding)**：
    硬约束通过 STINActionShield 实现，在动作执行前自动修正违规动作：
    - 电池 < 20%：限制 CPU/TX 功率
    - 队列满：提高本地处理比例
"""
import logging
from typing import TYPE_CHECKING, Any, Optional, List
import numpy as np
from gymnasium import spaces

from bsk_rl.act.continuous_actions import ContinuousAction 

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats.satellite import Satellite
    from bsk_rl.utils.shields import STINActionShield
    class TaskSlice:
        def __init__(self, task_id: int, data_size: float, workload: float, max_delay: float, origin_position: np.ndarray, uplink_distance: float, uplink_rate: float):
            pass


logger = logging.getLogger(__name__)

class STINContinuousAction(ContinuousAction):
    """
    STIN 多智能体连续动作，总维度 N=16。
    实现了 ContinuousAction 的抽象方法。
    
    **安全屏蔽**：
        启用 `enable_shield=True` 后，动作会在执行前经过 STINActionShield 检查：
        - 电池 < 20%：限制 CPU/TX 功率到安全范围
        - 队列满：强制提高本地处理比例
    """
    # 动作维度定义为类属性
    # 16D: [0-2] 资源, [3-4] α_local/α_cloud, [5] priority_threshold, [6:11] high_split, [11:16] low_split
    REQUIRED_ACTION_DIMS = 16 

    def __init__(
        self, 
        name: str = "stin_continuous_act",
        enable_shield: bool = True,
        battery_threshold: float = 0.15,
        max_neighbors: int = 4,  # 最大邻居数量（用于协作卸载和观测）
    ) -> None:
        """
        初始化动作类。
        
        Args:
            name: 动作名称。
            enable_shield: 是否启用安全屏蔽（硬约束）。
            battery_threshold: 触发电池保护的 SOC 阈值。
            max_neighbors: 最大邻居数量，用于协作卸载决策。默认 4。
        """
        super().__init__(name=name) 
        self.neighbor_satellites: List['Satellite'] = []
        self.max_neighbors = max_neighbors  # 供 gym.py 读取
        
        # 安全屏蔽配置（默认值，会在 link_satellite 时从 sat_args 覆盖）
        self.enable_shield = enable_shield
        self.battery_threshold = battery_threshold
        self._shield = None  # 延迟初始化，避免循环导入
        self._shield_config = {}  # Shield 详细配置
        
        # 屏蔽统计
        self.shield_interventions = 0

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
        
        logger.debug(f"[{satellite.name}] Shield config: enable={self.enable_shield}, threshold={self.battery_threshold}")
        

    @property
    def space(self) -> spaces.Box:
        """
        实现 ContinuousAction 的抽象方法: 返回动作空间。
        """
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.REQUIRED_ACTION_DIMS,),
            dtype=np.float32,
        )

    @property
    def action_description(self) -> List[str]:
        """
        实现 ContinuousAction 的抽象方法: 返回描述。
        
        **动作维度定义 (16D - 基于优先级的差异化切分)**:
            [0-2]: 资源分配 (CPU 频率、发射功率、平台功耗)
            [3-4]: 一级任务切分 (UD 本地、云端偏好) - 统一应用
            [5]: 优先级阈值 (高/低优先级任务的分界线)
            [6-10]: 高优先级任务的卫星间切分 (自身 + 4 个邻居)
            [11-15]: 低优先级任务的卫星间切分 (自身 + 4 个邻居)
        """
        return [
            "[0] CPU_Ratio: CPU 频率分配比例 [0,1] → [f_min, f_max]",
            "[1] TX_Power_Ratio: 发射功率分配比例 [0,1]",
            "[2] Platform_Power_Ratio: 平台功耗管理比例 [0,1]",
            "[3] Alpha_Local: UD 本地保留比例 [0,1] (统一应用)",
            "[4] Alpha_Cloud: 云端偏好因子 [0,1] (统一应用)",
            "[5] Priority_Threshold: 优先级阈值 [0,1] (高/低分界)",
            "[6] High_x_0: 高优先级-自身处理比例",
            "[7] High_x_1: 高优先级-邻居1协作比例",
            "[8] High_x_2: 高优先级-邻居2协作比例",
            "[9] High_x_3: 高优先级-邻居3协作比例",
            "[10] High_x_4: 高优先级-邻居4协作比例",
            "[11] Low_x_0: 低优先级-自身处理比例",
            "[12] Low_x_1: 低优先级-邻居1协作比例",
            "[13] Low_x_2: 低优先级-邻居2协作比例",
            "[14] Low_x_3: 低优先级-邻居3协作比例",
            "[15] Low_x_4: 低优先级-邻居4协作比例",
        ]


    def set_action(self, action: np.ndarray) -> None:
        """
        实现 ContinuousAction 的抽象方法: 解析连续动作并触发卫星逻辑。
        
        **安全屏蔽流程**：
            1. 如果启用 shield，先对原始动作进行安全检查和修正
            2. 解析修正后的动作分量
            3. 执行资源分配和协作调度
        
        **动作向量解析**:
            action[0:3]: 资源分配参数
            action[3:5]: 一级切分参数 (层次化)
            action[5:10]: 二级切分参数 (卫星间协作)
        
        Args:
            action: 10D 动作向量，范围 [0, 1]
        """
        if len(action) != self.REQUIRED_ACTION_DIMS:
            raise ValueError(
                f"Action vector dimension mismatch. Expected {self.REQUIRED_ACTION_DIMS}, got {len(action)}"
            )

        # ====== 安全屏蔽（硬约束）======
        safe_action = action
        if self.enable_shield:
            safe_action = self._apply_shield(action)
        
        # 解析动作分量（顺序必须与 action_description 一致）
        cpu_ratio = safe_action[0]              # [0] CPU 频率比例
        tx_power_ratio = safe_action[1]         # [1] 发射功率比例
        platform_power_ratio = safe_action[2]   # [2] 平台功耗比例
        alpha_local = safe_action[3]            # [3] UD 本地保留比例
        alpha_cloud = safe_action[4]            # [4] 云端偏好因子
        priority_threshold = safe_action[5]     # [5] 优先级阈值
        high_split_ratios = safe_action[6:11]   # [6:11] 高优先级任务的卫星间切分
        low_split_ratios = safe_action[11:16]   # [11:16] 低优先级任务的卫星间切分
        
        # ✅ 修复1: 记录模式A执行前的切片队列长度，确保模式B只处理观测时已存在的切片
        initial_slice_count = len(self.satellite.slice_queue) if hasattr(self.satellite, 'slice_queue') else 0
        
        # 1. 资源分配（本地控制）
        self.satellite.set_resource_allocation(cpu_ratio, tx_power_ratio, platform_power_ratio)

        # 2. 模式 A: 切分原始任务（Softmax）
        # 新切片会加入 slice_queue 末端，但本 step 不处理它们
        self.satellite.schedule_collaboration_action(
            alpha_local=alpha_local,
            alpha_cloud=alpha_cloud,
            priority_threshold=priority_threshold,
            high_split_ratios=high_split_ratios,
            low_split_ratios=low_split_ratios,
            neighbor_satellites=self.neighbor_satellites
        )
        
        # 3. 模式 B: 路由切片队列（Argmax）
        # ✅ 修复2: 只处理 step 开始时就存在的切片（观测-动作一致性）
        # ✅ 修复3: 对每个切片按其优先级选择高/低路由比例
        if initial_slice_count > 0:
            self.satellite.process_slice_queue(
                high_split_ratios=high_split_ratios,
                low_split_ratios=low_split_ratios,
                priority_threshold=priority_threshold,
                neighbor_satellites=self.neighbor_satellites,
                max_slices=initial_slice_count  # ✅ 只处理前 N 个切片
            )
    
    def _apply_shield(self, action: np.ndarray) -> np.ndarray:
        """应用安全屏蔽，返回修正后的动作。

        硬约束检查：
            1. 电池 < threshold：限制 CPU/TX/Cloud 动作
            2. 队列满：提高云端比例
        """
        # 延迟初始化 shield（避免循环导入）
        if self._shield is None:
            try:
                from bsk_rl.utils.shields import STINActionShield
                self._shield = STINActionShield(
                    battery_threshold=self.battery_threshold,
                    log_interventions=False,  # 训练时关闭日志
                )
            except ImportError:
                logger.warning("Cannot import STINActionShield, disabling shield")
                self.enable_shield = False
                return action
        
        # 应用屏蔽
        safe_action = self._shield.apply(self.satellite, action)
        
        # 统计干预次数
        if not np.array_equal(action, safe_action):
            self.shield_interventions += 1
        
        return safe_action


        
    def reset_overwrite_previous(self) -> None:
        """重置动作状态，清除邻居缓存和屏蔽统计。"""
        super().reset_overwrite_previous()
        self.neighbor_satellites.clear()
        self.shield_interventions = 0
        if self._shield is not None:
            self._shield.reset_stats()
        
        
    def add_neighbor(self, neighbor: 'Satellite') -> None:
        """供环境在 reset 时调用，以填充当前卫星的协作邻居列表。"""
        self.neighbor_satellites.append(neighbor)
