"""
STINContinuousAction: 定义星地融合网络的全连续动作空间。
继承自 ContinuousAction 以遵循 BSK-RL 框架。
"""
import logging
from typing import TYPE_CHECKING, Any, Optional, List
import numpy as np
from gymnasium import spaces

# 假设从正确路径导入 ContinuousAction。
# 根据您的目录结构，它应位于 bsk_rl/src/bsk_rl/act/continuous_actions.py
from bsk_rl.act.continuous_actions import ContinuousAction 

if TYPE_CHECKING:  # pragma: no cover
    from bsk_rl.sats.satellite import Satellite
    # 假设 TaskSlice 结构已在 ComputationSatellite 或辅助文件中定义
    class TaskSlice:
        def __init__(self, task_id: int, data_size: float, workload: float, max_delay: float, origin_position: np.ndarray, uplink_distance: float, uplink_rate: float):
            pass


logger = logging.getLogger(__name__)

class STINContinuousAction(ContinuousAction):
    """
    STIN 多智能体连续动作，总维度 N=10。
    实现了 ContinuousAction 的抽象方法。
    """
    # 动作维度定义为类属性
    REQUIRED_ACTION_DIMS = 10 

    def __init__(self, name: str = "stin_continuous_act") -> None:
        """
        初始化动作类。ContinuousAction 基类会使用此处的实例来构建动作空间。
        """
        # 只需要传递名称给基类
        super().__init__(name=name) 
        # 明确类型为 List[Satellite]
        self.neighbor_satellites: List['Satellite'] = []
        

    @property
    def space(self) -> spaces.Box:
        """
        实现 ContinuousAction 的抽象方法: 返回动作空间。
        """
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.REQUIRED_ACTION_DIMS,),
            dtype=np.float64,
        )

    @property
    def action_description(self) -> List[str]:
        """
        实现 ContinuousAction 的抽象方法: 返回描述。
        
        **动作维度定义 (10D)**:
            [0-2]: 资源分配 (CPU 频率、发射功率、平台功耗)
            [3-4]: 一级任务切分 (UD 本地、云端偏好)
            [5-9]: 二级卫星间切分 (自身 + 4 个邻居)
        """
        return [
            "[0] CPU_Ratio: CPU 频率分配比例 [0,1] → [f_min, f_max]",
            "[1] TX_Power_Ratio: 发射功率分配比例 [0,1]",
            "[2] Platform_Power_Ratio: 平台功耗管理比例 [0,1]",
            "[3] Alpha_Local: UD 本地保留比例 [0,1] (第一层切分)",
            "[4] Alpha_Cloud: 云端偏好因子 [0,1] (第二层切分偏好)",
            "[5] x_0: 自身处理比例 (归一化后)",
            "[6] x_1: 邻居 1 协作比例 (归一化后)",
            "[7] x_2: 邻居 2 协作比例 (归一化后)",
            "[8] x_3: 邻居 3 协作比例 (归一化后)",
            "[9] x_4: 邻居 4 协作比例 (归一化后)",
        ]


    def set_action(self, action: np.ndarray) -> None:
        """
        实现 ContinuousAction 的抽象方法: 解析连续动作并触发卫星逻辑。
        
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

        # 解析动作分量（顺序必须与 action_description 一致）
        cpu_ratio = action[0]              # [0] CPU 频率比例
        tx_power_ratio = action[1]         # [1] 发射功率比例
        platform_power_ratio = action[2]   # [2] 平台功耗比例
        alpha_local = action[3]            # [3] UD 本地保留比例
        alpha_cloud = action[4]            # [4] 云端偏好因子
        split_ratios = action[5:]          # [5:10] 卫星间切分比例
        
        # 1. 资源分配（本地控制）
        # self.satellite 是由 ActionBuilder 自动注入的 ComputationSatellite 实例
        self.satellite.set_resource_allocation(cpu_ratio, tx_power_ratio, platform_power_ratio)

        # 2. 层次化协作切分（一级 + 二级）
        self.satellite.schedule_collaboration_action(
            alpha_local=alpha_local,
            alpha_cloud=alpha_cloud,
            split_ratios=split_ratios,
            neighbor_satellites=self.neighbor_satellites
        )
        
        # 3. FSW 姿态控制 - 暂时禁用以避免 RW 警告
        # 任务处理流程不依赖姿态机动，可见性检查已在其他地方处理
        # if hasattr(self.satellite, 'fsw') and hasattr(self.satellite.fsw, 'action_nadir_scan'):
        #     try:
        #         self.satellite.fsw.action_nadir_scan()
        #     except Exception as e:
        #         if hasattr(self.satellite.fsw, 'action_charge'):
        #             self.satellite.fsw.action_charge()


        
    def reset_overwrite_previous(self) -> None:
        """重置动作状态，清除邻居缓存。"""
        super().reset_overwrite_previous() # <--- 新增: 调用父类方法
        # 注意: ContinuousAction 基类没有这个属性，但 Action 类有，这里保持 List.clear() 兼容性
        # 更好的写法是重新初始化，或者您当前使用 .clear() 也可以接受。
        self.neighbor_satellites.clear()
        
        
    def add_neighbor(self, neighbor: 'Satellite') -> None:
        """供环境在 reset 时调用，以填充当前卫星的协作邻居列表。"""
        self.neighbor_satellites.append(neighbor)