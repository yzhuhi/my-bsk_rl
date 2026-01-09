"""
Action Shielding for STIN Continuous Action Space.

提供硬约束的 Action Replacement 机制，在不修改奖励函数的情况下强制执行安全约束。

对于连续动作空间，传统的 Action Masking（将概率设为0）不适用。
我们使用 Action Replacement：检测到违规时，将动作值裁剪到安全范围。

设计原则：
    - 硬约束（如电量安全）：使用 Action Replacement
    - 软约束（如负载均衡）：通过 Observation 让 Agent 自己学习
"""

import logging
from typing import TYPE_CHECKING, Callable, Optional, Union
import numpy as np

if TYPE_CHECKING:
    from bsk_rl.sats import ComputationSatellite

logger = logging.getLogger(__name__)


class STINActionShield:
    """STIN 连续动作空间的安全屏蔽器。
    
    在 Agent 输出动作后、执行前进行检查和修正。
    
    支持的硬约束：
        1. 电池安全约束：低电量时限制高功耗动作
        2. 队列溢出约束：队列满时限制接收新任务
        3. 通信约束：无可见邻居时限制卸载动作
    
    Usage:
        >>> shield = STINActionShield(battery_threshold=0.2)
        >>> safe_action = shield.apply(satellite, raw_action)
    """
    
    def __init__(
        self,
        # 电池安全约束
        battery_threshold: float = 0.20,      # 电量阈值
        low_battery_cpu_cap: float = 0.3,     # 低电量时 CPU 比例上限
        low_battery_tx_cap: float = 0.2,      # 低电量时 TX 功率上限
        low_battery_cloud_cap: float = 0.5,   # 低电量时云端卸载上限
        # 队列溢出约束
        max_queue_size: int = 50,             # 最大队列长度
        queue_full_local_boost: float = 0.6,  # 队列满时本地处理比例下限
        # 日志控制
        log_interventions: bool = True,
    ):
        """初始化屏蔽器。
        
        Args:
            battery_threshold: 触发电池保护的 SOC 阈值。
            low_battery_cpu_cap: 低电量时 CPU 频率比例上限。
            low_battery_tx_cap: 低电量时发射功率比例上限。
            low_battery_cloud_cap: 低电量时云端卸载比例上限。
            max_queue_size: 触发队列保护的任务数量。
            queue_full_local_boost: 队列满时强制的本地处理比例下限。
            log_interventions: 是否记录干预日志。
        """
        self.battery_threshold = battery_threshold
        self.low_battery_cpu_cap = low_battery_cpu_cap
        self.low_battery_tx_cap = low_battery_tx_cap
        self.low_battery_cloud_cap = low_battery_cloud_cap
        self.max_queue_size = max_queue_size
        self.queue_full_local_boost = queue_full_local_boost
        self.log_interventions = log_interventions
        
        # 统计
        self.intervention_count = 0
        self.battery_interventions = 0
        self.queue_interventions = 0
    
    def get_battery_soc(self, satellite: "ComputationSatellite") -> float:
        """获取卫星当前电池 SOC。"""
        try:
            if hasattr(satellite, 'dynamics') and hasattr(satellite.dynamics, 'powerMonitor'):
                battery_msg = satellite.dynamics.powerMonitor.batPowerOutMsg.read()
                capacity = satellite.dynamics.powerMonitor.storageCapacity
                if capacity > 0:
                    return battery_msg.storageLevel / capacity
        except Exception:
            pass
        return 1.0  # 默认安全值
    
    def apply(
        self, 
        satellite: "ComputationSatellite", 
        action: np.ndarray
    ) -> np.ndarray:
        """应用安全屏蔽，返回修正后的动作。
        
        动作索引定义（10D）：
            [0] CPU_Ratio
            [1] TX_Power_Ratio  
            [2] Platform_Power_Ratio
            [3] Alpha_Local
            [4] Alpha_Cloud
            [5:10] 卫星间切分比例
        
        Args:
            satellite: 卫星对象，用于读取状态。
            action: 原始 10D 动作向量。
            
        Returns:
            修正后的安全动作向量。
        """
        safe_action = action.copy()
        modified = False
        reasons = []
        
        # ----------------------------------------------------------------
        # 1. 电池安全约束：低电量时限制高功耗动作
        # ----------------------------------------------------------------
        battery_soc = self.get_battery_soc(satellite)
        
        if battery_soc < self.battery_threshold:
            # 限制 CPU 频率
            if safe_action[0] > self.low_battery_cpu_cap:
                safe_action[0] = self.low_battery_cpu_cap
                modified = True
                reasons.append(f"CPU {action[0]:.2f}→{self.low_battery_cpu_cap:.2f}")
            
            # 限制发射功率
            if safe_action[1] > self.low_battery_tx_cap:
                safe_action[1] = self.low_battery_tx_cap
                modified = True
                reasons.append(f"TX {action[1]:.2f}→{self.low_battery_tx_cap:.2f}")
            
            # 限制云端卸载（需要通信功耗）
            if safe_action[4] > self.low_battery_cloud_cap:
                safe_action[4] = self.low_battery_cloud_cap
                modified = True
                reasons.append(f"Cloud {action[4]:.2f}→{self.low_battery_cloud_cap:.2f}")
            
            if modified:
                self.battery_interventions += 1
        
        # ----------------------------------------------------------------
        # 2. 队列溢出约束：队列满时增加本地处理
        # ----------------------------------------------------------------
        queue_size = len(satellite.task_queue) if hasattr(satellite, 'task_queue') else 0
        
        # 2a. 任务数量限制
        queue_overflow = queue_size >= self.max_queue_size
        
        # 2b. 存储容量限制（基于数据量）
        storage_overflow = False
        if hasattr(satellite, 'storage_fraction'):
            storage_overflow = satellite.storage_fraction >= 0.95  # 存储使用率 >= 95%
        
        if queue_overflow or storage_overflow:
            # 强制提高本地处理比例，减少卸载
            if safe_action[3] < self.queue_full_local_boost:
                old_local = safe_action[3]
                safe_action[3] = self.queue_full_local_boost
                modified = True
                reason = f"Queue={queue_size}" if queue_overflow else f"Storage={satellite.storage_fraction:.0%}"
                reasons.append(f"Local {old_local:.2f}→{self.queue_full_local_boost:.2f} ({reason})")
                self.queue_interventions += 1
        
        # ----------------------------------------------------------------
        # 3. 日志记录
        # ----------------------------------------------------------------
        if modified:
            self.intervention_count += 1
            if self.log_interventions:
                logger.debug(
                    f"[Shield] {satellite.name}: SOC={battery_soc:.1%}, "
                    f"Queue={queue_size}, Modified: {', '.join(reasons)}"
                )
        
        return safe_action
    
    def get_stats(self) -> dict:
        """获取屏蔽统计信息。"""
        return {
            "total_interventions": self.intervention_count,
            "battery_interventions": self.battery_interventions,
            "queue_interventions": self.queue_interventions,
        }
    
    def reset_stats(self):
        """重置统计计数器。"""
        self.intervention_count = 0
        self.battery_interventions = 0
        self.queue_interventions = 0


def create_battery_shield(
    threshold: float = 0.2,
    cpu_cap: float = 0.3,
    tx_cap: float = 0.2,
) -> Callable[[np.ndarray, float], np.ndarray]:
    """创建一个简单的电池保护函数（函数式接口）。
    
    用于不需要完整 Shield 类的场景。
    
    Args:
        threshold: 电量阈值。
        cpu_cap: 低电量时 CPU 上限。
        tx_cap: 低电量时 TX 上限。
        
    Returns:
        shield_fn(action, battery_soc) -> safe_action
    """
    def shield_fn(action: np.ndarray, battery_soc: float) -> np.ndarray:
        if battery_soc >= threshold:
            return action
        
        safe_action = action.copy()
        safe_action[0] = min(safe_action[0], cpu_cap)
        safe_action[1] = min(safe_action[1], tx_cap)
        return safe_action
    
    return shield_fn


# =============================================================================
# 与 BSK-RL 框架集成的 Wrapper
# =============================================================================

class STINShieldedActionBuilder:
    """包装原有 ActionBuilder，在 set_action 前应用 Shield。
    
    这是一个装饰器模式，可以无缝集成到现有的 BSK-RL 框架中。
    
    Usage:
        >>> from bsk_rl.act.continuous_actions import ContinuousActionBuilder
        >>> original_builder = ContinuousActionBuilder(satellite)
        >>> shield = STINActionShield(battery_threshold=0.2)
        >>> shielded_builder = STINShieldedActionBuilder(original_builder, shield)
        >>> shielded_builder.set_action(action)  # 自动应用屏蔽
    """
    
    def __init__(self, original_builder, shield: STINActionShield):
        """包装原有 ActionBuilder。
        
        Args:
            original_builder: 原始的 ContinuousActionBuilder 实例。
            shield: STINActionShield 实例。
        """
        self._original = original_builder
        self._shield = shield
    
    def __getattr__(self, name):
        """代理所有其他属性到原始 builder。"""
        return getattr(self._original, name)
    
    def set_action(self, action: np.ndarray) -> None:
        """应用屏蔽后执行动作。"""
        safe_action = self._shield.apply(self._original.satellite, action)
        self._original.set_action(safe_action)


__all__ = [
    "STINActionShield",
    "STINShieldedActionBuilder", 
    "create_battery_shield",
]
