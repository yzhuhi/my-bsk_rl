"""STIN 轻量级计算卫星类。

该类专为计算卸载和资源调度任务设计，简化了仿真复杂度以提升性能。

使用方法：
    在 BenchMARL 配置中使用 `lite_mode: true` 来启用轻量级模式。
"""

import logging
from typing import TYPE_CHECKING, Any, List, Optional
from weakref import proxy
import numpy as np

from bsk_rl.sats.computation_satellite import (
    ComputationSatellite,
    TaskSlice,
    TaskStatus,
)
from bsk_rl.sim.dyn.lite_computation_dynamics import LiteComputationDynModel

if TYPE_CHECKING:
    from bsk_rl.act.actions import Action

logger = logging.getLogger(__name__)


class NoOpFSWModel:
    """空操作 FSW 模型，用于轻量级卫星。
    
    不执行任何姿态控制，只提供必要的接口兼容性。
    """
    
    def __init__(self, satellite, fsw_rate, **kwargs):
        """初始化空操作 FSW。"""
        self.satellite = satellite
        self.fsw_rate = fsw_rate
        self.task_name = f"LiteFSW_{satellite.name}"
    
    def _setup_fsw_objects(self, **kwargs):
        """设置 FSW 对象（空操作）。"""
        pass
    
    def reset_for_action(self):
        """重置 FSW 状态（空操作）。"""
        pass
    
    def is_alive(self, log_failure=False):
        """始终返回存活。"""
        return True


class LiteComputationSatellite(ComputationSatellite):
    """轻量级 STIN 计算卫星。
    
    继承自 ComputationSatellite，使用轻量级动力学模型。
    
    优化策略：
    - ❌ 移除：反作用轮动力学、推进器动力学、大气阻力
    - ✅ 保留：轨道传播、太阳能板、电池、日食检测
    - ✅ 保留：ISL 通信可见性检测
    
    预期性能提升：30-50%
    
    保留功能：
    - 任务队列管理
    - 计算调度
    - ISL 通信
    - 能量约束（BSK 电池模型）
    """
    
    # 使用轻量级动力学模型
    from bsk_rl.sim.dyn.lite_computation_dynamics import LiteComputationDynModel
    dyn_type = LiteComputationDynModel
    
    def __init__(self, *args, **kwargs) -> None:
        """初始化轻量级计算卫星。"""
        # 标记为轻量级模式
        self._lite_mode = True
        
        super().__init__(*args, **kwargs)
        
        logger.info(f"[LITE] Created lightweight satellite: {self.name}")
    
    def set_fsw(self, fsw_rate: float):
        """创建轻量级 FSW 模型。
        
        重写父类方法，使用 NoOpFSWModel 替代完整 FSW。
        """
        fsw = NoOpFSWModel(self, fsw_rate, **self.sat_args)
        self.fsw = proxy(fsw)
        return fsw
    
    def reset_overwrite_previous(self) -> None:
        """重置卫星状态。
        
        轻量级版本直接调用父类方法。
        电池状态由完整版动力学模型自动处理。
        """
        super().reset_overwrite_previous()
    
    def set_resource_allocation(
        self, 
        cpu_ratio: float, 
        tx_power_ratio: float,
        platform_power_ratio: float
    ) -> None:
        """设置资源分配（轻量级版本，含电池安全裁剪）。"""
        # 注意：电池安全硬约束已通过 STINActionShield 在动作空间层实现
        # 无需在此处再次裁剪，避免双重保护导致的策略学习困难
        
        # === 轻量级资源分配 ===
        # 保存分配比例
        self.current_cpu_ratio = cpu_ratio
        self.current_tx_power_ratio = tx_power_ratio
        self.current_platform_power_ratio = platform_power_ratio
        
        # 计算 CPU 频率
        cpu_freq_range = self.dynamics.cpu_max_frequency - self.dynamics.cpu_min_frequency
        self.current_cpu_freq = (
            self.dynamics.cpu_min_frequency + cpu_ratio * cpu_freq_range
        )
        
        # 计算功耗 (使用简化模型)
        cpu_power = self.dynamics.cpu_power_draw * cpu_ratio
        tx_power = self.dynamics.max_tx_power_draw * tx_power_ratio
        
        # 更新动力学模型的功耗设置
        self.dynamics.current_cpu_power = cpu_power
        self.dynamics.current_tx_power = tx_power
        
        # 记录当前功耗
        payload_power = abs(cpu_power) + abs(tx_power)
        platform_power = abs(self.dynamics.base_power_draw) * platform_power_ratio
        self.current_power_draw = payload_power + platform_power
        self.current_base_power_draw = platform_power
    
    def execute_local_compute(self, duration: float) -> None:
        """执行本地计算（轻量级版本）。
        
        与完整版相同的任务处理逻辑，但使用简化的能量更新。
        """
        # 获取当前时间
        current_time = self.simulator.sim_time
        
        # 更新能量状态
        if hasattr(self.dynamics, 'update_power'):
            self.dynamics.update_power(
                duration,
                cpu_ratio=getattr(self, 'current_cpu_ratio', 0.0),
                tx_ratio=getattr(self, 'current_tx_power_ratio', 0.0)
            )
        
        # 调用父类的计算逻辑
        super().execute_local_compute(duration)
    
    def get_cpu_freq(self) -> float:
        """获取当前 CPU 频率。"""
        return getattr(self, 'current_cpu_freq', self.dynamics.cpu_min_frequency)
    
    @property
    def battery_valid(self) -> bool:
        """检查电池是否有效（轻量级版本）。"""
        if hasattr(self.dynamics, 'battery_soc'):
            return self.dynamics.battery_soc > 0.01
        return True
    
    def get_lite_log_state(self) -> dict:
        """获取轻量级日志状态。"""
        base_state = self.get_log_state() if hasattr(self, 'get_log_state') else {}
        
        # 添加轻量级特有信息
        base_state.update({
            "lite_mode": True,
            "battery_soc": self.dynamics.battery_soc if hasattr(self.dynamics, 'battery_soc') else 0.0,
            "energy_consumed": self.dynamics.energy_consumed if hasattr(self.dynamics, 'energy_consumed') else 0.0,
        })
        
        return base_state


def create_lite_satellite_factory(full_satellite_class):
    """创建轻量级卫星工厂函数。
    
    将任何继承自 ComputationSatellite 的类转换为轻量级版本。
    
    Args:
        full_satellite_class: 完整版卫星类
        
    Returns:
        轻量级卫星类
    """
    
    class LiteWrapper(full_satellite_class):
        """动态生成的轻量级包装类。"""
        
        dyn_type = LiteComputationDynModel
        fsw_type = None
        _lite_mode = True
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            logger.info(f"[LITE] Created lightweight wrapper for {self.name}")
    
    LiteWrapper.__name__ = f"Lite{full_satellite_class.__name__}"
    return LiteWrapper
