"""STIN 轻量级动力学模型。

保留能量系统（太阳能板、电池、日食检测），
移除姿态控制相关模块（反作用轮、推进器、大气阻力）。

预期性能提升：30-50%
"""

from typing import TYPE_CHECKING, Iterable, Optional
import numpy as np

from Basilisk.simulation import simplePowerSink
from Basilisk.utilities import orbitalMotion

from bsk_rl.sim.dyn.base import BasicDynamicsModel
from bsk_rl.sim.dyn.relative_motion import LOSCommDynModel
from bsk_rl.utils.functional import default_args

if TYPE_CHECKING:
    from bsk_rl.sats import Satellite


class LiteBasicDynamicsModel(BasicDynamicsModel):
    """轻量级基础动力学模型。
    
    继承自 BasicDynamicsModel，但跳过姿态控制相关模块：
    - ❌ 反作用轮动力学
    - ❌ 推进器动力学  
    - ❌ 大气阻力
    - ❌ 反作用轮功耗
    - ❌ 推进器功耗
    
    保留：
    - ✅ 轨道传播（用于 ISL 可见性）
    - ✅ 太阳能板（能量输入）
    - ✅ 电池（能量存储）
    - ✅ 日食检测（影响充电）
    - ✅ 基础功耗
    - ✅ 导航模块
    """
    
    def _setup_dynamics_objects(self, **kwargs) -> None:
        """设置动力学对象 - 跳过姿态控制模块。"""
        # ✅ 轨道传播和物理属性（必须）
        self.setup_spacecraft_hub(**kwargs)
        
        # ✅ 导航模块（用于状态读取）
        self.setup_simple_nav_object()
        
        # ✅ 能量系统（用户要求保留）
        self.setup_eclipse_object()
        self.setup_solar_panel(**kwargs)
        self.setup_battery(**kwargs)
        self.setup_power_sink(**kwargs)
        
        # ❌ 跳过姿态控制相关模块
        # self.setup_drag_effector(**kwargs)
        # self.setup_reaction_wheel_dyn_effector(**kwargs)
        # self.setup_thruster_dyn_effector()
        # self.setup_reaction_wheel_power(**kwargs)
        # self.setup_thruster_power(**kwargs)
    
    # 提供假的 wheel_speeds 属性以兼容现有代码
    @property
    def wheel_speeds(self):
        """假的轮速（轻量级模式无反作用轮）。"""
        return np.zeros(3)
    
    @property
    def wheel_speeds_fraction(self):
        """假的轮速比例。"""
        return np.zeros(3)
    
    def rw_speeds_valid(self) -> bool:
        """轻量级模式始终返回 True。"""
        return True


class LiteLOSCommDynModel(LiteBasicDynamicsModel):
    """轻量级 LOS 通信动力学模型。
    
    继承 LiteBasicDynamicsModel，添加 LOS 通信功能。
    注意：这个类用于内部继承，不直接使用。
    """
    
    def _setup_dynamics_objects(self, **kwargs) -> None:
        """设置动力学对象。"""
        super()._setup_dynamics_objects(**kwargs)
        # LOS 通信会在 LiteComputationDynModel 中设置


class LiteComputationDynModel(LOSCommDynModel):
    """轻量级计算卸载动力学模型。
    
    继承自 LOSCommDynModel（必须，用于通过类型检查）。
    重写 _setup_dynamics_objects 来跳过姿态控制模块。
    
    优化策略：
    - ❌ 移除：反作用轮、推进器、大气阻力及其功耗
    - ✅ 保留：轨道传播、能量系统、LOS 通信
    
    预期性能提升：30-50%
    """
    
    def _setup_dynamics_objects(self, **kwargs) -> None:
        """设置动力学对象 - 轻量级版本。"""
        # ✅ 轨道传播和物理属性（必须）
        self.setup_spacecraft_hub(**kwargs)
        
        # ✅ 导航模块（用于状态读取）
        self.setup_simple_nav_object()
        
        # ✅ 能量系统（用户要求保留）
        self.setup_eclipse_object()
        self.setup_solar_panel(**kwargs)
        self.setup_battery(**kwargs)
        self.setup_power_sink(**kwargs)
        
        # ✅ LOS 通信（继承自 LOSCommDynModel）
        self.setup_los_comms(**kwargs)
        
        # ✅ CPU 参数
        self._setup_cpu_parameters(**kwargs)
        
        # ❌ 跳过姿态控制相关模块
        # self.setup_drag_effector(**kwargs)
        # self.setup_reaction_wheel_dyn_effector(**kwargs)
        # self.setup_thruster_dyn_effector()
        # self.setup_reaction_wheel_power(**kwargs)
        # self.setup_thruster_power(**kwargs)
    
    @default_args(
        # CPU 参数
        cpuMinFrequency=0.1e9,
        cpuMaxFrequency=1.0e9,
        cpuWorkload=1000.0,
        cpuPowerDraw=-20.0,
        basePowerDraw=-50.0,  # 平台基础功耗
        taskMinimumElevation=0.0,
        transmitterBaudRate=-50e6,
        transmitterPowerDraw=-15.0,
        dataStorageCapacity=8e12,
    )
    def _setup_cpu_parameters(
        self,
        cpuMinFrequency: float,
        cpuMaxFrequency: float,
        cpuWorkload: float,
        cpuPowerDraw: float,
        basePowerDraw: float,
        taskMinimumElevation: float,
        transmitterBaudRate: float,
        transmitterPowerDraw: float,
        dataStorageCapacity: float,
        **kwargs
    ) -> None:
        """设置 CPU 相关参数。"""
        self.cpu_min_frequency = cpuMinFrequency
        self.cpu_max_frequency = cpuMaxFrequency
        self.cpu_workload = cpuWorkload
        self.cpu_power_draw = cpuPowerDraw
        self.base_power_draw = basePowerDraw  # 添加这个属性
        self.task_minimum_elevation = taskMinimumElevation
        self.transmitter_baud_rate = transmitterBaudRate
        self.max_tx_power_draw = transmitterPowerDraw
        self.data_storage_capacity = dataStorageCapacity
        
        # 设置 CPU 功耗节点
        self.cpuPowerSink = simplePowerSink.SimplePowerSink()
        self.cpuPowerSink.ModelTag = "cpuPowerSink" + self.satellite.name
        self.cpuPowerSink.nodePowerOut = 0.0
        self.simulator.AddModelToTask(
            self.task_name, self.cpuPowerSink, ModelPriority=896
        )
        self.powerMonitor.addPowerNodeToModel(self.cpuPowerSink.nodePowerOutMsg)
        
        # 设置发射机功耗节点（动态，根据 Agent 决策）
        self.txPowerSink = simplePowerSink.SimplePowerSink()
        self.txPowerSink.ModelTag = "txPowerSink" + self.satellite.name
        self.txPowerSink.nodePowerOut = 0.0  # 动态设置
        self.simulator.AddModelToTask(
            self.task_name, self.txPowerSink, ModelPriority=895
        )
        self.powerMonitor.addPowerNodeToModel(self.txPowerSink.nodePowerOutMsg)
        
        # 🆕 设置接收机功耗节点（静态，论文 LEO=8W）
        # 接收功率主要由 LNA 放大器决定，与数据量无关
        self.rxPowerSink = simplePowerSink.SimplePowerSink()
        self.rxPowerSink.ModelTag = "rxPowerSink" + self.satellite.name
        self.rxPowerSink.nodePowerOut = 0.0  # 初始关闭，有邻居时开启
        self.rx_power_draw = -8.0  # [W] 静态接收功率（论文 LEO=8W）
        self.simulator.AddModelToTask(
            self.task_name, self.rxPowerSink, ModelPriority=894
        )
        self.powerMonitor.addPowerNodeToModel(self.rxPowerSink.nodePowerOutMsg)
    
    @property
    def cpu_process_rate(self) -> float:
        """计算处理速率 (bits/s) = F_max / c_t。"""
        return self.cpu_max_frequency / self.cpu_workload
    
    def set_rx_power(self, enabled: bool = True) -> None:
        """开关接收机功耗。
        
        Args:
            enabled: True 开启接收功耗，False 关闭。
        """
        if hasattr(self, 'rxPowerSink') and self.rxPowerSink is not None:
            self.rxPowerSink.nodePowerOut = self.rx_power_draw if enabled else 0.0
    
    # 提供假的 storage 属性以兼容观察规格
    @property
    def storage_level(self) -> float:
        """存储级别（轻量级模式返回 0）。"""
        return 0.0
    
    @property
    def storage_level_fraction(self) -> float:
        """存储使用率（轻量级模式返回 0）。"""
        return 0.0
    
    # 提供假的 wheel_speeds 属性以兼容现有代码
    @property
    def wheel_speeds(self):
        """假的轮速（轻量级模式无反作用轮）。"""
        return np.zeros(3)
    
    @property
    def wheel_speeds_fraction(self):
        """假的轮速比例。"""
        return np.zeros(3)
    
    def rw_speeds_valid(self) -> bool:
        """轻量级模式始终返回 True。"""
        return True
