"""Models for computation satellite dynamics."""

from typing import TYPE_CHECKING

from Basilisk.simulation import simplePowerSink
from bsk_rl.sim.dyn.ground_imaging import GroundStationDynModel
from bsk_rl.sim.dyn import LOSCommDynModel
from bsk_rl.utils.functional import default_args


if TYPE_CHECKING:
    from bsk_rl.sats import Satellite


class ComputationDynModel(GroundStationDynModel):
    """
    STIN 计算卫星动力学模型。
    
    继承自 GroundStationDynModel，自动获得：
    - 电池/太阳能板 (BasicDynamicsModel) 
    - 数据存储/发射机 (ImagingDynModel)  
    - 地面站连接 (GroundStationDynModel) 设置了动态计算地面站距离
    
    新增：
    - CPU 功耗节点 (用于模拟计算耗能)
    - 平台功耗动态控制
    - 任务接入仰角设置
    Note:
        卫星间通信 (ISL) 由环境层 `communicator` 参数配置，
        如需 LOS 通信检测，可创建子类多继承 LOSCommDynModel。
    """

    def __init__(self, *args, **kwargs) -> None:
        """初始化计算动力学模型。
        
        Args:
            satellite: 卫星实例引用。
            dyn_rate: [s] 动力学仿真步长。
            kwargs: 包含 SMEC 配置的参数字典。
        """
        super().__init__(*args, **kwargs)
        
        # 保存 SMEC 参数供卫星层访问
        self.cpu_min_frequency = kwargs.get("cpuMinFrequency", 0.1e9)
        self.cpu_max_frequency = kwargs.get("cpuMaxFrequency", 1.0e9)
        self.cpu_workload = kwargs.get("cpuWorkload", 1000.0)
        self.cpu_power_draw = kwargs.get("cpuPowerDraw", -20.0)
        self.task_minimum_elevation = kwargs.get("taskMinimumElevation", 0.0)
        
        # 初始化 CPU 引用占位符
        self.cpuPowerSink = None

    def _setup_dynamics_objects(self, **kwargs) -> None:
        """
        初始化所有动力学对象。
        先调用父类初始化基础组件 (电池, 存储, 发射机, LOS检测等)，
        然后初始化我们独有的 CPU 组件。
        """
        super()._setup_dynamics_objects(**kwargs)
        self.setup_cpu_power_sink(**kwargs)

    @default_args(
        # --- 计算模块参数 (SMEC Config) ---
        cpuMinFrequency=0.1e9,        # [Hz] 最小 CPU 频率 (避免 f=0)
        cpuMaxFrequency=1.0e9,        # [Hz] F: 最大 CPU 频率
        cpuWorkload=1000.0,           # [cycles/bit] c_t: 默认工作负载
        cpuPowerDraw=-20.0,           # [W] CPU 满载功耗
        # --- 任务接入参数 ---
        taskMinimumElevation=0.0,     # [rad] 任务可见性最小仰角 (0 = 地平线)
    )
    def setup_cpu_power_sink(
        self, 
        cpuMinFrequency: float,
        cpuMaxFrequency: float,
        cpuWorkload: float,
        cpuPowerDraw: float,
        taskMinimumElevation: float,
        priority: int = 890, 
        **kwargs
    ) -> None:
        """设置 CPU 功耗负载节点。

        当执行计算任务时，可以通过设置 nodePowerOut 开启此负载。

        Args:
            cpuMinFrequency: [Hz] 最小 CPU 频率。
            cpuMaxFrequency: [Hz] 最大 CPU 频率。
            cpuWorkload: [cycles/bit] 计算复杂度。
            cpuPowerDraw: [W] CPU 满载功耗。
            taskMinimumElevation: [rad] 任务可见性最小仰角。
            priority: 模型优先级。
            kwargs: 传递给其他设置函数的参数。
        """
        # 保存 SMEC 参数供卫星层访问
        self.cpu_min_frequency = cpuMinFrequency
        self.cpu_max_frequency = cpuMaxFrequency
        self.cpu_workload = cpuWorkload
        self.cpu_power_draw = cpuPowerDraw
        self.task_minimum_elevation = taskMinimumElevation
        
        if self.cpu_power_draw > 0:
            self.logger.warning(
                "cpuPowerDraw should probably be zero or negative (consuming power)."
            )
            
        # 1. 实例化 Basilisk 的 SimplePowerSink 模块
        self.cpuPowerSink = simplePowerSink.SimplePowerSink()
        self.cpuPowerSink.ModelTag = "cpuPowerSink" + self.satellite.name
        
        # 2. 设置默认功耗状态 (默认为 0/关闭，由 FSW 或 Action 在运行时修改)
        self.cpuPowerSink.nodePowerOut = 0.0 
        
        # 3. 将模块加入仿真任务循环
        self.simulator.AddModelToTask(
            self.task_name, self.cpuPowerSink, ModelPriority=priority
        )
        
        # 4. 关键：将此负载连接到电池系统 (powerMonitor)
        if self.powerMonitor:
            self.powerMonitor.addPowerNodeToModel(self.cpuPowerSink.nodePowerOutMsg)
        else:
            self.logger.error("Battery (powerMonitor) not initialized before CPU setup!")
        
        # 🆕 设置接收机功耗节点（静态，论文 LEO=8W）
        self.rxPowerSink = simplePowerSink.SimplePowerSink()
        self.rxPowerSink.ModelTag = "rxPowerSink" + self.satellite.name
        self.rxPowerSink.nodePowerOut = 0.0  # 初始关闭
        self.rx_power_draw = -8.0  # [W] 静态接收功率（论文 LEO=8W）
        self.simulator.AddModelToTask(
            self.task_name, self.rxPowerSink, ModelPriority=priority - 1
        )
        if self.powerMonitor:
            self.powerMonitor.addPowerNodeToModel(self.rxPowerSink.nodePowerOutMsg)

    def set_cpu_power(self, power: float) -> None:
        """动态设置 CPU 功耗。
        
        Args:
            power: [W] 功耗值（应为负值表示消耗）。
        """
        if self.cpuPowerSink is not None:
            self.cpuPowerSink.nodePowerOut = power

    def set_base_power(self, power: float) -> None:
        """动态设置平台基础功耗。
        
        Args:
            power: [W] 功耗值（应为负值表示消耗）。
        """
        if hasattr(self, 'basePowerSink') and self.basePowerSink is not None:
            self.basePowerSink.nodePowerOut = power
    
    def set_rx_power(self, enabled: bool = True) -> None:
        """开关接收机功耗。
        
        Args:
            enabled: True 开启接收功耗，False 关闭。
        """
        if hasattr(self, 'rxPowerSink') and self.rxPowerSink is not None:
            self.rxPowerSink.nodePowerOut = self.rx_power_draw if enabled else 0.0

    @property
    def cpu_process_rate(self) -> float:
        """计算处理速率 (bits/s) = F_max / c_t。"""
        return self.cpu_max_frequency / self.cpu_workload
    


class LoSComputationDynaModel(ComputationDynModel, LOSCommDynModel):
    '''
        由于继承了LOScommDynMode,卫星运行的时候会进行可见性分析，
        那么我们一开始设置的4颗邻居卫星就可能会出现错误，
        因此，这里我们进行了相应的优化：
        动作空间维度保持固定（10D），RL 训练友好
        多余的比例自动回流到本地处理（自身）
        Agent 不需要"知道"当前有几个邻居可见，它只需要学习"如果这个比例分给不存在的邻居，就等于分给自己"
        
        实际邻居数	动作空间	处理方式
        4	[x_0, x_1, x_2, x_3, x_4]	正常分配到 5 个节点
        3	[x_0, x_1, x_2, x_3, x_4]	x_4 的比例加到 x_0（自身）  
        2	[x_0, x_1, x_2, x_3, x_4]	x_3 + x_4 加到 x_0
        1	[x_0, x_1, x_2, x_3, x_4]	x_2~x_4 全加到 x_0
        0	[x_0, x_1, x_2, x_3, x_4]	只有自己，x_0=1.0
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
