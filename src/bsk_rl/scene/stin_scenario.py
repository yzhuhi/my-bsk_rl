"""
STINScenario: 定义 STIN 场景中的计算任务分布。

设计说明:
    - 继承自 Scenario 基类（参考 UniformTargets 的设计）
    - 只负责生成计算任务（用户终端请求），不负责卫星轨道配置
    - 卫星轨道配置应通过 `sat_arg_randomizer=walker_delta_args(...)` 完成
    - 计算任务包含：数据量、计算复杂度、最大时延等属性

使用示例:
    >>> from bsk_rl import ConstellationTasking
    >>> from bsk_rl.utils.orbital import walker_delta_args
    >>> from bsk_rl.scene import STINTaskScenario
    >>> from bsk_rl.sats import ComputationSatellite
    >>>
    >>> env = ConstellationTasking(
    ...     satellites=[ComputationSatellite() for _ in range(6)],
    ...     sat_arg_randomizer=walker_delta_args(  # 星座在这里配置！
    ...         n_planes=1,
    ...         altitude=800,
    ...         inc=60,
    ...         clusterspacing=5,
    ...     ),
    ...     scenario=STINTaskScenario(n_tasks=10),  # 场景只负责任务
    ... )
"""

import logging
from typing import TYPE_CHECKING, Callable, Optional, Union
from dataclasses import dataclass

import numpy as np
from scipy.stats import truncnorm
from Basilisk.utilities import orbitalMotion

from bsk_rl.scene import Scenario
from bsk_rl.utils import vizard
from bsk_rl.utils.orbital import lla2ecef
from bsk_rl.utils.constants import DEFAULT_UPLINK_RATE, DEFAULT_FIBER_DISTANCE, DEFAULT_CLOUD_CPU_FREQ, DEFAULT_UD_CPU_FREQ

if TYPE_CHECKING:
    from bsk_rl.sats import Satellite

logger = logging.getLogger(__name__)


@dataclass
class ComputationTask:
    """STIN 场景中的计算任务。
    
    表示用户终端（UD）发起的计算请求。
    
    时延模型所需属性:
        - origin_position: 任务来源位置 (= r_LP_P)
        - uplink_distance: 上行链路距离（需要在任务分配时计算）
        
    并行路径时延模型:
        T_total = max(T_UD, T_SAT, T_CLOUD)
        
        - T_UD: UD 本地处理时延 = (alpha_local × data × workload) / f_ud
        - T_SAT: 卫星路径时延 = T_tx_up(α_sat) + T_prop_up + T_queue + T_compute + T_isl + T_prop_down
        - T_CLOUD: 云端路径时延 = T_tx_up(α_cloud) + T_tx_sgl(α_cloud) + 2×(T_prop_up + T_prop_sgl + T_fiber) + T_compute_cloud
    """
    name: str
    r_LP_P: np.ndarray        # [m] 任务来源位置（Planet-fixed）
    data_size: float          # [bits] 输入数据量
    workload: float           # [cycles/bit] 计算复杂度
    max_delay: float          # [s] 最大容忍时延
    priority: float           # 优先级 [0, 1]，详见下方说明 / 12-26 改动
    # ------------------------------------------------------------------
    # priority 字段设计说明：
    # 
    # ✅ 已实现功能：
    #   - 可视化标记大小 (markerScale = sqrt(priority))
    #   - 任务到达优先级：高优先级任务更可能先到达 (process_task_arrivals)
    #   - 分配优先级：高优先级任务分配给资源更充足的卫星 (assign_tasks_to_satellites)
    #   - Agent 观测：current_task_priority 已加入观测空间 (41D)
    #   - 奖励函数：超时惩罚已使用 priority 加权
    # 
    # 🔮 Future Work（卫星层优先级队列）：
    #   - 卫星端按优先级处理：在 task_queue 中使用优先级队列替代 FIFO
    #     实现方式：max(task_queue, key=lambda t: t.priority) 或 heapq
    #   - 奖励函数权重：完成高优先级任务获得更高奖励 （已实现）
    #   - 分级处理 (SLA tiers)：根据优先级设置不同的 max_delay 乘数
    #   - 抛弃策略：资源紧张时优先抛弃低优先级任务
    # ------------------------------------------------------------------
    uplink_rate: float = DEFAULT_UPLINK_RATE  # [bps] 上行传输速率
    
    # 云端路径参数
    fiber_distance: float = DEFAULT_FIBER_DISTANCE    # [m] 网关到云服务器的光纤距离，默认 500 km
    cloud_cpu_freq: float = DEFAULT_CLOUD_CPU_FREQ       # [Hz] 云端 CPU 频率，默认 10 GHz
    
    # UD 本地参数
    ud_cpu_freq: float = DEFAULT_UD_CPU_FREQ           # [Hz] UD 本地 CPU 频率，默认 1 GHz
    
    @property
    def origin_position(self) -> np.ndarray:
        """任务来源位置别名（用于 TaskSlice 兼容性）。"""
        return self.r_LP_P
    
    @property
    def id(self) -> str:
        """获取唯一标识符。"""
        try:
            return self._id
        except AttributeError:
            self._id = f"{self.name}_{id(self)}"
            return self._id
    
    @property
    def task_id(self) -> int:
        """数值型任务 ID（从名称中提取）。"""
        try:
            return int(self.name.split('-')[-1])
        except (ValueError, IndexError):
            return hash(self.name) % 100000
    
    @property
    def total_cycles(self) -> float:
        """计算任务所需的总 CPU 周期数。"""
        return self.data_size * self.workload
    
    def __repr__(self) -> str:
        return f"ComputationTask({self.name}, {self.data_size/1e6:.1f}Mb)"


class STINTaskScenario(Scenario):
    """STIN 计算任务场景：定义计算任务的分布。
    
    任务属性包括：
    - 地理位置（用户终端位置）
    - 数据量分布 截断正态分布
    - 计算复杂度分布
    - 时延要求分布
    
    注意：卫星轨道配置应通过 `sat_arg_randomizer` 参数完成，
    而不是在此场景中定义。
    """

    def __init__(
        self,
        n_tasks: Union[int, tuple[int, int]] = 10,
        data_size_range: tuple[float, float] = (1e6, 10e6),  # (a, b) 截断范围 [bits]
        data_size_mean: Optional[float] = None,  # 均值 μ，默认为范围中心
        data_size_std: Optional[float] = None,   # 标准差 σ，默认为范围的1/4
        workload_range: tuple[float, float] = (100, 1000),    # cycles/bit
        max_delay_range: tuple[float, float] = (1.0, 10.0),   # 1-10 seconds
        priority_distribution: Optional[Callable] = None, # 12-26 改动：优先级生成函数被真正输入
        task_arrival_rate: float = 0.1,  # 任务到达率 (tasks/second)
        radius: float = orbitalMotion.REQ_EARTH * 1e3,
        # --- 位置模糊化参数（模拟 GPS 误差/用户移动）---
        enable_location_uncertainty: bool = False,  # 是否启用位置模糊化
        location_uncertainty_radius: float = 5000.0,  # [m] 位置不确定性半径（高斯标准差）
    ) -> None:
        """初始化 STIN 计算任务场景。
        
        Args:
            n_tasks: 每个 episode 的任务数量，可以是固定值或范围 (low, high)。
            data_size_range: [bits] 任务数据量截断范围 (a, b)，对应论文公式 (3)。
            data_size_mean: [bits] 数据量均值 μ，默认为 (a+b)/2。
            data_size_std: [bits] 数据量标准差 σ，默认为 (b-a)/4。
            workload_range: [cycles/bit] 计算复杂度范围。
            max_delay_range: [s] 最大时延要求范围。
            priority_distribution: 优先级生成函数，默认为 uniform(0, 1)。 
            task_arrival_rate: [tasks/s] 任务到达率（泊松过程）。
            radius: [m] 任务来源位置的地球半径。
            enable_location_uncertainty: 是否启用位置模糊化模式。
            location_uncertainty_radius: [m] 位置不确定性半径（高斯 1σ）。
        """
        super().__init__()
        self._n_tasks = n_tasks
        self.data_size_range = data_size_range
        # 截断正态分布参数 (论文公式 3) 12-26 改动：允许自定义均值和标准差
        a, b = data_size_range
        self.data_size_mean = data_size_mean if data_size_mean is not None else (a + b) / 2
        self.data_size_std = data_size_std if data_size_std is not None else (b - a) / 4
        
        self.workload_range = workload_range
        self.max_delay_range = max_delay_range
        self.task_arrival_rate = task_arrival_rate
        self.radius = radius
        
        # 动态到达率参数（默认静态）
        self.dynamic_arrival_mode = 'static'  # 'static', 'sinusoidal', 'burst', 'time_of_day'
        self.arrival_amplitude = 0.3          # 正弦波幅度（相对基线）
        self.arrival_period = 1800.0          # 正弦波周期 [s]
        self.burst_probability = 0.01         # 突发事件概率
        self.burst_multiplier = 3.0           # 突发事件倍率
        
        # 位置模糊化参数
        self.enable_location_uncertainty = enable_location_uncertainty
        self.location_uncertainty_radius = location_uncertainty_radius
        
        if priority_distribution is None:
            priority_distribution = lambda: np.random.rand()
        self.priority_distribution = priority_distribution
        
        # 任务列表
        self.tasks: list[ComputationTask] = []
        # 泊松到达机制：任务池（待到达）和已到达队列
        self.task_pool: list[ComputationTask] = []  # 待到达任务池
        self.arrived_tasks: list[ComputationTask] = []  # 已到达任务队列

    def get_arrival_rate(self, sim_time: float) -> float:
        """根据仿真时间计算动态任务到达率。
        
        Args:
            sim_time: 当前仿真时间 [s]
            
        Returns:
            当前时刻的任务到达率 [tasks/s]
        """
        base_rate = self.task_arrival_rate
        
        if self.dynamic_arrival_mode == 'static':
            return base_rate
        
        elif self.dynamic_arrival_mode == 'sinusoidal':
            # 正弦波动：rate = base × (1 + amplitude × sin(2π × t / period))
            phase = 2 * np.pi * sim_time / self.arrival_period
            factor = 1 + self.arrival_amplitude * np.sin(phase)
            return base_rate * max(0.1, factor)  # 最低 10% 基线
        
        elif self.dynamic_arrival_mode == 'burst':
            # 突发模式：随机触发突发事件
            if np.random.rand() < self.burst_probability:
                return base_rate * self.burst_multiplier
            return base_rate
        
        elif self.dynamic_arrival_mode == 'time_of_day':
            # 分时段模式：模拟早晚高峰
            # 假设一个轨道周期约 5400s，分成 6 个时段
            period = 6000.0  # 轨道周期
            phase = (sim_time % period) / period  # [0, 1)
            
            if 0.1 <= phase < 0.3:      # 早高峰
                return base_rate * 1.5
            elif 0.6 <= phase < 0.8:    # 晚高峰
                return base_rate * 1.8
            elif 0.8 <= phase < 0.9:    # 夜间低谷
                return base_rate * 0.5
            else:                        # 平时
                return base_rate
        
        else:
            return base_rate

    def reset_overwrite_previous(self) -> None:
        """清除上一个 episode 的任务列表。"""
        self.tasks = []
        self.task_pool = []
        self.arrived_tasks = []

    def reset_pre_sim_init(self) -> None:
        """在仿真初始化前生成任务集。

        注意：任务生成后放入task_pool，等待泊松过程动态到达
        """
        # 确定任务数量
        if isinstance(self._n_tasks, int):
            self.n_tasks = self._n_tasks
        else:
            self.n_tasks = np.random.randint(self._n_tasks[0], self._n_tasks[1] + 1) # yaml中改成序列就行 [1000,10000]

        logger.info(f"Generating {self.n_tasks} computation tasks for Poisson arrival")
        self._regenerate_tasks()
        
        # 将所有任务放入待到达池
        self.task_pool = self.tasks.copy()
        self.arrived_tasks = []
        
        # 注意：不再在这里注册任务的access checking
        # 任务将在到达时动态注册（在process_task_arrivals中）

    def reset_during_sim_init(self) -> None:
        """在仿真初始化期间可视化任务。"""
        for task in self.tasks:
            self._visualize_task(task)

    @vizard.visualize
    def _visualize_task(self, task, vizSupport=None, vizInstance=None):
        """在 Vizard 中可视化任务位置。
        
        可视化规则（参考 Agile EOS 视频风格）：
            - 任务大小 (markerScale) → 表示数据量（数据越大，标记越大）
            - 任务颜色 (color) → 表示优先级（绿色=低，黄色=中，红色=高）
            - 任务完成后 → 颜色变蓝（通过 mark_task_completed 方法）
        
        Args:
            task: 要可视化的 ComputationTask 对象。
            vizSupport: Vizard 支持模块（由装饰器注入）。
            vizInstance: Vizard 实例（由装饰器注入）。
        """
        # 优先级到颜色的映射（绿→黄→红渐变）
        color_rgba = self._priority_to_color(task.priority, vizSupport)
        
        # 数据量到大小的映射
        # 标准化到 [0.3, 1.5] 范围，避免太小或太大
        data_min, data_max = self.data_size_range
        normalized_size = (task.data_size - data_min) / (data_max - data_min + 1e-6)
        marker_scale = 0.3 + normalized_size * 1.2  # [0.3, 1.5]
        
        vizSupport.addLocation(
            vizInstance,
            stationName=task.name,           # 任务名称（显示在 Vizard 中）
            parentBodyName="earth",          # 附着在地球上
            r_GP_P=list(task.r_LP_P),        # 任务在地球固连系中的位置 [m]
            fieldOfView=np.arctan(500/800),  # 视场角（约 32°）
            color=color_rgba,                # 优先级颜色（绿→红渐变）
            range=1000.0 * 1000,             # 可见范围 1000 km
            markerScale=marker_scale,        # 标记大小 = 数据量映射
        )
        if vizInstance.settings.showLocationCones == 0:
            vizInstance.settings.showLocationCones = -1   # 禁用锥形范围显示

        if vizInstance.settings.showLocationCommLines == 0:
            vizInstance.settings.showLocationCommLines = 1  # 启用蓝线

        if vizInstance.settings.showLocationLabels == 0:
            vizInstance.settings.showLocationLabels = -1  # 禁用标签显示
    
    def _priority_to_color(self, priority: float, vizSupport) -> list:
        """将优先级 [0, 1] 映射到颜色（绿→黄→红渐变）。
        
        颜色方案：
            - priority = 0.0: 绿色 (0, 255, 0)
            - priority = 0.5: 黄色 (255, 255, 0)
            - priority = 1.0: 红色 (255, 0, 0)
        
        Args:
            priority: 任务优先级 [0, 1]
            vizSupport: Vizard 支持模块
            
        Returns:
            RGBA255 颜色列表 [R, G, B, A]
        """
        priority = max(0.0, min(1.0, priority))  # 裁剪到 [0, 1]
        
        if priority < 0.5:
            # 绿→黄：G 保持 255，R 从 0 增到 255
            t = priority * 2  # [0, 1]
            r = int(255 * t)
            g = 255
            b = 0
        else:
            # 黄→红：R 保持 255，G 从 255 减到 0
            t = (priority - 0.5) * 2  # [0, 1]
            r = 255
            g = int(255 * (1 - t))
            b = 0
        
        return [r, g, b, 255]  # RGBA255 格式
    
    @vizard.visualize
    def mark_task_completed(self, task, vizSupport=None, vizInstance=None):
        """将任务标记为已完成（颜色变为蓝色）。
        
        任务完成后调用此方法，会在 Vizard 中将该任务的标记颜色更新为蓝色。
        
        注意：Vizard 的 addLocation 不支持直接更新颜色，
        但可以通过 liveSettings 或重新添加 location 来实现更新。
        这里我们使用 createTargetLine 从任务连接到地心来表示"已完成"状态。
        
        Args:
            task: 已完成的 ComputationTask 对象
            vizSupport: Vizard 支持模块（由装饰器注入）
            vizInstance: Vizard 实例（由装饰器注入）
        """
        try:
            # 方法 1: 用 createTargetLine 画一条蓝色目标线表示完成
            # 由于 addLocation 不能动态更新颜色，我们用目标线覆盖显示完成状态
            vizSupport.createTargetLine(
                vizInstance,
                toBodyName=task.name,      # 目标：任务 location
                lineColor="blue",          # 蓝色表示已完成
                fromBodyName="earth",      # 从地心（或可用其他参考点）
            )
        except Exception as e:
            # 静默失败，不影响仿真
            pass

    def _regenerate_tasks(self) -> None:
        """生成均匀分布的计算任务。
        
        可以在子类中重写此方法以实现其他分布（如城市分布）。
        """
        self.tasks = []
        
        # 计算截断正态分布的标准化边界 (论文公式 3) 12-26 改动
        a, b = self.data_size_range
        mu, sigma = self.data_size_mean, self.data_size_std
        # ✅ 防止除零：sigma 至少为 1.0
        sigma = max(sigma, 1.0)
        a_std = (a - mu) / sigma  # 标准化下界
        b_std = (b - mu) / sigma  # 标准化上界
        
        for i in range(self.n_tasks):
            # 均匀分布的地理位置（基准位置）
            x = np.random.normal(size=3)
            x *= self.radius / np.linalg.norm(x)
            
            # 位置模糊化：添加高斯扰动模拟 GPS 误差/用户移动
            if self.enable_location_uncertainty:
                # 在切平面内添加扰动（保持在地球表面）
                perturbation = np.random.normal(0, self.location_uncertainty_radius, 3)
                x = x + perturbation
                x *= self.radius / np.linalg.norm(x)  # 投影回地球表面
            
            # 截断正态分布采样数据量 (论文公式 3)
            data_size = truncnorm.rvs(a_std, b_std, loc=mu, scale=sigma)
            data_size = np.clip(data_size, a, b)  # 确保在范围内
            
            # 其他属性仍用均匀分布
            workload = np.random.uniform(*self.workload_range)
            max_delay = np.random.uniform(*self.max_delay_range)
            
            task = ComputationTask(
                name=f"task-{i}",
                r_LP_P=x,
                data_size=data_size,
                workload=workload,
                max_delay=max_delay,
                priority=self.priority_distribution(),
            )
            self.tasks.append(task)

    def sample_task(self) -> Optional[ComputationTask]:
        """根据优先级加权采样一个任务。
        
        **当前状态**: 已实现但未被调用（保留接口）。
        
        **设计用途**: 
            实现优先级驱动的任务选择，高优先级任务被选中的概率更高。
        
        **启用方法**:
            在 process_task_arrivals() 中使用此方法替代随机选择::
            
                # 当前代码（随机选择）:
                arrived_indices = np.random.choice(len(self.task_pool), ...)
                
                # 改为优先级驱动:
                for _ in range(actual_arrivals):
                    task = self.sample_task_from_pool()  # 需要实现此变体
                    ...
        
        Returns:
            采样的任务，如果没有任务则返回 None。
            
        Note:
            采样概率 P(task_i) = priority_i / sum(priorities)
        """
        if not self.tasks:
            return None
        
        # 按优先级加权采样
        priorities = np.array([t.priority for t in self.tasks])
        probabilities = priorities / priorities.sum()
        idx = np.random.choice(len(self.tasks), p=probabilities)
        return self.tasks[idx]
    
    def process_task_arrivals(self, step_duration: float) -> int:
        """ 改
        处理当前步的任务到达（泊松过程）。
        
        根据task_arrival_rate和step_duration，使用泊松分布采样本步到达的任务数，
        并将任务从task_pool移到arrived_tasks，同时注册到卫星的access checking。
        目前实现：
        1.任务到达优先级 高优先级任务更可能先从 task_pool 进入 arrived_tasks
        2.任务分配优先级 高优先级任务更可能先从 arrived_tasks 分配给卫星（已经设置了一个计算分数的函数）
        3.未实现：卫星处理任务的优先级队列，在任务上传的阶段设置了优先级
        Args:
            step_duration: 当前步的时长(秒)
        
        Returns:
            本步到达的任务数

        Note: 
            如果lambda是静态的，稳态流量；如果是动态的，存在高峰低谷模式（增加了系统性的事变趋势）；

            纯泊松（static）:
            rate (恒定 1.4)
            
            正弦波动（sinusoidal）:
            rate (在 1.0-1.8 之间波动)
            
            分时段（time_of_day）:
            rate (早晚高峰)

            纯泊松：已经有随机性，适合稳态场景
            动态到达率：增加系统性的时变趋势（如早晚高峰）
        """
        if len(self.task_pool) == 0:
            return 0
        
        # 获取当前仿真时间（从任意卫星的 simulator 获取）
        current_sim_time = 0.0
        if self.satellites and hasattr(self.satellites[0], 'simulator'):
            current_sim_time = self.satellites[0].simulator.sim_time
        
        # 计算期望到达数：λ = task_arrival_rate (tasks/s) × step_duration (s)
        current_rate = self.get_arrival_rate(current_sim_time)
        expected_arrivals = current_rate * step_duration
        
        # 泊松采样实际到达数
        actual_arrivals = np.random.poisson(expected_arrivals)
        # 不能超过任务池剩余数量
        actual_arrivals = min(actual_arrivals, len(self.task_pool))
        
        # 从任务池中按优先级加权选择到达的任务 (创新点: 优先级驱动)
        if actual_arrivals > 0:
            # 按优先级加权采样（高优先级任务更可能先到达）
            priorities = np.array([t.priority for t in self.task_pool])
            probabilities = priorities / priorities.sum()
            arrived_indices = np.random.choice(
                len(self.task_pool), 
                size=actual_arrivals, 
                replace=False,
                p=probabilities
            )
            
            # 从后往前删除，避免索引变化
            for idx in sorted(arrived_indices, reverse=True):
                task = self.task_pool.pop(idx)
                self.arrived_tasks.append(task)
                
                # 注册任务到所有卫星的access checking
                for satellite in self.satellites:
                    if hasattr(satellite, "add_location_for_access_checking"):
                        min_elev = satellite.sat_args_generator.get(
                            "taskMinimumElevation",
                            satellite.sat_args_generator.get(
                                "imageTargetMinimumElevation", 0.0
                            )
                        )
                        satellite.add_location_for_access_checking(
                            object=task,
                            r_LP_P=task.r_LP_P,
                            min_elev=min_elev,
                            type="task",
                        )
            
            # 🔍 调试信息：每500个任务或任务池<500时输出
            if actual_arrivals > 0 and (len(self.task_pool) % 500 == 0 or len(self.task_pool) < 500):
                # 计算实际到达率
                actual_rate = len(self.arrived_tasks) / max(1, self.satellites[0].simulator.sim_time) if self.satellites else 0
                logger.debug(
                    f"[Poisson] +{actual_arrivals} tasks | "
                    f"Pool: {len(self.task_pool)} | "
                    f"Arrived: {len(self.arrived_tasks)} | "
                    f"Config rate: {self.task_arrival_rate:.2f}/s | "
                    f"Actual rate: {actual_rate:.2f}/s"
                )
        
        return actual_arrivals
    
    def assign_tasks_to_satellites(self) -> None:
        """ 改
        将任务分配给具有可见性的卫星（优先级驱动）。
        
        创新点：
        1. 按优先级排序处理任务（高优先级任务优先分配）
        2. 优先级感知的卫星选择（高优先级任务分配给资源更充足的卫星）
        """
        current_time = self.satellites[0].simulator.sim_time
        
        # 统计本次分配
        assigned_count = 0
        
        # 按优先级降序排列待分配任务（高优先级优先）
        pending_tasks = [
            t for t in self.arrived_tasks 
            if not (hasattr(t, '_assigned') and t._assigned)
        ]
        pending_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        for task in pending_tasks:
            # 从任务获取 UD 位置
            task_position = getattr(task, 'r_LP_P', None)
            if task_position is None:
                continue
                
            # 找到能看到这个任务的卫星列表
            visible_satellites = []
            for satellite in self.satellites:
                if not hasattr(satellite, 'receive_task_from_scenario'):
                    continue
                if not hasattr(satellite, '_check_ud_visibility'):
                    continue
                    
                # 检查可见性
                if satellite._check_ud_visibility(task_position):
                    visible_satellites.append(satellite)
            
            # 如果有可见的卫星，根据优先级选择最优卫星
            if visible_satellites:
                def _random_tiebreak(candidates, scores):
                    scores = np.asarray(scores, dtype=np.float64)
                    min_score = np.min(scores)
                    idxs = np.where(np.isclose(scores, min_score, rtol=0.0, atol=1e-9))[0]
                    return candidates[int(np.random.choice(idxs))]

                # 高优先级任务：选择资源最充足的卫星（队列短 + 电量高）
                # 低优先级任务：选择队列最短的卫星（简单负载均衡）
                if task.priority > 0.5:  # 高优先级阈值
                    # 综合评分：队列越短越好，电量越高越好
                    def sat_score(s):
                        queue_len = len(s.task_queue) if hasattr(s, 'task_queue') else 0
                        battery_ratio = getattr(s, 'battery_charge_ratio', 1.0)
                        return queue_len - battery_ratio * 10  # 电量每10%抵消1个队列任务
                    scores = [sat_score(s) for s in visible_satellites]
                    best_sat = _random_tiebreak(visible_satellites, scores)
                else:
                    # 普通任务：简单负载均衡
                    scores = [
                        len(s.task_queue) if hasattr(s, 'task_queue') else 0
                        for s in visible_satellites
                    ]
                    best_sat = _random_tiebreak(visible_satellites, scores)
                best_sat.receive_task_from_scenario(task)
                task._assigned = True
                assigned_count += 1
        
        if assigned_count > 0:
            logger.debug(
                f"[DIAG] ✓ Assigned {assigned_count} tasks (priority-driven) at t={current_time:.2f}s"
            )



class CityTaskScenario(STINTaskScenario):
    """基于城市分布的 STIN 计算任务场景。
    
    任务来源于人口密集区域（城市），更符合实际的用户终端分布。
    """

    def __init__(
        self,
        n_tasks: Union[int, tuple[int, int]] = 10,
        n_select_from: Optional[int] = 100,  # 从前 N 个最大城市中选择
        location_offset: float = 50000,  # [m] 位置偏移
        city_task_ratio: float = 1.0,  # 城市任务占比（其余为均匀分布）
        **kwargs,
    ) -> None:
        """初始化城市分布的计算任务场景。
        
        Args:
            n_tasks: 任务数量。
            n_select_from: 从最大的 N 个城市中采样，None 表示全部。
            location_offset: [m] 任务位置相对城市中心的随机偏移。
            city_task_ratio: 城市任务占比（其余为均匀分布）。
            **kwargs: 传递给 STINTaskScenario 的其他参数。
        """
        super().__init__(n_tasks=n_tasks, **kwargs)
        self.n_select_from = n_select_from
        self.location_offset = location_offset
        self.city_task_ratio = city_task_ratio

    def _regenerate_tasks(self) -> None:
        """基于城市分布生成计算任务。"""
        import os
        import sys
        from pathlib import Path
        import pandas as pd
        
        self.tasks = []
        
        # 加载城市数据库
        cities = pd.read_csv(
            Path(os.path.realpath(__file__)).parent.parent
            / "_dat"
            / "simplemaps_worldcities"
            / "worldcities.csv",
        )
        
        n_select = self.n_select_from
        if n_select is None or n_select > len(cities):
            n_select = len(cities)
        
        # 计算截断正态分布的标准化边界 (论文公式 3)
        a, b = self.data_size_range
        mu, sigma = self.data_size_mean, self.data_size_std
        # ✅ 防止除零：sigma 至少为 1.0
        sigma = max(sigma, 1.0)
        a_std = (a - mu) / sigma
        b_std = (b - mu) / sigma
        
        # 任务分布：城市占比 + 均匀分布（海上等）
        city_ratio = float(self.city_task_ratio) if self.city_task_ratio is not None else 1.0
        city_ratio = max(0.0, min(1.0, city_ratio))
        n_city = int(round(self.n_tasks * city_ratio))
        n_city = min(max(n_city, 0), self.n_tasks)
        n_uniform = self.n_tasks - n_city

        task_idx = 0

        def _sample_task(location, name_prefix: str) -> None:
            nonlocal task_idx
            # 添加随机偏移（城市内部的空间分布）
            loc = np.array(location, dtype=np.float64)
            # 位置模糊化：添加高斯扰动模拟 GPS 误差/用户移动
            if self.enable_location_uncertainty:
                perturbation = np.random.normal(0, self.location_uncertainty_radius, 3)
                loc = loc + perturbation
                loc *= self.radius / np.linalg.norm(loc)  # 投影回地球表面

            # 截断正态分布采样数据量 (论文公式 3)
            data_size = truncnorm.rvs(a_std, b_std, loc=mu, scale=sigma)
            data_size = np.clip(data_size, a, b)

            # 其他属性仍用均匀分布
            workload = np.random.uniform(*self.workload_range)
            max_delay = np.random.uniform(*self.max_delay_range)

            task = ComputationTask(
                name=f"{name_prefix}-{task_idx}",
                r_LP_P=loc,
                data_size=data_size,
                workload=workload,
                max_delay=max_delay,
                priority=self.priority_distribution(),
            )
            self.tasks.append(task)
            task_idx += 1

        # 城市任务：允许重复采样
        if n_city > 0 and n_select > 0:
            for i in np.random.choice(n_select, n_city, replace=True):
                city = cities.iloc[i]
                location = lla2ecef(city["lat"], city["lng"], self.radius)

                # 添加随机偏移（城市内部的空间分布）
                if self.location_offset > 0:
                    offset = np.random.normal(size=3)
                    offset /= np.linalg.norm(offset)
                    offset *= self.location_offset * np.random.rand()
                    location += offset
                    location /= np.linalg.norm(location)
                    location *= self.radius

                _sample_task(location, f"{city['city']}, {city['iso2']}".replace("'", ""))

        # 均匀分布任务（海上等）
        for _ in range(n_uniform):
            x = np.random.normal(size=3)
            x *= self.radius / np.linalg.norm(x)
            _sample_task(x, "task")


__doc_title__ = "STIN Task Scenarios"
__all__ = ["ComputationTask", "STINTaskScenario", "CityTaskScenario"]
